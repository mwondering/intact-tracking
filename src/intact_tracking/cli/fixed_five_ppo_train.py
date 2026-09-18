"""Train four fixed-class residual PPOs and an on-policy all-class baseline."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils import resolve_callable

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import capture_original_rewards, assert_rewards_unchanged
from intact_tracking.cli.residual_policy_train import _build_train_configuration, _seed_everything
from intact_tracking.environment.runtime import prepare_rollout, _load_saved_config
from intact_tracking.fixed_five_ppo import VERSION, ROLES, FixedPartition, FixedFivePPO, mirror_baseline_physics
from intact_tracking.fixed_five_ppo import randomize_paired_episode_phase, split_episode_endings
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.distributed import DistributedContext
from intact_tracking.limb_context_grouped_dr import PHYSICS_FIELDS
from intact_tracking.limb_context_dr import TRACKER_DR, configure_limb_dr, audit_limb_dr
from intact_tracking.limb_context_protocol import TRACKER, TRACKER_SHA256, FULL_DATASET
from intact_tracking.limb_context_terminations import configure_training_terminations, audit_runtime_terminations
from intact_tracking.memory350_inference import Memory350Inference, load_memory350_checkpoint
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.residual_uniform_protocol import configure_models, wrist_articulation, MAX_MASSES
from intact_tracking.rollout.mjlab_adapter import _sha256, _robot_raw_state
from intact_tracking.rollout.online import _capture_privileged_dynamics_targets, _capture_randomized_model_fields, _read_motion_resample_boundary


DEFAULT_ROOT = Path("runs/limb_context_20260917_all_fixed_dr_half_nominal_50s/analysis4")
DEFAULT_ENCODER = "runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/update_015000.pt"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--num-envs", type=int, default=4096, help="DR worlds per arm PER RANK; simulator uses twice this count on each GPU")
    p.add_argument("--iterations", type=int, default=1000, help="Target completed joint rounds, five independent PPO updates per round")
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--mini-batches", type=int, default=4)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--entropy-coef", type=float, default=.0002)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=121)
    p.add_argument("--tracker-checkpoint", default=TRACKER)
    p.add_argument("--context-checkpoint", default=DEFAULT_ENCODER)
    p.add_argument("--partition-reference", type=Path, default=DEFAULT_ROOT / "environment_assignments.npz")
    m = p.add_mutually_exclusive_group()
    m.add_argument("--motion-file")
    m.add_argument("--motion-path")
    p.add_argument("--resume", type=Path)
    return p


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write_progress(out, value):
    write_json(out / "progress.json", value)
    if out.name == "rank_00":
        write_json(out.parent / "progress.json", value)


def aggregate_record(record, system):
    if not system.distributed:
        return record
    rows = [None] * system.world_size
    torch.distributed.all_gather_object(rows, record)
    result = copy.deepcopy(record)
    result["world_size"] = system.world_size
    for key in ("collect_seconds", "learn_seconds"):
        result[key] = max(row[key] for row in rows)
    for role in ROLES:
        values = [row["roles"][role] for row in rows]
        total = sum(v["worlds"] for v in values)
        merged = {k: sum(v[k]*v["worlds"] for v in values)/total for k in
                  ("value", "surrogate", "entropy", "raw_advantage_std", "actor_lr", "critic_lr", "mean_step_reward")}
        merged.update(worlds=total, updates=record["completed_updates"],
                      transitions=sum(v["transitions"] for v in values),
                      episode_endings=sum(v["episode_endings"] for v in values),
                      timeouts=sum(v["timeouts"] for v in values),
                      failure_terminations=sum(v["failure_terminations"] for v in values))
        result["roles"][role] = merged
    return result


def physics_snapshot(env):
    result = _capture_randomized_model_fields(env)
    result["encoder_bias"] = env.scene["robot"].data.encoder_bias.detach().clone()
    result["privileged_physics"] = _capture_privileged_dynamics_targets(env).values.detach().clone()
    return result


def assert_physics(env, expected):
    for name, value in physics_snapshot(env).items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


@torch.inference_mode()
def warmup(env, wrapped, actor, checkpoint, n, reference, boundaries, out):
    if abs(env.step_dt - .02) > 1e-9:
        raise ValueError("The 100s protocol requires dt=.02")
    memory = Memory350Inference(checkpoint, n, batch_size=256, use_bfloat16=False)
    episode = torch.zeros(n, dtype=torch.long, device=env.device)
    count = torch.zeros(n, dtype=torch.long, device=env.device)
    sums = torch.zeros(n, 64, dtype=torch.float64, device=env.device)
    command = env.command_manager.get_term("motion")
    obs = wrapped.get_observations()
    query_motion = []
    started = time.monotonic()
    for step in range(1, 5001):
        before = {"robot_state": _robot_raw_state(env)[:n].clone(), "episode_id": episode.clone(),
                  "episode_step": env.episode_length_buf[:n].clone(),
                  "motion_id": command.motion_idx[:n].clone(), "motion_step": command.time_steps[:n].clone()}
        action = actor.tracker(obs)
        env.action_manager.get_term("joint_pos").record_policy_mean(action)
        obs, _, dones, _ = wrapped.step(action)
        boundary = dones.bool() | _read_motion_resample_boundary(command, dones.bool())
        before.update(next_robot_state=_robot_raw_state(env)[:n].clone(),
                      joint_target=env.scene["robot"].data.joint_pos_target[:n].clone(), reset_boundary=boundary[:n])
        memory.append(before)
        episode += boundary[:n].long()
        if step % 100 == 0:
            bank = memory.memory
            valid = (bank.short_count == 50) & ((bank.total_chunks - bank.session_start) >= 30)
            z = memory.encode().double()
            z /= torch.linalg.vector_norm(z, dim=-1, keepdim=True)
            if not bool(torch.isfinite(z).all()):
                raise RuntimeError("Nonfinite warmup latent")
            sums[valid] += z[valid]
            count += valid.long()
            query_motion.append(command.motion_idx[:n].cpu().clone())
            progress = {"phase": "tracker_warmup", "step": step, "total_steps": 5000,
                        "simulated_seconds_per_world": step*env.step_dt, "valid_queries_min": int(count.min()),
                        "elapsed_seconds": time.monotonic()-started}
            write_progress(out, progress)
            if step % 500 == 0 and int(os.environ.get("RANK", "0")) == 0:
                print(json.dumps(progress), flush=True)
    partition = FixedPartition.from_sums(sums, count, reference, boundaries)
    atomic_save(out / "partition.pt", partition.state_dict())
    np.savez_compressed(out / "partition.npz", centers=partition.centers.cpu().numpy(),
                        distance=torch.linalg.vector_norm(partition.centers-reference, dim=-1).cpu().numpy(),
                        class_id=partition.labels.cpu().numpy(), full_query_count=count.cpu().numpy(),
                        query_motion=torch.stack(query_motion).numpy(),
                        nominal_center=reference.cpu().numpy(), boundaries=boundaries.cpu().numpy())
    return partition


def main():
    args = build_parser().parse_args()
    source_hashes = {str(Path(p).resolve()): _sha256(Path(p)) for p in
                     (__file__, str(Path(__file__).parents[1] / "fixed_five_ppo.py"))}
    distributed = DistributedContext.initialize(requested_device=args.device)
    args.device = str(distributed.device)
    if min(args.num_envs, args.iterations, args.epochs, args.mini_batches, args.save_interval) < 1 or args.rollout_steps < 2:
        raise ValueError("Positive counts and at least two rollout steps are required")
    if not args.device.startswith("cuda"):
        raise ValueError("Use mjwarp GPU execution")
    root = args.output_dir.resolve()
    out = root / f"rank_{distributed.rank:02d}" if distributed.enabled else root
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    configure_policy_precision("fp32")
    rank_seed = args.seed + 1000003 * distributed.rank
    _seed_everything(rank_seed)
    tracker = Path(args.tracker_checkpoint).resolve()
    if _sha256(tracker) != TRACKER_SHA256:
        raise ValueError("Wrong tracker checkpoint")
    context = load_memory350_checkpoint(args.context_checkpoint, device=args.device, expected_tracker_sha256=TRACKER_SHA256)
    reference_meta = json.loads(args.partition_reference.with_name("summary.json").read_text())
    if reference_meta["protocol"]["checkpoint_sha256"] != context.sha256:
        raise ValueError("Distance reference was calibrated using a different encoder")
    with np.load(args.partition_reference) as saved:
        reference = torch.as_tensor(saved["nominal_center"], dtype=torch.float64, device=args.device)
        boundaries = torch.as_tensor(saved["edges"][1:-1], dtype=torch.float64, device=args.device)
    resume = args.resume
    if resume and distributed.enabled:
        if resume.is_dir():
            resume = resume / f"rank_{distributed.rank:02d}" / "checkpoint_final.pt"
        elif resume.parent.name.startswith("rank_"):
            resume = resume.parent.parent / f"rank_{distributed.rank:02d}" / resume.name
        else:
            raise ValueError("Distributed resume requires the run directory or a rank checkpoint")
    saved_run = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if saved_run:
        if saved_run["metadata"].get("world_size", 1) != distributed.world_size:
            raise ValueError("Resume changed the number of training ranks")
        old = saved_run["metadata"]["arguments"]
        for key in arguments.keys() - {"resume", "output_dir", "iterations", "save_interval", "device"}:
            if old[key] != arguments[key]:
                raise ValueError(f"Resume changed {key}")
        if saved_run["metadata"]["context_sha256"] != context.sha256:
            raise ValueError("Resume changed encoder")
        torch.testing.assert_close(saved_run["partition"]["reference"], reference.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(saved_run["partition"]["boundaries"], boundaries.cpu(), rtol=0, atol=0)
    n = args.num_envs
    prepared = prepare_rollout(checkpoint_file=str(tracker), num_envs=2*n,
                               motion_path=None if args.motion_file else (args.motion_path or FULL_DATASET),
                               motion_file=args.motion_file)
    prepared.env.seed = rank_seed
    prepared.env.episode_length_s = 20.
    prepared.env.scene.entities["robot"].articulation = wrist_articulation(prepared.env.scene.entities["robot"].articulation)
    source = _load_saved_config(tracker)
    rewards = capture_original_rewards(prepared.env)
    physics = configure_limb_dr(prepared.env, rank_seed, profile=TRACKER_DR,
                                max_masses_kg=MAX_MASSES, nominal_probability=.5)
    configure_training_starts(prepared.env, "original")
    termination = configure_training_terminations(prepared.env, source, "original")
    assert_rewards_unchanged(rewards, prepared.env)
    train = _build_train_configuration(source, tracker_checkpoint=tracker,
        tracker_actor_kwargs=prepared.actor_kwargs, tracker_obs_groups=prepared.obs_groups,
        baseline="no-latent", dynamics_latent_dim=64, residual_hidden_dims=(512, 256, 128), residual_scale=1.,
        iterations=args.iterations, num_steps_per_env=args.rollout_steps, save_interval=args.save_interval,
        seed=args.seed, logger="tensorboard", wandb_project="intact-preview-v2",
        actor_learning_rate=args.actor_lr, critic_learning_rate=args.critic_lr, check_for_nan=True)
    train = configure_models(train, "baseline", scratch_seed=args.seed)
    train["algorithm"].update(schedule="fixed", entropy_coef=args.entropy_coef,
        num_learning_epochs=args.epochs, num_mini_batches=args.mini_batches, adaptive_critic_learning_rate=False)
    _seed_everything(rank_seed)
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=args.device)
    try:
        run(env, prepared, train, context, reference, boundaries, physics, rewards,
            termination, args, arguments, saved_run, out, source_hashes)
    finally:
        env.close()
        distributed.close()


def run(env, prepared, train, context, reference, boundaries, physics, rewards,
        termination, args, arguments, saved_run, out, source_hashes):
    n = args.num_envs
    pairing = mirror_baseline_physics(env)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
    obs = wrapped.get_observations()
    physical_audit = audit_limb_dr(env, physics)
    audit_runtime_terminations(env.termination_manager, termination)
    initial_physics = physics_snapshot(env)
    # Derived GPU-reduction constants need not be bitwise reproducible across
    # fresh simulator processes. Resume identity uses all physical inputs.
    physical_sha = tensor_digest(sorted((k, v) for k, v in initial_physics.items()
                                       if k in (*PHYSICS_FIELDS, "encoder_bias", "privileged_physics")))
    for value in initial_physics.values():
        if value.ndim and value.shape[0] == 2*n:
            torch.testing.assert_close(value[:n], value[n:], atol=0, rtol=0)
    actor_cfg, critic_cfg = copy.deepcopy(train["actor"]), copy.deepcopy(train["critic"])
    actor_type = resolve_callable(actor_cfg.pop("class_name"))
    critic_type = resolve_callable(critic_cfg.pop("class_name"))
    actor = actor_type(obs, train["obs_groups"], "actor", 29, **actor_cfg).to(args.device)
    critic = critic_type(obs, train["obs_groups"], "critic", 1, **critic_cfg).to(args.device)
    tracker_state_sha = tensor_digest(actor.tracker.state_dict().items())
    if saved_run:
        if saved_run["metadata"]["physics_sha256"] != physical_sha:
            raise ValueError("Resume regenerated different physical environments")
        partition = FixedPartition.from_state_dict(saved_run["partition"], args.device)
    else:
        partition = warmup(env, wrapped, actor, context, n, reference, boundaries, out)
    assert_physics(env, initial_physics)
    resume_update = int(saved_run["five_ppo"]["completed_updates"]) if saved_run else 0
    phase_seed = args.seed + 1000003 * int(os.environ.get("RANK", "0")) + 70000027 + resume_update
    phase = randomize_paired_episode_phase(env, seed=phase_seed)
    phase_audit = {"version": "paired_uniform_episode_phase_v1", "seed": phase_seed,
                   "max_episode_steps": env.max_episode_length, "completed_updates_at_start": resume_update,
                   "applied_after_fixed_partition": True, "expert_baseline_clocks_paired": True,
                   "initial_age_min": int(phase.min()), "initial_age_max": int(phase.max()),
                   "initial_age_mean": float(phase.float().mean())}
    np.savez_compressed(out / f"episode_phase_u{resume_update:06d}.npz", initial_age=phase.cpu().numpy(),
                        class_id=partition.labels.cpu().numpy(), max_episode_steps=env.max_episode_length)
    obs = wrapped.get_observations()
    system = FixedFivePPO(actor, critic, obs, partition.labels, rollout_steps=args.rollout_steps,
                          algorithm_cfg=train["algorithm"], device=args.device)
    if not saved_run:
        with torch.no_grad():
            for ids, alg in zip(system.indices, system.algorithms, strict=True):
                if len(ids):
                    selected = obs[ids]
                    actual = alg.actor(selected)
                    torch.testing.assert_close(actual, alg.actor.last_base_action, rtol=0, atol=0)
                    alg.actor.distribution.update(actual)
                    torch.testing.assert_close(alg.actor.output_std, torch.full_like(actual, .25), rtol=0, atol=0)
    if saved_run:
        system.load_state_dict(saved_run["five_ppo"])
    rank_agreement = system.audit_rank_agreement()
    motion_count = torch.tensor(len(env.command_manager.get_term("motion").motion_files), device=args.device)
    if system.distributed:
        torch.distributed.all_reduce(motion_count)
    metadata = {"version": VERSION, "arguments": arguments, "training_configuration": train,
        "rank": system.rank, "world_size": system.world_size, "rank_agreement": rank_agreement,
        "global_world_counts": system.global_counts,
        "global_motion_count": int(motion_count),
        "episode_phase": phase_audit,
        "termination_metrics": "episode_endings = timeouts + failure_terminations; failures take precedence on overlap",
        "context_sha256": context.sha256, "tracker_sha256": TRACKER_SHA256,
        "tracker_state_sha256": tracker_state_sha,
        "physics_sha256": physical_sha, "pairing": pairing, "physics": physical_audit,
        "reward_contract": rewards, "termination": termination,
        "independence": system.initial_audit, "warmup_seconds": 100,
        "partition_fixed_through_resets_and_updates": True,
        "baseline_data": "Independent actions and trajectories in matched replicas of all four classes",
        "equal_interactions_per_arm": True, "wrist_pitch_yaw_effort_limit_nm": 10.,
        "motion_files": list(env.command_manager.get_term("motion").motion_files),
        "source_sha256": source_hashes}
    write_json(out / "run_config.json", metadata)
    if system.distributed and system.rank == 0:
        write_json(out.parent / "run_config.json", metadata)
    np.savez_compressed(out / "physics.npz", parameters=initial_physics["privileged_physics"].cpu().numpy(),
                        encoder_bias=initial_physics["encoder_bias"].cpu().numpy(),
                        class_id=partition.labels.cpu().numpy())
    stop = [False]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    def save(filename):
        assert_physics(env, initial_physics)
        torch.testing.assert_close(system.labels, partition.labels, atol=0, rtol=0)
        if tensor_digest(actor.tracker.state_dict().items()) != tracker_state_sha:
            raise RuntimeError("The shared frozen tracker changed during PPO")
        agreement = system.audit_rank_agreement()
        atomic_save(out / filename, {"metadata": metadata, "partition": partition.state_dict(),
                                    "five_ppo": system.state_dict(), "rank_agreement": agreement})
        if system.distributed:
            torch.distributed.barrier()
            if system.rank == 0:
                write_json(out.parent / Path(filename).with_suffix(".json"),
                           {"completed_updates": system.completed_updates, "rank_agreement": agreement,
                            "rank_checkpoints": [f"rank_{rank:02d}/{filename}" for rank in range(system.world_size)]})
    try:
        if not saved_run:
            save("checkpoint_initial.pt")
        obs = system.prepare_observations(obs)
        while system.completed_updates < args.iterations:
            if system.distributed:
                stop_tensor = torch.tensor(int(stop[0]), device=args.device)
                torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
                stop[0] = bool(stop_tensor)
            if stop[0]:
                break
            start = time.monotonic()
            reward_sums = [0.] * 5
            done_sums = [0] * 5
            timeout_sums = [0] * 5
            failure_sums = [0] * 5
            with torch.inference_mode():
                for _ in range(args.rollout_steps):
                    actions, means = system.act(obs)
                    env.action_manager.get_term("joint_pos").record_policy_mean(means)
                    obs, reward, dones, extras = wrapped.step(actions)
                    timeouts, failures = split_episode_endings(dones, env.reset_time_outs, env.reset_terminated)
                    if not bool(torch.isfinite(reward).all()) or not all(bool(torch.isfinite(v).all()) for v in obs.values()):
                        raise RuntimeError("Nonfinite simulator observations/rewards")
                    obs = system.prepare_observations(obs)
                    system.process_step(obs, reward, dones, extras)
                    for i, ids in enumerate(system.indices):
                        reward_sums[i] += float(reward[ids].sum())
                        done_sums[i] += int(dones[ids].sum())
                        timeout_sums[i] += int(timeouts[ids].sum())
                        failure_sums[i] += int(failures[ids].sum())
            collect = time.monotonic()-start
            start = time.monotonic()
            metrics = system.update(obs)
            record = {"phase": "ppo", "completed_updates": system.completed_updates,
                      "collect_seconds": collect, "learn_seconds": time.monotonic()-start,
                      "roles": metrics}
            for i, role in enumerate(ROLES):
                denom = max(1, len(system.indices[i])*args.rollout_steps)
                record["roles"][role].update(mean_step_reward=reward_sums[i]/denom, episode_endings=done_sums[i],
                                              timeouts=timeout_sums[i], failure_terminations=failure_sums[i])
                if done_sums[i] != timeout_sums[i] + failure_sums[i]:
                    raise RuntimeError(f"Episode-ending accounting mismatch for {role}")
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            combined = aggregate_record(record, system)
            write_json(out / "progress.json", record)
            if system.distributed and system.rank == 0:
                with (out.parent / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(combined, allow_nan=False) + "\n")
                write_json(out.parent / "progress.json", combined)
            if system.rank == 0:
                print(json.dumps(combined), flush=True)
            if system.completed_updates == resume_update + 1:
                agreement = system.audit_rank_agreement()
                if system.rank == 0:
                    audit_name = "first_update_rank_agreement.json" if not saved_run else f"resume_u{resume_update:06d}_rank_agreement.json"
                    write_json((out.parent if system.distributed else out) / audit_name, agreement)
            if system.completed_updates % args.save_interval == 0:
                save(f"checkpoint_update_{system.completed_updates:06d}.pt")
        save("checkpoint_final.pt")
        state = {"complete": system.completed_updates >= args.iterations,
                   "completed_updates": system.completed_updates, "stopped": stop[0],
                   "independence_audit": system.audit_independence(),
                   "world_size": system.world_size, "global_world_counts": system.global_counts,
                   "fixed_physics_and_assignments_verified": True,
                   "resume": "Restore five models/optimizers and fixed assignments; simulator episodes restart"}
        write_json(out / "state.json", state)
        if system.distributed and system.rank == 0:
            write_json(out.parent / "state.json", state)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
