"""Matched residual PPO with original inputs and optional frozen context."""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import capture_original_rewards, assert_rewards_unchanged
from intact_tracking.cli.residual_policy_train import (
    _build_train_configuration, _prepare_output, _seed_everything,
)
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.distributed import DistributedContext
from intact_tracking.limb_context_distributed import (
    audit_rank_agreement, main_process_call, synchronize_critic_normalization, tensor_digest,
)
from intact_tracking.limb_context_env import LimbContextWrapper
from intact_tracking.limb_context_policy import (
    FUSIONS, SCRATCH_INITIALIZATION, SCRATCH_ACTION_STD, configure_context_models,
)
from intact_tracking.limb_context_sampling import (
    SamplingCheckpoint, checkpoint_configuration, configure_motion_sampling,
    phase_budget, state_digest, validate_sampling_resume,
)
from intact_tracking.limb_context_terminations import (
    PROFILES, audit_runtime_terminations, configure_training_terminations,
    resolve_profile, validate_termination_resume,
)
from intact_tracking.limb_context_protocol import (
    VERSION, FULL_DATASET, TRACKER, TRACKER_SHA256, PROJECT_ROOT,
)
from intact_tracking.limb_context_dr import (
    DR_PROFILES, TRACKER_DR, configure_limb_dr, audit_limb_dr, resolve_dr_profile, validate_context_dr,
)
from intact_tracking.preview_protocol import dataset_identity
from intact_tracking.residual_context import load_frozen_context_checkpoint
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import _sha256


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", choices=FUSIONS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-checkpoint")
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--dr-profile", choices=DR_PROFILES,
                        help="Fresh runs default to legacy load_only; resumes inherit their saved DR profile")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-file")
    motion.add_argument("--motion-path")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Target total COMPLETED PPO updates (also on resume)")
    parser.add_argument("--num-envs", type=int, default=8192,
                        help="Environment count PER GPU/process")
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--resume")
    parser.add_argument("--training-terminations", choices=PROFILES,
                        help="New experiments default to no_ee_body_pos; pinned experiments/resumes preserve their profile")
    parser.add_argument("--motion-sampling", choices=("uniform", "adaptive"), default="uniform")
    parser.add_argument("--adaptive-after-update", type=int, default=1000,
                        help="Uniform prefix, then restart from this exact checkpoint with adaptive sampling")
    parser.add_argument("--endpoint-eval-protocol",
                        help="Fixed 0/4 kg endpoint evaluation at each 100 completed updates")
    return parser


def audit_initial_models(actor, critic, obs, fusion):
    with torch.no_grad():
        features, base_action = actor._base_features_and_action(obs)
        action = actor(obs)
        torch.testing.assert_close(action, base_action, atol=0, rtol=0)
        torch.testing.assert_close(action, actor.tracker(obs), atol=0, rtol=0)
        actor.distribution.update(action)
        torch.testing.assert_close(actor.output_std, torch.full_like(actor.output_std, SCRATCH_ACTION_STD),
                                   atol=0, rtol=0)
        normal = critic.obs_normalizer(critic._flat_obs(obs))
        original = critic.mlp if fusion == "baseline" else critic.mlp.base
        torch.testing.assert_close(critic(obs), original(normal), atol=1e-6, rtol=1e-6)
        if fusion != "baseline":
            alternate = obs.clone()
            alternate["dynamics_latent"] = torch.randn_like(alternate["dynamics_latent"])
            torch.testing.assert_close(actor(alternate), action, atol=0, rtol=0)
            torch.testing.assert_close(critic(alternate), critic(obs), atol=1e-6, rtol=1e-6)
    return {"actor_original_features": features.shape[-1], "critic_original_features": critic.obs_dim,
            "latent_dimensions": 0 if fusion == "baseline" else 64,
            "original_obs_groups": {"actor": list(actor.tracker.obs_groups),
                                    "critic": list(critic.obs_groups)},
            "identical_initial_action": True, "identical_initial_value": True,
            "action_std_initialization": SCRATCH_ACTION_STD,
            "actor_initialization_seed": actor.initialization_seed,
            "critic_initialization_seed": critic.initialization_seed,
            "actor_common_trunk_sha256": tensor_digest(
                (actor.residual_mlp if fusion == "baseline" else actor.residual_mlp.base).named_parameters()),
            "critic_common_trunk_sha256": tensor_digest(original.named_parameters()),
            "critic_initial_normalizer_sha256": tensor_digest(critic.obs_normalizer.state_dict().items()),
            "critic_initial_normalizer_count": float(critic.obs_normalizer.count),
            "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
            "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad)}


def attach_json_logger(runner, output, distributed):
    original = runner.logger.log
    original_process = runner.logger.process_env_step

    def process(*args, **kwargs):
        # RSL guards buffer collection on writer presence. Non-main ranks need
        # those buffers too, but must never create a writer or checkpoint.
        writer = runner.logger.writer
        if writer is None:
            runner.logger.writer = object()
        try:
            return original_process(*args, **kwargs)
        finally:
            runner.logger.writer = writer

    runner.logger.process_env_step = process

    def log(**kw):
        losses = distributed.mean_scalars({k: float(v) for k, v in kw["loss_dict"].items()})
        moments = torch.tensor([sum(runner.logger.rewbuffer), len(runner.logger.rewbuffer),
                                sum(runner.logger.lenbuffer), len(runner.logger.lenbuffer)],
                               device=distributed.device, dtype=torch.float64)
        distributed.all_reduce_sum(moments)
        durations = torch.tensor([kw["collect_time"], kw["learn_time"]], device=distributed.device)
        if distributed.enabled:
            torch.distributed.all_reduce(durations, op=torch.distributed.ReduceOp.MAX)
        record = {"completed_updates": kw["it"] + 1, "unix_time": time.time(),
                  "world_size": distributed.world_size, "num_envs_per_rank": runner.env.num_envs,
                  "global_num_envs": runner.env.num_envs * distributed.world_size,
                  "global_transitions": (kw["it"] + 1) * runner.env.num_envs * distributed.world_size * runner.cfg["num_steps_per_env"],
                  "collect_seconds": float(durations[0]), "learn_seconds": float(durations[1]),
                  "learning_rate": kw["learning_rate"],
                  "loss": losses,
                  "action_std": float(kw["action_std"].mean()),
                  "mean_reward": float(moments[0] / moments[1]) if moments[1] else None,
                  "mean_episode_length": float(moments[2] / moments[3]) if moments[3] else None,
                  "episode_window_count_across_ranks": int(moments[1])}
        sampling = getattr(runner, "residual_metadata", {}).get("motion_sampling")
        terminations = getattr(runner, "residual_metadata", {}).get("training_terminations")
        if terminations:
            record["training_terminations"] = {
                "ee_body_pos_enabled": "ee_body_pos" not in terminations["disabled_terms"]}
        if sampling:
            command = runner.env.unwrapped.command_manager.get_term("motion")
            record["motion_sampling"] = {
                "adaptive_enabled": command.cfg.sampling_mode == "adaptive",
                "failure_rewind_enabled": command.cfg.rewind.enabled,
                "adaptive_after_update": sampling["adaptive_after_update"]}
        endpoint = getattr(runner, "pending_endpoint_evaluation", None)
        if endpoint is not None:
            record["endpoint_eval"] = endpoint
            record["evaluation_seconds"] = runner.pending_evaluation_seconds
        local_keys = sorted({key for row in runner.logger.ep_extras for key in row})
        keys = sorted({key for names in distributed.all_gather_object(local_keys) for key in names})
        if keys:
            stats = torch.zeros(len(keys), 2, dtype=torch.float64, device=distributed.device)
            for i, key in enumerate(keys):
                values = [torch.as_tensor(row[key], device=distributed.device).reshape(-1)
                          for row in runner.logger.ep_extras if key in row]
                if values:
                    flat = torch.cat(values).double()
                    stats[i, 0], stats[i, 1] = flat.sum(), flat.numel()
            distributed.all_reduce_sum(stats)
            record["episode_metrics"] = {key: float(stats[i, 0] / stats[i, 1])
                                          for i, key in enumerate(keys) if stats[i, 1]}
        if distributed.is_main:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        runner.pending_endpoint_evaluation = None
        runner.pending_evaluation_seconds = 0.
        original(**{**kw, "loss_dict": losses, "collect_time": float(durations[0]), "learn_time": float(durations[1])})
        if not distributed.is_main:
            runner.logger.ep_extras.clear()

    runner.logger.log = log


def _run(args, distributed):
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    for key in ("iterations", "num_envs", "rollout_steps", "epochs", "mini_batches", "save_interval"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if (args.fusion in ("film", "concat")) != bool(args.context_checkpoint):
        raise ValueError("Film/concat require a context checkpoint; baseline/constant do not")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Outputs must remain in the project directory")
    def validate_output():
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(output)
    main_process_call(distributed, validate_output)
    device = str(distributed.device)
    previous = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    args.dr_profile = resolve_dr_profile(args.dr_profile, previous["residual_policy"] if previous else None)
    resume_update = previous["completed_updates"] if previous else 0
    args.training_terminations = resolve_profile(args.training_terminations, output, PROJECT_ROOT, previous)
    rank_seed = args.seed + 1000003 * distributed.rank
    motion_file = str(Path(args.motion_file).resolve()) if args.motion_file else None
    motion_path = None if motion_file else str(Path(args.motion_path or FULL_DATASET).resolve())
    files, dataset = dataset_identity(motion_file, motion_path)
    tracker = Path(args.tracker_checkpoint).resolve()
    tracker_hash = _sha256(tracker)
    if tracker_hash != TRACKER_SHA256:
        raise ValueError("Wrong frozen tracker checkpoint")
    _seed_everything(rank_seed)
    prepared = prepare_rollout(checkpoint_file=str(tracker), num_envs=args.num_envs,
                               motion_file=motion_file, motion_path=motion_path)
    source = _load_saved_config(tracker)
    rewards = capture_original_rewards(prepared.env)
    physics = configure_limb_dr(prepared.env, rank_seed, profile=args.dr_profile)
    terminations = configure_training_terminations(prepared.env, source, args.training_terminations)
    sampling = configure_motion_sampling(prepared.env, args.motion_sampling,
                                         args.adaptive_after_update, resume_update)
    prepared.env.seed = rank_seed
    starts = configure_training_starts(prepared.env, "original" if args.dr_profile == TRACKER_DR else "reference")
    train = _build_train_configuration(
        source, tracker_checkpoint=tracker, tracker_actor_kwargs=prepared.actor_kwargs,
        tracker_obs_groups=prepared.obs_groups, baseline="no-latent", dynamics_latent_dim=64,
        residual_hidden_dims=(512, 256, 128), residual_scale=args.residual_scale,
        iterations=args.iterations, num_steps_per_env=args.rollout_steps,
        save_interval=args.save_interval, seed=args.seed, logger="tensorboard",
        wandb_project="limb-context", actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr, check_for_nan=True,
    )
    train = configure_context_models(train, args.fusion, scratch_seed=args.seed)
    train["algorithm"].update(schedule="fixed", entropy_coef=args.entropy_coef,
                              num_learning_epochs=args.epochs, num_mini_batches=args.mini_batches,
                              adaptive_critic_learning_rate=False,
                              class_name="intact_tracking.limb_context_distributed:DistributedResidualPPO")
    context = (load_frozen_context_checkpoint(args.context_checkpoint, device=device,
                                              expected_tracker_sha256=tracker_hash)
               if args.context_checkpoint else None)
    if context is not None:
        context_meta = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)
        validate_context_dr(context_meta, args.dr_profile)
        del context_meta
    if context is not None and context.config.dynamics_latent_dim != 64:
        raise ValueError("This experiment requires a 64-dimensional frozen context")
    metadata = {
        "version": VERSION, "fusion": args.fusion, "arguments": vars(args),
        "dr_profile": args.dr_profile,
        "execution_protocol": "four_gpu_8192_scratch_v3" if distributed.world_size == 4 else "two_gpu_8192_scratch_v3",
        "initialization_protocol": SCRATCH_INITIALIZATION,
        "distributed": {"world_size": distributed.world_size, "num_envs_per_rank": args.num_envs,
                        "global_num_envs": args.num_envs * distributed.world_size,
                        "rank_seed_formula": "seed + 1000003 * rank",
                        "gradient_synchronization": "mean across all ranks at each optimizer step",
                        "advantage_normalization": "global rollout mean and sample standard deviation",
                        "critic_normalization": "global pooled sums, squared sums and count at every environment step"},
        "tracker_checkpoint": str(tracker), "tracker_sha256": tracker_hash,
        "context_checkpoint": context.path if context else None,
        "context_sha256": context.sha256 if context else None,
        "dataset": dataset, "motion_count": len(files), "motion_file": motion_file,
        "motion_path": motion_path, "physics": physics, "training_starts": starts,
        "reward_contract": rewards, "reward_changes": {},
        "motion_sampling": sampling,
        "training_terminations": terminations,
        "actor_initialization": "fresh residual MLP; random hidden layers, zero output layer, fresh scalar action std 0.25",
        "critic_initialization": "fresh random MLP; no source critic weights",
        "critic_normalization_initialization": "fresh moments from one pooled initial observation batch; then updated online",
        "initialization_seeds": {"actor": args.seed + 10007, "critic": args.seed + 20003},
        "frozen_tracker_role": "pretrained base action and original 1645-dimensional preprocessing",
        "encoder_frozen": True, "context_normalization_frozen": True,
        "predictor_executed_in_ppo": False, "extra_current_state_physics_privilege": False,
        "stop_contract": "finish current update and save; total completed count is authoritative",
        "resume_contract": "restore model optimizer normalizers and update count; restart simulator episodes",
        "research_source_sha256": {str(p.relative_to(PROJECT_ROOT)): _sha256(p) for p in (
            Path(__file__).resolve(), *sorted((PROJECT_ROOT / "src/intact_tracking").glob("limb_context_*.py")),
            PROJECT_ROOT / "src/intact_tracking/residual_policy.py",
            PROJECT_ROOT / "src/intact_tracking/residual_runner.py")},
    }
    endpoint_evaluator = None
    if args.endpoint_eval_protocol:
        from intact_tracking.limb_context_checkpoint_eval import PeriodicEndpointEvaluator, digest
        endpoint_evaluator = PeriodicEndpointEvaluator(output, args.endpoint_eval_protocol, distributed)
        metadata["periodic_evaluation"] = {"protocol": endpoint_evaluator.protocol,
            "protocol_file": str(Path(args.endpoint_eval_protocol).resolve()),
            "protocol_sha256": digest(args.endpoint_eval_protocol),
            "checkpoint_counting": "completed PPO updates; checkpoint_update_XXXXXX.pt",
            "execution": "all ranks pause PPO; separate endpoint simulators on the first two allocated GPUs"}
    if args.resume:
        old = previous["residual_policy"]
        validate_termination_resume(previous, terminations)
        previous_sampling_mode = validate_sampling_resume(old, sampling, previous["completed_updates"])
        expected_resume_digest = state_digest(previous.get("rsl_rl", previous))
        for key in ("version", "fusion", "tracker_sha256", "context_sha256", "reward_contract", "initialization_protocol"):
            if old.get(key) != metadata[key]:
                raise ValueError(f"Resume changed {key}")
        if old["dataset"]["manifest_sha256"] != dataset["manifest_sha256"]:
            raise ValueError("Resume changed dataset")
        for key in ("seed", "num_envs", "rollout_steps"):
            if old["arguments"][key] != vars(args)[key]:
                raise ValueError(f"Resume changed {key}")
        if old.get("distributed") != metadata["distributed"]:
            raise ValueError("Resume changed distributed training scale")
        old_train = OmegaConf.to_container(previous["cfg"].agent, resolve=True)
        for key in ("actor", "critic", "algorithm", "obs_groups"):
            if old_train[key] != train[key]:
                raise ValueError(f"Resume changed {key}")
        metadata["resume_history"] = [*copy.deepcopy(old.get("resume_history", [])), {
            "checkpoint": str(Path(args.resume).resolve()), "checkpoint_sha256": _sha256(Path(args.resume)),
            "completed_updates": previous["completed_updates"], "unix_time": time.time(),
            "motion_sampling_from": previous_sampling_mode, "motion_sampling_to": sampling["active_mode"],
            "model_optimizer_and_normalization_restored": True, "simulator_episodes_restarted": True}]
        if endpoint_evaluator is not None:
            original = old.get("periodic_evaluation", {})
            if original and original["protocol_sha256"] != metadata["periodic_evaluation"]["protocol_sha256"]:
                raise ValueError("Resume changed the periodic endpoint protocol")
            metadata["periodic_evaluation"]["enabled_after_update"] = original.get(
                "enabled_after_update", previous["completed_updates"])
        del previous
    elif endpoint_evaluator is not None:
        metadata["periodic_evaluation"]["enabled_after_update"] = 0
    assert_rewards_unchanged(rewards, prepared.env)
    _seed_everything(rank_seed)
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=device)
    handlers = {}
    try:
        termination_audit = audit_runtime_terminations(env.termination_manager, terminations)
        metadata["training_termination_runtime_audits"] = distributed.all_gather_object(termination_audit)
        command = env.command_manager.get_term("motion")
        if command.cfg.sampling_mode != sampling["active_mode"] or command.cfg.rewind.enabled:
            raise ValueError("Runtime motion sampling differs from the recorded configuration")
        expected = files[distributed.rank::distributed.world_size] if len(files) > 1 else files
        if not distributed.all_true(tuple(map(str, expected)) == command.motion_files):
            raise ValueError("Loaded rank shard differs from the full requested motion catalog")
        rank_audit = {"rank": distributed.rank, "seed": rank_seed,
                      "motion_count": command.motion.num_files,
                      "loaded_frames": int(command.motion.file_lengths.sum()),
                      "physics": audit_limb_dr(env, physics)}
        ranks = distributed.all_gather_object(rank_audit)
        dataset.update(loaded_motion_count=sum(r["motion_count"] for r in ranks) if len(files) > 1 else 1,
                       loaded_frames=sum(r["loaded_frames"] for r in ranks) if len(files) > 1 else ranks[0]["loaded_frames"],
                       rank_partition="sorted files [rank::world_size]; single-file smoke replicated",
                       loaded_motion_counts_per_rank=[r["motion_count"] for r in ranks],
                       reference_storage_mode=command.cfg.reference_storage_mode)
        physics["runtime_audits_by_rank"] = ranks
        physics = distributed.broadcast_object(physics)
        metadata["physics"] = physics
        if dataset["loaded_motion_count"] != len(files):
            raise ValueError("Distributed motion shards do not cover the full dataset")
        wrapped = (RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions) if args.fusion == "baseline"
                   else LimbContextWrapper(env, prepared.clip_actions, context))
        _seed_everything(args.seed)
        cfg = checkpoint_configuration(source, train, metadata)
        main_process_call(distributed, lambda: output.mkdir(parents=True, exist_ok=True))
        runner = ResidualOnPolicyRunner(wrapped, train, str(output), device,
                                       checkpoint_cfg=cfg, residual_metadata=metadata)
        if distributed.enabled:
            synchronize_critic_normalization(runner.alg.critic)
        initial_obs = wrapped.get_observations()
        runner.alg.critic.train()
        # Bootstrap fresh statistics from the current DR worlds before the first
        # value estimate. Frozen tracker/context normalization is not changed.
        if not distributed.all_true(float(runner.alg.critic.obs_normalizer.count) == 0):
            raise ValueError("Scratch critic inherited old observation statistics")
        runner.alg.critic.update_normalization(initial_obs)
        metadata["input_audit"] = audit_initial_models(
            runner.alg.actor, runner.alg.critic, initial_obs, args.fusion)
        if args.resume:
            runner.load(args.resume, map_location=device)
            runner.current_learning_iteration = runner.completed_learning_updates
            restored_digest = state_digest(runner.alg.save())
            if not distributed.all_true(restored_digest == expected_resume_digest):
                raise ValueError("Model/optimizer/normalization restoration differs from the selected checkpoint")
            metadata["resume_state_audit"] = {"passed": True, "checkpoint_update": resume_update,
                                              "model_optimizer_state_sha256": restored_digest}
            metadata["resume_history"][-1]["restoration_audit"] = copy.deepcopy(metadata["resume_state_audit"])
            # Preserve the true original initialization audit across simulator restarts.
            metadata["input_audit"] = copy.deepcopy(old["input_audit"])
        sampler_checkpoint = SamplingCheckpoint(output, distributed, sampling)
        if args.resume:
            restored_sampling = sampler_checkpoint.restore(runner, previous_sampling_mode)
            metadata["sampling_resume_audits"] = distributed.all_gather_object(restored_sampling)
        runner.checkpoint_state_preparer = sampler_checkpoint.prepare
        cfg = checkpoint_configuration(source, train, metadata)
        runner.checkpoint_cfg, runner.residual_metadata = cfg, copy.deepcopy(metadata)
        main_process_call(distributed, lambda: _prepare_output(output, resume=args.resume, run_config=metadata, checkpoint_config=cfg))
        attach_json_logger(runner, output, distributed)
        if not args.resume:
            main_process_call(distributed, lambda: runner.save(str(output / "checkpoint_initial.pt")))

        def stop(signum, _frame):
            runner.request_stop()
            os.write(2, f"Signal {signum}: saving after the current complete PPO update.\n".encode())

        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, stop)
        if endpoint_evaluator is not None:
            runner.checkpoint_evaluator = endpoint_evaluator
            # The first evaluation after enabling this feature also covers the
            # current saved model. Older unsaved 100-update snapshots cannot be reconstructed.
            endpoint_evaluator(runner, force=True)
        _seed_everything(rank_seed + 1 + runner.completed_learning_updates)
        remaining = phase_budget(args.motion_sampling, args.adaptive_after_update,
                                 runner.completed_learning_updates, args.iterations)
        print(json.dumps({"phase": "ppo_ready", "fusion": args.fusion,
                          "input_audit": metadata["input_audit"], "remaining_updates": remaining,
                          "training_terminations": termination_audit,
                          "motion_sampling": sampling}), flush=True)
        if remaining:
            runner.learn(remaining, init_at_random_ep_len=True)
        agreement = audit_rank_agreement(runner, distributed)
        stopped = not distributed.all_true(not runner.stop_requested)
        planned_transition = (not stopped and args.motion_sampling == "adaptive"
                              and sampling["active_mode"] == "uniform"
                              and runner.completed_learning_updates == args.adaptive_after_update
                              and args.iterations > args.adaptive_after_update)
        if planned_transition:
            boundary = output / f"checkpoint_update_{runner.completed_learning_updates:06d}.pt"
            main_process_call(distributed, lambda: runner.save(str(boundary)) if not boundary.exists() else None)
        main_process_call(distributed, lambda: (output / "completion.json").write_text(json.dumps({
            "target_updates": args.iterations, "completed_updates": runner.completed_learning_updates,
            "complete": runner.completed_learning_updates >= args.iterations,
            "stopped": stopped, "distributed_parameter_agreement": agreement,
            "planned_sampling_transition": planned_transition, "motion_sampling": sampling,
            "training_terminations": terminations,
            "initialization_protocol": SCRATCH_INITIALIZATION,
            "distributed": metadata["distributed"], "unix_time": time.time()}, indent=2) + "\n"))
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        env.close()


def run(args):
    distributed = DistributedContext.initialize(requested_device=args.device,
                                                requested_backend=args.distributed_backend)
    if distributed.device.type == "cuda":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(distributed.local_rank)
    try:
        return _run(args, distributed)
    finally:
        distributed.close()


if __name__ == "__main__":
    run(build_parser().parse_args())
