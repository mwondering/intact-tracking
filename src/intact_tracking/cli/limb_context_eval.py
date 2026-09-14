"""Paired fixed-start evaluation with full failure, reward and history accounting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from intact_tracking.adaptation_policy import physics_observation
from intact_tracking.adaptation_reward_contract import capture_original_rewards, assert_fixed_reward_checkpoint
from intact_tracking.cli.adaptation_eval import METRICS, fixed_starts, reset_finished_worlds, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.limb_context_env import LimbContextWrapper
from intact_tracking.limb_context_policy import LimbContextResidualActor
from intact_tracking.limb_context_protocol import (
    VERSION, TRACKER, TRACKER_SHA256, PROJECT_ROOT, FULL_DATASET, PAYLOAD_EVENT,
)
from intact_tracking.limb_context_dr import (
    DR_PROFILES, TRACKER_DR, configure_limb_dr, audit_limb_dr, resolve_dr_profile,
)
from intact_tracking.residual_context import load_frozen_context_checkpoint
from intact_tracking.rollout.mjlab_adapter import _sha256

EVAL_PROTOCOL = "limb_context_paired_fixed_starts_v1"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--dr-profile", choices=DR_PROFILES,
                        help="Inherit residual checkpoint DR; specify explicitly for frozen-tracker evaluations")
    parser.add_argument("--motion-manifest", required=True)
    parser.add_argument("--motion-path", default=FULL_DATASET)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fixed-masses", nargs=4, type=float)
    parser.add_argument("--latent-mode", choices=("correct", "zero", "paired-swap"), default="correct")
    parser.add_argument("--paired-starts", action="store_true")
    return parser


def run(args):
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    output = Path(args.output).resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Evaluation outputs must remain in the project")
    if output.exists():
        raise FileExistsError("Use a fresh output or verify the existing evaluation externally")
    files = [str(Path(line).resolve()) for line in Path(args.motion_manifest).read_text().splitlines() if line.strip()]
    if not files or len(set(files)) != len(files) or any(not Path(p).is_file() for p in files):
        raise ValueError("Manifest must contain unique existing motions")
    n = len(files) * args.repeats
    if not 0 < n <= 4096 or args.steps <= 0:
        raise ValueError("Bound evaluations to 4096 worlds and a positive horizon")
    if (args.paired_starts or args.latent_mode == "paired-swap") and args.repeats != 2:
        raise ValueError("Matched donor diagnostics require two copies of each motion")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    args.dr_profile = resolve_dr_profile(args.dr_profile, state["residual_policy"] if state else None)
    if args.dr_profile == TRACKER_DR and _sha256(Path(args.tracker_checkpoint)) != TRACKER_SHA256:
        raise ValueError("Wrong frozen tracker checkpoint for the original DR profile")
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=args.tracker_checkpoint, num_envs=n,
                               motion_file=None, motion_path=args.motion_path)
    cfg = prepared.env
    rewards = capture_original_rewards(cfg)
    physics = configure_limb_dr(cfg, args.seed, profile=args.dr_profile, fixed_masses=args.fixed_masses)
    command_cfg = cfg.commands["motion"]
    command_cfg.motion_manifest_file = str(Path(args.motion_manifest).resolve())
    command_cfg.resample_on_motion_end = False
    command_cfg.resampling_time_range = (1e9, 1e9)
    command_cfg.init_noise, command_cfg.pose_range, command_cfg.velocity_range = {}, {}, {}
    command_cfg.joint_position_range = (0., 0.)
    cfg.seed, cfg.auto_reset = args.seed, False
    cfg.episode_length_s = (args.steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    fusion, context = "frozen", None
    if state:
        meta = state["residual_policy"]
        if meta["version"] != VERSION:
            raise ValueError("Only this experiment's residual checkpoints are allowed")
        assert_fixed_reward_checkpoint(state, rewards)
        fusion = meta["fusion"]
        if fusion in ("film", "concat"):
            context = load_frozen_context_checkpoint(meta["context_checkpoint"], device=args.device,
                                                      expected_tracker_sha256=meta["tracker_sha256"])
            if context.sha256 != meta["context_sha256"]:
                raise ValueError("Frozen context checkpoint changed")
    if args.latent_mode != "correct" and fusion not in ("film", "concat"):
        raise ValueError("Latent interventions require a learned-context policy")
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        physics["runtime_audit"] = audit_limb_dr(env, physics)
        command = env.command_manager.get_term("motion")
        if tuple(files) != tuple(command.motion_files):
            raise ValueError("Evaluation loaded a different motion catalog")
        ids, starts = fixed_starts(command.motion.file_lengths, args.repeats, args.steps, args.seed)
        if args.paired_starts or args.latent_mode == "paired-swap":
            _, single = fixed_starts(command.motion.file_lengths, 1, args.steps, args.seed)
            starts = single.repeat(2)
        ids, starts = ids.to(env.device), starts.to(env.device)

        def sample(self, env_ids):
            self.motion_idx[env_ids] = ids[env_ids]
            self.motion_length[env_ids] = self.motion.file_lengths[ids[env_ids]]
            self.time_steps[env_ids] = starts[env_ids]

        command._uniform_sampling = MethodType(sample, command)
        _seed_everything(args.seed + 1000000)
        wrapped = (LimbContextWrapper(env, prepared.clip_actions, context)
                   if fusion in ("film", "concat", "constant")
                   else RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions))
        obs = wrapped.get_observations()
        if state:
            agent = OmegaConf.to_container(state["cfg"].agent, resolve=True)
            kwargs = copy.deepcopy(agent["actor"])
            if kwargs.pop("class_name") != "intact_tracking.limb_context_policy:LimbContextResidualActor":
                raise ValueError("Actor checkpoint contract differs")
            actor = LimbContextResidualActor(obs, agent["obs_groups"], "actor", wrapped.num_actions, **kwargs)
            actor.to(env.device)
            actor.load_state_dict(state["actor_state_dict"], strict=True)
            completed_updates = state["completed_updates"]
            del state
        else:
            actor = load_actor(None, prepared, obs, wrapped)
            completed_updates = None
        actor.eval().requires_grad_(False)
        initial_steps = command.time_steps.clone()
        fingerprints = [hashlib.sha256(row.tobytes()).hexdigest()
                        for row in physics_observation(env).detach().cpu().numpy()]
        masses = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func.observe().cpu().tolist()
        horizons = (command.motion_length - initial_steps - 1).clamp(1, args.steps)
        totals = torch.zeros(n, len(METRICS), device=env.device, dtype=torch.float64)
        returns = torch.zeros(n, device=env.device, dtype=torch.float64)
        counts = torch.zeros(n, device=env.device, dtype=torch.long)
        failed = torch.zeros(n, device=env.device, dtype=torch.bool)
        active = ~failed
        names = [key for key in env.termination_manager.active_terms
                 if not env.termination_manager.get_term_cfg(key).time_out]
        failure_terms = torch.zeros(n, len(names), dtype=torch.bool, device=env.device)
        trace = torch.zeros(n, args.steps, 2, dtype=torch.float32, device=env.device)
        valid_context_steps = torch.zeros(n, device=env.device, dtype=torch.long)
        swapped_steps = torch.zeros_like(valid_context_steps)
        residual_saturation = torch.zeros(n, device=env.device, dtype=torch.float64)
        donor = torch.arange(n, device=env.device).roll(len(files))
        start_time = time.time()
        reset_batches = 0
        with torch.inference_mode():
            for step in range(args.steps):
                _seed_everything(args.seed + 2000000 + step)
                policy_obs = obs
                if context is not None:
                    full = wrapped.context.history_valid.all(0)
                    valid_context_steps += (active & full).long()
                    if args.latent_mode == "zero":
                        policy_obs = obs.clone(recurse=False)
                        policy_obs["dynamics_latent"] = torch.zeros_like(obs["dynamics_latent"])
                    elif args.latent_mode == "paired-swap":
                        # Swap only between active equal-phase worlds with complete histories.
                        eligible = active & active[donor] & full & full[donor]
                        eligible &= command.time_steps == command.time_steps[donor]
                        swapped_steps += eligible.long()
                        policy_obs = obs.clone(recurse=False)
                        policy_obs["dynamics_latent"] = torch.where(
                            eligible[:, None], obs["dynamics_latent"][donor], obs["dynamics_latent"])
                action = actor(policy_obs)
                if hasattr(actor, "last_residual_mean") and actor.last_residual_mean is not None:
                    fraction = (actor.last_residual_mean.abs() >= .95 * actor.residual_scale).float().mean(-1)
                    residual_saturation[active] += fraction[active].double()
                action = action.masked_fill(~active[:, None], 0.)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, reward, dones, extras = wrapped.step(action)
                if not torch.equal(command.time_steps[active], initial_steps[active] + step + 1):
                    raise RuntimeError("Surviving-world reference timeline drifted")
                command._update_metrics()
                values = torch.stack([command.metrics[key] for key in METRICS], -1)
                if not torch.isfinite(values[active]).all() or not torch.isfinite(reward[active]).all():
                    raise RuntimeError("Nonfinite evaluation metric/reward")
                totals[active] += values[active].double()
                trace[active, step] = values[active, :2]
                returns[active] += reward[active].double()
                counts += active.long()
                failed |= active & env.reset_terminated.bool()
                for i, name in enumerate(names):
                    failure_terms[:, i] |= active & env.termination_manager.get_term(name)
                active &= ~dones.bool() & (counts < horizons)
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if done_ids.numel():
                    reset_finished_worlds(env, done_ids, active)
                    reset_batches += 1
                if (step + 1) % 100 == 0:
                    print(json.dumps({"step": step + 1, "active": int(active.sum()),
                                      "failed": int(failed.sum()), "seconds": time.time() - start_time}), flush=True)
                if not active.any():
                    break
        means = totals / counts[:, None].clamp_min(1)
        result = {
            "protocol": EVAL_PROTOCOL, "arguments": vars(args), "fusion": fusion,
            "checkpoint": args.checkpoint or args.tracker_checkpoint,
            "checkpoint_sha256": _sha256(Path(args.checkpoint or args.tracker_checkpoint)),
            "completed_training_updates": completed_updates, "context_sha256": context.sha256 if context else None,
            "physics": physics, "physics_world_fingerprints": fingerprints, "actual_limb_masses_kg": masses,
            "seed": args.seed, "motions": len(files), "episodes": n, "repeats_per_motion": args.repeats,
            "motion_files": files, "motion_ids": ids.cpu().tolist(), "start_frames": initial_steps.cpu().tolist(),
            "horizons": horizons.cpu().tolist(), "max_steps": args.steps,
            "metric_names": list(METRICS), "per_episode_metrics": means.cpu().tolist(),
            "mean": dict(zip(METRICS, means.mean(0).cpu().tolist(), strict=True)),
            "episode_lengths": counts.cpu().tolist(), "episode_returns": returns.cpu().tolist(),
            "mean_episode_return": float(returns.mean()), "reward_contract": rewards,
            "failed": failed.cpu().tolist(), "failure_rate": float(failed.float().mean()),
            "failure_term_names": names, "failure_terms": failure_terms.cpu().tolist(),
            "coverage_fraction": float((counts.float() / horizons).mean()),
            "context_full_steps": valid_context_steps.cpu().tolist(), "swapped_steps": swapped_steps.cpu().tolist(),
            "residual_saturation_fraction": float((residual_saturation / counts.clamp_min(1)).mean()),
            "reference_timeline_audited": True, "partial_reset_survivor_state_audited": True,
            "partial_reset_survivor_history_audited": True, "partial_reset_batches": reset_batches,
            "latent_intervention": args.latent_mode,
            "metric_convention": "episode means and returns; errors are failure-truncated, report failures and coverage alongside",
            "wall_seconds": time.time() - start_time,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output.with_suffix(".traces.npz"), body_joint=trace.cpu().numpy(),
                            lengths=counts.cpu().numpy())
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
        print(json.dumps({k: result[k] for k in ("mean", "failure_rate", "coverage_fraction", "mean_episode_return")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    run(build_parser().parse_args())
