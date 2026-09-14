"""Five-step simulator oracle consistency and noninterference audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.cli.adaptation_eval import TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.rollout.mjlab_adapter import _reference_snapshot, _robot_raw_state
from intact_tracking.simulator_preview import FrozenTrackerPreview


def snapshot(env):
    result = {"sim/" + name: getattr(env.sim.data, name).clone() for name in FrozenTrackerPreview.DATA_FIELDS
              if hasattr(env.sim.data, name)}
    owners = {"env": env, "command": env.command_manager.get_term("motion"), "action": env.action_manager}
    owners.update({"action/" + name: value for name, value in env.action_manager._terms.items()})
    for group, cfgs in env.observation_manager._group_obs_class_term_cfgs.items():
        owners.update({f"observation/{group}/{index}": cfg.func for index, cfg in enumerate(cfgs)})
    for group, terms in env.observation_manager._group_obs_term_history_buffer.items():
        owners.update({f"history/{group}/{name}": value for name, value in terms.items()})
    for prefix, owner in owners.items():
        for name, value in vars(owner).items():
            if isinstance(value, torch.Tensor):
                result[f"{prefix}/{name}"] = value.clone()
    result["rng/cpu"] = torch.get_rng_state()
    result["rng/cuda"] = torch.cuda.get_rng_state(env.device)
    return result


def run(args):
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=TRACKER, motion_file=args.motion_file,
                               motion_path=args.motion_path, num_envs=args.num_envs)
    cfg = prepared.env
    physics = configure_physics(cfg, args.physics, args.dr_profile)
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.terminations = {}
    command = cfg.commands["motion"]
    command.sampling_mode = "uniform"
    command.rewind.enabled = False
    command.resample_on_motion_end = False
    command.resampling_time_range = (1e9, 1e9)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    preview = None
    try:
        wrapped = RslRlVecEnvWrapper(env, prepared.clip_actions)
        obs = wrapped.get_observations()
        actor = load_actor(None, prepared, obs, wrapped).eval().requires_grad_(False)
        preview = FrozenTrackerPreview(env, prepared.clip_actions, full_sim_state=args.full_sim_state)
        if args.physics == "dr" and args.dr_profile == "hands-shins-2-4kg":
            from intact_tracking.preview_protocol import audit_limb_payloads

            physics["runtime_payload_audit"] = audit_limb_payloads(env)
        records = []
        with torch.no_grad():
            for trial in range(args.trials):
                # Queries must work off the frozen tracker's own trajectory as well.
                for _ in range(8):
                    action = actor(obs) + 0.1 * torch.randn(env.num_envs, wrapped.num_actions, device=env.device)
                    for term in env.action_manager._terms.values():
                        term.record_policy_mean(action)
                    obs, *_ = wrapped.step(action)
                before = snapshot(env)
                preview.query(actor, obs)
                after = snapshot(env)
                changed = [name for name in before if not torch.equal(before[name], after[name])]
                if changed:
                    raise RuntimeError(f"Oracle changed real state/RNG: {changed}")
                predicted = preview.last_states.clone()
                predicted_actions = preview.last_actions.clone()
                predicted_bodies = preview.last_body_positions.clone()
                targets = preview.last_targets.clone()
                model_different = [name for name in preview.model_fields if not torch.equal(
                    getattr(env.sim.model, name), getattr(preview.shadow.sim.model, name)
                )]
                if model_different:
                    raise RuntimeError(f"Shadow physics mismatch: {model_different}")
                preview.query(actor, obs)
                repeat_error = float((predicted - preview.last_states).abs().max())
                states, actions, references, bodies, body_tracking = [], [], [], [], []
                for _ in range(5):
                    action = actor(obs)
                    for term in env.action_manager._terms.values():
                        term.record_policy_mean(action)
                    obs, *_ = wrapped.step(action)
                    states.append(_robot_raw_state(env))
                    actions.append(action.clone())
                    references.append(_reference_snapshot(env)[1])
                    command = env.command_manager.get_term("motion")
                    bodies.append(command.robot_body_pos_w.clone())
                    body_tracking.append((command.body_pos_relative_w - command.robot_body_pos_w).norm(dim=-1).mean(-1))
                difference = (torch.stack(states, 1) - predicted).abs()
                body_prediction = (torch.stack(bodies, 1) - predicted_bodies).norm(dim=-1)
                joint_prediction = difference[..., 13:42].norm(dim=-1)
                joint_tracking = (torch.stack(states, 1)[..., 13:42] - targets[..., 13:42]).norm(dim=-1)
                pose_indices = list(range(7)) + list(range(13, 42))
                records.append({
                    "trial": trial, "real_state_and_rng_unchanged": True,
                    "randomized_model_fields_identical": True,
                    "first_action_max": float((actions[0] - predicted_actions[:, 0]).abs().max()),
                    "body_prediction_mean_m": float(body_prediction.mean()),
                    "body_prediction_max_m": float(body_prediction.max()),
                    "joint_prediction_mean_l2": float(joint_prediction.mean()),
                    "joint_prediction_max_l2": float(joint_prediction.max()),
                    "body_prediction_to_tracking": float(body_prediction.mean() / torch.stack(body_tracking).mean().clamp_min(1e-6)),
                    "joint_prediction_to_tracking": float(joint_prediction.mean() / joint_tracking.mean().clamp_min(1e-6)),
                    "repeat_max": repeat_error,
                    "state_max_by_step": difference.amax(dim=(0, 2)).tolist(),
                    "pose_max": float(difference[..., pose_indices].max()),
                    "full_state_max": float(difference.max()),
                    "action_max": float((torch.stack(actions, 1) - predicted_actions).abs().max()),
                    "reference_max": float((torch.stack(references, 1) - targets).abs().max()),
                })
                print(json.dumps(records[-1]), flush=True)
        result = {"physics": args.physics, "motion_file": args.motion_file, "num_envs": args.num_envs,
                  "motion_path": args.motion_path, "motion_count": command.motion.num_files,
                  "physics_configuration": physics, "shared_immutable_motion": True,
                  "seed": args.seed, "horizon": 5, "full_sim_state": args.full_sim_state, "records": records,
                  "passed": all(r["pose_max"] <= 1e-5 and r["full_state_max"] <= 1e-4
                                and r["reference_max"] <= 1e-5 and r["repeat_max"] <= 1e-4 for r in records)}
        result["strict_replay_passed"] = result["passed"]
        result["bounded_preview_passed"] = all(
            r["first_action_max"] <= 1e-6 and r["reference_max"] <= 1e-5
            and r["body_prediction_to_tracking"] <= 0.01
            and r["joint_prediction_to_tracking"] <= 0.01
            and r["body_prediction_max_m"] <= 0.001
            and r["joint_prediction_max_l2"] <= 0.005
            for r in records
        )
        result["acceptance_mode"] = args.acceptance
        if args.acceptance == "bounded":
            result["passed"] = result["bounded_preview_passed"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            raise RuntimeError("Preview consistency threshold failed; do not start efficacy training")
    finally:
        if preview is not None:
            preview.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-file")
    motion.add_argument("--motion-path")
    parser.add_argument("--dr-profile", choices=("right-hand-1-3kg", "hands-shins-2-4kg"),
                        default="right-hand-1-3kg")
    parser.add_argument("--physics", choices=("nominal", "dr"), default="dr")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--full-sim-state", action="store_true")
    parser.add_argument("--acceptance", choices=("strict", "bounded"), default="strict")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
