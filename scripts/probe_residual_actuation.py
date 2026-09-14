"""Training-style diagnostic of residual bounds by joint; not an acceptance eval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_policy import (
    PrivilegedAdaptationWrapper,
    configure_oracle_proprioception,
    expanded_model_field,
)
from intact_tracking.adaptation_reward_contract import (
    assert_fixed_reward_checkpoint,
    capture_original_rewards,
)
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.rollout.mjlab_adapter import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19831)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.steps <= 50 or args.num_envs < 1:
        parser.error("Need positive environments and more than50 steps for the diagnostic window")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("MUJOCO_GL", "egl")
    _seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_config = checkpoint["cfg"].agent.actor
    if not actor_config["class_name"].endswith(":PrivilegedAdaptationActor"):
        raise ValueError("This diagnostic requires a privileged teacher")
    prepared = prepare_rollout(checkpoint_file=TRACKER, num_envs=args.num_envs, motion_path=DATASET, motion_file=None)
    assert_fixed_reward_checkpoint(checkpoint, capture_original_rewards(prepared.env))
    configure_physics(prepared.env, "dr")
    configure_training_starts(prepared.env, "reference")
    prepared.env.seed = args.seed
    prepared.env.commands["motion"].sampling_mode = "uniform"
    if actor_config.get("oracle_clean_proprio"):
        configure_oracle_proprioception(prepared.env)
    env = ManagerBasedRlEnv(cfg=prepared.env, device="cuda:0")
    try:
        wrapped = PrivilegedAdaptationWrapper(env, clip_actions=prepared.clip_actions,
                                              privilege_schema=actor_config["privilege_schema"])
        obs = wrapped.get_observations()
        actor = load_actor(str(args.checkpoint), prepared, obs, wrapped)
        action_term, robot = env.action_manager.get_term("joint_pos"), env.scene["robot"]
        controls = []
        for joint in robot.indexing.joint_ids[action_term.target_ids].cpu().tolist():
            ids = np.flatnonzero(env.sim.mj_model.actuator_trnid[:, 0] == joint)
            if len(ids) != 1:
                raise ValueError("Expected one position actuator per policy action")
            controls.append(int(ids[0]))
        gain, _ = expanded_model_field(env, "actuator_gainprm")
        limits, _ = expanded_model_field(env, "actuator_forcerange")
        torque_per_action = gain[:, controls, 0] * torch.as_tensor(action_term._scale, device=env.device)
        residuals, torques = [], []
        with torch.inference_mode():
            for step in range(args.steps):
                action = actor(obs)
                base = actor._base_features_and_action(obs)[1]
                delta = action - base
                if step >= 50:
                    residuals.append(delta.abs().cpu())
                    torques.append((delta * torque_per_action).abs().cpu())
                action_term.record_policy_mean(action)
                obs, _, _, _ = wrapped.step(action)
        values, torque = torch.cat(residuals), torch.cat(torques)
        rows = []
        for index, name in enumerate(action_term._target_names):
            rows.append({"joint": name,
                         "fraction_over_95pct_residual_bound": float((values[:, index] > 0.95 * actor.residual_scale).float().mean()),
                         "mean_absolute_residual_action": float(values[:, index].mean()),
                         "p99_absolute_residual_action": float(values[:, index].quantile(0.99)),
                         "mean_absolute_unsaturated_pd_torque_delta_Nm": float(torque[:, index].mean()),
                         "p99_absolute_unsaturated_pd_torque_delta_Nm": float(torque[:, index].quantile(0.99)),
                         "unsaturated_pd_torque_bound_Nm": float((torque_per_action[:, index].abs() * actor.residual_scale).mean()),
                         "actual_actuator_force_limit_Nm": float(limits[:, controls[index]].abs().amin(-1).mean())})
        rows.sort(key=lambda row: row["fraction_over_95pct_residual_bound"], reverse=True)
        result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint),
                  "seed": args.seed, "samples_per_joint": len(values), "residual_scale": actor.residual_scale,
                  "action_clip": prepared.clip_actions,
                  "scope": "Training-style uniform/reference DR rollout with ordinary resets, first50 steps omitted. Diagnostic only, not v2 acceptance or a causal performance comparison. Torque deltas are fixed-state unsaturated PD increments; physical actuator limits remain unchanged.",
                  "joints": rows}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
