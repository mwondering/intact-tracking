"""Read-only latent identifiability probe with a held-out physical-world split.

Physics labels are used by the diagnostic regression only, never by policy
inference. No policy weights are fitted or modified by this program.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.adaptation_identification import nominal_pd_contract, sensor_pd_bias_proxy
from intact_tracking.adaptation_policy import (
    ContextAdaptationActor,
    compact_physics_observation,
    expanded_model_field,
)
from intact_tracking.adaptation_sensors import IMU_HISTORY, configure_context_sensors
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.rollout.mjlab_adapter import _sha256


def physical_labels(env):
    model = env.sim.mj_model

    def body_id(suffix):
        matches = [i for i in range(model.nbody) if model.body(i).name.split("/")[-1] == suffix]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous probe body {suffix}: {matches}")
        return matches[0]

    wrist = body_id("right_wrist_yaw_link")
    torso = body_id("torso_link")
    mass, default_mass = expanded_model_field(env, "body_mass")
    com, default_com = expanded_model_field(env, "body_ipos")
    friction, _ = expanded_model_field(env, "geom_friction")
    armature, default_armature = expanded_model_field(env, "dof_armature")
    feet = [i for i in range(model.ngeom) if "_foot" in model.geom(i).name]
    dofs = default_armature > 0
    names = [
        "payload_kg",
        "torso_delta_kg",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "foot_friction",
        "mean_armature_ratio",
    ]
    values = torch.cat(
        (
            (mass[:, wrist] - default_mass[wrist])[:, None],
            (mass[:, torso] - default_mass[torso])[:, None],
            com[:, torso] - default_com[torso],
            friction[:, feet, 0].mean(-1, keepdim=True),
            (armature[:, dofs] / default_armature[dofs]).mean(-1, keepdim=True),
        ),
        dim=-1,
    )
    return names, values.cpu()


def ridge_probe(features, targets, train_ids, test_ids, ridge=0.1):
    """All time samples of a world stay together; no row-wise leakage."""
    train_x = features[:, train_ids].flatten(0, 1).double()
    test_x = features[:, test_ids].flatten(0, 1).double()
    train_y = targets[train_ids].repeat(features.shape[0], 1).double()
    test_y = targets[test_ids].repeat(features.shape[0], 1).double()
    mean_x = train_x.mean(0)
    std_x = train_x.std(0).clamp_min(1e-5)
    train_x = (train_x - mean_x) / std_x
    test_x = (test_x - mean_x) / std_x
    mean_y = train_y.mean(0)
    gram = train_x.T @ train_x / len(train_x)
    weights = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
        train_x.T @ (train_y - mean_y) / len(train_x),
    )
    prediction = test_x @ weights + mean_y
    mse = (prediction - test_y).square().mean(0)
    variance = (test_y - test_y.mean(0)).square().mean(0)
    return {
        "heldout_r2": (1 - mse / variance.clamp_min(1e-12)).tolist(),
        "heldout_rmse": mse.sqrt().tolist(),
        "constant_train_mean_rmse": (test_y - mean_y).square().mean(0).sqrt().tolist(),
    }


def direct_compact_code_probe(features, targets, train_ids, test_ids):
    """Score the actual seven actor inputs without fitting a diagnostic decoder."""
    if features.shape[-1] != 7 or targets.shape[-1] != 7:
        raise ValueError("Direct compact-code diagnostic requires exactly seven coordinates")
    prediction = features[:, test_ids].flatten(0, 1).double()
    truth = targets[test_ids].repeat(features.shape[0], 1).double()
    error = prediction - truth
    scale = torch.tensor([3.0, 1.0, 0.075, 0.075, 0.075, 2.0, 0.2], dtype=torch.float64)
    mse = error.square().mean(0)
    constant = targets[train_ids].double().mean(0)
    return {
        "normalized_code_rmse": mse.sqrt().tolist(),
        "physical_rmse": (mse.sqrt() * scale).tolist(),
        "physical_bias": (error.mean(0) * scale).tolist(),
        "r2": (1 - mse / truth.var(0, unbiased=False).clamp_min(1e-12)).tolist(),
        "constant_train_mean_physical_rmse": ((truth - constant).square().mean(0).sqrt() * scale).tolist(),
        "physical_units": ["kg", "kg", "m", "m", "m", "friction coefficient", "armature ratio"],
        "contract": "Actual causal actor code; no fitted readout and no label substitution in actions",
    }


def run(args):
    _seed_everything(args.seed)
    prepared = prepare_rollout(
        checkpoint_file=TRACKER, num_envs=args.num_envs, motion_path=DATASET, motion_file=None
    )
    configure_physics(prepared.env, "dr")
    actor_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"].agent.actor
    if actor_cfg.get("context_imu_accel", False):
        configure_context_sensors(prepared.env)
    prepared.env.seed = args.seed
    prepared.env.commands["motion"].sampling_mode = "uniform"
    prepared.env.commands["motion"].rewind.enabled = False
    env = ManagerBasedRlEnv(cfg=prepared.env, device=args.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        obs = wrapped.get_observations()
        actor = load_actor(args.checkpoint, prepared, obs, wrapped)
        if not isinstance(actor, ContextAdaptationActor) or getattr(actor, "correction_use_privilege", False):
            raise TypeError("This probe requires a deployable context actor")
        if actor.context_latent_mean:
            from intact_tracking.adaptation_context_memory import CausalContextMeanWrapper

            wrapped = CausalContextMeanWrapper(
                wrapped, actor.adaptation_latent_dim, actor.context_mean_horizon
            )
            wrapped.bind(actor)
            obs = wrapped.attach(obs)
        names, targets = physical_labels(env)
        compact_targets = (
            compact_physics_observation(env).cpu()
            if actor.physics_bilinear or actor.physics_low_rank else None
        )
        encoder_bias_targets = (
            env.scene["robot"].data.encoder_bias.detach().cpu().clone()
            if args.encoder_bias_diagnostic else None
        )
        features, sensors = [], []
        readout_predictions, readout_targets = [], []
        pd_estimates = []
        pd_contract = nominal_pd_contract(env) if args.pd_bias_diagnostic else None
        payload_kg = targets[:, 0].to(env.device)
        with torch.inference_mode():
            for step in range(args.steps):
                deployable = obs.select(*actor.deployable_observation_groups)
                if step >= 50 and step % 25 == 0:
                    latent = actor.context_latent(deployable)
                    features.append(latent.cpu())
                    if actor.context_auxiliary_head is not None:
                        from intact_tracking.cli.adaptation_distill import context_auxiliary_targets

                        robot = env.scene["robot"].data
                        readout_predictions.append(
                            actor.context_auxiliary_head(actor.instant_context_latent(deployable)).cpu()
                        )
                        readout_targets.append(
                            context_auxiliary_targets(
                                robot.root_link_lin_vel_w, robot.root_link_quat_w, payload_kg
                            ).cpu()
                        )
                    raw_history = deployable["estimator_history"]
                    if pd_contract is not None:
                        pd_estimates.append(sensor_pd_bias_proxy(raw_history, pd_contract)[0].cpu())
                    if actor.context_imu_accel:
                        raw_history = torch.cat((raw_history, deployable[IMU_HISTORY]), dim=-1)
                    sensors.append(actor.context_encoder.frames(raw_history)[:, -1].cpu())
                action = actor(deployable)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, _, _, _ = wrapped.step(action)
        order = torch.randperm(
            args.num_envs, generator=torch.Generator().manual_seed(args.seed + 7)
        )
        train_ids, test_ids = order[: 3 * args.num_envs // 4], order[3 * args.num_envs // 4 :]
        latent_features = torch.stack(features)
        sensor_features = torch.stack(sensors)
        result = {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": _sha256(Path(args.checkpoint)),
            "seed": args.seed,
            "targets": names,
            "train_worlds": len(train_ids),
            "heldout_worlds": len(test_ids),
            "samples_per_world": len(features),
            "context_latent_mean": actor.context_latent_mean,
            "context_mean_only": actor.context_mean_only,
            "context_mean_horizon": actor.context_mean_horizon,
            "latent_probe": ridge_probe(latent_features, targets, train_ids, test_ids),
            "current_sensor_probe": ridge_probe(sensor_features, targets, train_ids, test_ids),
            "latent_episode_average_probe": ridge_probe(
                latent_features.mean(0, keepdim=True), targets, train_ids, test_ids
            ),
            "contract": "Policy saw deployable observations only; regression labels never enter actions; physical worlds held out as groups",
            "limitation": "Diagnostic on one new physics seed, not proof that the policy depends on the latent or that every parameter is identifiable",
        }
        if compact_targets is not None:
            result["direct_compact_code"] = direct_compact_code_probe(
                latent_features, compact_targets, train_ids, test_ids
            )
        if pd_estimates:
            estimates = torch.stack(pd_estimates)[:, test_ids].flatten(0, 1)
            truth = env.scene["robot"].data.encoder_bias.cpu()[test_ids].repeat(len(features), 1)
            result["pd_bias_diagnostic"] = {
                "joint_names": list(env.scene["robot"].joint_names),
                "joint_rmse_rad": (estimates - truth).square().mean(0).sqrt().tolist(),
                "zero_baseline_rmse_rad": truth.square().mean(0).sqrt().tolist(),
                "mean_rmse_rad": float((estimates - truth).square().mean().sqrt()),
                "mean_zero_baseline_rmse_rad": float(truth.square().mean().sqrt()),
                "contract": "Diagnostic only. Proxy uses noisy joint/torque/action history and fixed nominal PD constants; randomized encoder bias is a scoring label only. Common-timestamp and controller-calibration assumptions are unvalidated on hardware.",
            }
        if encoder_bias_targets is not None:
            result["encoder_bias_probe"] = {
                "joint_names": list(env.scene["robot"].joint_names),
                "instant_context": ridge_probe(latent_features, encoder_bias_targets, train_ids, test_ids),
                "offline_episode_mean_context": ridge_probe(
                    latent_features.mean(0, keepdim=True), encoder_bias_targets, train_ids, test_ids
                ),
                "current_sensor": ridge_probe(sensor_features, encoder_bias_targets, train_ids, test_ids),
                "contract": "Held-out-world diagnostic only, true biases are regression labels and never enter policy actions. Full-episode averages are offline evidence, not initial-time deployable inputs.",
            }
        if readout_predictions:
            all_predictions = torch.stack(readout_predictions)[:, test_ids].flatten(0, 1)
            predictions = all_predictions[..., :4]
            labels = torch.stack(readout_targets)[:, test_ids].flatten(0, 1)
            result["trained_auxiliary_readout"] = {
                "labels": ["body_vx_m_s", "body_vy_m_s", "body_vz_m_s", "payload_kg_minus2"],
                "heldout_world_rmse": (predictions - labels).square().mean(0).sqrt().tolist(),
                "heldout_label_std": labels.std(0).tolist(),
                "contract": "Frozen trained head on a new physics seed; no diagnostic fitting; labels excluded from all actions",
            }
            if all_predictions.shape[-1] >= 10:
                from intact_tracking.cli.adaptation_distill import (
                    normalized_context_physics_targets,
                )

                physics_targets = normalized_context_physics_targets(targets)[test_ids].repeat(len(features), 1)
                result["trained_physics_readout"] = {
                    "labels": names[1:],
                    "normalized_rmse": (all_predictions[..., 4:10] - physics_targets).square().mean(0).sqrt().tolist(),
                    "normalized_heldout_label_std": physics_targets.std(0).tolist(),
                    "contract": "Frozen training-only decoder, new physical worlds; privileged labels never enter policy inference",
                }
            if all_predictions.shape[-1] == 39:
                bias_targets = env.scene["robot"].data.encoder_bias.cpu()[test_ids].repeat(len(features), 1)
                result["trained_encoder_bias_readout"] = {
                    "joint_names": list(env.scene["robot"].joint_names),
                    "rmse_rad": (all_predictions[..., 10:39] * 0.01 - bias_targets).square().mean(0).sqrt().tolist(),
                    "zero_prior_rmse_rad": bias_targets.square().mean(0).sqrt().tolist(),
                    "contract": "Frozen trained head, no diagnostic fitting, worlds held out; true bias is a scoring label only",
                }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=14643)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--encoder-bias-diagnostic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pd-bias-diagnostic", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
