"""Frozen-policy interaction traces for expert geometry, physics, and routing audits."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking.adaptation_policy import expanded_model_field, physics_observation
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT, TRACKER_SHA256
from intact_tracking.limb_context_sampling import configure_motion_sampling
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_moe_critic_action import TrackerActionContextWrapper
from intact_tracking.memory350_online_kmeans import normalized_latent, squared_distances
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.residual_uniform_protocol import configure_physics, audit_physics, physics_contract


def audit_routed_minimum(latent, centers, ids):
    # Reuse the actor's exact strided input. An equal-valued contiguous copy can
    # change reduction rounding; topk also need not share argmin's tie ordering.
    unit = normalized_latent(latent.reshape(-1, 64))
    distances = squared_distances(unit, centers)
    closest, top_ids = distances.topk(2, largest=False)
    selected = distances.gather(1, ids.reshape(-1, 1)).squeeze(1)
    excess = selected - closest[:, 0]
    if not bool((excess <= 2e-6).all()):
        raise RuntimeError(f"Actor chose a non-minimum center: excess={float(excess.max())}")
    return unit, closest[:, 1] - closest[:, 0], {
        "topk_index_disagreements": int((top_ids[:, 0] != ids).sum()),
        "argmin_index_disagreements": int((distances.argmin(-1) != ids).sum()),
        "maximum_selected_distance_excess": float(excess.max()),
    }


def static_parameters(env):
    """All independently randomized static coordinates, in explicit physical units.

    Payload inertias/COMs follow deterministically from these four masses and
    the fixed attachment geometry. Do not count their derived entries again.
    """
    robot, model = env.scene["robot"], env.sim.mj_model
    torso = [i for i in range(model.nbody) if model.body(i).name.split("/")[-1] == "torso_link"]
    feet = [i for i in range(model.ngeom)
            if re.fullmatch(r"(?:left|right)_foot[1-7]_collision", model.geom(i).name.split("/")[-1])]
    if len(torso) != 1 or len(feet) != 14:
        raise ValueError(f"Unexpected torso/foot indices: {torso}, {feet}")
    masses = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func.observe()
    mass, mass0 = expanded_model_field(env, "body_mass")
    com, com0 = expanded_model_field(env, "body_ipos")
    friction, _ = expanded_model_field(env, "geom_friction")
    armature, armature0 = expanded_model_field(env, "dof_armature")
    dofs = robot.indexing.joint_v_adr.long()
    joint_names = [model.joint(int(i)).name.split("/")[-1] for i in robot.indexing.joint_ids]
    if len(dofs) != 29 or len(joint_names) != 29:
        raise ValueError("Expected 29 ordered robot joints")
    feet_values = friction[:, feet, 0]
    torch.testing.assert_close(feet_values, feet_values[:, :1].expand_as(feet_values), atol=1e-6, rtol=0)
    blocks = [
        ("limb_load", masses, ["left_hand_kg", "right_hand_kg", "left_shin_kg", "right_shin_kg"],
         [0., 0., 0., 0.], [2.5, 2.5, 4., 4.]),
        ("torso_mass", mass[:, torso] - mass0[torso], ["torso_added_mass_kg"], [-1.], [1.]),
        ("torso_com", com[:, torso[0]] - com0[torso[0]], [f"torso_com_{x}_m" for x in "xyz"],
         [-.075] * 3, [.075] * 3),
        ("foot_friction", feet_values[:, :1], ["foot_sliding_friction"], [.3], [2.]),
        ("armature", armature[:, dofs] / armature0[dofs], [f"{j}_armature_scale" for j in joint_names],
         [.8] * 29, [1.2] * 29),
        ("encoder_bias", robot.data.encoder_bias, [f"{j}_encoder_bias_rad" for j in joint_names],
         [-.01] * 29, [.01] * 29),
    ]
    values, names, lows, highs, groups, offset = [], [], [], [], {}, 0
    for group, value, labels, low, high in blocks:
        assert value.shape == (env.num_envs, len(labels))
        values.append(value)
        names.extend(labels)
        lows.extend(low)
        highs.extend(high)
        groups[group] = list(range(offset, offset + len(labels)))
        offset += len(labels)
    raw = torch.cat(values, -1).detach().cpu().numpy()
    low, high = np.asarray(lows), np.asarray(highs)
    if not np.isfinite(raw).all() or (raw < low - 1e-5).any() or (raw > high + 1e-5).any():
        raise ValueError("Measured DR coordinate outside its configured training range")
    return raw, {"names": names, "lower": lows, "upper": highs, "groups": groups,
                 "normalization": "(value-lower)/(upper-lower), one full training range per coordinate",
                 "foot_friction_shared_across_14_geometries_verified": True,
                 "derived_payload_inertias_and_COMs": "determined by load masses and fixed attachment geometry"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=30401)
    parser.add_argument("--action-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--latent-stride", type=int, default=10)
    parser.add_argument("--motion-manifest", help="Optional small catalog for smoke checks; formal runs use the full training catalog")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.latent_stride) <= 0 or args.warmup_steps < 0:
        raise ValueError("Invalid trace dimensions")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    precision = configure_policy_precision("fp32")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    meta = state["residual_policy"]
    if meta["fusion"] != "concat" or meta["physics"].get("residual_physics_contract") != physics_contract():
        raise ValueError("Expected the uniform DR, unbounded residual MoE checkpoint")
    if meta["tracker_sha256"] != TRACKER_SHA256 or file_sha256(meta["tracker_checkpoint"]) != TRACKER_SHA256:
        raise ValueError("Unexpected tracker checkpoint")
    agent = OmegaConf.to_container(state["cfg"].agent, resolve=True)
    for key, value in state["actor_state_dict"].items():
        if key.startswith("residual_mlp.router."):
            torch.testing.assert_close(value, state["critic_state_dict"][key.replace("residual_mlp.", "mlp.", 1)], atol=0, rtol=0)
    _seed_everything(args.seed)
    prepared = prepare_rollout(checkpoint_file=meta["tracker_checkpoint"], num_envs=args.num_envs,
                               motion_file=None, motion_path=meta["motion_path"])
    cfg = prepared.env
    cfg.episode_length_s = meta["episode_length_control_steps"] * cfg.decimation * cfg.sim.mujoco.timestep
    physics = configure_physics(cfg, args.seed)
    configure_motion_sampling(cfg, "uniform", 0, 0)
    # Keep the source tracker reset, noise, push and termination configuration,
    # just as the uniform training entry point does for the original profile.
    from intact_tracking.adaptation_curriculum import configure_training_starts
    from intact_tracking.limb_context_terminations import configure_training_terminations
    configure_training_starts(cfg, "original")
    configure_training_terminations(cfg, _load_saved_config(Path(meta["tracker_checkpoint"])), "original")
    cfg.seed, cfg.auto_reset = args.seed, True
    if args.motion_manifest:
        cfg.commands["motion"].motion_manifest_file = str(Path(args.motion_manifest).resolve())
    context = load_memory350_checkpoint(meta["context_checkpoint"], device=args.device,
                                        expected_tracker_sha256=meta["tracker_sha256"])
    if context.sha256 != meta["context_sha256"]:
        raise ValueError("Context checkpoint changed")
    _seed_everything(args.seed)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        physics_audit = audit_physics(env, physics)
        raw_dr, schema = static_parameters(env)
        static_before = physics_observation(env).detach().clone()
        command = env.command_manager.get_term("motion")
        wrapped = TrackerActionContextWrapper(env, prepared.clip_actions, context)
        obs = wrapped.get_observations()
        kwargs = copy.deepcopy(agent["actor"])
        module, name = kwargs.pop("class_name").split(":")
        if module not in ("intact_tracking.residual_uniform_protocol", "intact_tracking.memory350_independent_moe_policy"):
            raise ValueError("Unsupported MoE actor class")
        actor = getattr(importlib.import_module(module), name)(obs, agent["obs_groups"], "actor", wrapped.num_actions, **kwargs)
        actor.to(env.device).load_state_dict(state["actor_state_dict"], strict=True)
        actor.eval().requires_grad_(False)
        router = actor.residual_mlp.router
        centers = router.centers.detach().clone()
        router_before = {k: v.clone() for k, v in router.state_dict().items()}
        result = {"protocol": "moe_interaction_routing_diagnostic_v1", "arguments": vars(args),
                  "checkpoint_sha256": file_sha256(args.checkpoint), "completed_updates": state["completed_updates"],
                  "context_sha256": context.sha256, "precision": precision,
                  "physics": physics, "physics_audit": physics_audit, "dr_schema": schema,
                  "control_dt_seconds": float(cfg.decimation * cfg.sim.mujoco.timestep),
                  "motion_count": len(command.motion_files), "episode_limit": env.max_episode_length,
                  "actor_critic_router_equal": True, "centers_updated_during_diagnostic": False,
                  "policy_parameters_updated": False,
                  "observation_timing": "Routes, latent, motion, episode and memory counts all recorded BEFORE the corresponding action",
                  "bootstrap": "Frozen tracker interaction; same memory retention rules as training, no query reset afterwards"}
        (output / "STARTED.json").write_text(json.dumps(result, indent=2) + "\n")
        del state
        # Capture the actual router result consumed inside actor.forward.
        routed = {}
        hook = router.register_forward_hook(lambda _m, inputs, route: routed.update(ids=route.detach(), latent=inputs[0].detach()))
        n, t = args.num_envs, args.steps
        routes = torch.empty(t, n, dtype=torch.uint8, device=env.device)
        episodes = torch.empty(t, n, dtype=torch.int32, device=env.device)
        motions = torch.empty_like(episodes)
        motion_steps = torch.empty_like(episodes)
        short = torch.empty_like(routes)
        long = torch.empty_like(routes)
        ended = torch.empty(t, n, dtype=torch.bool, device=env.device)
        failed = torch.empty_like(ended)
        norms = torch.empty(t, n, dtype=torch.float32, device=env.device)
        margins = torch.empty_like(norms)
        residual_rms = torch.empty_like(norms)
        sampled_steps, latent_samples = [], []
        route_audit = {"topk_index_disagreements": 0, "argmin_index_disagreements": 0,
                       "maximum_selected_distance_excess": 0., "allowed_squared_distance_roundoff": 2e-6}
        bootstrap_routes = torch.empty(args.warmup_steps, n, dtype=torch.uint8, device=env.device)
        timer = time.monotonic()
        _seed_everything(args.seed + 2300000)
        with torch.inference_mode():
            for step in range(args.warmup_steps):
                bootstrap_routes[step] = router(obs["dynamics_latent"])
                action = actor.tracker(obs)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, _, _, _ = wrapped.step(action)
                if (step + 1) % 100 == 0:
                    print(json.dumps({"stage": "bootstrap", "step": step + 1, "seconds": time.monotonic() - timer}), flush=True)
            _seed_everything(args.seed + 2400000)
            for step in range(t):
                memory = wrapped.context.memory
                episodes[step] = wrapped.episode_ids
                motions[step] = command.motion_idx
                motion_steps[step] = command.time_steps
                short[step] = memory.short_count
                long[step] = (memory.total_chunks - memory.session_start).clamp_max(30)
                z = obs["dynamics_latent"]
                action = actor(obs, stochastic_output=args.action_mode == "sample")
                if not torch.equal(routed["latent"], z):
                    raise RuntimeError("Actor routed a different latent than the diagnostic recorded")
                routes[step] = routed["ids"]
                unit, margin, checked = audit_routed_minimum(routed["latent"], router.centers, routed["ids"])
                route_audit["topk_index_disagreements"] += checked["topk_index_disagreements"]
                route_audit["argmin_index_disagreements"] += checked["argmin_index_disagreements"]
                route_audit["maximum_selected_distance_excess"] = max(
                    route_audit["maximum_selected_distance_excess"], checked["maximum_selected_distance_excess"])
                if checked["topk_index_disagreements"] or checked["argmin_index_disagreements"]:
                    print(json.dumps({"stage": "routing_roundoff_audit", "step": step, **checked}), flush=True)
                norms[step] = z.norm(dim=-1)
                margins[step] = margin
                residual_rms[step] = actor.last_residual_mean.square().mean(-1).sqrt()
                if step % args.latent_stride == 0:
                    sampled_steps.append(step)
                    latent_samples.append(unit.clone())
                mean = actor.last_base_action + actor.last_residual_mean
                env.action_manager.get_term("joint_pos").record_policy_mean(mean)
                obs, _, dones, _ = wrapped.step(action)
                ended[step] = dones.bool()
                failed[step] = env.reset_terminated.bool()
                if (step + 1) % 100 == 0:
                    progress = {"stage": "policy", "step": step + 1, "steps": t,
                                "seconds": time.monotonic() - timer,
                                "long_full_fraction": float((long[step] == 30).float().mean()),
                                "short_full_fraction": float((short[step] == 50).float().mean())}
                    (output / "progress.json").write_text(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
        hook.remove()
        torch.testing.assert_close(static_before, physics_observation(env), atol=0, rtol=0)
        after_dr, after_schema = static_parameters(env)
        np.testing.assert_array_equal(raw_dr, after_dr)
        assert schema == after_schema
        for key, value in router_before.items():
            torch.testing.assert_close(value, router.state_dict()[key], atol=0, rtol=0)
        data = {"routes": routes, "episode_ids": episodes, "motion_ids": motions, "motion_steps": motion_steps,
                "short_count": short, "long_count": long, "dones_after_action": ended,
                "failed_after_action": failed, "raw_latent_norm": norms, "center_squared_margin": margins,
                "residual_rms": residual_rms, "bootstrap_routes": bootstrap_routes,
                "centers": centers, "unit_latent_samples": torch.stack(latent_samples)}
        arrays = {key: value.cpu().numpy() for key, value in data.items()}
        arrays.update(raw_dr=raw_dr, sampled_steps=np.asarray(sampled_steps),
                      motion_lengths=command.motion.file_lengths.cpu().numpy())
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Nonfinite diagnostic trace")
        np.savez_compressed(output / "traces.npz", **arrays)
        result.update(wall_seconds=time.monotonic() - timer, completed_steps=t,
                      static_DR_unchanged_verified=True, router_state_unchanged_verified=True,
                      actual_actor_dispatch_verified_every_step=True, routing_minimum_audit=route_audit,
                      final_memory=wrapped.latent_metrics,
                      source_sha256={str(Path(__file__).resolve()): file_sha256(__file__)})
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"completed": str(output), "seconds": result["wall_seconds"]}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
