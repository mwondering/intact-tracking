"""Motion-balanced evaluation with fixed starts and explicit failure accounting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import MethodType

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from intact_tracking.cli.residual_policy_train import (
    _audit_nominal_runtime,
    _configure_nominal_physics,
    _seed_everything,
)
from intact_tracking.environment.policy import SPV52HeightContactEstimatorActor
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.residual_policy import FrozenTrackerResidualActor
from intact_tracking.rollout.mjlab_adapter import (
    _clear_missing_motion_exclusions,
    _filter_disturbance_events,
    _sha256,
)
from intact_tracking.rollout.online import (
    DEFAULT_PAYLOAD_BODY_NAME,
    DEFAULT_PAYLOAD_POSITION_BODY_M,
    DEFAULT_PAYLOAD_SIZE_M,
    _add_payload_startup_event,
)

DATASET = "/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong"
TRACKER = (
    "/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/"
    "2026-09-02_04-48-42_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_"
    "8gpu_12288env_motion_data_correct/checkpoint_72000.pt"
)
METRICS = (
    "error_body_pos",
    "error_joint_pos",
    "error_anchor_pos",
    "error_anchor_rot",
    "error_body_rot",
    "error_joint_vel",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_lin_vel",
    "error_body_ang_vel",
)

PROTOCOL = "balanced_fixed_starts_v2_isolated_resets"


def body_error_components(command):
    """Read-only per-body decomposition of the four aggregate body metrics."""
    from mjlab.utils.lab_api.math import quat_error_magnitude

    from intact_tracking.environment.mdp.motion_fk import quat_apply_inverse

    if command._uses_qpos_only_actor_fk():
        root = command._qpos_actor_root_velocity_state((0,))
        reference = command.gather_reference_body_state_b((0,))
        robot_lin = quat_apply_inverse(
            root.root_quat_w[:, 0, None],
            command.robot_body_lin_vel_w - root.root_lin_vel_w[:, 0, None],
        )
        robot_ang = quat_apply_inverse(
            root.root_quat_w[:, 0, None],
            command.robot_body_ang_vel_w - root.root_ang_vel_w[:, 0, None],
        )
        lin = (reference.lin_vel_b[:, 0] - robot_lin).norm(dim=-1)
        ang = (reference.ang_vel_b[:, 0] - robot_ang).norm(dim=-1)
    else:
        lin = (command.body_lin_vel_w - command.robot_body_lin_vel_w).norm(dim=-1)
        ang = (command.body_ang_vel_w - command.robot_body_ang_vel_w).norm(dim=-1)
    return torch.stack(
        (
            (command.body_pos_relative_w - command.robot_body_pos_w).norm(dim=-1),
            quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w),
            lin,
            ang,
        ),
        dim=-1,
    )


def reset_finished_worlds(env, done_ids, survivors):
    """Reset inactive worlds without advancing surviving references/history.

    Public env.reset() calls command.compute(dt=0), which nevertheless advances
    this motion command, and pushes observation history for ALL worlds. Inactive
    worlds need no fresh observation: their actions are masked to zero. The next
    regular simulator step performs forward dynamics on their reset states.
    """
    command = env.command_manager.get_term("motion")
    before = {
        "cursor": command.time_steps[survivors].clone(),
        "qpos": env.sim.data.qpos[survivors].clone(),
        "qvel": env.sim.data.qvel[survivors].clone(),
    }
    warmstart = getattr(env.sim.data, "qacc_warmstart", None)
    if isinstance(warmstart, torch.Tensor):
        before["warmstart"] = warmstart[survivors].clone()
    histories = []
    manager = getattr(env, "observation_manager", None)
    for group in getattr(manager, "_group_obs_term_history_buffer", {}).values():
        for buffer in group.values():
            if buffer._buffer is not None:
                histories.append(
                    (
                        buffer,
                        buffer._pointer,
                        buffer._num_pushes[survivors].clone(),
                        buffer._buffer[:, survivors].clone(),
                    )
                )
    env._reset_idx(done_ids)
    env.scene.write_data_to_sim()
    after = {
        "cursor": command.time_steps[survivors],
        "qpos": env.sim.data.qpos[survivors],
        "qvel": env.sim.data.qvel[survivors],
    }
    if "warmstart" in before:
        after["warmstart"] = warmstart[survivors]
    for name in before:
        if not torch.equal(before[name], after[name]):
            raise RuntimeError(f"Partial reset changed surviving-world {name}")
    for buffer, pointer, pushes, values in histories:
        if (
            pointer != buffer._pointer
            or not torch.equal(pushes, buffer._num_pushes[survivors])
            or not torch.equal(values, buffer._buffer[:, survivors])
        ):
            raise RuntimeError("Partial reset changed surviving-world observation history")


def configure_physics(cfg, physics: str, dr_profile: str = "right-hand-1-3kg") -> dict:
    if physics == "hardest":
        from intact_tracking.tracker_finetune import configure_abc_physics

        return configure_abc_physics(cfg, "B", cfg.seed)
    from intact_tracking.preview_protocol import DR_PROFILES, LIMB_PROFILE, add_limb_payloads

    if dr_profile not in DR_PROFILES and not (dr_profile == "abc-load-only" and physics == "nominal"):
        raise ValueError(f"Unknown DR profile: {dr_profile}")
    removed = _filter_disturbance_events(cfg)
    cleared = _clear_missing_motion_exclusions(cfg)
    command = cfg.commands["motion"]
    # The original checkpoint can name filters outside this dataset. Evaluation
    # and adaptation training explicitly use the complete requested dataset.
    for name in ("excluded_motion_files", "motion_exclude_files", "motion_exclude_file"):
        if hasattr(command, name):
            setattr(command, name, "" if isinstance(getattr(command, name), str) else [])
    if physics == "nominal":
        physics_info = _configure_nominal_physics(cfg)
    elif dr_profile == LIMB_PROFILE:
        physics_info = add_limb_payloads(cfg)
    else:
        args = argparse.Namespace(
            payload_enabled=True,
            payload_body_name=DEFAULT_PAYLOAD_BODY_NAME,
            payload_mass_range_kg=(1.0, 3.0),
            payload_position_body_m=DEFAULT_PAYLOAD_POSITION_BODY_M,
            payload_size_m=DEFAULT_PAYLOAD_SIZE_M,
        )
        physics_info = {"payload": _add_payload_startup_event(cfg, args)}
    return {
        "physics": physics,
        "dr_profile": dr_profile,
        "details": physics_info,
        "removed_disturbances": removed,
        "cleared_exclusions": cleared,
    }


def fixed_starts(lengths: torch.Tensor, repeats: int, horizon: int, seed: int):
    """One common CPU RNG makes starts independent of environment/actor RNG use."""
    lengths = lengths.cpu()
    ids = torch.arange(len(lengths)).repeat(repeats)
    g = torch.Generator().manual_seed(seed)
    upper = (lengths[ids] - horizon - 21).clamp_min(0)
    strata = torch.arange(repeats).repeat_interleave(len(lengths))
    fractions = (strata + torch.rand(len(ids), generator=g)) / repeats
    starts = (fractions * upper).long()
    return ids, starts


def load_actor(checkpoint, prepared, obs, wrapped):
    if checkpoint is None:
        actor = SPV52HeightContactEstimatorActor(
            obs, prepared.obs_groups, "actor", wrapped.num_actions, **prepared.actor_kwargs
        )
        state = torch.load(prepared.checkpoint_path, map_location="cpu", weights_only=False)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = OmegaConf.to_container(state["cfg"].agent, resolve=True)
        kwargs = copy.deepcopy(cfg["actor"])
        class_name = kwargs.pop("class_name")
        if class_name.endswith(":PrivilegedAdaptationActor"):
            from intact_tracking.adaptation_policy import PrivilegedAdaptationActor

            cls = PrivilegedAdaptationActor
        elif class_name.endswith(":ContextPhysicsCorrectionActor"):
            from intact_tracking.adaptation_correction import ContextPhysicsCorrectionActor

            cls = ContextPhysicsCorrectionActor
        elif class_name.endswith(":ContextAdaptationActor"):
            from intact_tracking.adaptation_policy import ContextAdaptationActor

            cls = ContextAdaptationActor
        elif class_name.endswith(":FrozenTrackerResidualActor"):
            cls = FrozenTrackerResidualActor
            if kwargs.get("use_dynamics_latent"):
                raise ValueError("Use the explicit student evaluator for context policies")
        elif class_name.endswith(":PreviewResidualActor"):
            from intact_tracking.simulator_preview_experiment import PreviewResidualActor

            cls = PreviewResidualActor
        elif class_name.endswith(":TrackerFinetuneActor"):
            from intact_tracking.tracker_finetune import TrackerFinetuneActor

            cls = TrackerFinetuneActor
        else:
            raise ValueError(f"Unsupported actor {class_name}")
        actor = cls(obs, cfg["obs_groups"], "actor", wrapped.num_actions, **kwargs)
    actor.to(wrapped.device)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    return actor


def evaluate(args):
    _seed_everything(args.seed)
    motion_file = getattr(args, "motion_file", None)
    manifest = getattr(args, "motion_manifest", None)
    if manifest:
        if motion_file:
            raise ValueError("Use either motion-file or motion-manifest")
        files = [Path(line.strip()) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not files or any(not path.is_absolute() or not path.is_file() or path.suffix != ".npz"
                            for path in files):
            raise ValueError("Evaluation manifest requires existing absolute NPZ paths")
        if len(set(files)) != len(files):
            raise ValueError("Duplicate motions in evaluation manifest")
    else:
        files = [Path(motion_file)] if motion_file else sorted(Path(args.motion_path).rglob("*.npz"))
    n = len(files) * args.repeats
    if n > getattr(args, "max_envs", 4096):
        raise ValueError("Evaluation exceeds max-envs; use scripts/evaluate_preview_dataset.py for bounded batches")
    prepared = prepare_rollout(
        checkpoint_file=args.tracker_checkpoint,
        num_envs=n,
        motion_path=None if motion_file else args.motion_path,
        motion_file=motion_file,
    )
    cfg = prepared.env
    cfg.commands["motion"].motion_manifest_file = str(Path(manifest).resolve()) if manifest else ""
    dr_profile = getattr(args, "dr_profile", None)
    from intact_tracking.adaptation_reward_contract import (
        assert_fixed_reward_checkpoint,
        capture_original_rewards,
    )

    reward_contract = capture_original_rewards(cfg)
    training_reward_audit = {
        "eligible_fixed_reward_candidate": False,
        "current_task_reward_sha256": reward_contract["sha256"],
        "reason": "Original tracker is a reference, not a newly trained fixed-reward candidate",
    }
    if args.checkpoint:
        checkpoint_info = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if dr_profile is None:
            dr_profile = checkpoint_info.get("residual_policy", {}).get("dr_profile", "right-hand-1-3kg")
        try:
            assert_fixed_reward_checkpoint(checkpoint_info, reward_contract)
            training_reward_audit.update(eligible_fixed_reward_candidate=True, reason="Matching original-reward signature and audited initialization lineage")
        except ValueError as error:
            training_reward_audit["reason"] = str(error)
        training_reward_audit["recorded_reward_changes"] = dict(checkpoint_info.get("residual_policy", {}).get("reward_changes", {}))
        if checkpoint_info["cfg"].agent.actor.get("oracle_clean_proprio", False):
            from intact_tracking.adaptation_policy import configure_oracle_proprioception

            configure_oracle_proprioception(cfg)
        if checkpoint_info["cfg"].agent.actor.get("context_imu_accel", False):
            from intact_tracking.adaptation_sensors import configure_context_sensors

            configure_context_sensors(cfg)
        del checkpoint_info
    cfg.seed = args.seed
    dr_profile = dr_profile or "right-hand-1-3kg"
    if dr_profile != "right-hand-1-3kg" and args.payload_gravity_compensation:
        raise ValueError("Legacy analytical gravity probe only supports the right-hand payload")
    physics = configure_physics(cfg, args.physics, dr_profile)
    command_cfg = cfg.commands["motion"]
    command_cfg.sampling_mode = "uniform"
    command_cfg.rewind.enabled = False
    command_cfg.resample_on_motion_end = False
    command_cfg.resampling_time_range = (1e9, 1e9)
    command_cfg.init_noise = {}
    command_cfg.pose_range = {}
    command_cfg.velocity_range = {}
    command_cfg.joint_position_range = (0.0, 0.0)
    cfg.auto_reset = False
    cfg.episode_length_s = (args.steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    compensation = None
    preview_wrapper = None
    try:
        if args.physics == "nominal":
            physics["runtime_audit"] = _audit_nominal_runtime(env)
        elif args.physics == "hardest":
            physics["runtime_payload_audit"] = env.event_manager.get_term_cfg("abc_payload").func.audit()
        elif dr_profile == "hands-shins-2-4kg":
            from intact_tracking.preview_protocol import audit_limb_payloads

            physics["runtime_payload_audit"] = audit_limb_payloads(env)
        command = env.command_manager.get_term("motion")
        if command.motion.num_files != len(files):
            raise RuntimeError(
                f"Dataset coverage mismatch: loaded {command.motion.num_files}/{len(files)}"
            )
        if tuple(str(path.resolve()) for path in files) != tuple(
            str(Path(path).resolve()) for path in command.motion_files
        ):
            raise ValueError("Evaluation motion catalog/order mismatch")
        ids, starts = fixed_starts(command.motion.file_lengths, args.repeats, args.steps, args.seed)
        ids, starts = ids.to(env.device), starts.to(env.device)

        def sample(self, env_ids):
            self.motion_idx[env_ids] = ids[env_ids]
            self.motion_length[env_ids] = self.motion.file_lengths[ids[env_ids]]
            self.time_steps[env_ids] = starts[env_ids]

        command._uniform_sampling = MethodType(sample, command)
        _seed_everything(args.seed + 1_000_000)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            oracle = "PrivilegedAdaptationActor" in str(state["cfg"].agent.actor.class_name)
            from intact_tracking.adaptation_correction import correction_config_requires_privilege

            correction_privileged = correction_config_requires_privilege(state["cfg"].agent.actor)
            privilege_schema = str(state["cfg"].agent.actor.get("privilege_schema", "critic"))
            preview_mode = state.get("residual_policy", {}).get("simulator_preview")
            preview_version = state.get("residual_policy", {}).get("version")
            preview_critic = state.get("residual_policy", {}).get("simulator_preview_critic", False)
            if getattr(args, "simulator_preview_override", None):
                if preview_mode is None:
                    raise ValueError("Preview override requires a preview-trained checkpoint")
                preview_mode = args.simulator_preview_override
            del state
        else:
            oracle = False
            correction_privileged = False
            preview_mode = None
            preview_version = None
            preview_critic = False
        if correction_privileged:
            from intact_tracking.adaptation_correction import StaticPhysicsCorrectionWrapper

            wrapped = StaticPhysicsCorrectionWrapper(RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions))
        elif preview_mode:
            from intact_tracking.simulator_preview_experiment import (
                EXPERIMENT_VERSION,
                PreviewObservationWrapper,
            )

            if preview_version == EXPERIMENT_VERSION:
                wrapped = preview_wrapper = PreviewObservationWrapper(
                    env, clip_actions=prepared.clip_actions, preview_mode=preview_mode,
                )
            else:
                from intact_tracking.simulator_preview import SimulatorPreviewWrapper

                wrapped = preview_wrapper = SimulatorPreviewWrapper(
                    env, clip_actions=prepared.clip_actions, preview_mode=preview_mode,
                    privilege_schema=privilege_schema,
                )
        elif oracle:
            from intact_tracking.adaptation_policy import PrivilegedAdaptationWrapper

            wrapped = PrivilegedAdaptationWrapper(
                env, clip_actions=prepared.clip_actions, privilege_schema=privilege_schema
            )
        else:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        obs = wrapped.get_observations()
        actor = load_actor(args.checkpoint, prepared, obs, wrapped)
        if preview_wrapper is not None:
            preview_wrapper.bind(actor)
            obs = wrapped.get_observations()
        if getattr(actor, "context_latent_mean", False):
            from intact_tracking.adaptation_context_memory import CausalContextMeanWrapper

            wrapped = CausalContextMeanWrapper(
                wrapped, actor.adaptation_latent_dim, actor.context_mean_horizon
            )
            wrapped.bind(actor)
            obs = wrapped.attach(obs)
        from intact_tracking.adaptation_policy import ContextAdaptationActor

        student = isinstance(actor, ContextAdaptationActor) and not getattr(actor, "correction_use_privilege", False)
        state_ablation = getattr(args, "teacher_remove_privilege", "normal")
        if state_ablation != "normal":
            if student or getattr(actor, "privilege_schema", None) != "state_physics":
                raise ValueError("All-path privilege replacement requires a state_physics teacher")
            actor.oracle_state_ablation = state_ablation
        if args.teacher_feature_ablation != "normal":
            if student or not getattr(actor, "oracle_tracking_features", False):
                raise ValueError("Feature-replacement diagnostics require an oracle-feature teacher")
            actor.oracle_feature_ablation = args.teacher_feature_ablation
        exported_policy = None
        exported_sha256 = None
        exported_cpu_threads = None
        exported_mean = exported_count = exported_reset = None
        if args.exported_policy:
            if not student or args.context_ablation != "normal":
                raise ValueError("Export evaluation requires a normal context-policy checkpoint")
            artifact = Path(args.exported_policy)
            manifest = json.loads((artifact.parent / "manifest.json").read_text())
            exported_sha256 = _sha256(artifact)
            if manifest["artifact_sha256"] != exported_sha256 or manifest[
                "checkpoint_sha256"
            ] != _sha256(Path(args.checkpoint)):
                raise ValueError("Export/checkpoint identity mismatch")
            exported_cpu_threads = manifest.get("required_cpu_threads")
            if exported_cpu_threads is not None:
                if exported_cpu_threads != 1 or len(manifest.get("independent_process_audits", [])) < 8:
                    raise ValueError("Export CPU execution contract lacks independent audits")
                torch.set_num_threads(exported_cpu_threads)
            exported_policy = torch.jit.load(str(artifact), map_location="cpu").eval()
            if bool(manifest.get("stateful_context_memory", False)) != actor.context_latent_mean:
                raise ValueError("Export/checkpoint causal-memory contract mismatch")
            if actor.context_latent_mean:
                contract = manifest["memory_contract"]
                if contract["horizon"] != actor.context_mean_horizon or contract["latent_dim"] != actor.adaptation_latent_dim:
                    raise ValueError("Export causal-memory dimensions/horizon mismatch")
                if bool(contract.get("mean_only", False)) != actor.context_mean_only:
                    raise ValueError("Export causal-memory controller code mismatch")
                exported_mean = torch.zeros(n, actor.adaptation_latent_dim)
                exported_count = torch.zeros(n, 1)
                exported_reset = torch.zeros(n, dtype=torch.bool)
        if student:
            actor.context_ablation = args.context_ablation
        if args.oracle_tracking_features and args.checkpoint:
            raise ValueError(
                "The oracle-feature probe only supports the original tracker; trained policies store their input mode"
            )
        if args.bias_compensation != 0.0 or args.payload_gravity_compensation != 0.0:
            if student:
                raise ValueError("Privileged bias compensation is forbidden for the student")
            from intact_tracking.oracle_compensation import (
                NominalBiasCompensation,
                PayloadGravityCompensation,
            )

            compensation = (
                PayloadGravityCompensation(env) if args.payload_gravity_compensation
                else NominalBiasCompensation(env)
            )
        initial_steps = command.time_steps.clone()
        from intact_tracking.adaptation_policy import physics_observation

        # Audit-only snapshot: NEVER added to student observations or actions.
        physical_rows = physics_observation(env).detach().cpu().numpy()
        physics_world_fingerprints = [hashlib.sha256(row.tobytes()).hexdigest() for row in physical_rows]
        termination_names = [
            name for name in env.termination_manager.active_terms
            if not env.termination_manager.get_term_cfg(name).time_out
        ]
        failure_terms = torch.zeros(n, len(termination_names), dtype=torch.bool, device=env.device)
        horizons = (command.motion_length - initial_steps - 1).clamp(1, args.steps)
        totals = torch.zeros(n, len(METRICS), device=env.device, dtype=torch.float64)
        joint_error_totals = torch.zeros_like(command.robot_joint_pos, dtype=torch.float64)
        force_saturation_totals = torch.zeros_like(
            env.scene["robot"].data.actuator_force, dtype=torch.float64
        )
        body_error_totals = None
        if args.body_diagnostics:
            body_error_totals = torch.zeros(
                n, len(command.cfg.body_names), 4, device=env.device, dtype=torch.float64
            )
        counts = torch.zeros(n, device=env.device, dtype=torch.long)
        failed = torch.zeros(n, device=env.device, dtype=torch.bool)
        active = ~failed
        compensation_rms = []
        compensation_max = 0.0
        reset_batches = 0
        previous_orientation = previous_gyro = None
        wall_start = time.time()
        with torch.inference_mode():
            for step in range(args.steps):
                _seed_everything(args.seed + 2_000_000 + step)
                # Actual student evaluation removes all privileged keys before
                # every action, in addition to the separate input-isolation test.
                policy_obs = obs.select(*actor.deployable_observation_groups) if student else obs
                if args.orientation_filter_weight != 1.0:
                    from intact_tracking.adaptation_filters import causal_orientation_filter

                    measured = policy_obs["robot_root_quat"]
                    gyro = policy_obs["estimator_history"][:, 3197:3200]
                    filtered = measured
                    if previous_orientation is not None:
                        filtered = causal_orientation_filter(
                            measured, gyro, previous_orientation, previous_gyro,
                            env.step_dt, args.orientation_filter_weight,
                        )
                    previous_orientation, previous_gyro = filtered.clone(), gyro.clone()
                    # Shallow clone prevents altering measured observations or
                    # histories. Evaluated worlds never restart active episodes.
                    policy_obs = policy_obs.clone(recurse=False)
                    policy_obs.set("robot_root_quat", filtered)
                if exported_policy is not None:
                    from intact_tracking.context_export import deployable_fields

                    packed = torch.cat(
                        [policy_obs[name] for name, _ in deployable_fields(actor)], dim=-1
                    )
                    if exported_mean is not None:
                        action, exported_mean, exported_count = exported_policy(
                            packed.cpu(), exported_mean, exported_count, exported_reset
                        )
                        action = action.to(env.device)
                    else:
                        action = exported_policy(packed.cpu()).to(env.device)
                elif args.oracle_tracking_features:
                    raw = actor._spv5_2_features(
                        obs,
                        obs[actor.estimator_target_group],
                        obs[actor.foot_contact_target_group],
                        obs[actor.reference_encoder_target_group],
                    )
                    action = actor.distribution.deterministic_output(
                        actor.mlp(actor.policy_normalizer(raw))
                    )
                else:
                    action = actor(policy_obs)
                if compensation is not None:
                    correction = compensation.correction()
                    compensation_rms.append(float(correction.square().mean().sqrt()))
                    compensation_max = max(compensation_max, float(correction.abs().max()))
                    if args.physics == "nominal" and compensation_max > 1e-5:
                        raise RuntimeError(
                            "Nominal twin failed the zero-compensation identity audit"
                        )
                    action = action + (args.bias_compensation or args.payload_gravity_compensation) * correction
                if args.action_filter_strength:
                    from intact_tracking.adaptation_filters import causal_action_filter

                    action = causal_action_filter(
                        action, obs["estimator_history"],
                        args.action_filter_strength, args.action_filter_order,
                    )
                action = action.masked_fill(~active[:, None], 0.0)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, _, dones, _ = wrapped.step(action)
                if exported_mean is not None:
                    exported_reset = dones.bool().cpu()
                if not torch.equal(command.time_steps[active], initial_steps[active] + step + 1):
                    raise RuntimeError("Surviving-world reference timeline drifted")
                command._update_metrics()
                values = torch.stack([command.metrics[key] for key in METRICS], dim=-1)
                if not torch.isfinite(values[active]).all():
                    raise RuntimeError("Nonfinite evaluation metric")
                totals[active] += values[active].double()
                if body_error_totals is not None:
                    body_error_totals[active] += body_error_components(command)[active].double()
                if args.joint_diagnostics:
                    joint_error_totals[active] += (
                        (command.joint_pos - command.robot_joint_pos)[active].square().double()
                    )
                    robot = env.scene["robot"]
                    force_range = env.sim.model.actuator_forcerange[:, robot.indexing.ctrl_ids]
                    force_limit = force_range.abs().amax(-1).clamp_min(1e-6)
                    saturated = robot.data.actuator_force.abs() >= 0.95 * force_limit
                    force_saturation_totals[active] += saturated[active].double()
                counts += active.long()
                failed |= active & env.reset_terminated.bool()
                for term_id, name in enumerate(termination_names):
                    failure_terms[:, term_id] |= active & env.termination_manager.get_term(name)
                active = active & ~dones.bool() & (counts < horizons)
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if done_ids.numel():
                    memory_before = (
                        (wrapped.mean[active].clone(), wrapped.count[active].clone())
                        if getattr(actor, "context_latent_mean", False) else None
                    )
                    reset_finished_worlds(env, done_ids, active)
                    if memory_before is not None and (
                        not torch.equal(memory_before[0], wrapped.mean[active])
                        or not torch.equal(memory_before[1], wrapped.count[active])
                    ):
                        raise RuntimeError("Partial reset changed surviving-world context memory")
                    reset_batches += 1
                if (step + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step + 1,
                                "active": int(active.sum()),
                                "failed": int(failed.sum()),
                                "seconds": time.time() - wall_start,
                            }
                        ),
                        flush=True,
                    )
                if not active.any():
                    break
        means = totals / counts[:, None].clamp_min(1)
        result = {
            "protocol": PROTOCOL,
            "reference_timeline_audited": True,
            "partial_reset_survivor_state_audited": True,
            "partial_reset_survivor_history_audited": True,
            "partial_reset_batches": reset_batches,
            "checkpoint": args.checkpoint or args.tracker_checkpoint,
            "checkpoint_sha256": _sha256(Path(args.checkpoint or args.tracker_checkpoint)),
            "training_reward_audit": training_reward_audit,
            "physics": physics,
            "physics_world_fingerprints": physics_world_fingerprints,
            "seed": args.seed,
            "motions": len(files),
            "episodes": n,
            "repeats_per_motion": args.repeats,
            "motion_file": motion_file,
            "simulator_preview": preview_mode,
            "simulator_preview_version": preview_version,
            "simulator_preview_horizon": 5 if preview_mode else None,
            "simulator_preview_critic": preview_critic,
            "simulator_preview_override": getattr(args, "simulator_preview_override", None),
            "max_steps": args.steps,
            "metric_convention": "per-step errors averaged within episode, then equally across episodes/motions",
            "actor_input_contract": "deployable groups only"
            if student
            else "checkpoint actor contract",
            "context_ablation": args.context_ablation if student else None,
            "correction_use_privilege": getattr(actor, "correction_use_privilege", None),
            "correction_scale": getattr(actor, "correction_scale", None),
            "context_latent_mean": getattr(actor, "context_latent_mean", False),
            "context_mean_only": getattr(actor, "context_mean_only", False),
            "actor_physics_input": getattr(actor, "actor_physics_input", "not_applicable"),
            "context_mean_horizon": getattr(actor, "context_mean_horizon", None),
            "exported_policy": args.exported_policy,
            "exported_policy_sha256": exported_sha256,
            "exported_cpu_threads": exported_cpu_threads,
            "privileged_bias_compensation": args.bias_compensation,
            "privileged_payload_gravity_compensation": args.payload_gravity_compensation,
            "oracle_tracking_features": args.oracle_tracking_features
            or getattr(actor, "oracle_tracking_features", False),
            "oracle_feature_mix": (
                float(actor.oracle_feature_mix)
                if hasattr(actor, "oracle_feature_mix") else None
            ),
            "oracle_clean_proprio": getattr(actor, "oracle_clean_proprio", False),
            "context_imu_accel": getattr(actor, "context_imu_accel", False),
            "context_root_orientation": getattr(actor, "context_root_orientation", False),
            "context_right_aligned": getattr(actor, "context_right_aligned", False),
            "context_key_body": getattr(actor, "context_key_body", False),
            "teacher_feature_ablation": args.teacher_feature_ablation,
            "teacher_remove_privilege": state_ablation,
            "action_filter_strength": args.action_filter_strength,
            "action_filter_order": args.action_filter_order,
            "orientation_filter_weight": args.orientation_filter_weight,
            "bias_correction_rms": sum(compensation_rms) / max(len(compensation_rms), 1),
            "bias_correction_abs_max": compensation_max,
            "mean": dict(zip(METRICS, means.mean(0).cpu().tolist(), strict=True)),
            "pooled_mean": dict(
                zip(METRICS, (totals.sum(0) / counts.sum()).cpu().tolist(), strict=True)
            ),
            "failure_rate": float(failed.float().mean()),
            "coverage_fraction": float((counts.float() / horizons).mean()),
            "episode_lengths": counts.cpu().tolist(),
            "failed": failed.cpu().tolist(),
            "failure_term_names": termination_names,
            "failure_terms": failure_terms.cpu().tolist(),
            "motion_ids": ids.cpu().tolist(),
            "start_frames": initial_steps.cpu().tolist(),
            "per_episode_metrics": means.cpu().tolist(),
            "metric_names": list(METRICS),
            "motion_files": list(command.motion_files),
            "wall_seconds": time.time() - wall_start,
        }
        if args.joint_diagnostics:
            result["joint_diagnostics"] = {
                "joint_names": list(env.scene["robot"].joint_names),
                "joint_rmse": (joint_error_totals / counts[:, None].clamp_min(1))
                .mean(0)
                .sqrt()
                .cpu()
                .tolist(),
                "actuator_names": list(env.scene["robot"].actuator_names),
                "actuator_saturation_fraction": (
                    force_saturation_totals / counts[:, None].clamp_min(1)
                )
                .mean(0)
                .cpu()
                .tolist(),
                "saturation_convention": "at least 95% force limit at final physics substep; not whole-substep duty cycle",
            }
        if body_error_totals is not None:
            body_means = (body_error_totals / counts[:, None, None].clamp_min(1)).mean(0)
            names = ("error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel")
            for index, name in enumerate(names):
                if abs(float(body_means[:, index].mean()) - result["mean"][name]) > 1e-6:
                    raise RuntimeError(f"Per-body decomposition differs from aggregate {name}")
            result["body_diagnostics"] = {
                "body_names": list(command.cfg.body_names),
                "metric_names": names,
                "per_body_mean": body_means.cpu().tolist(),
                "aggregate_identity_audited": True,
            }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("mean", "failure_rate", "coverage_fraction", "wall_seconds")
                }
            ),
            flush=True,
        )
    finally:
        if preview_wrapper is not None:
            preview_wrapper.close_preview()
        if compensation is not None:
            compensation.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--checkpoint")
    parser.add_argument("--motion-path", default=DATASET)
    parser.add_argument("--motion-file", help="Evaluate exactly one motion")
    parser.add_argument("--motion-manifest", help="Bounded batch of absolute NPZ paths")
    parser.add_argument("--max-envs", type=int, default=4096)
    parser.add_argument("--dr-profile", choices=("right-hand-1-3kg", "hands-shins-2-4kg"),
                        help="Default: checkpoint profile, or legacy profile for frozen tracker")
    parser.add_argument("--simulator-preview-override", choices=("true", "zero", "shuffle"))
    parser.add_argument("--physics", choices=("nominal", "dr", "hardest"), required=True)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-tracking-features", action="store_true")
    parser.add_argument("--joint-diagnostics", action="store_true")
    parser.add_argument("--body-diagnostics", action="store_true")
    parser.add_argument("--payload-gravity-compensation", type=float, default=0.0, help="Analytical payload-only gravity feasibility probe; never a trained-policy acceptance result")
    parser.add_argument("--action-filter-strength", type=float, default=0.0)
    parser.add_argument("--action-filter-order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--orientation-filter-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-remove-privilege", choices=("normal", "height", "contact", "height_contact"),
        default="normal", help="Replace selected true labels in BOTH tracking frontend and privileged latent by deployable estimator predictions",
    )
    parser.add_argument(
        "--teacher-feature-ablation",
        choices=("normal", "estimated_height_contact", "estimated_reference", "estimated_all"),
        default="normal",
    )
    parser.add_argument(
        "--exported-policy", help="Audited CPU TorchScript artifact for a context actor"
    )
    parser.add_argument(
        "--bias-compensation",
        type=float,
        default=0.0,
        help="Privileged analytic feasibility probe; not a learned-policy result",
    )
    parser.add_argument(
        "--context-ablation", choices=("normal", "zero", "shuffle"), default="normal"
    )
    args = parser.parse_args()
    if not math.isfinite(args.payload_gravity_compensation) or (
        args.payload_gravity_compensation and args.bias_compensation
    ):
        parser.error("Payload gravity probe must be finite and separate from nominal-bias compensation")
    if args.repeats < 1 or args.steps < 1:
        parser.error("repeats and steps must be positive")
    if not 0 < args.orientation_filter_weight <= 1:
        parser.error("orientation-filter-weight must be in (0,1]")
    if args.orientation_filter_weight != 1.0 and args.oracle_tracking_features:
        parser.error("Orientation diagnostic cannot combine with raw oracle-tracker mode")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    evaluate(args)


if __name__ == "__main__":
    main()
