from __future__ import annotations

from collections import OrderedDict
from typing import Any

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from intact_tracking.environment import mdp
from intact_tracking.environment.assets.robots import (
    get_g1_tracking_bfm_mesh_arm_spv1_robot_cfg,
    get_g1_tracking_bfm_robot_cfg,
    get_g1_tracking_bfm_spv1_robot_cfg,
)
from intact_tracking.environment.mdp import randomizations as sp_randomizations
from intact_tracking.environment.mdp import sp as sp_mdp
from intact_tracking.environment.mdp import spv1 as spv1_mdp
from intact_tracking.environment.mdp import spv2 as spv2_mdp
from intact_tracking.environment.mdp import spv3 as spv3_mdp
from intact_tracking.environment.mdp import spv4 as spv4_mdp
from intact_tracking.environment.mdp import spv5 as spv5_mdp
from intact_tracking.environment.mdp import spv5_2 as spv5_2_mdp
from intact_tracking.environment.mdp import spv6 as spv6_mdp
from intact_tracking.environment.mdp import spv7 as spv7_mdp
from intact_tracking.environment.mdp import spv8 as spv8_mdp
from intact_tracking.environment.mdp.actions import (
    ObservationHistoryJointPositionActionCfg,
    SpTrackingJointPositionActionCfg,
)
from intact_tracking.environment.mdp.multi_command_largedataset import (
    MotionCommandCfg as LargeDatasetMotionCommandCfg,
)
from intact_tracking.environment.mdp.multi_commands import (
    AdaptiveSamplingCfg,
    RewindCfg,
)
from intact_tracking.environment.mdp.multi_commands import (
    MotionCommandCfg as MultiMotionCommandCfg,
)

OBS_TERMS = {
    "generated_commands": mdp.generated_commands,
    "reference_joint_state_window": mdp.reference_joint_state_window,
    "ref_limb_ee_pose_b": mdp.ref_limb_ee_pose_b,
    "robot_limb_ee_pose_b": mdp.robot_limb_ee_pose_b,
    "motion_ref_ang_vel": mdp.motion_ref_ang_vel,
    "motion_anchor_pos_b": mdp.motion_anchor_pos_b,
    "motion_anchor_ori_b": mdp.motion_anchor_ori_b,
    "gradient_test_motion_label": mdp.gradient_test_motion_label,
    "gradient_test_motion_phase": mdp.gradient_test_motion_phase,
    "motion_resample_boundary": mdp.motion_resample_boundary,
    "robot_body_pos_b": mdp.robot_body_pos_b,
    "robot_body_ori_b": mdp.robot_body_ori_b,
    "builtin_sensor": mjlab_mdp.builtin_sensor,
    "projected_gravity": mjlab_mdp.projected_gravity,
    "joint_pos_rel": mjlab_mdp.joint_pos_rel,
    "joint_vel_rel": mjlab_mdp.joint_vel_rel,
    "last_action": mjlab_mdp.last_action,
    "boot_indicator_state_obs": sp_mdp.boot_indicator_state_obs,
    "command_obs": sp_mdp.command_obs,
    "target_joint_pos_obs": sp_mdp.target_joint_pos_obs,
    "target_root_z_obs": sp_mdp.target_root_z_obs,
    "target_projected_gravity_b_obs": sp_mdp.target_projected_gravity_b_obs,
    "target_pos_b_obs": sp_mdp.target_pos_b_obs,
    "target_rot_b_obs": sp_mdp.target_rot_b_obs,
    "target_linvel_b_obs": sp_mdp.target_linvel_b_obs,
    "target_angvel_b_obs": sp_mdp.target_angvel_b_obs,
    "current_keypoint_pos_b_obs": sp_mdp.current_keypoint_pos_b_obs,
    "current_keypoint_rot_b_obs": sp_mdp.current_keypoint_rot_b_obs,
    "current_keypoint_linvel_b_obs": sp_mdp.current_keypoint_linvel_b_obs,
    "current_keypoint_angvel_b_obs": sp_mdp.current_keypoint_angvel_b_obs,
    "target_keypoints_pos_b_obs": sp_mdp.target_keypoints_pos_b_obs,
    "target_keypoints_rot_b_obs": sp_mdp.target_keypoints_rot_b_obs,
    "root_linvel_b_history": sp_mdp.root_linvel_b_history,
    "root_angvel_b_history": sp_mdp.root_angvel_b_history,
    "projected_gravity_history": sp_mdp.projected_gravity_history,
    "joint_pos_history": sp_mdp.joint_pos_history,
    "joint_vel_history": sp_mdp.joint_vel_history,
    "prev_actions": sp_mdp.prev_actions,
    "applied_action": sp_mdp.applied_action,
    "applied_torque": sp_mdp.applied_torque,
    "body_z_termination_obs": sp_mdp.body_z_termination_obs,
    "gravity_dir_termination_obs": sp_mdp.gravity_dir_termination_obs,
    "feet_contact_state": sp_mdp.feet_contact_state,
    "feet_contact_binary_state": sp_mdp.feet_contact_binary_state,
    "target_feet_contact_state_obs": sp_mdp.target_feet_contact_state_obs,
    "domain_motor_params_implicit": sp_mdp.domain_motor_params_implicit,
    "domain_perturb_body_materials": sp_mdp.domain_perturb_body_materials,
    "domain_random_joint_offset": sp_mdp.domain_random_joint_offset,
    "domain_perturb_gravity": sp_mdp.domain_perturb_gravity,
    "spv1_root_pos_command": spv1_mdp.root_pos_command,
    "spv1_root_ori_command": spv1_mdp.root_ori_command,
    "spv1_ref_joint_pos": spv1_mdp.ref_joint_pos,
    "spv1_ref_joint_vel": spv1_mdp.ref_joint_vel,
    "spv1_ref_projected_gravity": spv1_mdp.ref_projected_gravity,
    "spv1_ref_base_ang_vel": spv1_mdp.ref_base_ang_vel,
    "spv1_joint_pos": spv1_mdp.joint_pos,
    "spv1_joint_vel": spv1_mdp.joint_vel,
    "spv1_projected_gravity": spv1_mdp.projected_gravity,
    "spv1_base_ang_vel": spv1_mdp.base_ang_vel,
    "spv1_joint_torque": spv1_mdp.joint_torque,
    "spv1_joint_pos_error": spv1_mdp.joint_pos_error,
    "spv1_joint_vel_error": spv1_mdp.joint_vel_error,
    "spv1_projected_gravity_error": spv1_mdp.projected_gravity_error,
    "spv1_base_ang_vel_error": spv1_mdp.base_ang_vel_error,
    "spv2_root_pos_command": spv2_mdp.root_pos_command,
    "spv2_root_ori_command": spv2_mdp.root_ori_command,
    "spv2_ref_joint_pos": spv2_mdp.ref_joint_pos,
    "spv2_ref_joint_vel": spv2_mdp.ref_joint_vel,
    "spv2_ref_projected_gravity": spv2_mdp.ref_projected_gravity,
    "spv2_ref_base_ang_vel": spv2_mdp.ref_base_ang_vel,
    "spv2_ref_root_height": spv2_mdp.ref_root_height,
    "spv2_ref_root_lin_vel": spv2_mdp.ref_root_lin_vel,
    "spv3_root_height_gt": spv3_mdp.root_height_gt,
    "spv3_root_lin_vel_b_gt": spv3_mdp.root_lin_vel_b_gt,
    "spv4_robot_key_body_state": spv4_mdp.robot_key_body_state,
    "spv4_ref_key_body_state": spv4_mdp.ref_key_body_state,
    "spv4_key_body_error": spv4_mdp.key_body_error,
    "spv5_reference_encoder_input": spv5_mdp.reference_encoder_input,
    "spv5_reference_encoder_target": spv5_mdp.reference_encoder_target,
    "spv5_robot_root_quat": spv5_mdp.robot_root_quat,
    "spv5_2_robot_key_body_state": spv5_2_mdp.robot_key_body_state,
    "spv6_normalized_physical_context": spv6_mdp.normalized_physical_context,
    "spv6_normalized_armature_context": spv6_mdp.normalized_armature_context,
    "spv6_normalized_past_force": spv6_mdp.normalized_past_force,
    "spv7_pmoe_reference_input": spv7_mdp.pmoe_reference_input,
    "spv7_v26_reference_input": spv7_mdp.v26_reference_input,
    "spv8_v29_reference_input": spv8_mdp.v29_reference_input,
}

REWARD_TERMS = {
    "motion_global_anchor_position_error_exp": mdp.motion_global_anchor_position_error_exp,
    "motion_global_anchor_orientation_error_exp": mdp.motion_global_anchor_orientation_error_exp,
    "motion_relative_body_position_error_exp": mdp.motion_relative_body_position_error_exp,
    "motion_relative_body_orientation_error_exp": mdp.motion_relative_body_orientation_error_exp,
    "motion_global_body_linear_velocity_error_exp": mdp.motion_global_body_linear_velocity_error_exp,
    "motion_global_body_angular_velocity_error_exp": mdp.motion_global_body_angular_velocity_error_exp,
    "action_rate_l2": mdp.action_rate_l2,
    "joint_action_rate_l2": mdp.joint_action_rate_l2,
    "joint_pos_limits": mdp.joint_pos_limits,
    "self_collision_cost": mdp.self_collision_cost,
    "root_pos_tracking": sp_mdp.root_pos_tracking,
    "global_link_height_tracking": sp_mdp.global_link_height_tracking,
    "root_rot_tracking": sp_mdp.root_rot_tracking,
    "root_vel_tracking": sp_mdp.root_vel_tracking,
    "root_ang_vel_tracking": sp_mdp.root_ang_vel_tracking,
    "keypoint_pos_tracking": sp_mdp.keypoint_pos_tracking,
    "keypoint_vel_tracking": sp_mdp.keypoint_vel_tracking,
    "keypoint_rot_tracking": sp_mdp.keypoint_rot_tracking,
    "keypoint_angvel_tracking": sp_mdp.keypoint_angvel_tracking,
    "joint_pos_tracking": sp_mdp.joint_pos_tracking,
    "joint_vel_tracking": sp_mdp.joint_vel_tracking,
    "survival": sp_mdp.survival,
    "joint_vel_l2": sp_mdp.joint_vel_l2,
    "action_rate_l2_sp": sp_mdp.action_rate_l2,
    "feet_air_time_ref": sp_mdp.feet_air_time_ref,
    "feet_air_time_ref_dense": sp_mdp.feet_air_time_ref_dense,
    "joint_pos_limits_sp": sp_mdp.joint_pos_limits,
    "joint_torque_limits": sp_mdp.joint_torque_limits,
    "loco_reward_group_schedule": sp_mdp.loco_reward_group_schedule,
}

TERMINATION_TERMS = {
    "time_out": mjlab_mdp.time_out,
    "bad_anchor_pos_z_only": mdp.bad_anchor_pos_z_only,
    "bad_anchor_ori": mdp.bad_anchor_ori,
    "bad_motion_body_pos_global": mdp.bad_motion_body_pos_global,
    "bad_motion_body_pos_z_only": mdp.bad_motion_body_pos_z_only,
    "body_z_termination": sp_mdp.body_z_termination,
    "gravity_dir_termination": sp_mdp.gravity_dir_termination,
    "motion_timeout": sp_mdp.motion_timeout,
    "motion_xy_range_termination": sp_mdp.motion_xy_range_termination,
}

EVENT_TERMS = {
    "perturb_body_com": sp_randomizations.perturb_body_com,
    "perturb_body_materials": sp_randomizations.perturb_body_materials,
    "motor_params_implicit": sp_randomizations.motor_params_implicit,
    "random_joint_offset": sp_randomizations.random_joint_offset,
    "perturb_root_vel": sp_randomizations.perturb_root_vel,
    "perturb_body_wrench": sp_randomizations.perturb_body_wrench,
    "perturb_gravity": sp_randomizations.perturb_gravity,
}

CURRICULUM_TERMS = {
    "sp_tracking_progress": sp_randomizations.sp_tracking_progress,
}

METRICS_TERMS = {
    # A per-substep sampler shared by SP's contact rewards and uncorrupted joint
    # histories.  It is opt-in in task YAML, so existing tasks remain unchanged.
    "substep_tracking_cache": sp_mdp.substep_tracking_cache,
}


def _to_container(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _to_tuple(value: Any) -> tuple[Any, ...]:
    value = _to_container(value)
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _optional_int(value: Any) -> int | None:
    value = _to_container(value)
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_auto_float(value: Any) -> float | str | None:
    value = _to_container(value)
    if value is None or value == "auto":
        return value
    return float(value)


def _params(raw: Any | None) -> dict[str, Any]:
    params = _to_container(raw) if raw is not None else {}
    params = dict(params)
    for key, value in list(params.items()):
        if key == "asset_cfg" and isinstance(value, dict):
            params[key] = _scene_entity_cfg(value)
        elif isinstance(value, list):
            params[key] = tuple(value)
        elif isinstance(value, dict):
            params[key] = {
                nested_key: tuple(nested_value) if isinstance(nested_value, list) else nested_value
                for nested_key, nested_value in value.items()
            }
    return params


def _scene_entity_cfg(data: dict[str, Any]) -> SceneEntityCfg:
    kwargs = dict(data)
    entity_name = kwargs.pop("entity_name")
    for key, value in list(kwargs.items()):
        if isinstance(value, list):
            kwargs[key] = tuple(value)
    return SceneEntityCfg(entity_name, **kwargs)


def _noise(raw: Any | None) -> UniformNoiseCfg | None:
    if raw is None:
        return None
    n_min, n_max = _to_container(raw)
    return UniformNoiseCfg(n_min=float(n_min), n_max=float(n_max))


def _build_observations(cfg: DictConfig) -> dict[str, ObservationGroupCfg]:
    obs_cfg = cfg.observations if "observations" in cfg else cfg.obs.observations
    groups = OrderedDict()
    for group_name, group_cfg in obs_cfg.items():
        if not bool(group_cfg.get("enabled", True)):
            continue
        disabled_terms = {str(name) for name in group_cfg.get("disabled_terms", ())}
        terms = OrderedDict()
        for item in group_cfg.terms:
            term_name = str(item.name)
            if term_name in disabled_terms or not bool(item.get("enabled", True)):
                continue
            term_key = str(item.term)
            terms[term_name] = ObservationTermCfg(
                func=OBS_TERMS[term_key],
                params=_params(item.get("params")),
                noise=_noise(item.get("noise")),
                history_length=int(item.get("history_length", 0)),
            )
        groups[group_name] = ObservationGroupCfg(
            terms=terms,
            concatenate_terms=bool(group_cfg.get("concatenate_terms", True)),
            enable_corruption=bool(group_cfg.get("enable_corruption", False)),
            nan_policy=str(group_cfg.get("nan_policy", "disabled")),
            nan_check_per_term=bool(group_cfg.get("nan_check_per_term", True)),
        )
    return groups


def _build_rewards(cfg: DictConfig) -> dict[str, RewardTermCfg]:
    reward_cfg = cfg.rewards if "rewards" in cfg else cfg.reward.rewards
    rewards = OrderedDict()
    for item in reward_cfg:
        rewards[str(item.name)] = RewardTermCfg(
            func=REWARD_TERMS[str(item.term)],
            weight=float(item.weight),
            params=_params(item.get("params")),
        )
    return rewards


def _build_terminations(cfg: DictConfig) -> dict[str, TerminationTermCfg]:
    terminations = OrderedDict()
    for item in cfg.terminations:
        terminations[str(item.name)] = TerminationTermCfg(
            func=TERMINATION_TERMS[str(item.term)],
            params=_params(item.get("params")),
            time_out=bool(item.get("time_out", False)),
        )
    return terminations


def _build_command(cfg: DictConfig):
    command_cfg = cfg.command.command if "command" in cfg.command else cfg.command
    kwargs = {
        "entity_name": str(command_cfg.entity_name),
        "motion_path": str(command_cfg.get("motion_path", "")),
        "motion_file": str(command_cfg.get("motion_file", "")),
        "motion_manifest_file": str(command_cfg.get("motion_manifest_file", "")),
        "motion_manifest_wait_timeout_s": float(
            command_cfg.get("motion_manifest_wait_timeout_s", 600.0)
        ),
        "motion_manifest_poll_interval_s": float(
            command_cfg.get("motion_manifest_poll_interval_s", 0.25)
        ),
        "motion_scan_backend": str(command_cfg.get("motion_scan_backend", "auto")),
        "motion_scan_workers": int(command_cfg.get("motion_scan_workers", 0)),
        "motion_scan_fd_executable": str(command_cfg.get("motion_scan_fd_executable", "fd")),
        "motion_scan_log_interval_s": float(command_cfg.get("motion_scan_log_interval_s", 10.0)),
        "excluded_motion_files": _to_tuple(command_cfg.get("excluded_motion_files")),
        "motion_exclude_files": _to_tuple(command_cfg.get("motion_exclude_files")),
        "motion_exclude_file": str(command_cfg.get("motion_exclude_file", "")),
        "extra_reference_motion_file": str(command_cfg.get("extra_reference_motion_file", "")),
        "gradient_test_mode": (
            str(command_cfg.gradient_test_mode)
            if command_cfg.get("gradient_test_mode") is not None
            else None
        ),
        "gradient_test_simple_motion_file": str(
            command_cfg.get("gradient_test_simple_motion_file", "")
        ),
        "gradient_test_hard_motion_file": str(
            command_cfg.get("gradient_test_hard_motion_file", "")
        ),
        "motion_type": str(command_cfg.motion_type),
        "fk_from_joint_pos": bool(command_cfg.get("fk_from_joint_pos", False)),
        "recompute_joint_vel_from_joint_pos": bool(
            command_cfg.get("recompute_joint_vel_from_joint_pos", False)
        ),
        "load_compact_qpos": bool(command_cfg.get("load_compact_qpos", False)),
        "reference_storage_mode": str(command_cfg.get("reference_storage_mode", "full")),
        "actor_reference_fps": float(command_cfg.get("actor_reference_fps", 50.0)),
        "qpos_body_pose_cache_tile_steps": int(
            command_cfg.get("qpos_body_pose_cache_tile_steps", 0)
        ),
        "qpos_body_pose_cache_range": tuple(
            int(step) for step in command_cfg.get("qpos_body_pose_cache_range", (-8, 20))
        ),
        "motion_origin_recenter": bool(command_cfg.get("motion_origin_recenter", False)),
        "sliding_root_xy_reward": bool(command_cfg.get("sliding_root_xy_reward", False)),
        "boot_indicator_max": int(command_cfg.get("boot_indicator_max", 0)),
        "termination_warmup_steps": int(command_cfg.get("termination_warmup_steps", 0)),
        "feet_standing_body_names": _to_tuple(command_cfg.get("feet_standing_body_names")),
        "feet_standing": _params(command_cfg.get("feet_standing")),
        "student_motion_randomization": _params(command_cfg.get("student_motion_randomization")),
        "resample_on_motion_end": bool(command_cfg.get("resample_on_motion_end", True)),
        "anchor_body_name": str(command_cfg.anchor_body_name),
        "body_names": _to_tuple(command_cfg.body_names),
        "pose_range": _params(command_cfg.get("pose_range")),
        "velocity_range": _params(command_cfg.get("velocity_range")),
        "joint_position_range": tuple(command_cfg.joint_position_range),
        "reset_root_lift_height": float(command_cfg.get("reset_root_lift_height", 0.0)),
        "reset_min_body_z": _optional_float(command_cfg.get("reset_min_body_z")),
        "reset_joint_vel_limit": _optional_float(command_cfg.get("reset_joint_vel_limit")),
        "future_steps": int(command_cfg.get("future_steps", 5)),
        "history_steps": int(command_cfg.get("history_steps", 5)),
        "reference_cache_enabled": bool(command_cfg.get("reference_cache_enabled", True)),
        "reference_cache_steps": (
            {
                str(field_name): tuple(int(step) for step in steps)
                for field_name, steps in command_cfg.reference_cache_steps.items()
            }
            if command_cfg.get("reference_cache_steps") is not None
            else None
        ),
        "adaptive_uniform_ratio": float(command_cfg.get("adaptive_uniform_ratio", 0.1)),
        "adaptive_sampling": AdaptiveSamplingCfg(
            **dict(_to_container(command_cfg.get("adaptive_sampling", {})) or {})
        ),
        "rewind": RewindCfg(**dict(_to_container(command_cfg.get("rewind", {})) or {})),
        "adaptive_bin_width_s": float(command_cfg.get("adaptive_bin_width_s", 1.0)),
        "adaptive_bin_width_steps": _optional_int(command_cfg.get("adaptive_bin_width_steps")),
        "adaptive_prior_visit_count": float(command_cfg.get("adaptive_prior_visit_count", 1.0)),
        "adaptive_prior_failure_count": float(command_cfg.get("adaptive_prior_failure_count", 0.0)),
        "adaptive_failure_rate_ema_iterations": _optional_int(
            command_cfg.get("adaptive_failure_rate_ema_iterations")
        ),
        # Deprecated input alias retained for old launch overrides/checkpoints.
        "adaptive_failure_rate_window_iterations": _optional_int(
            command_cfg.get("adaptive_failure_rate_window_iterations")
        ),
        # ``adaptive_failure_rate_max_over_mean`` used to clip the raw signal.
        # Retain it only as an input alias for old configs that do not yet define
        # the final-distribution probability cap.
        "adaptive_probability_max_over_mean": float(
            command_cfg.get(
                "adaptive_probability_max_over_mean",
                command_cfg.get("adaptive_failure_rate_max_over_mean", 200.0),
            )
        ),
        "adaptive_sequence_length_agnostic": bool(
            command_cfg.get("adaptive_sequence_length_agnostic", False)
        ),
        "adaptive_max_prob_per_bin": _optional_auto_float(
            command_cfg.get("adaptive_max_prob_per_bin", "auto")
        ),
        "adaptive_max_prob_per_motion": _optional_auto_float(
            command_cfg.get("adaptive_max_prob_per_motion", "auto")
        ),
        "adaptive_pre_failure_sample_window_steps": int(
            command_cfg.get("adaptive_pre_failure_sample_window_steps", 200)
        ),
        "adaptive_bin_snapshot_interval_iterations": int(
            command_cfg.get("adaptive_bin_snapshot_interval_iterations", 0)
        ),
        "adaptive_bin_snapshot_dir": str(command_cfg.get("adaptive_bin_snapshot_dir", "")),
        "sampling_mode": str(command_cfg.get("sampling_mode", "adaptive")),
        "if_log_metrics": bool(command_cfg.get("if_log_metrics", True)),
        "collection_preforward_enabled": bool(
            command_cfg.get("collection_preforward_enabled", False)
        ),
        "resampling_time_range": tuple(command_cfg.resampling_time_range),
        "debug_vis": bool(command_cfg.get("debug_vis", True)),
    }
    if command_cfg.type == "large_dataset":
        if kwargs["reference_storage_mode"] != "full":
            raise ValueError(
                "reference_storage_mode=qpos_only_actor_fk is supported only by the "
                "multi command, not large_dataset"
            )
        kwargs.update(
            {
                "active_subset_size": int(command_cfg.get("active_subset_size", 20_000)),
                "subset_refresh_count": int(command_cfg.get("subset_refresh_count", 10)),
                "subset_min_resident_iterations": int(
                    command_cfg.get("subset_min_resident_iterations", 50)
                ),
                "subset_adaptive_refresh_ratio": float(
                    command_cfg.get("subset_adaptive_refresh_ratio", 0.5)
                ),
                "subset_adaptive_candidate_pool_size": int(
                    command_cfg.get("subset_adaptive_candidate_pool_size", 10_000)
                ),
                "adaptive_bin_pool_reset_interval_iterations": int(
                    command_cfg.get("adaptive_bin_pool_reset_interval_iterations", 0)
                ),
                "adaptive_bin_snapshot_num_buckets": int(
                    command_cfg.get("adaptive_bin_snapshot_num_buckets", 2048)
                ),
                "motion_metadata_cache_file": str(
                    command_cfg.get("motion_metadata_cache_file", "")
                ),
                "motion_metadata_cache_wait_timeout_s": float(
                    command_cfg.get("motion_metadata_cache_wait_timeout_s", 7200.0)
                ),
                "motion_metadata_cache_poll_interval_s": float(
                    command_cfg.get("motion_metadata_cache_poll_interval_s", 0.25)
                ),
                "motion_metadata_read_workers": int(
                    command_cfg.get("motion_metadata_read_workers", 0)
                ),
                "motion_metadata_read_backend": str(
                    command_cfg.get("motion_metadata_read_backend", "thread")
                ),
                "motion_metadata_read_chunksize": int(
                    command_cfg.get("motion_metadata_read_chunksize", 64)
                ),
            }
        )
        return LargeDatasetMotionCommandCfg(**kwargs)
    if command_cfg.type == "multi":
        return MultiMotionCommandCfg(**kwargs)
    raise ValueError(f"Unsupported command type: {command_cfg.type}")


def _build_events(cfg: DictConfig) -> dict[str, EventTermCfg]:
    if not bool(cfg.get("domain_randomization", True)):
        return {}

    command_cfg = cfg.command.command if "command" in cfg.command else cfg.command
    velocity_range = _params(command_cfg.velocity_range)
    robot = cfg.robot
    events = OrderedDict()
    if cfg.events.get("push_robot") and cfg.events.push_robot.enabled:
        push_cfg = cfg.events.push_robot
        push_kind = str(push_cfg.get("kind", "velocity_kick"))
        if push_kind == "body_force_pulse":
            events["push_robot"] = EventTermCfg(
                func=sp_randomizations.body_force_pulse,
                mode="step",
                params={
                    "body_name": str(push_cfg.get("body_name", "torso_link")),
                    "interval_range_s": tuple(push_cfg.interval_range_s),
                    "duration_range_s": tuple(push_cfg.duration_range_s),
                    "force_abs_range_n": tuple(push_cfg.force_abs_range_n),
                },
            )
        elif push_kind == "velocity_kick":
            events["push_robot"] = EventTermCfg(
                func=(
                    sp_randomizations.recorded_push_by_setting_velocity
                    if bool(push_cfg.get("record_history", False))
                    else mdp.push_by_setting_velocity
                ),
                mode="interval",
                interval_range_s=tuple(push_cfg.interval_range_s),
                params={"velocity_range": velocity_range},
            )
        else:
            raise ValueError(f"Unsupported push_robot kind: {push_kind}")
    if cfg.events.get("base_com") and cfg.events.base_com.enabled:
        events["base_com"] = EventTermCfg(
            mode=str(cfg.events.base_com.get("mode", "startup")),
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=_to_tuple(robot.base_com_body_names)
                ),
                "operation": "add",
                "ranges": _params(cfg.events.base_com.ranges),
            },
        )
    if cfg.events.get("base_mass") and cfg.events.base_mass.enabled:
        events["base_mass"] = EventTermCfg(
            mode=str(cfg.events.base_mass.get("mode", "startup")),
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=_to_tuple(cfg.events.base_mass.body_names)
                ),
                "operation": "add",
                "ranges": tuple(cfg.events.base_mass.ranges),
            },
        )
    if cfg.events.get("encoder_bias") and cfg.events.encoder_bias.enabled:
        events["encoder_bias"] = EventTermCfg(
            mode=str(cfg.events.encoder_bias.get("mode", "startup")),
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": tuple(cfg.events.encoder_bias.bias_range),
            },
        )
    if cfg.events.get("foot_friction") and cfg.events.foot_friction.enabled:
        events["foot_friction"] = EventTermCfg(
            mode=str(cfg.events.foot_friction.get("mode", "startup")),
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=str(robot.foot_geom_pattern)),
                "operation": "abs",
                "ranges": tuple(cfg.events.foot_friction.ranges),
                "shared_random": bool(cfg.events.foot_friction.shared_random),
            },
        )
    for name, event_cfg in cfg.events.items():
        if name in {"push_robot", "base_com", "base_mass", "encoder_bias", "foot_friction"}:
            continue
        if not event_cfg.get("enabled", True):
            continue
        events[str(name)] = EventTermCfg(
            mode=str(event_cfg.mode),
            func=EVENT_TERMS[str(event_cfg.term)],
            interval_range_s=(
                tuple(event_cfg.interval_range_s) if event_cfg.get("interval_range_s") else None
            ),
            is_global_time=bool(event_cfg.get("is_global_time", False)),
            min_step_count_between_reset=int(event_cfg.get("min_step_count_between_reset", 0)),
            params=_params(event_cfg.get("params")),
        )
    return events


def _build_action(cfg: DictConfig):
    scale = (
        G1_ACTION_SCALE
        if cfg.action.scale == "g1_action_scale"
        else _to_container(cfg.action.scale)
    )
    action_type = str(cfg.action.get("type", "joint_position"))
    if action_type == "sp_tracking":
        cfg_cls = SpTrackingJointPositionActionCfg
    elif action_type == "joint_position_with_mean_history":
        cfg_cls = ObservationHistoryJointPositionActionCfg
    else:
        cfg_cls = JointPositionActionCfg
    extra_kwargs = {}
    if cfg_cls is SpTrackingJointPositionActionCfg:
        extra_kwargs = {
            "max_delay": int(cfg.action.get("max_delay", 2)),
            "delay_full_progress": float(cfg.action.get("delay_full_progress", 0.8)),
            "alpha": tuple(cfg.action.get("alpha", (0.8, 1.0))),
            "torque_limit_scale_range": tuple(
                cfg.action.get("torque_limit_scale_range", (1.0, 1.0))
            ),
            "torque_limit_progress_range": tuple(
                cfg.action.get("torque_limit_progress_range", (0.0, 0.8))
            ),
            "raw_action_clip": _optional_float(cfg.action.get("raw_action_clip")),
            "boot_delay_steps": int(cfg.action.get("boot_delay_steps", 0)),
            "curriculum_mode": str(cfg.action.get("curriculum_mode", "progressive")),
            "prev_action_obs": str(cfg.action.get("prev_action_obs", "sampled")),
            "action_rate_source": str(cfg.action.get("action_rate_source", "sampled")),
            "joint_name_order": (
                _to_tuple(cfg.action.joint_name_order)
                if cfg.action.get("joint_name_order") is not None
                else None
            ),
        }
    elif cfg_cls is ObservationHistoryJointPositionActionCfg:
        extra_kwargs = {
            "observation_history_steps": int(cfg.action.get("observation_history_steps", 8)),
            "joint_name_order": (
                _to_tuple(cfg.action.joint_name_order)
                if cfg.action.get("joint_name_order") is not None
                else None
            ),
        }
    return {
        "joint_pos": cfg_cls(
            entity_name=str(cfg.robot.entity_name),
            actuator_names=_to_tuple(cfg.action.actuator_names),
            scale=scale,
            use_default_offset=bool(cfg.action.use_default_offset),
            **extra_kwargs,
        )
    }


def _build_curriculum(cfg: DictConfig) -> dict[str, CurriculumTermCfg]:
    terms = OrderedDict()
    for name, term_cfg in cfg.get("curriculum", {}).items():
        if not term_cfg.get("enabled", True):
            continue
        terms[str(name)] = CurriculumTermCfg(
            func=CURRICULUM_TERMS[str(term_cfg.term)],
            params=_params(term_cfg.get("params")),
        )
    return terms


def _build_metrics(cfg: DictConfig) -> dict[str, MetricsTermCfg]:
    """Build optional reusable per-step/per-substep task instrumentation."""
    terms = OrderedDict()
    for name, term_cfg in cfg.get("metrics", {}).items():
        if not term_cfg.get("enabled", True):
            continue
        terms[str(name)] = MetricsTermCfg(
            func=METRICS_TERMS[str(term_cfg.term)],
            params=_params(term_cfg.get("params")),
            per_substep=bool(term_cfg.get("per_substep", False)),
            reduce=str(term_cfg.get("reduce", "mean")),
        )
    return terms


def _configured_contact_sensor_names(cfg: DictConfig) -> set[str]:
    """Return contact-sensor names explicitly requested by active task terms.

    Observation, reward, and metric modules can all consume a contact sensor.
    Keeping this dependency in the YAML term parameters means mixed ablations do
    not need task-name-specific sensor logic.  In particular, the SP substep
    cache consumes ``contact_forces`` even when both SP observations and rewards
    are replaced by their legacy counterparts.
    """
    sensor_names: set[str] = set()

    def add_sensor_name(term_cfg: Any) -> None:
        params = term_cfg.get("params")
        if params is None:
            return
        sensor_name = params.get("sensor_name")
        if sensor_name is not None:
            sensor_names.add(str(sensor_name))

    obs_cfg = cfg.observations if "observations" in cfg else cfg.obs.observations
    for group_cfg in obs_cfg.values():
        if not bool(group_cfg.get("enabled", True)):
            continue
        disabled_terms = {str(name) for name in group_cfg.get("disabled_terms", ())}
        for term_cfg in group_cfg.terms:
            if str(term_cfg.name) in disabled_terms or not bool(term_cfg.get("enabled", True)):
                continue
            add_sensor_name(term_cfg)

    reward_cfg = cfg.rewards if "rewards" in cfg else cfg.reward.rewards
    for term_cfg in reward_cfg:
        add_sensor_name(term_cfg)

    for term_cfg in cfg.get("metrics", {}).values():
        if term_cfg.get("enabled", True):
            add_sensor_name(term_cfg)

    return sensor_names


def _build_sensors(cfg: DictConfig):
    """Select sensors from active modules, rather than from a task name.

    This lets a future ablation put SP observations, rewards, or metrics on
    ``tracking_bfm`` without accidentally retaining only the old self-collision
    sensor.  Mixed presets get every contact sensor their active terms declare.
    """
    requested_sensors = _configured_contact_sensor_names(cfg)
    needs_contact_forces = "contact_forces" in requested_sensors
    needs_self_collision = "self_collision" in requested_sensors
    sensors = []
    if needs_contact_forces:
        sensors.append(
            ContactSensorCfg(
                name="contact_forces",
                primary=ContactMatch(
                    mode="subtree",
                    pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                    entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                global_frame=True,
                num_slots=1,
                track_air_time=True,
                history_length=3,
            )
        )
    if needs_self_collision or not sensors:
        sensors.append(
            ContactSensorCfg(
                name="self_collision",
                primary=ContactMatch(
                    mode="subtree",
                    pattern=str(cfg.robot.self_collision_primary_pattern),
                    entity="robot",
                ),
                secondary=ContactMatch(
                    mode="subtree",
                    pattern=str(cfg.robot.self_collision_primary_pattern),
                    entity="robot",
                ),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            )
        )
    return tuple(sensors)


def _build_robot(cfg: DictConfig):
    asset = str(cfg.robot.get("asset", "tracking_bfm_g1"))
    if asset == "tracking_bfm_g1":
        robot_cfg = get_g1_tracking_bfm_robot_cfg()
    elif asset == "tracking_bfm_spv1_g1":
        robot_cfg = get_g1_tracking_bfm_spv1_robot_cfg()
    elif asset == "tracking_bfm_mesh_arm_spv1_g1":
        robot_cfg = get_g1_tracking_bfm_mesh_arm_spv1_robot_cfg()
    else:
        raise ValueError(f"Unsupported in-repository G1 rollout asset: {asset!r}")
    joint_name_order = cfg.robot.get("joint_name_order")
    if joint_name_order is not None:
        # Policy-facing metadata only. It does not modify XML order, articulation,
        # action dynamics or any simulator parameter.
        robot_cfg.joint_name_order = _to_tuple(joint_name_order)
    return robot_cfg


def build_env_cfg(cfg: DictConfig | dict[str, Any]) -> ManagerBasedRlEnvCfg:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type=str(cfg.scene.terrain_type)),
        num_envs=int(cfg.num_envs),
        env_spacing=float(cfg.scene.env_spacing),
        entities={"robot": _build_robot(cfg)},
        sensors=_build_sensors(cfg),
    )
    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name=str(cfg.robot.viewer_body_name),
        distance=float(cfg.viewer.distance),
        fovy=float(cfg.viewer.fovy),
        elevation=float(cfg.viewer.elevation),
        azimuth=float(cfg.viewer.azimuth),
    )
    sim = SimulationCfg(
        nconmax=int(cfg.sim.nconmax),
        njmax=int(cfg.sim.njmax),
        contact_sensor_maxmatch=int(cfg.sim.get("contact_sensor_maxmatch", 64)),
        mujoco=MujocoCfg(
            timestep=float(cfg.sim.timestep),
            iterations=int(cfg.sim.iterations),
            ls_iterations=int(cfg.sim.ls_iterations),
        ),
    )
    return ManagerBasedRlEnvCfg(
        decimation=int(cfg.decimation),
        scene=scene,
        observations=_build_observations(cfg),
        actions=_build_action(cfg),
        commands={"motion": _build_command(cfg)},
        events=_build_events(cfg),
        rewards=_build_rewards(cfg),
        terminations=_build_terminations(cfg),
        curriculum=_build_curriculum(cfg),
        metrics=_build_metrics(cfg),
        viewer=viewer,
        sim=sim,
        episode_length_s=float(cfg.episode_length_s),
        seed=int(cfg.seed),
    )
