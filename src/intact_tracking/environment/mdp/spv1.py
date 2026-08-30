from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch

from intact_tracking.environment.assets.robots import SPV1_JOINT_TORQUE_SENSOR_PREFIX

from . import sp as sp_mdp
from .multi_commands import MultiMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


SPV1_REFERENCE_STEPS = (0, 1, 2, 3, 4, 5, 6)


def _command(env: ManagerBasedRlEnv, command_name: str) -> MultiMotionCommand:
    return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


def _root_reference(
    env: ManagerBasedRlEnv,
    command_name: str,
    field_name: str,
    root_body_name: str | None,
) -> torch.Tensor:
    return sp_mdp._root_motion(
        env,
        command_name,
        field_name,
        SPV1_REFERENCE_STEPS,
        horizon="teacher",
        root_body_name=root_body_name,
    )


def _shared_actor_observation(
    env: ManagerBasedRlEnv,
    command_name: str,
    key: tuple[object, ...],
    value: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    """Share one noisy sensor sample between history and explicit-error terms."""
    command = _command(env, command_name)
    cache = getattr(command, "_shared_spv1_observation_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        command._shared_spv1_observation_cache = cache
    cached = cache.get(key)
    if cached is None:
        cached = sp_mdp._uniform_noise(value, noise_std)
        cache[key] = cached
    return cached


def root_pos_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    """Six reference-internal future pelvis offsets in the current reference frame."""
    pos_w = _root_reference(env, command_name, "body_pos_w", root_body_name)
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    offsets_w = pos_w[:, 1:] - pos_w[:, :1]
    offsets_b = sp_mdp._quat_apply_inverse(quat_w[:, :1], offsets_w)
    return offsets_b.reshape(env.num_envs, -1)


def root_ori_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    """Six future reference rotations relative to the current reference pelvis."""
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    relative = sp_mdp._quat_in_frame(quat_w[:, :1], quat_w[:, 1:])
    return sp_mdp._rot6d(relative).reshape(env.num_envs, -1)


def ref_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    target = sp_mdp._gather(env, command_name, "joint_pos", SPV1_REFERENCE_STEPS)
    return target.reshape(env.num_envs, -1)


def ref_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    target = sp_mdp._gather(env, command_name, "joint_vel", SPV1_REFERENCE_STEPS)
    return target.reshape(env.num_envs, -1)


def ref_projected_gravity(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    return sp_mdp._projected_gravity(quat_w).reshape(env.num_envs, -1)


def ref_base_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    """Reference angular velocity in each reference pelvis's own local frame."""
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    ang_vel_w = _root_reference(env, command_name, "body_ang_vel_w", root_body_name)
    return sp_mdp._quat_apply_inverse(quat_w, ang_vel_w).reshape(env.num_envs, -1)


def joint_pos(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    biased: bool = True,
    noise_std: float = 0.0,
) -> torch.Tensor:
    asset = env.scene["robot"]
    data = asset.data
    measured = (
        data.joint_pos_biased
        if biased and isinstance(getattr(data, "joint_pos_biased", None), torch.Tensor)
        else data.joint_pos
    )
    default = data.default_joint_pos
    value = measured - default
    return _shared_actor_observation(
        env, command_name, ("joint_pos", bool(biased), float(noise_std)), value, noise_std
    )


def joint_vel(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    noise_std: float = 0.0,
) -> torch.Tensor:
    data = env.scene["robot"].data
    value = data.joint_vel - data.default_joint_vel
    return _shared_actor_observation(
        env, command_name, ("joint_vel", float(noise_std)), value, noise_std
    )


def projected_gravity(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    noise_std: float = 0.0,
) -> torch.Tensor:
    value = env.scene["robot"].data.projected_gravity_b
    return _shared_actor_observation(
        env, command_name, ("projected_gravity", float(noise_std)), value, noise_std
    )


def base_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    sensor_name: str = "robot/imu_ang_vel",
    noise_std: float = 0.0,
) -> torch.Tensor:
    value = env.scene[sensor_name].data
    return _shared_actor_observation(
        env,
        command_name,
        ("base_ang_vel", sensor_name, float(noise_std)),
        value,
        noise_std,
    )


def _instant_joint_torque(
    env: ManagerBasedRlEnv,
    sensor_prefix: str,
) -> torch.Tensor:
    values = []
    for joint_name in env.scene["robot"].joint_names:
        value = env.scene[f"robot/{sensor_prefix}{joint_name}"].data
        values.append(value.reshape(env.num_envs, -1))
    return torch.cat(values, dim=-1)


def joint_torque(
    env: ManagerBasedRlEnv,
    sensor_prefix: str = SPV1_JOINT_TORQUE_SENSOR_PREFIX,
    sample_mode: Literal["substep_average", "latest"] = "substep_average",
) -> torch.Tensor:
    """Read joint torque as a substep mean or one control-rate latest sample."""
    if sample_mode == "latest":
        return _instant_joint_torque(env, sensor_prefix)
    if sample_mode != "substep_average":
        raise ValueError(
            f"joint_torque sample_mode must be 'substep_average' or 'latest', got {sample_mode!r}"
        )
    cache = getattr(env, "_sp_substep_tracking_cache", None)
    get_average = getattr(cache, "joint_torque_average", None)
    if callable(get_average):
        return get_average()
    return _instant_joint_torque(env, sensor_prefix)


def joint_pos_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    biased: bool = True,
    noise_std: float = 0.0,
) -> torch.Tensor:
    target = sp_mdp._gather(env, command_name, "joint_pos", (0,))[:, 0]
    return target - joint_pos(env, command_name, biased, noise_std)


def joint_vel_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    noise_std: float = 0.0,
) -> torch.Tensor:
    target = sp_mdp._gather(env, command_name, "joint_vel", (0,))[:, 0]
    return target - joint_vel(env, command_name, noise_std)


def projected_gravity_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
    noise_std: float = 0.0,
) -> torch.Tensor:
    target_quat = _root_reference(env, command_name, "body_quat_w", root_body_name)[:, 0]
    target = sp_mdp._projected_gravity(target_quat)
    return target - projected_gravity(env, command_name, noise_std)


def base_ang_vel_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
    sensor_name: str = "robot/imu_ang_vel",
    noise_std: float = 0.0,
) -> torch.Tensor:
    target_quat = _root_reference(env, command_name, "body_quat_w", root_body_name)[:, 0]
    target_w = _root_reference(env, command_name, "body_ang_vel_w", root_body_name)[:, 0]
    target_b = sp_mdp._quat_apply_inverse(target_quat, target_w)
    return target_b - base_ang_vel(env, command_name, sensor_name, noise_std)
