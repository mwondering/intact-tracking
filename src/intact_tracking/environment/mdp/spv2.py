from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import sp as sp_mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


SPV2_REFERENCE_STEPS = (0, 1, 2, 3, 4)


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
        SPV2_REFERENCE_STEPS,
        horizon="teacher",
        root_body_name=root_body_name,
    )


def root_pos_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    """Four reference-internal future pelvis offsets in the current reference frame."""
    pos_w = _root_reference(env, command_name, "body_pos_w", root_body_name)
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    offsets_w = pos_w[:, 1:] - pos_w[:, :1]
    offsets_b = sp_mdp._quat_apply_inverse(quat_w[:, :1], offsets_w)
    return offsets_b.reshape(env.num_envs, -1)


def root_ori_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """HEFT-style robot-to-reference pelvis rotations at steps zero through four."""
    robot_quat_w = sp_mdp._perturb_quaternion(
        env.scene["robot"].data.root_link_quat_w, noise_std
    ).unsqueeze(1)
    ref_quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    relative = sp_mdp._quat_in_frame(robot_quat_w.expand_as(ref_quat_w), ref_quat_w)
    return sp_mdp._rot6d(relative).reshape(env.num_envs, -1)


def ref_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    target = sp_mdp._gather(env, command_name, "joint_pos", SPV2_REFERENCE_STEPS)
    return target.reshape(env.num_envs, -1)


def ref_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    target = sp_mdp._gather(env, command_name, "joint_vel", SPV2_REFERENCE_STEPS)
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
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    ang_vel_w = _root_reference(env, command_name, "body_ang_vel_w", root_body_name)
    return sp_mdp._quat_apply_inverse(quat_w, ang_vel_w).reshape(env.num_envs, -1)


def ref_root_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    """Reference pelvis height at the current and next four control steps."""
    pos_w = _root_reference(env, command_name, "body_pos_w", root_body_name)
    return pos_w[..., 2].reshape(env.num_envs, -1)


def ref_root_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    """Reference linear velocity in each reference pelvis's own local frame."""
    quat_w = _root_reference(env, command_name, "body_quat_w", root_body_name)
    lin_vel_w = _root_reference(env, command_name, "body_lin_vel_w", root_body_name)
    return sp_mdp._quat_apply_inverse(quat_w, lin_vel_w).reshape(env.num_envs, -1)
