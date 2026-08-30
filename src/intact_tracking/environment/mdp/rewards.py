from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)

from .multi_commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(command: MotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def _reference_body_positions_w(command: MotionCommand, body_indexes: list[int]) -> torch.Tensor:
    gather_pos = getattr(command, "gather_reference_body_pos_w", None)
    if callable(gather_pos):
        return gather_pos(body_indexes, (0,))[:, 0]
    return command.body_pos_w[:, body_indexes]


def _reference_body_quaternions_w(command: MotionCommand, body_indexes: list[int]) -> torch.Tensor:
    gather_quat = getattr(command, "gather_reference_body_quat_w", None)
    if callable(gather_quat):
        return gather_quat(body_indexes, (0,))[:, 0]
    return command.body_quat_w[:, body_indexes]


def _reference_body_poses_w(
    command: MotionCommand, body_indexes: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _reference_body_positions_w(command, body_indexes),
        _reference_body_quaternions_w(command, body_indexes),
    )


def _anchor_pose(
    command: MotionCommand, anchor_body_name: str | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchor_body_name is None:
        return (
            command.anchor_pos_w,
            command.anchor_quat_w,
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
        )
    body_indexes = _get_body_indexes(command, (anchor_body_name,))
    if len(body_indexes) != 1:
        raise ValueError(f"Anchor body '{anchor_body_name}' is absent from the command reference.")
    body_index = body_indexes[0]
    reference_pos_w, reference_quat_w = _reference_body_poses_w(command, [body_index])
    return (
        reference_pos_w[:, 0],
        reference_quat_w[:, 0],
        command.robot_body_pos_w[:, body_index],
        command.robot_body_quat_w[:, body_index],
    )


def _relative_reference_body_poses(
    command: MotionCommand,
    body_indexes: list[int],
    anchor_body_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match command-relative poses for an optional named reference view.

    ``MultiMotionCommand`` stores its default relative body poses using the
    primary command anchor.  A term may select a different named reference view,
    so recompute the exact yaw-aligned construction for that alternate view here.
    """
    if anchor_body_name is None:
        return (
            command.body_pos_relative_w[:, body_indexes],
            command.body_quat_relative_w[:, body_indexes],
        )

    anchor_pos_w, anchor_quat_w, robot_anchor_pos_w, robot_anchor_quat_w = _anchor_pose(
        command, anchor_body_name
    )
    num_bodies = len(body_indexes)
    delta_pos_w = robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    delta_pos_w[..., 2] = anchor_pos_w[:, None, 2]
    delta_ori_w = (
        yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))
        .unsqueeze(1)
        .expand(-1, num_bodies, -1)
    )
    body_pos_w, body_quat_w = _reference_body_poses_w(command, body_indexes)
    return (
        delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w[:, None, :]),
        quat_mul(delta_ori_w, body_quat_w),
    )


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    anchor_pos_w, _, robot_anchor_pos_w, _ = _anchor_pose(command, anchor_body_name)
    error = torch.sum(torch.square(anchor_pos_w - robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    _, anchor_quat_w, _, robot_anchor_quat_w = _anchor_pose(command, anchor_body_name)
    error = quat_error_magnitude(anchor_quat_w, robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    reference_pos_w, _ = _relative_reference_body_poses(command, body_indexes, anchor_body_name)
    error = torch.sum(
        torch.square(reference_pos_w - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    _, reference_quat_w = _relative_reference_body_poses(command, body_indexes, anchor_body_name)
    error = (
        quat_error_magnitude(
            reference_quat_w,
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def _pelvis_limb_ee_pose_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...],
    anchor_body_name: str = "pelvis",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    anchor_index = tuple(command.cfg.body_names).index(anchor_body_name)

    num_bodies = len(body_indexes)
    ref_anchor_pos_w = command.body_pos_w[:, anchor_index : anchor_index + 1, :].repeat(
        1, num_bodies, 1
    )
    ref_anchor_quat_w = command.body_quat_w[:, anchor_index : anchor_index + 1, :].repeat(
        1, num_bodies, 1
    )
    robot_anchor_pos_w = command.robot_body_pos_w[:, anchor_index : anchor_index + 1, :].repeat(
        1, num_bodies, 1
    )
    robot_anchor_quat_w = command.robot_body_quat_w[:, anchor_index : anchor_index + 1, :].repeat(
        1, num_bodies, 1
    )

    ref_pos_b, ref_quat_b = subtract_frame_transforms(
        ref_anchor_pos_w,
        ref_anchor_quat_w,
        command.body_pos_w[:, body_indexes],
        command.body_quat_w[:, body_indexes],
    )
    robot_pos_b, robot_quat_b = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.robot_body_pos_w[:, body_indexes],
        command.robot_body_quat_w[:, body_indexes],
    )
    return ref_pos_b, ref_quat_b, robot_pos_b, robot_quat_b


def motion_pelvis_limb_ee_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...],
    anchor_body_name: str = "pelvis",
) -> torch.Tensor:
    ref_pos_b, _, robot_pos_b, _ = _pelvis_limb_ee_pose_b(
        env,
        command_name=command_name,
        body_names=body_names,
        anchor_body_name=anchor_body_name,
    )
    pos_error = torch.sum(torch.square(ref_pos_b - robot_pos_b), dim=-1).mean(-1)
    return torch.exp(-pos_error / std**2)


def motion_pelvis_limb_ee_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...],
    anchor_body_name: str = "pelvis",
) -> torch.Tensor:
    _, ref_quat_b, _, robot_quat_b = _pelvis_limb_ee_pose_b(
        env,
        command_name=command_name,
        body_names=body_names,
        anchor_body_name=anchor_body_name,
    )
    ori_error = quat_error_magnitude(ref_quat_b, robot_quat_b).square().mean(-1)
    return torch.exp(-ori_error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_height_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_name: str,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_index = tuple(command.cfg.body_names).index(body_name)
    error = torch.square(
        command.body_pos_w[:, body_index, 2] - command.robot_body_pos_w[:, body_index, 2]
    )
    return torch.exp(-error / std**2)


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor provides force history (from ``history_length > 0``),
    counts substeps where any contact force exceeds *force_threshold*.
    Falls back to the instantaneous ``found`` count otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.squeeze(-1)


def joint_action_rate_l2(
    env: ManagerBasedRlEnv,
    asset_cfg,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize raw action-rate changes for selected action targets."""
    action_term = env.action_manager.get_term(action_name)
    target_names = action_term.target_names
    selected = torch.zeros(len(target_names), dtype=torch.bool, device=env.device)
    patterns = getattr(asset_cfg, "joint_names", None)
    if patterns is None:
        patterns = (".*",)
    if isinstance(patterns, str):
        patterns = (patterns,)
    for pattern in patterns:
        regex = re.compile(pattern)
        for idx, name in enumerate(target_names):
            if regex.fullmatch(name) or regex.search(name):
                selected[idx] = True
    diff = action_term.raw_action - env.action_manager.get_term(action_name).raw_action
    if hasattr(env.action_manager, "prev_action"):
        action_start = 0
        for term_name in env.action_manager.active_terms:
            term = env.action_manager.get_term(term_name)
            if term_name == action_name:
                break
            action_start += term.action_dim
        prev = env.action_manager.prev_action[
            :, action_start : action_start + action_term.action_dim
        ]
        diff = action_term.raw_action - prev
    return torch.sum(torch.square(diff[:, selected]), dim=1)
