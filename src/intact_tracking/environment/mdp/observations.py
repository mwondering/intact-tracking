from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    subtract_frame_transforms,
)

from .multi_commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _reference_body_indices(
    command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    configured_names = tuple(command.cfg.body_names)
    missing = [name for name in body_names if name not in configured_names]
    if missing:
        raise ValueError(
            f"Requested observation body names are absent from the command reference: {missing}"
        )
    return [configured_names.index(name) for name in body_names]


def _anchor_pose(
    command: MotionCommand, anchor_body_name: str | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reference and robot anchor poses for an optional named view."""
    if anchor_body_name is None:
        return (
            command.anchor_pos_w,
            command.anchor_quat_w,
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
        )
    body_index = _reference_body_indices(command, (anchor_body_name,))[0]
    return (
        command.body_pos_w[:, body_index],
        command.body_quat_w[:, body_index],
        command.robot_body_pos_w[:, body_index],
        command.robot_body_quat_w[:, body_index],
    )


def gradient_test_motion_label(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return the behavior-time simple/hard label without policy exposure."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    label = getattr(command, "gradient_test_motion_label", None)
    if not isinstance(label, torch.Tensor):
        raise RuntimeError("gradient_test_motion_label requires a gradient-test motion command")
    return label.to(dtype=torch.float32).unsqueeze(-1)


def gradient_test_motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return normalized phase for later within-motion conflict analysis."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    denominator = (command.motion_length - 1).clamp_min(1).to(torch.float32)
    phase = command.time_steps.to(torch.float32) / denominator
    return phase.clamp(0.0, 1.0).unsqueeze(-1)


def motion_resample_boundary(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return a one-step pulse after a mid-episode motion resample."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    boundary = getattr(command, "motion_resample_boundary", None)
    if not isinstance(boundary, torch.Tensor):
        raise RuntimeError("motion_resample_boundary requires a multi-motion command")
    return boundary.to(dtype=torch.float32).unsqueeze(-1)


def reference_joint_state_window(
    env: ManagerBasedRlEnv,
    command_name: str,
    history_steps: int = 5,
    future_steps: int = 5,
) -> torch.Tensor:
    """Return the wbteleop reference joint window without mutating command config.

    The layout matches ``MotionCommand.command`` from the source tracking_bfm
    task: all reference joint positions first, followed by all velocities.  A
    5-step history and ``future_steps=5`` produce ten frames total (past five,
    current, future four), or 580 values for the 29-DoF G1.
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    history_steps = int(history_steps)
    future_steps = int(future_steps)
    if history_steps < 0 or future_steps < 1:
        raise ValueError(
            "reference_joint_state_window requires history_steps >= 0 and future_steps >= 1"
        )
    offsets = [*range(-history_steps, 0), 0, *range(1, future_steps)]
    relative_steps = torch.tensor(offsets, device=command.time_steps.device, dtype=torch.long)
    time_steps = command.time_steps.unsqueeze(1) + relative_steps.unsqueeze(0)
    joint_pos = command._gather_motion_field("joint_pos", command.motion_idx, time_steps).reshape(
        env.num_envs, -1
    )
    joint_vel = command._gather_motion_field("joint_vel", command.motion_idx, time_steps).reshape(
        env.num_envs, -1
    )
    return torch.cat((joint_pos, joint_vel), dim=-1)


def _reference_body_pose_window(
    command: MotionCommand,
    history_steps: int,
    future_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a reference body-pose window in source WBTeleop order."""
    history_steps = int(history_steps)
    future_steps = int(future_steps)
    if history_steps < 0 or future_steps < 1:
        raise ValueError(
            "Reference body-pose windows require history_steps >= 0 and future_steps >= 1."
        )
    if history_steps == 0 and future_steps == 1:
        return command.body_pos_w.unsqueeze(1), command.body_quat_w.unsqueeze(1)

    offsets = [*range(-history_steps, 0), 0, *range(1, future_steps)]
    relative_steps = torch.as_tensor(offsets, device=command.time_steps.device, dtype=torch.long)
    time_steps = command.time_steps.unsqueeze(1) + relative_steps.unsqueeze(0)
    body_pos_w = command._gather_motion_field("body_pos_w", command.motion_idx, time_steps)
    body_quat_w = command._gather_motion_field("body_quat_w", command.motion_idx, time_steps)
    body_pos_w = body_pos_w + command._env.scene.env_origins[:, None, None, :]
    return body_pos_w, body_quat_w


def _limb_pose_in_anchor_frame(
    env: ManagerBasedRlEnv,
    command: MotionCommand,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_names: tuple[str, ...],
    anchor_body_name: str,
) -> torch.Tensor:
    body_indexes = _reference_body_indices(command, body_names)
    anchor_index = _reference_body_indices(command, (anchor_body_name,))[0]
    num_steps = body_pos_w.shape[1]
    num_bodies = len(body_indexes)
    anchor_pos_w = body_pos_w[:, :, anchor_index : anchor_index + 1].expand(-1, -1, num_bodies, -1)
    anchor_quat_w = body_quat_w[:, :, anchor_index : anchor_index + 1].expand(
        -1, -1, num_bodies, -1
    )
    pos_b, quat_b = subtract_frame_transforms(
        anchor_pos_w,
        anchor_quat_w,
        body_pos_w[:, :, body_indexes],
        body_quat_w[:, :, body_indexes],
    )
    rot6d = matrix_from_quat(quat_b)[..., :2].reshape(env.num_envs, num_steps, num_bodies, 6)
    return torch.cat((pos_b, rot6d), dim=-1).reshape(env.num_envs, -1)


def ref_limb_ee_pose_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...],
    anchor_body_name: str = "pelvis",
    history_steps: int = 0,
    future_steps: int = 1,
) -> torch.Tensor:
    """Reference limb end-effector poses expressed in the pelvis frame."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_pos_w, body_quat_w = _reference_body_pose_window(command, history_steps, future_steps)
    return _limb_pose_in_anchor_frame(
        env,
        command,
        body_pos_w,
        body_quat_w,
        body_names,
        anchor_body_name,
    )


def robot_limb_ee_pose_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...],
    anchor_body_name: str = "pelvis",
) -> torch.Tensor:
    """Current limb end-effector poses expressible through robot FK."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return _limb_pose_in_anchor_frame(
        env,
        command,
        command.robot_body_pos_w.unsqueeze(1),
        command.robot_body_quat_w.unsqueeze(1),
        body_names,
        anchor_body_name,
    )


def motion_ref_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor angular velocity from the deployable motion command."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return command.anchor_ang_vel_w


def motion_anchor_pos_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    ref_pos_w, ref_quat_w, robot_pos_w, robot_quat_w = _anchor_pose(command, anchor_body_name)

    pos, _ = subtract_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        ref_pos_w,
        ref_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    ref_pos_w, ref_quat_w, robot_pos_w, robot_quat_w = _anchor_pose(command, anchor_body_name)

    _, ori = subtract_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        ref_pos_w,
        ref_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...] | None = None,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indices = _reference_body_indices(command, body_names)
    _, _, robot_anchor_pos_w, robot_anchor_quat_w = _anchor_pose(command, anchor_body_name)

    num_bodies = len(body_indices)
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    body_names: tuple[str, ...] | None = None,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indices = _reference_body_indices(command, body_names)
    _, _, robot_anchor_pos_w, robot_anchor_quat_w = _anchor_pose(command, anchor_body_name)

    num_bodies = len(body_indices)
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
