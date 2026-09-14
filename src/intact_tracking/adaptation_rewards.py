"""Optional training rewards aligned with the fixed tracking-error definitions.

These change the optimization objective, never evaluation or physical dynamics.
Default timing follows the environment's existing reward contract. An optional
one-frame target advance accounts for rewards running before command.compute,
whereas fixed tracking evaluation reads the next reference after that compute.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    yaw_quat,
)

from intact_tracking.environment.mdp.motion_fk import quat_apply_inverse

AUXILIARY_SCALES = (0.25, 0.075, 0.125, 4.4, 0.21, 0.45, 0.267, 1.07)


def failure_event(env):
    """Training-only failure cost; normal timeouts never count as failures."""
    return env.termination_manager.terminated.float()


def reference_value(env, command, field, offset):
    """Read another reference frame without changing cursors or global caches."""
    if offset == 0:
        return getattr(command, field)
    if field.startswith("anchor_"):
        value = command.gather_root_reference("body_" + field[7:], (offset,))[:, 0]
    else:
        value = command.gather_reference(field, (offset,))[:, 0]
    if field == "anchor_pos_w":
        value = value + env.scene.env_origins
    elif field == "body_pos_w":
        value = value + env.scene.env_origins[:, None]
    return value


def error_reward(error, scale, shape):
    if scale <= 0:
        raise ValueError("Tracking reward scale must be positive")
    if shape == "exp":
        return torch.exp(-error / scale)
    if shape == "linear":
        # Linear in the acceptance error until catastrophic outliers; an upper
        # cap on penalty keeps recovery episodes from dominating critic fitting.
        return 1.0 - (error / scale).clamp_max(5.0)
    raise ValueError(f"Unknown error reward shape {shape}")


def joint_l2_tracking(env, command_name="motion", scale=0.4, shape="exp", reference_offset=0):
    command = env.command_manager.get_term(command_name)
    target = reference_value(env, command, "joint_pos", reference_offset)
    error = torch.linalg.vector_norm(target - command.robot_joint_pos, dim=-1)
    return error_reward(error, scale, shape)


def balanced_body_tracking(env, command_name="motion", scale=0.03, shape="exp", reference_offset=0):
    command = env.command_manager.get_term(command_name)
    anchor = reference_value(env, command, "anchor_pos_w", reference_offset)
    anchor_quat = reference_value(env, command, "anchor_quat_w", reference_offset)
    origin = torch.cat((command.robot_anchor_pos_w[:, :2], anchor[:, 2:]), dim=-1)
    rotation = yaw_quat(quat_mul(command.robot_anchor_quat_w, quat_inv(anchor_quat)))
    # Recompute alignment against the current robot pose. The command's stored
    # body_pos_relative_w was made before the action and is stale inside rewards.
    reference_offsets = (
        reference_value(env, command, "body_pos_w", reference_offset) - anchor[:, None]
    )
    target = origin[:, None] + quat_apply(
        rotation[:, None].expand(-1, reference_offsets.shape[1], -1), reference_offsets
    )
    error = torch.linalg.vector_norm(target - command.robot_body_pos_w, dim=-1).mean(-1)
    return error_reward(error, scale, shape)


def auxiliary_tracking(
    env, command_name="motion", reference_offset=0, metric_weights=None, shape="exp"
):
    """Equal-scale reward for the eight additional reported tracking errors.

    Scales were fixed from nominal development magnitudes, not confirmation data.
    The qpos-only body velocity convention matches MultiMotionCommand metrics.
    """
    command = env.command_manager.get_term(command_name)
    n = env.num_envs
    current = command.cfg.history_steps if reference_offset == 0 else 0
    reference_lin = reference_value(env, command, "anchor_lin_vel_w", reference_offset).reshape(
        n, -1, 3
    )[:, current]
    reference_ang = reference_value(env, command, "anchor_ang_vel_w", reference_offset).reshape(
        n, -1, 3
    )[:, current]
    anchor_pos = reference_value(env, command, "anchor_pos_w", reference_offset)
    anchor_quat = reference_value(env, command, "anchor_quat_w", reference_offset)
    rotation = yaw_quat(quat_mul(command.robot_anchor_quat_w, quat_inv(anchor_quat)))
    body_quat = reference_value(env, command, "body_quat_w", reference_offset)
    aligned_quat = quat_mul(rotation[:, None].expand_as(body_quat), body_quat)
    if command._uses_qpos_only_actor_fk():
        root = command._qpos_actor_root_velocity_state((reference_offset,))
        body = command.gather_reference_body_state_b((reference_offset,))
        robot_lin = quat_apply_inverse(
            root.root_quat_w[:, 0, None],
            command.robot_body_lin_vel_w - root.root_lin_vel_w[:, 0, None],
        )
        robot_ang = quat_apply_inverse(
            root.root_quat_w[:, 0, None],
            command.robot_body_ang_vel_w - root.root_ang_vel_w[:, 0, None],
        )
        body_lin_error = (body.lin_vel_b[:, 0] - robot_lin).norm(dim=-1).mean(-1)
        body_ang_error = (body.ang_vel_b[:, 0] - robot_ang).norm(dim=-1).mean(-1)
    else:
        body_lin_error = (
            (
                reference_value(env, command, "body_lin_vel_w", reference_offset)
                - command.robot_body_lin_vel_w
            )
            .norm(dim=-1)
            .mean(-1)
        )
        body_ang_error = (
            (
                reference_value(env, command, "body_ang_vel_w", reference_offset)
                - command.robot_body_ang_vel_w
            )
            .norm(dim=-1)
            .mean(-1)
        )
    errors = torch.stack(
        (
            (anchor_pos - command.robot_anchor_pos_w).norm(dim=-1),
            quat_error_magnitude(anchor_quat, command.robot_anchor_quat_w),
            quat_error_magnitude(aligned_quat, command.robot_body_quat_w).mean(-1),
            (
                reference_value(env, command, "joint_vel", reference_offset)
                - command.robot_joint_vel
            ).norm(dim=-1),
            (reference_lin - command.robot_anchor_lin_vel_w).norm(dim=-1),
            (reference_ang - command.robot_anchor_ang_vel_w).norm(dim=-1),
            body_lin_error,
            body_ang_error,
        ),
        dim=-1,
    )
    normalized = errors / errors.new_tensor(AUXILIARY_SCALES)
    if shape == "exp":
        scores = torch.exp(-normalized)
    elif shape == "linear":
        scores = 1.0 - normalized.clamp_max(5.0)
    else:
        raise ValueError(f"Unknown auxiliary tracking reward shape {shape}")
    if metric_weights is None:
        return scores.mean(-1)
    if len(metric_weights) != 8 or any(weight < 0 for weight in metric_weights):
        raise ValueError("Auxiliary tracking requires eight nonnegative weights")
    if sum(metric_weights) <= 0:
        raise ValueError("At least one auxiliary tracking weight must be positive")
    weights = errors.new_tensor(metric_weights)
    return (scores * weights).sum(-1) / weights.sum()


def _right_arm_indices(env, command):
    indices = getattr(env, "_adaptation_right_arm_indices", None)
    if indices is None:
        indices = [
            index
            for index, name in enumerate(command.cfg.body_names)
            if name.split("/")[-1].startswith("right_")
            and any(part in name for part in ("shoulder", "elbow", "wrist"))
        ]
        if not indices:
            raise ValueError("No right-arm bodies in the tracking command")
        indices = torch.tensor(indices, device=command.robot_body_ang_vel_w.device)
        env._adaptation_right_arm_indices = indices
    return indices


def right_arm_rotation_tracking(env, command_name="motion", scale=0.15, reference_offset=0, shape="linear"):
    """Payload-arm pose, with exactly the evaluation's yaw-alignment convention."""
    command = env.command_manager.get_term(command_name)
    indices = _right_arm_indices(env, command)
    anchor = reference_value(env, command, "anchor_quat_w", reference_offset)
    rotation = yaw_quat(quat_mul(command.robot_anchor_quat_w, quat_inv(anchor)))
    reference = reference_value(env, command, "body_quat_w", reference_offset)
    aligned = quat_mul(rotation[:, None].expand_as(reference), reference)
    error = quat_error_magnitude(aligned[:, indices], command.robot_body_quat_w[:, indices]).mean(-1)
    return error_reward(error, scale, shape)


def right_arm_angular_tracking(env, command_name="motion", scale=1.0, reference_offset=0):
    """Target the measured payload-arm bottleneck; evaluation still averages all bodies."""
    command = env.command_manager.get_term(command_name)
    indices = _right_arm_indices(env, command)
    if command._uses_qpos_only_actor_fk():
        root = command._qpos_actor_root_velocity_state((reference_offset,))
        body = command.gather_reference_body_state_b((reference_offset,))
        actual = quat_apply_inverse(
            root.root_quat_w[:, 0, None],
            command.robot_body_ang_vel_w - root.root_ang_vel_w[:, 0, None],
        )
        delta = body.ang_vel_b[:, 0] - actual
    else:
        delta = (
            reference_value(env, command, "body_ang_vel_w", reference_offset)
            - command.robot_body_ang_vel_w
        )
    error = delta[:, indices].norm(dim=-1).mean(-1)
    return error_reward(error, scale, "exp")
