"""Read-only errors against the motion's unaligned world-frame reference."""

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude


VERSION = "unaligned_world_tracking_v1"
METRICS = (
    "error_body_pos_global",
    "error_body_rot_global",
    "error_body_lin_vel_global",
    "error_body_ang_vel_global",
    "error_anchor_pos_global",
    "error_anchor_xy_global",
    "error_anchor_height_global",
    "error_anchor_rot_global",
    "error_anchor_lin_vel_global",
    "error_anchor_ang_vel_global",
)
UNITS = ("m", "rad", "m/s", "rad/s", "m", "m", "m", "rad", "m/s", "rad/s")


def values(command):
    """Per-world instantaneous errors; no robot-dependent reference alignment.

    The command's public world-position properties include environment origins.
    In qpos-only mode they reconstruct the reference with its own root pose.
    Body errors average linkwise Euclidean norms or SO(3) angular distances.
    Root/anchor velocities explicitly select the current reference frame.
    """
    position = command.body_pos_w - command.robot_body_pos_w
    anchor = command.anchor_pos_w - command.robot_anchor_pos_w
    return torch.stack((
        position.norm(dim=-1).mean(dim=-1),
        quat_error_magnitude(command.body_quat_w, command.robot_body_quat_w).mean(dim=-1),
        (command.body_lin_vel_w - command.robot_body_lin_vel_w).norm(dim=-1).mean(dim=-1),
        (command.body_ang_vel_w - command.robot_body_ang_vel_w).norm(dim=-1).mean(dim=-1),
        anchor.norm(dim=-1),
        anchor[:, :2].norm(dim=-1),
        anchor[:, 2].abs(),
        quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w),
        (command.gather_root_reference("body_lin_vel_w", (0,))[:, 0]
         - command.robot_anchor_lin_vel_w).norm(dim=-1),
        (command.gather_root_reference("body_ang_vel_w", (0,))[:, 0]
         - command.robot_anchor_ang_vel_w).norm(dim=-1),
    ), dim=-1)


def contract(command):
    return {
        "version": VERSION,
        "metric_names": list(METRICS),
        "units": dict(zip(METRICS, UNITS, strict=True)),
        "body_names": list(command.cfg.body_names),
        "anchor_body_name": command.cfg.anchor_body_name,
        "reference": "World-frame motion reference including environment origins; no robot-dependent translation or yaw alignment",
        "position_reduction": "Euclidean norm per link, then equal-weight link mean; anchor metrics use the named anchor only",
        "rotation_reduction": "SO(3) shortest angular distance in radians, then equal-weight link mean",
        "velocity_reference": "Current frame world velocities, including the reference root motion",
        "timeline": "After the same action step as the existing tracking metrics, before inactive-world resets",
    }
