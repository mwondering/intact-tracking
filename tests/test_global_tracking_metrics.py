from types import SimpleNamespace
import math

import torch

from intact_tracking.global_tracking_metrics import METRICS, values


def command():
    zero = torch.zeros(1, 2, 3)
    identity = torch.tensor([1., 0., 0., 0.]).expand(1, 2, 4).clone()
    return SimpleNamespace(
        body_pos_w=zero.clone(), robot_body_pos_w=zero.clone(),
        body_quat_w=identity.clone(), robot_body_quat_w=identity.clone(),
        body_lin_vel_w=zero.clone(), robot_body_lin_vel_w=zero.clone(),
        body_ang_vel_w=zero.clone(), robot_body_ang_vel_w=zero.clone(),
        anchor_pos_w=zero[:, 0].clone(), robot_anchor_pos_w=zero[:, 0].clone(),
        anchor_quat_w=identity[:, 0].clone(), robot_anchor_quat_w=identity[:, 0].clone(),
        robot_anchor_lin_vel_w=zero[:, 0].clone(), robot_anchor_ang_vel_w=zero[:, 0].clone(),
        gather_root_reference=lambda field, steps: torch.zeros(1, len(steps), 3),
    )


def measured(c):
    return dict(zip(METRICS, values(c)[0].tolist(), strict=True))


def test_global_position_keeps_rigid_translation_and_ignores_aligned_reference():
    c = command()
    # Both links translated by the same 3-4-12 vector: local pose can be perfect.
    c.robot_body_pos_w[:] = torch.tensor([3., 4., 12.])
    c.robot_anchor_pos_w[:] = torch.tensor([3., 4., 12.])
    c.body_pos_relative_w = c.robot_body_pos_w.clone()
    m = measured(c)
    assert m["error_body_pos_global"] == 13.
    assert m["error_anchor_pos_global"] == 13.
    assert m["error_anchor_xy_global"] == 5.
    assert m["error_anchor_height_global"] == 12.
    # Changing world origin equally on both sides leaves the error unchanged.
    for name in ("body_pos_w", "robot_body_pos_w", "anchor_pos_w", "robot_anchor_pos_w"):
        getattr(c, name).add_(100.)
    assert measured(c) == m


def test_global_rotation_keeps_yaw_and_quaternion_sign_invariance():
    c = command()
    yaw90 = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    c.robot_body_quat_w[:] = yaw90
    c.robot_anchor_quat_w[:] = -yaw90
    m = measured(c)
    assert math.isclose(m["error_body_rot_global"], math.pi / 2, abs_tol=1e-6)
    assert math.isclose(m["error_anchor_rot_global"], math.pi / 2, abs_tol=1e-6)


def test_world_velocities_include_root_motion_and_select_current_frame():
    c = command()
    c.body_lin_vel_w[:] = torch.tensor([3., 4., 0.])
    c.body_ang_vel_w[:] = torch.tensor([0., 0., 2.])

    def reference(field, steps):
        assert steps == (0,), "Do not mix future or historical reference velocity"
        vector = [3., 4., 0.] if field == "body_lin_vel_w" else [0., 0., 2.]
        return torch.tensor(vector).reshape(1, 1, 3)

    c.gather_root_reference = reference
    m = measured(c)
    assert m["error_body_lin_vel_global"] == m["error_anchor_lin_vel_global"] == 5.
    assert m["error_body_ang_vel_global"] == m["error_anchor_ang_vel_global"] == 2.
