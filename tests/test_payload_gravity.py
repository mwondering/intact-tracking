import torch

from intact_tracking.oracle_compensation import payload_gravity_joint_torque


def test_payload_gravity_virtual_work_sign_mask_and_zero_nominal():
    center = torch.tensor([[2.0, 0.0, 0.0]])
    anchors = torch.zeros(1, 3, 3)
    axes = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    mask = torch.tensor([True, True, False])
    gravity = torch.tensor([0.0, 0.0, -9.81])
    torque = payload_gravity_joint_torque(center, anchors, axes, torch.tensor([3.0]), gravity, mask)
    torch.testing.assert_close(torque, torch.tensor([[-58.86, 0.0, 0.0]]))
    assert payload_gravity_joint_torque(center, anchors, axes, torch.zeros(1), gravity, mask).count_nonzero() == 0
    # d(m*g*z)/dq for Ry(q)[2,0,0] at q=0 equals -2*m*g.
    q = torch.tensor(0.0, requires_grad=True)
    potential = 3.0 * 9.81 * (-2.0 * q.sin())
    torch.testing.assert_close(torch.autograd.grad(potential, q)[0], torque[0, 0])
