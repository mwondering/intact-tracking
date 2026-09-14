"""Causal controller diagnostics using only previously executed actions."""

import torch


def causal_orientation_filter(measured_quat, gyro, previous_quat, previous_gyro, dt, weight):
    """Gyro-predicted complementary filter; all inputs are measured or past estimates.

    Quaternions are scalar-first and gyro is in the body frame. Caller handles
    episode initialization explicitly; no simulator state or labels are read.
    """
    if not 0 < weight <= 1 or dt <= 0:
        raise ValueError("Orientation measurement weight must be in (0,1], dt positive")
    from mjlab.utils.lab_api.math import quat_mul

    rotation = (gyro + previous_gyro) * (0.5 * dt)
    angle = rotation.norm(dim=-1, keepdim=True)
    delta = torch.cat(
        ((0.5 * angle).cos(), 0.5 * torch.sinc(angle / (2 * torch.pi)) * rotation), -1
    )
    predicted = quat_mul(previous_quat, delta)
    sign = torch.where((predicted * measured_quat).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    mixed = (1 - weight) * predicted + weight * sign * measured_quat
    return mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def causal_action_filter(action, history, strength=0.0, order=1):
    """Blend with held or linearly predicted last actions; reset histories bypass.

    Order2 has unity DC gain and zero low-frequency first-order delay. Both
    recurrences are stable for0<=strength<1. No extra simulator state is used.
    """
    if not 0 <= strength < 1 or order not in (1, 2):
        raise ValueError("Expected0<=strength<1 and filter order1or2")
    if not strength:
        return action
    # Term-major50-frame history: q29,qdot29,gravity3,gyro3,action29,torque29.
    previous = history[..., 3200:4650].reshape(*history.shape[:-1], 50, 29)
    latest, older = previous[..., -1, :], previous[..., -2, :]
    prediction = latest
    if order == 2:
        prediction = torch.where(
            older.abs().sum(-1, keepdim=True) > 0, 2 * latest - older, latest
        )
    filtered = (1 - strength) * action + strength * prediction
    return torch.where(latest.abs().sum(-1, keepdim=True) > 0, filtered, action)
