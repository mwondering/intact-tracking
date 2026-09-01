"""Losses and diagnostics for context-conditioned residual tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_model import ResidualTrackingModel


@dataclass(frozen=True)
class ResidualLossConfig:
    forward_weight: float = 2.0
    backward_weight: float = 2.0
    tracking_weight: float = 1.0
    residual_l2_weight: float = 1.0e-3
    residual_smooth_weight: float = 1.0e-3
    root_position_weight: float = 1.0
    root_orientation_weight: float = 1.0
    root_linear_velocity_weight: float = 1.0
    root_angular_velocity_weight: float = 1.0
    joint_position_weight: float = 1.0
    joint_velocity_weight: float = 1.0
    action_clip: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "action_clip":
                if value is not None and value <= 0:
                    raise ValueError("action_clip must be positive when configured")
            elif value < 0:
                raise ValueError(f"{name} must be non-negative")


_STATE_SLICES = {
    "root_position": slice(0, 3),
    "root_orientation": slice(3, 7),
    "root_linear_velocity": slice(7, 10),
    "root_angular_velocity": slice(10, 13),
    "joint_position": slice(13, 42),
    "joint_velocity": slice(42, 71),
}


def _state_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    config: ResidualLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.size(-1) != 71:
        raise ValueError(
            "State prediction and target must share [B,T,71], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    component: dict[str, torch.Tensor] = {}
    for name, section in _STATE_SLICES.items():
        if name == "root_orientation":
            predicted_quat = prediction[..., section] * state_std[section] + state_mean[section]
            target_quat = target[..., section] * state_std[section] + state_mean[section]
            predicted_quat = torch.nn.functional.normalize(predicted_quat, dim=-1, eps=1e-8)
            target_quat = torch.nn.functional.normalize(target_quat, dim=-1, eps=1e-8)
            dot = (predicted_quat * target_quat).sum(dim=-1).clamp(-1.0, 1.0)
            component[name] = (1.0 - dot.square()).mean()
        else:
            component[name] = (prediction[..., section] - target[..., section]).square().mean()
    total = (
        config.root_position_weight * component["root_position"]
        + config.root_orientation_weight * component["root_orientation"]
        + config.root_linear_velocity_weight * component["root_linear_velocity"]
        + config.root_angular_velocity_weight * component["root_angular_velocity"]
        + config.joint_position_weight * component["joint_position"]
        + config.joint_velocity_weight * component["joint_velocity"]
    )
    return total, component


class ResidualTrainingObjective(nn.Module):
    """One DDP-safe objective implementing the agreed gradient routing."""

    def __init__(
        self,
        model: ResidualTrackingModel,
        loss_config: ResidualLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_config = loss_config or ResidualLossConfig()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {
            "context",
            "context_mask",
            "state",
            "reference_state",
            "action",
            "previous_action",
            "tracker_action",
            "policy_observation",
            "action_mean",
            "action_std",
            "state_mean",
            "state_std",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Residual training batch is missing fields: {missing}")
        if any(not torch.isfinite(batch[name]).all() for name in required):
            raise ValueError("Residual training batch contains non-finite values")

        cfg = self.loss_config
        model = self.model
        world = model.encode_context(batch["context"], batch["context_mask"])
        state = batch["state"]
        action = batch["action"]
        previous_action = batch["previous_action"]
        state_mean = batch["state_mean"]
        state_std = batch["state_std"]

        forward_prediction = model.predict_future(
            world,
            state[:, 0],
            previous_action,
            action,
        )
        forward_loss, forward_components = _state_losses(
            forward_prediction,
            state[:, 1:],
            state_mean,
            state_std,
            cfg,
        )

        backward_previous = torch.cat((previous_action[:, None], action[:, :-1]), dim=1)
        backward_prediction = model.backward_predictor(
            world,
            state[:, :-1],
            state[:, 1:],
            backward_previous,
        )
        backward_loss = (backward_prediction - action).square().mean()

        # Context and Forward parameters are constants on this branch. The
        # functional Forward call retains the derivative with respect to the
        # candidate actions, which is the signal used by the residual policy.
        policy_world = world.detach()
        residual = model.residual_action(policy_world, batch["policy_observation"])
        candidate_unclipped = batch["tracker_action"] + residual
        candidate = candidate_unclipped
        if cfg.action_clip is not None:
            candidate = candidate.clamp(-cfg.action_clip, cfg.action_clip)
        action_mean = batch["action_mean"]
        action_std = batch["action_std"]
        candidate_normalized = (candidate - action_mean) / action_std
        policy_prediction = model.predict_future_with_frozen_dynamics(
            policy_world,
            state[:, 0].detach(),
            previous_action.detach(),
            candidate_normalized,
        )
        tracking_loss, tracking_components = _state_losses(
            policy_prediction,
            batch["reference_state"],
            state_mean,
            state_std,
            cfg,
        )
        residual_l2 = residual.square().mean()
        residual_smooth = (residual[:, 1:] - residual[:, :-1]).square().mean()

        weighted_forward = cfg.forward_weight * forward_loss
        weighted_backward = cfg.backward_weight * backward_loss
        weighted_tracking = cfg.tracking_weight * tracking_loss
        weighted_residual_l2 = cfg.residual_l2_weight * residual_l2
        weighted_residual_smooth = cfg.residual_smooth_weight * residual_smooth
        total = (
            weighted_forward
            + weighted_backward
            + weighted_tracking
            + weighted_residual_l2
            + weighted_residual_smooth
        )

        with torch.no_grad():
            if world.size(0) > 1:
                shuffled_world = world.roll(1, dims=0)
                shuffled_forward = model.predict_future(
                    shuffled_world,
                    state[:, 0],
                    previous_action,
                    action,
                )
                shuffled_forward_loss, _ = _state_losses(
                    shuffled_forward,
                    state[:, 1:],
                    state_mean,
                    state_std,
                    cfg,
                )
                shuffled_backward = model.backward_predictor(
                    shuffled_world,
                    state[:, :-1],
                    state[:, 1:],
                    backward_previous,
                )
                shuffled_backward_loss = (shuffled_backward - action).square().mean()
            else:
                shuffled_forward_loss = forward_loss.detach()
                shuffled_backward_loss = backward_loss.detach()
            clipped = (candidate_unclipped - candidate).abs() > 1e-7
            target_variance = state[:, 1:].float().var(unbiased=False)

        output = {
            "loss": total,
            "forward_loss": forward_loss,
            "backward_loss": backward_loss,
            "tracking_loss": tracking_loss,
            "residual_l2": residual_l2,
            "residual_smooth": residual_smooth,
            "weighted_forward_loss": weighted_forward.detach(),
            "weighted_backward_loss": weighted_backward.detach(),
            "weighted_tracking_loss": weighted_tracking.detach(),
            "weighted_residual_l2": weighted_residual_l2.detach(),
            "weighted_residual_smooth": weighted_residual_smooth.detach(),
            "forward_nmse": forward_loss.detach() / target_variance.clamp_min(1e-8),
            "forward_context_shuffle_ratio": (
                shuffled_forward_loss / forward_loss.detach().clamp_min(1e-8)
            ),
            "backward_context_shuffle_ratio": (
                shuffled_backward_loss / backward_loss.detach().clamp_min(1e-8)
            ),
            "residual_abs_mean": residual.detach().abs().mean(),
            "residual_rms": residual.detach().square().mean().sqrt(),
            "residual_abs_max": residual.detach().abs().max(),
            "candidate_action_clipped_fraction": clipped.float().mean(),
            "candidate_action_change_abs_mean": (
                candidate.detach() - batch["tracker_action"]
            ).abs().mean(),
        }
        for name, value in forward_components.items():
            output[f"forward_{name}_loss"] = value.detach()
        for name, value in tracking_components.items():
            output[f"tracking_{name}_loss"] = value.detach()
        return output
