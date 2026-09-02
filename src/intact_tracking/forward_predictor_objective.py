"""Five-step recurrent full-state loss for the nominal Forward Predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .forward_predictor import (
    DELTA_JOINT_POSITION,
    DELTA_JOINT_VELOCITY,
    DELTA_ROOT_ANGULAR_VELOCITY,
    DELTA_ROOT_LINEAR_VELOCITY,
    DELTA_ROOT_POSITION,
    DELTA_ROOT_ROTATION_VECTOR,
    JOINT_POSITION,
    JOINT_VELOCITY,
    ROOT_ANGULAR_VELOCITY,
    ROOT_LINEAR_VELOCITY,
    ROOT_ORIENTATION,
    ROOT_POSITION,
    ForwardDynamicsMLP,
    physical_state_delta,
)


@dataclass(frozen=True)
class ForwardPredictorLossConfig:
    root_position_weight: float = 1.0
    root_orientation_weight: float = 1.0
    root_linear_velocity_weight: float = 1.0
    root_angular_velocity_weight: float = 1.0
    joint_position_weight: float = 1.0
    joint_velocity_weight: float = 1.0

    def __post_init__(self) -> None:
        invalid = {name: value for name, value in vars(self).items() if value < 0.0}
        if invalid:
            raise ValueError(f"Forward Predictor loss weights must be non-negative: {invalid}")


def _physical_state(
    normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return normalized * std + mean


def _state_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    delta_std: torch.Tensor,
    config: ForwardPredictorLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.size(-1) != 71:
        raise ValueError(
            "Prediction and target must have equal [...,71] shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    prediction_physical = _physical_state(prediction, state_mean, state_std)
    target_physical = _physical_state(target, state_mean, state_std)
    normalized_error = physical_state_delta(target_physical, prediction_physical) / delta_std
    component = {
        "root_position": normalized_error[..., DELTA_ROOT_POSITION].square().mean(),
        "root_orientation": normalized_error[..., DELTA_ROOT_ROTATION_VECTOR].square().mean(),
        "root_linear_velocity": normalized_error[..., DELTA_ROOT_LINEAR_VELOCITY].square().mean(),
        "root_angular_velocity": normalized_error[..., DELTA_ROOT_ANGULAR_VELOCITY].square().mean(),
        "joint_position": normalized_error[..., DELTA_JOINT_POSITION].square().mean(),
        "joint_velocity": normalized_error[..., DELTA_JOINT_VELOCITY].square().mean(),
    }
    total = (
        config.root_position_weight * component["root_position"]
        + config.root_orientation_weight * component["root_orientation"]
        + config.root_linear_velocity_weight * component["root_linear_velocity"]
        + config.root_angular_velocity_weight * component["root_angular_velocity"]
        + config.joint_position_weight * component["joint_position"]
        + config.joint_velocity_weight * component["joint_velocity"]
    )
    return total, component


def _physical_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = _physical_state(prediction, state_mean, state_std)
    target = _physical_state(target, state_mean, state_std)
    prediction_quaternion = torch.nn.functional.normalize(
        prediction[..., ROOT_ORIENTATION], dim=-1, eps=1.0e-8
    )
    target_quaternion = torch.nn.functional.normalize(
        target[..., ROOT_ORIENTATION], dim=-1, eps=1.0e-8
    )
    orientation_dot = (prediction_quaternion * target_quaternion).sum(dim=-1).abs().clamp(max=1.0)
    return {
        "root_position_error_m": torch.linalg.vector_norm(
            prediction[..., ROOT_POSITION] - target[..., ROOT_POSITION], dim=-1
        ).mean(),
        "root_orientation_error_rad": (2.0 * torch.acos(orientation_dot)).mean(),
        "root_linear_velocity_error_mps": torch.linalg.vector_norm(
            prediction[..., ROOT_LINEAR_VELOCITY] - target[..., ROOT_LINEAR_VELOCITY],
            dim=-1,
        ).mean(),
        "root_angular_velocity_error_radps": torch.linalg.vector_norm(
            prediction[..., ROOT_ANGULAR_VELOCITY] - target[..., ROOT_ANGULAR_VELOCITY],
            dim=-1,
        ).mean(),
        "joint_position_error_rad": (prediction[..., JOINT_POSITION] - target[..., JOINT_POSITION])
        .abs()
        .mean(),
        "joint_velocity_error_radps": (
            prediction[..., JOINT_VELOCITY] - target[..., JOINT_VELOCITY]
        )
        .abs()
        .mean(),
    }


class ForwardPredictorObjective(nn.Module):
    """Train one shared transition model through its complete five-step rollout graph."""

    def __init__(
        self,
        model: ForwardDynamicsMLP,
        loss_config: ForwardPredictorLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_config = loss_config or ForwardPredictorLossConfig()

    @staticmethod
    def _validate_batch(batch: dict[str, torch.Tensor]) -> None:
        required = {
            "state",
            "action",
            "state_mean",
            "state_std",
            "delta_mean",
            "delta_std",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Forward Predictor batch is missing fields: {missing}")
        if any(not torch.isfinite(batch[name]).all() for name in required):
            raise ValueError("Forward Predictor batch contains non-finite values")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._validate_batch(batch)
        state = batch["state"]
        action = batch["action"]
        if state.ndim != 3 or state.shape[1:] != (6, 71):
            raise ValueError(f"State batch must be [batch,6,71], got {tuple(state.shape)}")
        if action.shape != (state.size(0), 5, 29):
            raise ValueError(f"Action batch must be [batch,5,29], got {tuple(action.shape)}")

        prediction, normalized_delta = self.model.rollout(
            state[:, 0],
            action,
            batch["state_mean"],
            batch["state_std"],
            batch["delta_mean"],
            batch["delta_std"],
        )
        target = state[:, 1:]
        rollout_loss, components = _state_losses(
            prediction,
            target,
            batch["state_mean"],
            batch["state_std"],
            batch["delta_std"],
            self.loss_config,
        )

        with torch.no_grad():
            unchanged = state[:, :1].expand_as(target)
            no_change_loss, _ = _state_losses(
                unchanged,
                target,
                batch["state_mean"],
                batch["state_std"],
                batch["delta_std"],
                self.loss_config,
            )
            horizon_metrics: dict[str, torch.Tensor] = {}
            for index in range(5):
                horizon_loss, _ = _state_losses(
                    prediction[:, index : index + 1],
                    target[:, index : index + 1],
                    batch["state_mean"],
                    batch["state_std"],
                    batch["delta_std"],
                    self.loss_config,
                )
                horizon_baseline, _ = _state_losses(
                    unchanged[:, index : index + 1],
                    target[:, index : index + 1],
                    batch["state_mean"],
                    batch["state_std"],
                    batch["delta_std"],
                    self.loss_config,
                )
                step = index + 1
                horizon_metrics[f"horizon_{step}_loss"] = horizon_loss
                horizon_metrics[f"horizon_{step}_nmse"] = horizon_loss / horizon_baseline.clamp_min(
                    1.0e-8
                )
            physical_metrics = _physical_errors(
                prediction,
                target,
                batch["state_mean"],
                batch["state_std"],
            )

        return {
            "loss": rollout_loss,
            "rollout_loss": rollout_loss.detach(),
            "rollout_no_change_loss": no_change_loss,
            "rollout_nmse": rollout_loss.detach() / no_change_loss.clamp_min(1.0e-8),
            "predicted_normalized_delta_rms": normalized_delta.detach().square().mean().sqrt(),
            **{f"{name}_loss": value.detach() for name, value in components.items()},
            **horizon_metrics,
            **physical_metrics,
        }
