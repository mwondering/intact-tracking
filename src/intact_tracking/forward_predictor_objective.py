"""Five-step recurrent full-state loss for the nominal Forward Predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

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

DEFAULT_RECURSIVE_WEIGHT = 0.5


@dataclass(frozen=True)
class ForwardPredictorLossConfig:
    root_position_weight: float = 1.0
    root_orientation_weight: float = 1.0
    root_linear_velocity_weight: float = 1.0
    root_angular_velocity_weight: float = 1.0
    joint_position_weight: float = 1.0
    joint_velocity_weight: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in vars(self).items()
            if name != "huber_delta" and value < 0.0
        }
        if invalid:
            raise ValueError(f"Forward Predictor loss weights must be non-negative: {invalid}")
        if self.huber_delta <= 0.0:
            raise ValueError("Forward Predictor huber_delta must be positive")


def _physical_state(
    normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return normalized * std + mean


def _normalized_state_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    delta_std: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.size(-1) != 71:
        raise ValueError(
            "Prediction and target must have equal [...,71] shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    prediction_physical = _physical_state(prediction, state_mean, state_std)
    target_physical = _physical_state(target, state_mean, state_std)
    return physical_state_delta(target_physical, prediction_physical) / delta_std


def _state_losses(
    normalized_error: torch.Tensor,
    config: ForwardPredictorLossConfig,
    *,
    huber: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def reduce(value: torch.Tensor) -> torch.Tensor:
        if huber:
            return F.huber_loss(
                value,
                torch.zeros_like(value),
                reduction="mean",
                delta=config.huber_delta,
            )
        return value.square().mean()

    component = {
        "root_position": reduce(normalized_error[..., DELTA_ROOT_POSITION]),
        "root_orientation": reduce(normalized_error[..., DELTA_ROOT_ROTATION_VECTOR]),
        "root_linear_velocity": reduce(normalized_error[..., DELTA_ROOT_LINEAR_VELOCITY]),
        "root_angular_velocity": reduce(normalized_error[..., DELTA_ROOT_ANGULAR_VELOCITY]),
        "joint_position": reduce(normalized_error[..., DELTA_JOINT_POSITION]),
        "joint_velocity": reduce(normalized_error[..., DELTA_JOINT_VELOCITY]),
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
    root_position_error = torch.linalg.vector_norm(
        prediction[..., ROOT_POSITION] - target[..., ROOT_POSITION], dim=-1
    )
    root_orientation_error = 2.0 * torch.acos(orientation_dot)
    root_linear_velocity_error = torch.linalg.vector_norm(
        prediction[..., ROOT_LINEAR_VELOCITY] - target[..., ROOT_LINEAR_VELOCITY],
        dim=-1,
    )
    root_angular_velocity_error = torch.linalg.vector_norm(
        prediction[..., ROOT_ANGULAR_VELOCITY] - target[..., ROOT_ANGULAR_VELOCITY],
        dim=-1,
    )
    joint_position_error = (prediction[..., JOINT_POSITION] - target[..., JOINT_POSITION]).abs()
    joint_velocity_error = (prediction[..., JOINT_VELOCITY] - target[..., JOINT_VELOCITY]).abs()

    return {
        "root_position_error_m": root_position_error.mean(),
        "root_position_error_p95_m": torch.quantile(root_position_error.float(), 0.95),
        "root_position_error_p99_m": torch.quantile(root_position_error.float(), 0.99),
        "root_orientation_error_rad": root_orientation_error.mean(),
        "root_orientation_error_p95_rad": torch.quantile(root_orientation_error.float(), 0.95),
        "root_orientation_error_p99_rad": torch.quantile(root_orientation_error.float(), 0.99),
        "root_linear_velocity_error_mps": root_linear_velocity_error.mean(),
        "root_angular_velocity_error_radps": root_angular_velocity_error.mean(),
        "joint_position_error_rad": joint_position_error.mean(),
        "joint_position_error_p95_rad": torch.quantile(joint_position_error.float(), 0.95),
        "joint_position_error_p99_rad": torch.quantile(joint_position_error.float(), 0.99),
        "joint_velocity_error_radps": joint_velocity_error.mean(),
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
            "history_state",
            "history_action",
            "history_valid",
            "state_mean",
            "state_std",
            "delta_mean",
            "delta_std",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Forward Predictor batch is missing fields: {missing}")
        finite_names = required.difference({"history_valid"})
        if any(not torch.isfinite(batch[name]).all() for name in finite_names):
            raise ValueError("Forward Predictor batch contains non-finite values")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        recursive_weight: float = DEFAULT_RECURSIVE_WEIGHT,
    ) -> dict[str, torch.Tensor]:
        self._validate_batch(batch)
        if recursive_weight < 0.0:
            raise ValueError("recursive_weight must be non-negative")
        state = batch["state"]
        action = batch["action"]
        if state.ndim != 3 or state.shape[1:] != (6, 71):
            raise ValueError(f"State batch must be [batch,6,71], got {tuple(state.shape)}")
        if action.shape != (state.size(0), 5, 29):
            raise ValueError(f"Action batch must be [batch,5,29], got {tuple(action.shape)}")

        history_arguments = {
            "history_state": batch["history_state"],
            "history_action": batch["history_action"],
            "history_valid": batch["history_valid"],
        }
        normalization_arguments = {
            "state_mean": batch["state_mean"],
            "state_std": batch["state_std"],
            "delta_mean": batch["delta_mean"],
            "delta_std": batch["delta_std"],
        }
        teacher_prediction, teacher_delta = self.model.teacher_forced(
            state,
            action,
            **history_arguments,
            **normalization_arguments,
        )
        if self.training and recursive_weight == 0.0:
            with torch.no_grad():
                recursive_prediction, recursive_delta = self.model.rollout(
                    state[:, 0],
                    action,
                    **history_arguments,
                    **normalization_arguments,
                )
        else:
            recursive_prediction, recursive_delta = self.model.rollout(
                state[:, 0],
                action,
                **history_arguments,
                **normalization_arguments,
            )
        target = state[:, 1:]
        teacher_error = _normalized_state_error(
            teacher_prediction,
            target,
            batch["state_mean"],
            batch["state_std"],
            batch["delta_std"],
        )
        recursive_error = _normalized_state_error(
            recursive_prediction,
            target,
            batch["state_mean"],
            batch["state_std"],
            batch["delta_std"],
        )
        teacher_loss, teacher_components = _state_losses(
            teacher_error,
            self.loss_config,
            huber=True,
        )
        recursive_loss, recursive_components = _state_losses(
            recursive_error,
            self.loss_config,
            huber=True,
        )
        total_loss = teacher_loss + float(recursive_weight) * recursive_loss

        with torch.no_grad():
            unchanged = state[:, :1].expand_as(target)
            unchanged_error = _normalized_state_error(
                unchanged,
                target,
                batch["state_mean"],
                batch["state_std"],
                batch["delta_std"],
            )
            teacher_mse, _ = _state_losses(
                teacher_error,
                self.loss_config,
                huber=False,
            )
            recursive_mse, _ = _state_losses(
                recursive_error,
                self.loss_config,
                huber=False,
            )
            no_change_mse, _ = _state_losses(
                unchanged_error,
                self.loss_config,
                huber=False,
            )
            horizon_metrics: dict[str, torch.Tensor] = {}
            for index in range(5):
                horizon_loss, _ = _state_losses(
                    recursive_error[:, index : index + 1],
                    self.loss_config,
                    huber=True,
                )
                horizon_mse, _ = _state_losses(
                    recursive_error[:, index : index + 1],
                    self.loss_config,
                    huber=False,
                )
                horizon_baseline, _ = _state_losses(
                    unchanged_error[:, index : index + 1],
                    self.loss_config,
                    huber=False,
                )
                step = index + 1
                horizon_metrics[f"horizon_{step}_loss"] = horizon_loss
                horizon_metrics[f"horizon_{step}_mse"] = horizon_mse
                horizon_metrics[f"horizon_{step}_nmse"] = horizon_mse / horizon_baseline.clamp_min(
                    1.0e-8
                )
            physical_metrics = _physical_errors(
                recursive_prediction,
                target,
                batch["state_mean"],
                batch["state_std"],
            )
            teacher_physical_metrics = _physical_errors(
                teacher_prediction,
                target,
                batch["state_mean"],
                batch["state_std"],
            )

        return {
            "loss": total_loss,
            "teacher_loss": teacher_loss.detach(),
            "recursive_loss": recursive_loss.detach(),
            "recursive_weight": teacher_loss.new_tensor(float(recursive_weight)),
            "teacher_mse": teacher_mse,
            "teacher_nmse": teacher_mse / no_change_mse.clamp_min(1.0e-8),
            "rollout_loss": recursive_loss.detach(),
            "rollout_mse": recursive_mse,
            "rollout_no_change_loss": no_change_mse,
            "rollout_nmse": recursive_mse / no_change_mse.clamp_min(1.0e-8),
            "teacher_normalized_delta_rms": teacher_delta.detach().square().mean().sqrt(),
            "predicted_normalized_delta_rms": recursive_delta.detach().square().mean().sqrt(),
            "history_valid_fraction": batch["history_valid"].float().mean(),
            **{
                f"teacher_{name}_loss": value.detach() for name, value in teacher_components.items()
            },
            **{f"{name}_loss": value.detach() for name, value in recursive_components.items()},
            **horizon_metrics,
            **physical_metrics,
            **{f"teacher_{name}": value for name, value in teacher_physical_metrics.items()},
        }
