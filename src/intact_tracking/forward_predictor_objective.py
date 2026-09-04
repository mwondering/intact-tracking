"""Five-step prediction and observable-dynamics representation objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .forward_predictor import ForwardDynamicsTransformer, physical_state_delta

DEFAULT_RECURSIVE_WEIGHT = 0.5


@dataclass(frozen=True)
class ForwardPredictorLossConfig:
    root_position_weight: float = 1.0
    root_orientation_weight: float = 1.0
    root_linear_velocity_weight: float = 1.0
    root_angular_velocity_weight: float = 1.0
    joint_position_weight: float = 1.0
    joint_velocity_weight: float = 1.0
    foot_weight: float = 1.0
    contact_force_weight: float = 1.0
    contact_binary_weight: float = 1.0
    representation_weight: float = 0.01
    response_distance_scale: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        weights = (
            "root_position_weight",
            "root_orientation_weight",
            "root_linear_velocity_weight",
            "root_angular_velocity_weight",
            "joint_position_weight",
            "joint_velocity_weight",
            "foot_weight",
            "contact_force_weight",
            "contact_binary_weight",
            "representation_weight",
        )
        invalid = {name: getattr(self, name) for name in weights if getattr(self, name) < 0.0}
        if invalid:
            raise ValueError(f"Forward Predictor loss weights must be non-negative: {invalid}")
        if self.huber_delta <= 0.0:
            raise ValueError("Forward Predictor huber_delta must be positive")
        if self.response_distance_scale <= 0.0:
            raise ValueError("response_distance_scale must be positive")


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


def _state_loss(
    normalized_error: torch.Tensor,
    config: ForwardPredictorLossConfig,
    *,
    huber: bool,
) -> torch.Tensor:
    slices = (
        (slice(0, 3), config.root_position_weight),
        (slice(3, 6), config.root_orientation_weight),
        (slice(6, 9), config.root_linear_velocity_weight),
        (slice(9, 12), config.root_angular_velocity_weight),
        (slice(12, 41), config.joint_position_weight),
        (slice(41, 70), config.joint_velocity_weight),
    )

    def reduce(value: torch.Tensor) -> torch.Tensor:
        if huber:
            return F.huber_loss(
                value,
                torch.zeros_like(value),
                reduction="mean",
                delta=config.huber_delta,
            )
        return value.square().mean()

    total = normalized_error.new_zeros(())
    for component, weight in slices:
        total = total + weight * reduce(normalized_error[..., component])
    return total


def _foot_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: ForwardPredictorLossConfig,
    *,
    huber: bool,
) -> torch.Tensor:
    if huber:
        return F.huber_loss(prediction, target, reduction="mean", delta=config.huber_delta)
    return (prediction - target).square().mean()


def _contact_loss(
    force_prediction: torch.Tensor,
    binary_logits: torch.Tensor,
    force_target: torch.Tensor,
    binary_target: torch.Tensor,
    config: ForwardPredictorLossConfig,
    *,
    huber: bool,
) -> torch.Tensor:
    if huber:
        force_loss = F.huber_loss(
            force_prediction,
            force_target,
            reduction="mean",
            delta=config.huber_delta,
        )
    else:
        force_loss = (force_prediction - force_target).square().mean()
    binary_loss = F.binary_cross_entropy_with_logits(
        binary_logits, binary_target.float(), reduction="mean"
    )
    return config.contact_force_weight * force_loss + config.contact_binary_weight * binary_loss


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def _masked_nmse(
    error: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        return error.square().mean() / baseline.square().mean().clamp_min(1.0e-8)
    return _masked_mean(error.square(), mask) / _masked_mean(baseline.square(), mask).clamp_min(
        1.0e-8
    )


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(value.square(), mask).sqrt()


def _pearson_correlation(
    left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    weight = valid.to(left.dtype)
    count = weight.sum()
    left_mean = (left * weight).sum() / count.clamp_min(1.0)
    right_mean = (right * weight).sum() / count.clamp_min(1.0)
    centered_left = (left - left_mean) * weight
    centered_right = (right - right_mean) * weight
    denominator = (
        (centered_left.square().sum() * centered_right.square().sum()).clamp_min(1.0e-12).sqrt()
    )
    correlation = (centered_left * centered_right).sum() / denominator
    usable = (count >= 2) & (denominator > 1.0e-6)
    return torch.where(usable, correlation, torch.zeros_like(correlation))


def _cross_world_partner(world_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a cheap broad-batch pairing for continuous response supervision."""

    batch_size = world_id.numel()
    indices = torch.arange(batch_size, device=world_id.device)
    if batch_size < 2:
        return indices, torch.zeros_like(world_id, dtype=torch.bool)
    partner = torch.roll(indices, shifts=max(1, batch_size // 2))
    valid = world_id != world_id.index_select(0, partner)
    return partner, valid


def _counterfactual_representation_loss(
    latent: torch.Tensor,
    positive_latent: torch.Tensor,
    response: torch.Tensor,
    positive_pair_valid: torch.Tensor,
    context_full: torch.Tensor,
    positive_context_full: torch.Tensor,
    world_id: torch.Tensor,
    *,
    response_distance_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Match latent geometry to the continuously observed nominal/DR response.

    There is no dynamics threshold and no categorical environment target. A
    pair with a tiny observable A-B response difference receives a tiny target
    latent distance; a pair with a large response difference receives a larger
    target distance. Exact +/-5-frame views of one environment provide the
    local invariance term.
    """

    if latent.shape != positive_latent.shape or latent.ndim != 2:
        raise ValueError("Latent and positive_latent must have equal [batch,dim] shapes")
    if response.ndim != 3 or response.shape[:2] != (latent.size(0), 5):
        raise ValueError("Counterfactual response must be [batch,5,70]")
    full_positive = positive_pair_valid & context_full & positive_context_full
    embedding = F.normalize(latent.float(), dim=-1, eps=1.0e-8)
    positive_embedding = F.normalize(positive_latent.float(), dim=-1, eps=1.0e-8)
    positive_cosine = (embedding * positive_embedding).sum(dim=-1)
    positive_loss = _masked_mean(1.0 - positive_cosine, full_positive)

    partner, different_world = _cross_world_partner(world_id)
    paired_embedding = embedding.index_select(0, partner)
    latent_distance = torch.linalg.vector_norm(embedding - paired_embedding, dim=-1)
    flat_response = response.float().flatten(start_dim=1)
    response_distance = (
        (flat_response - flat_response.index_select(0, partner)).square().mean(dim=-1).sqrt()
    )
    target_distance = 2.0 * response_distance / (response_distance + float(response_distance_scale))
    relation_valid = context_full & context_full.index_select(0, partner) & different_world
    relation_error = F.smooth_l1_loss(latent_distance, target_distance, reduction="none", beta=0.25)
    relation_loss = _masked_mean(relation_error, relation_valid)
    representation_loss = positive_loss + relation_loss
    metrics = {
        "latent_positive_cosine": _masked_mean(positive_cosine, full_positive),
        "latent_response_correlation": _pearson_correlation(
            latent_distance, response_distance, relation_valid
        ),
    }
    return representation_loss, metrics, partner


class ForwardPredictorObjective(nn.Module):
    """Train one shared transition model on both nominal and DR A trajectories."""

    def __init__(
        self,
        model: ForwardDynamicsTransformer,
        loss_config: ForwardPredictorLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_config = loss_config or ForwardPredictorLossConfig()

    @staticmethod
    def _validate_batch(batch: dict[str, torch.Tensor]) -> None:
        required = {
            "state",
            "nominal_state",
            "action",
            "foot",
            "contact_force",
            "contact_binary",
            "history_state",
            "history_action",
            "history_foot",
            "history_contact_force",
            "history_contact_binary",
            "history_valid",
            "positive_current_state",
            "positive_history_state",
            "positive_history_action",
            "positive_history_valid",
            "positive_pair_valid",
            "is_nominal",
            "world_id",
            "state_mean",
            "state_std",
            "delta_mean",
            "delta_std",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Forward Predictor batch is missing fields: {missing}")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        recursive_weight: float = DEFAULT_RECURSIVE_WEIGHT,
        *,
        compute_metrics: bool = True,
        validate_batch: bool = True,
    ) -> dict[str, torch.Tensor]:
        if validate_batch:
            self._validate_batch(batch)
        if recursive_weight < 0.0:
            raise ValueError("recursive_weight must be non-negative")
        state = batch["state"]
        action = batch["action"]
        batch_size = state.size(0)
        if tuple(state.shape[1:]) != (6, 71):
            raise ValueError(f"State batch must be [batch,6,71], got {tuple(state.shape)}")
        if tuple(action.shape) != (batch_size, 5, 29):
            raise ValueError(f"Action batch must be [batch,5,29], got {tuple(action.shape)}")
        context_steps = self.model.config.context_history_steps
        expected = {
            "nominal_state": (batch_size, 5, 71),
            "foot": (batch_size, 6, 8),
            "contact_force": (batch_size, 6, 6),
            "contact_binary": (batch_size, 6, 2),
            "history_state": (batch_size, context_steps, 71),
            "history_action": (batch_size, context_steps, 29),
            "history_valid": (batch_size, context_steps),
            "positive_current_state": (batch_size, 71),
            "positive_history_state": (batch_size, context_steps, 71),
            "positive_history_action": (batch_size, context_steps, 29),
            "positive_history_valid": (batch_size, context_steps),
            "positive_pair_valid": (batch_size,),
            "is_nominal": (batch_size,),
            "world_id": (batch_size,),
        }
        invalid = {
            name: (tuple(batch[name].shape), shape)
            for name, shape in expected.items()
            if tuple(batch[name].shape) != shape
        }
        if invalid:
            raise ValueError(f"Invalid Forward Predictor batch shapes: {invalid}")

        combined_latent = self.model.encode_context(
            torch.cat((batch["history_state"], batch["positive_history_state"]), dim=0),
            torch.cat((batch["history_action"], batch["positive_history_action"]), dim=0),
            torch.cat((state[:, 0], batch["positive_current_state"]), dim=0),
            torch.cat((batch["history_valid"], batch["positive_history_valid"]), dim=0),
        )
        latent, positive_latent = combined_latent.split(batch_size, dim=0)
        history_arguments = {
            "history_state": batch["history_state"],
            "history_action": batch["history_action"],
            "history_foot": batch["history_foot"],
            "history_contact_force": batch["history_contact_force"],
            "history_contact_binary": batch["history_contact_binary"],
            "history_valid": batch["history_valid"],
        }
        normalization_arguments = {
            "state_mean": batch["state_mean"],
            "state_std": batch["state_std"],
            "delta_mean": batch["delta_mean"],
            "delta_std": batch["delta_std"],
        }
        teacher_prediction, _, teacher_foot, teacher_force, teacher_binary = (
            self.model.teacher_forced(
                state,
                action,
                batch["foot"],
                batch["contact_force"],
                batch["contact_binary"],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=latent,
            )
        )
        recursive_prediction, _, recursive_foot, recursive_force, recursive_binary = (
            self.model.rollout(
                state[:, 0],
                action,
                batch["foot"][:, 0],
                batch["contact_force"][:, 0],
                batch["contact_binary"][:, 0],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=latent,
            )
        )
        target = state[:, 1:]
        target_foot = batch["foot"][:, 1:]
        target_force = batch["contact_force"][:, 1:]
        target_binary = batch["contact_binary"][:, 1:]
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

        def prediction_branch_loss(
            error: torch.Tensor,
            foot_prediction: torch.Tensor,
            force_prediction: torch.Tensor,
            binary_prediction: torch.Tensor,
        ) -> torch.Tensor:
            return (
                _state_loss(error, self.loss_config, huber=True)
                + self.loss_config.foot_weight
                * _foot_loss(foot_prediction, target_foot, self.loss_config, huber=True)
                + _contact_loss(
                    force_prediction,
                    binary_prediction,
                    target_force,
                    target_binary,
                    self.loss_config,
                    huber=True,
                )
            )

        teacher_loss = prediction_branch_loss(
            teacher_error, teacher_foot, teacher_force, teacher_binary
        )
        recursive_loss = prediction_branch_loss(
            recursive_error, recursive_foot, recursive_force, recursive_binary
        )
        prediction_loss = teacher_loss + float(recursive_weight) * recursive_loss
        counterfactual_response = _normalized_state_error(
            target,
            batch["nominal_state"],
            batch["state_mean"],
            batch["state_std"],
            batch["delta_std"],
        )
        representation_loss, representation_metrics, partner = _counterfactual_representation_loss(
            latent,
            positive_latent,
            counterfactual_response,
            batch["positive_pair_valid"].bool(),
            batch["history_valid"].all(dim=1),
            batch["positive_history_valid"].all(dim=1),
            batch["world_id"],
            response_distance_scale=self.loss_config.response_distance_scale,
        )
        total_loss = prediction_loss + self.loss_config.representation_weight * representation_loss
        if not compute_metrics:
            return {
                "loss": total_loss,
                "prediction_loss": prediction_loss.detach(),
                "representation_loss": representation_loss.detach(),
            }

        with torch.no_grad():
            unchanged = state[:, :1].expand_as(target)
            unchanged_error = _normalized_state_error(
                unchanged,
                target,
                batch["state_mean"],
                batch["state_std"],
                batch["delta_std"],
            )
            shuffled_prediction = self.model.rollout(
                state[:, 0],
                action,
                batch["foot"][:, 0],
                batch["contact_force"][:, 0],
                batch["contact_binary"][:, 0],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=latent.index_select(0, partner),
            )[0]
            shuffled_error = _normalized_state_error(
                shuffled_prediction,
                target,
                batch["state_mean"],
                batch["state_std"],
                batch["delta_std"],
            )
            nominal = batch["is_nominal"].bool()
            dr = ~nominal
            shuffle_valid = dr & (batch["world_id"] != batch["world_id"].index_select(0, partner))
            true_dr_mse = _masked_mean(recursive_error[:, -1].square(), shuffle_valid)
            shuffled_dr_mse = _masked_mean(shuffled_error[:, -1].square(), shuffle_valid)
            latent_shuffle_ratio = shuffled_dr_mse / true_dr_mse.clamp_min(1.0e-8)

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss.detach(),
            "representation_loss": representation_loss.detach(),
            "one_step_nmse": _masked_nmse(recursive_error[:, 0], unchanged_error[:, 0]),
            "nominal_five_step_nmse": _masked_nmse(
                recursive_error[:, -1], unchanged_error[:, -1], nominal
            ),
            "dr_five_step_nmse": _masked_nmse(recursive_error[:, -1], unchanged_error[:, -1], dr),
            "latent_shuffle_dr_error_ratio": latent_shuffle_ratio,
            "nominal_counterfactual_rms": _masked_rms(counterfactual_response, nominal),
            "dr_counterfactual_rms": _masked_rms(counterfactual_response, dr),
            **representation_metrics,
        }
