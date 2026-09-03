"""Five-step dynamics and matched contrastive losses for the Forward Predictor."""

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
    ForwardDynamicsTransformer,
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
    foot_weight: float = 1.0
    contact_force_weight: float = 1.0
    contact_binary_weight: float = 1.0
    contrastive_weight: float = 0.01
    contrastive_temperature: float = 0.1
    contrastive_negative_distance: float = 1.25
    contrastive_hard_negative_count: int = 255
    contrastive_phase_distance_scale: float = 50.0
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
            "contrastive_weight",
        )
        invalid = {name: getattr(self, name) for name in weights if getattr(self, name) < 0.0}
        if invalid:
            raise ValueError(f"Forward Predictor loss weights must be non-negative: {invalid}")
        if self.huber_delta <= 0.0:
            raise ValueError("Forward Predictor huber_delta must be positive")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive")
        if self.contrastive_negative_distance < 0.0:
            raise ValueError("contrastive_negative_distance must be non-negative")
        if self.contrastive_hard_negative_count < 1:
            raise ValueError("contrastive_hard_negative_count must be positive")
        if self.contrastive_phase_distance_scale <= 0.0:
            raise ValueError("contrastive_phase_distance_scale must be positive")


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


def _matched_hard_negative_contrastive_loss(
    dynamics_latent: torch.Tensor,
    context_valid: torch.Tensor,
    dynamics_id: torch.Tensor,
    cohort_id: torch.Tensor,
    normalized_theta: torch.Tensor,
    motion_id: torch.Tensor,
    motion_step: torch.Tensor,
    episode_id: torch.Tensor,
    episode_step: torch.Tensor,
    normalized_state: torch.Tensor,
    normalized_action: torch.Tensor,
    contact_binary: torch.Tensor,
    *,
    temperature: float,
    negative_distance: float,
    hard_negative_count: int,
    phase_distance_scale: float,
    positive_nonoverlap_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrast same-dynamics histories against nuisance-matched theta-far classes."""

    if dynamics_latent.ndim != 3 or context_valid.shape != dynamics_latent.shape[:2]:
        raise ValueError("Dynamics latents and context-valid mask must be [batch,horizon,...]")
    batch_size = dynamics_latent.size(0)
    if dynamics_id.shape != (batch_size,) or normalized_theta.shape[0] != batch_size:
        raise ValueError("Dynamics IDs and theta targets must share the latent batch dimension")
    vector_fields = {
        "motion_id": motion_id,
        "motion_step": motion_step,
        "episode_id": episode_id,
        "episode_step": episode_step,
        "cohort_id": cohort_id,
    }
    invalid_vectors = {
        name: tuple(value.shape)
        for name, value in vector_fields.items()
        if value.shape != (batch_size,)
    }
    if invalid_vectors:
        raise ValueError(f"Contrastive metadata must be batch vectors: {invalid_vectors}")
    if normalized_state.ndim != 2 or normalized_state.size(0) != batch_size:
        raise ValueError("Current normalized state must be [batch,state_dim]")
    if normalized_action.ndim != 2 or normalized_action.size(0) != batch_size:
        raise ValueError("Current normalized action must be [batch,action_dim]")
    if contact_binary.ndim != 2 or contact_binary.size(0) != batch_size:
        raise ValueError("Current contact state must be [batch,contact_dim]")
    if hard_negative_count < 1 or phase_distance_scale <= 0.0:
        raise ValueError("Hard-negative count and phase-distance scale must be positive")
    if positive_nonoverlap_steps < 1:
        raise ValueError("Positive non-overlap distance must be positive")

    valid_weight = context_valid.unsqueeze(-1).to(dynamics_latent.dtype)
    pooled_latent = (dynamics_latent * valid_weight).sum(dim=1) / valid_weight.sum(dim=1).clamp_min(
        1.0
    )
    embedding = F.normalize(pooled_latent.float(), dim=-1, eps=1.0e-8)
    cosine = embedding @ embedding.transpose(0, 1)
    logits = cosine / float(temperature)

    theta = normalized_theta.detach().float()
    theta_distance = torch.cdist(theta, theta) / float(theta.size(-1)) ** 0.5
    # A reset-padded history is useful to the transition predictor, but it is
    # not a reliable environment identity observation.  Representation pairs
    # therefore require every one of the long-context frames to be valid.
    sample_valid = context_valid.all(dim=1)
    pair_valid = sample_valid[:, None] & sample_valid[None, :]
    pair_valid &= ~torch.eye(batch_size, dtype=torch.bool, device=dynamics_latent.device)
    same_dynamics = dynamics_id[:, None] == dynamics_id[None, :]
    positive = pair_valid & same_dynamics
    different_dynamics = pair_valid & ~same_dynamics
    theta_far = theta_distance >= float(negative_distance)
    theta_far_pair = different_dynamics & theta_far
    ignored_near = different_dynamics & ~theta_far

    same_motion = motion_id[:, None] == motion_id[None, :]
    phase_gap = (motion_step[:, None] - motion_step[None, :]).abs()
    contact_mismatch = (
        (contact_binary[:, None].bool() != contact_binary[None, :].bool()).float().mean(dim=-1)
    )
    same_cohort = cohort_id[:, None] == cohort_id[None, :]
    hard_candidate = theta_far_pair & same_cohort

    state_action = torch.cat((normalized_state, normalized_action), dim=-1).detach().float()
    state_action_distance = (
        torch.cdist(state_action, state_action) / float(state_action.size(-1)) ** 0.5
    )
    hard_score = (
        state_action_distance + phase_gap.float() / float(phase_distance_scale) + contact_mismatch
    )
    # Exact motion/phase cohorts are always selected before easier cross-motion
    # negatives.  If the configured budget exceeds the matched 127 classes,
    # theta-far contexts from other motions/phases fill the remainder.
    hard_score = hard_score + (~same_cohort).to(hard_score.dtype) * 1.0e6
    hard_score = hard_score.masked_fill(~theta_far_pair, torch.inf)
    selected_count = min(int(hard_negative_count), max(batch_size - 1, 1))
    selected_score, selected_indices = torch.topk(
        hard_score,
        k=selected_count,
        dim=1,
        largest=False,
    )
    negative = torch.zeros_like(hard_candidate)
    negative.scatter_(1, selected_indices, torch.isfinite(selected_score))
    candidate = positive | negative

    masked_logits = logits.masked_fill(~candidate, -1.0e4)
    log_probability = logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    valid_anchor = (positive_count > 0) & (negative_count > 0)
    mean_positive_log_probability = log_probability.masked_fill(~positive, 0.0).sum(
        dim=1
    ) / positive_count.clamp_min(1)
    anchor_weight = valid_anchor.to(logits.dtype)
    loss = -(mean_positive_log_probability * anchor_weight).sum() / anchor_weight.sum().clamp_min(
        1.0
    )

    positive_pairs = positive.sum().to(logits.dtype)
    negative_pairs = negative.sum().to(logits.dtype)
    different_pairs = different_dynamics.sum().to(logits.dtype)
    near_pairs = ignored_near.sum().to(logits.dtype)
    theta_far_pairs = theta_far_pair.sum().to(logits.dtype)
    hard_candidate_pairs = hard_candidate.sum().to(logits.dtype)
    cross_motion_positive = positive & ~same_motion
    different_episode = episode_id[:, None] != episode_id[None, :]
    episode_gap = (episode_step[:, None] - episode_step[None, :]).abs()
    nonoverlap_positive = positive & (
        ~same_motion | different_episode | (episode_gap >= int(positive_nonoverlap_steps))
    )
    cross_motion_positive_pairs = cross_motion_positive.sum().to(logits.dtype)
    nonoverlap_positive_pairs = nonoverlap_positive.sum().to(logits.dtype)
    metrics = {
        "contrastive_valid_anchor_fraction": anchor_weight.mean(),
        "contrastive_positive_pair_count": positive_pairs,
        "contrastive_negative_pair_count": negative_pairs,
        "contrastive_negative_pair_fraction": negative_pairs / different_pairs.clamp_min(1.0),
        "contrastive_ignored_near_pair_fraction": near_pairs / different_pairs.clamp_min(1.0),
        "contrastive_theta_far_pair_fraction": theta_far_pairs / different_pairs.clamp_min(1.0),
        "contrastive_hard_candidate_pair_fraction": hard_candidate_pairs
        / theta_far_pairs.clamp_min(1.0),
        "contrastive_matched_negative_fraction": (negative & same_cohort).sum().to(logits.dtype)
        / negative_pairs.clamp_min(1.0),
        "contrastive_cross_motion_positive_fraction": cross_motion_positive_pairs
        / positive_pairs.clamp_min(1.0),
        "contrastive_nonoverlap_positive_fraction": nonoverlap_positive_pairs
        / positive_pairs.clamp_min(1.0),
        "contrastive_positive_cosine": cosine.masked_fill(~positive, 0.0).sum()
        / positive_pairs.clamp_min(1.0),
        "contrastive_negative_cosine": cosine.masked_fill(~negative, 0.0).sum()
        / negative_pairs.clamp_min(1.0),
        "contrastive_hard_negative_phase_gap": phase_gap.float().masked_fill(~negative, 0.0).sum()
        / negative_pairs.clamp_min(1.0),
        "contrastive_hard_negative_state_action_rms": state_action_distance.masked_fill(
            ~negative, 0.0
        ).sum()
        / negative_pairs.clamp_min(1.0),
        "contrastive_hard_negative_theta_rms": theta_distance.masked_fill(~negative, 0.0).sum()
        / negative_pairs.clamp_min(1.0),
    }
    return loss, metrics


def _contact_losses(
    normalized_force_prediction: torch.Tensor,
    binary_logits: torch.Tensor,
    normalized_force_target: torch.Tensor,
    binary_target: torch.Tensor,
    config: ForwardPredictorLossConfig,
    *,
    huber: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if normalized_force_prediction.shape != normalized_force_target.shape:
        raise ValueError("Contact-force prediction and target shapes must match")
    if binary_logits.shape != binary_target.shape:
        raise ValueError("Contact-binary prediction and target shapes must match")
    if huber:
        force_loss = F.huber_loss(
            normalized_force_prediction,
            normalized_force_target,
            reduction="mean",
            delta=config.huber_delta,
        )
    else:
        force_loss = (normalized_force_prediction - normalized_force_target).square().mean()
    binary_loss = F.binary_cross_entropy_with_logits(
        binary_logits,
        binary_target.float(),
        reduction="mean",
    )
    components = {
        "contact_force": force_loss,
        "contact_binary": binary_loss,
    }
    total = config.contact_force_weight * force_loss + config.contact_binary_weight * binary_loss
    return total, components


def _foot_loss(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    config: ForwardPredictorLossConfig,
    *,
    huber: bool,
) -> torch.Tensor:
    if normalized_prediction.shape != normalized_target.shape:
        raise ValueError("Foot prediction and target shapes must match")
    if huber:
        return F.huber_loss(
            normalized_prediction,
            normalized_target,
            reduction="mean",
            delta=config.huber_delta,
        )
    return (normalized_prediction - normalized_target).square().mean()


def _privileged_physical_errors(
    normalized_foot_prediction: torch.Tensor,
    normalized_foot_target: torch.Tensor,
    force_prediction: torch.Tensor,
    force_target: torch.Tensor,
    binary_logits: torch.Tensor,
    binary_target: torch.Tensor,
    foot_mean: torch.Tensor,
    foot_std: torch.Tensor,
    contact_force_mean: torch.Tensor,
    contact_force_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_foot = (normalized_foot_prediction * foot_std + foot_mean).reshape(
        *normalized_foot_prediction.shape[:-1], 2, 4
    )
    target_foot = (normalized_foot_target * foot_std + foot_mean).reshape(
        *normalized_foot_target.shape[:-1], 2, 4
    )
    height_error = (predicted_foot[..., 0] - target_foot[..., 0]).abs()
    velocity_error = torch.linalg.vector_norm(
        predicted_foot[..., 1:] - target_foot[..., 1:], dim=-1
    )

    predicted_force = force_prediction * contact_force_std + contact_force_mean
    target_force = force_target * contact_force_std + contact_force_mean
    force_error = torch.linalg.vector_norm(
        (predicted_force - target_force).reshape(*predicted_force.shape[:-1], 2, 3),
        dim=-1,
    )
    binary_probability = torch.sigmoid(binary_logits)
    binary_prediction = binary_probability >= 0.5
    binary_target_bool = binary_target.bool()
    return {
        "foot_height_error_m": height_error.mean(),
        "foot_height_error_p95_m": torch.quantile(height_error.float(), 0.95),
        "foot_velocity_error_mps": velocity_error.mean(),
        "foot_velocity_error_p95_mps": torch.quantile(velocity_error.float(), 0.95),
        "contact_force_error_n": force_error.mean(),
        "contact_force_error_p95_n": torch.quantile(force_error.float(), 0.95),
        "contact_binary_accuracy": (binary_prediction == binary_target_bool).float().mean(),
        "contact_binary_brier": (binary_probability - binary_target.float()).square().mean(),
    }


class ForwardPredictorObjective(nn.Module):
    """Train one shared transition model through its complete five-step rollout graph."""

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
            "state_mean",
            "state_std",
            "foot_mean",
            "foot_std",
            "contact_force_mean",
            "contact_force_std",
            "delta_mean",
            "delta_std",
            "dynamics_id",
            "motion_id",
            "motion_step",
            "motion_group_id",
            "cohort_id",
            "episode_id",
            "episode_step",
            "privileged_dynamics",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Forward Predictor batch is missing fields: {missing}")
        finite_names = required.difference(
            {"history_valid", "contact_binary", "history_contact_binary"}
        )
        finite = torch.stack([torch.isfinite(batch[name]).all() for name in finite_names]).all()
        if not bool(finite):
            raise ValueError("Forward Predictor batch contains non-finite values")

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
        if state.ndim != 3 or state.shape[1:] != (6, 71):
            raise ValueError(f"State batch must be [batch,6,71], got {tuple(state.shape)}")
        if action.shape != (state.size(0), 5, 29):
            raise ValueError(f"Action batch must be [batch,5,29], got {tuple(action.shape)}")
        if (
            batch["privileged_dynamics"].ndim != 2
            or batch["privileged_dynamics"].size(0) != state.size(0)
            or batch["privileged_dynamics"].size(1) < 1
        ):
            raise ValueError(
                "Theta mining labels must have shape [batch,theta_dim>0], got "
                f"{tuple(batch['privileged_dynamics'].shape)}"
            )
        metadata_names = (
            "dynamics_id",
            "motion_group_id",
            "cohort_id",
            "motion_id",
            "motion_step",
            "episode_id",
            "episode_step",
        )
        invalid_metadata = {
            name: tuple(batch[name].shape)
            for name in metadata_names
            if batch[name].shape != (state.size(0),)
        }
        if invalid_metadata:
            raise ValueError(f"Contrastive metadata must be batch vectors: {invalid_metadata}")
        expected_privileged = {
            "foot": (state.size(0), 6, 8),
            "contact_force": (state.size(0), 6, 6),
            "contact_binary": (state.size(0), 6, 2),
            "history_foot": (state.size(0), 10, 8),
            "history_contact_force": (state.size(0), 10, 6),
            "history_contact_binary": (state.size(0), 10, 2),
        }
        invalid_privileged = {
            name: (tuple(batch[name].shape), shape)
            for name, shape in expected_privileged.items()
            if tuple(batch[name].shape) != shape
        }
        if invalid_privileged:
            raise ValueError(f"Invalid Forward Predictor privileged shapes: {invalid_privileged}")
        expected_context = {
            "history_state": (
                state.size(0),
                self.model.config.context_history_steps,
                self.model.config.state_dim,
            ),
            "history_action": (
                state.size(0),
                self.model.config.context_history_steps,
                self.model.config.action_dim,
            ),
            "history_valid": (
                state.size(0),
                self.model.config.context_history_steps,
            ),
        }
        invalid_context = {
            name: (tuple(batch[name].shape), shape)
            for name, shape in expected_context.items()
            if tuple(batch[name].shape) != shape
        }
        if invalid_context:
            raise ValueError(f"Invalid Forward Predictor context shapes: {invalid_context}")

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
        dynamics_latent_seed = self.model.encode_context(
            batch["history_state"],
            batch["history_action"],
            state[:, 0],
            batch["history_valid"],
        )
        (
            teacher_prediction,
            teacher_delta,
            teacher_foot,
            teacher_force,
            teacher_binary_logits,
            dynamics_latent,
            context_valid,
        ) = self.model.teacher_forced(
            state,
            action,
            batch["foot"],
            batch["contact_force"],
            batch["contact_binary"],
            **history_arguments,
            **normalization_arguments,
            dynamics_latent=dynamics_latent_seed,
            return_context=True,
        )
        if self.training and recursive_weight == 0.0:
            with torch.no_grad():
                (
                    recursive_prediction,
                    recursive_delta,
                    recursive_foot,
                    recursive_force,
                    recursive_binary_logits,
                ) = self.model.rollout(
                    state[:, 0],
                    action,
                    batch["foot"][:, 0],
                    batch["contact_force"][:, 0],
                    batch["contact_binary"][:, 0],
                    **history_arguments,
                    **normalization_arguments,
                    dynamics_latent=dynamics_latent_seed,
                )
        else:
            (
                recursive_prediction,
                recursive_delta,
                recursive_foot,
                recursive_force,
                recursive_binary_logits,
            ) = self.model.rollout(
                state[:, 0],
                action,
                batch["foot"][:, 0],
                batch["contact_force"][:, 0],
                batch["contact_binary"][:, 0],
                **history_arguments,
                **normalization_arguments,
                dynamics_latent=dynamics_latent_seed,
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
        teacher_state_loss, teacher_components = _state_losses(
            teacher_error,
            self.loss_config,
            huber=True,
        )
        recursive_state_loss, recursive_components = _state_losses(
            recursive_error,
            self.loss_config,
            huber=True,
        )
        teacher_foot_loss = _foot_loss(
            teacher_foot,
            target_foot,
            self.loss_config,
            huber=True,
        )
        recursive_foot_loss = _foot_loss(
            recursive_foot,
            target_foot,
            self.loss_config,
            huber=True,
        )
        teacher_contact_loss, teacher_contact_components = _contact_losses(
            teacher_force,
            teacher_binary_logits,
            target_force,
            target_binary,
            self.loss_config,
            huber=True,
        )
        recursive_contact_loss, recursive_contact_components = _contact_losses(
            recursive_force,
            recursive_binary_logits,
            target_force,
            target_binary,
            self.loss_config,
            huber=True,
        )
        teacher_loss = (
            teacher_state_loss
            + self.loss_config.foot_weight * teacher_foot_loss
            + teacher_contact_loss
        )
        recursive_loss = (
            recursive_state_loss
            + self.loss_config.foot_weight * recursive_foot_loss
            + recursive_contact_loss
        )
        contrastive_loss, contrastive_metrics = _matched_hard_negative_contrastive_loss(
            dynamics_latent,
            context_valid,
            batch["dynamics_id"],
            batch["cohort_id"],
            batch["privileged_dynamics"],
            batch["motion_id"],
            batch["motion_step"],
            batch["episode_id"],
            batch["episode_step"],
            state[:, 0],
            action[:, 0],
            batch["contact_binary"][:, 0],
            temperature=self.loss_config.contrastive_temperature,
            negative_distance=self.loss_config.contrastive_negative_distance,
            hard_negative_count=self.loss_config.contrastive_hard_negative_count,
            phase_distance_scale=self.loss_config.contrastive_phase_distance_scale,
            positive_nonoverlap_steps=(
                self.model.config.context_history_steps + self.model.config.horizon
            ),
        )
        total_loss = (
            teacher_loss
            + float(recursive_weight) * recursive_loss
            + self.loss_config.contrastive_weight * contrastive_loss
        )

        if not compute_metrics:
            return {
                "loss": total_loss,
                "teacher_loss": teacher_loss.detach(),
                "recursive_loss": recursive_loss.detach(),
                "contrastive_loss": contrastive_loss.detach(),
            }

        with torch.no_grad():
            unchanged = state[:, :1].expand_as(target)
            unchanged_foot = batch["foot"][:, :1].expand_as(target_foot)
            unchanged_force = batch["contact_force"][:, :1].expand_as(target_force)
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
            teacher_foot_mse = _foot_loss(
                teacher_foot,
                target_foot,
                self.loss_config,
                huber=False,
            )
            recursive_foot_mse = _foot_loss(
                recursive_foot,
                target_foot,
                self.loss_config,
                huber=False,
            )
            no_change_foot_mse = _foot_loss(
                unchanged_foot,
                target_foot,
                self.loss_config,
                huber=False,
            )
            teacher_force_mse = (teacher_force - target_force).square().mean()
            recursive_force_mse = (recursive_force - target_force).square().mean()
            no_change_force_mse = (unchanged_force - target_force).square().mean()
            horizon_metrics: dict[str, torch.Tensor] = {}
            for index in range(5):
                horizon_state_loss, _ = _state_losses(
                    recursive_error[:, index : index + 1],
                    self.loss_config,
                    huber=True,
                )
                horizon_contact_loss, _ = _contact_losses(
                    recursive_force[:, index : index + 1],
                    recursive_binary_logits[:, index : index + 1],
                    target_force[:, index : index + 1],
                    target_binary[:, index : index + 1],
                    self.loss_config,
                    huber=True,
                )
                horizon_foot_loss = _foot_loss(
                    recursive_foot[:, index : index + 1],
                    target_foot[:, index : index + 1],
                    self.loss_config,
                    huber=True,
                )
                horizon_loss = (
                    horizon_state_loss
                    + self.loss_config.foot_weight * horizon_foot_loss
                    + horizon_contact_loss
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
            privileged_metrics = _privileged_physical_errors(
                recursive_foot,
                target_foot,
                recursive_force,
                target_force,
                recursive_binary_logits,
                target_binary,
                batch["foot_mean"],
                batch["foot_std"],
                batch["contact_force_mean"],
                batch["contact_force_std"],
            )
            teacher_privileged_metrics = _privileged_physical_errors(
                teacher_foot,
                target_foot,
                teacher_force,
                target_force,
                teacher_binary_logits,
                target_binary,
                batch["foot_mean"],
                batch["foot_std"],
                batch["contact_force_mean"],
                batch["contact_force_std"],
            )
        return {
            "loss": total_loss,
            "teacher_loss": teacher_loss.detach(),
            "recursive_loss": recursive_loss.detach(),
            "recursive_weight": teacher_loss.new_tensor(float(recursive_weight)),
            "contrastive_loss": contrastive_loss.detach(),
            "contrastive_weight": teacher_loss.new_tensor(self.loss_config.contrastive_weight),
            "contrastive_temperature": teacher_loss.new_tensor(
                self.loss_config.contrastive_temperature
            ),
            "contrastive_negative_distance": teacher_loss.new_tensor(
                self.loss_config.contrastive_negative_distance
            ),
            "contrastive_hard_negative_count": teacher_loss.new_tensor(
                self.loss_config.contrastive_hard_negative_count
            ),
            "contrastive_phase_distance_scale": teacher_loss.new_tensor(
                self.loss_config.contrastive_phase_distance_scale
            ),
            **contrastive_metrics,
            "context_full_history_fraction": context_valid.float().mean(),
            "dynamics_latent_rms": dynamics_latent.detach().square().mean().sqrt(),
            "teacher_mse": teacher_mse,
            "teacher_nmse": teacher_mse / no_change_mse.clamp_min(1.0e-8),
            "teacher_foot_loss": teacher_foot_loss.detach(),
            "teacher_foot_mse": teacher_foot_mse,
            "teacher_foot_nmse": teacher_foot_mse / no_change_foot_mse.clamp_min(1.0e-8),
            "teacher_contact_force_mse": teacher_force_mse,
            "teacher_contact_force_nmse": teacher_force_mse / no_change_force_mse.clamp_min(1.0e-8),
            "rollout_loss": recursive_loss.detach(),
            "rollout_mse": recursive_mse,
            "rollout_no_change_loss": no_change_mse,
            "rollout_nmse": recursive_mse / no_change_mse.clamp_min(1.0e-8),
            "foot_loss": recursive_foot_loss.detach(),
            "foot_mse": recursive_foot_mse,
            "foot_no_change_mse": no_change_foot_mse,
            "foot_nmse": recursive_foot_mse / no_change_foot_mse.clamp_min(1.0e-8),
            "contact_force_mse": recursive_force_mse,
            "contact_force_no_change_mse": no_change_force_mse,
            "contact_force_nmse": recursive_force_mse / no_change_force_mse.clamp_min(1.0e-8),
            "teacher_normalized_delta_rms": teacher_delta.detach().square().mean().sqrt(),
            "predicted_normalized_delta_rms": recursive_delta.detach().square().mean().sqrt(),
            "teacher_normalized_foot_rms": teacher_foot.detach().square().mean().sqrt(),
            "predicted_normalized_foot_rms": recursive_foot.detach().square().mean().sqrt(),
            "history_valid_fraction": batch["history_valid"].float().mean(),
            **{
                f"teacher_{name}_loss": value.detach() for name, value in teacher_components.items()
            },
            **{
                f"teacher_{name}_loss": value.detach()
                for name, value in teacher_contact_components.items()
            },
            **{f"{name}_loss": value.detach() for name, value in recursive_components.items()},
            **{
                f"{name}_loss": value.detach()
                for name, value in recursive_contact_components.items()
            },
            **horizon_metrics,
            **physical_metrics,
            **privileged_metrics,
            **{f"teacher_{name}": value for name, value in teacher_physical_metrics.items()},
            **{f"teacher_{name}": value for name, value in teacher_privileged_metrics.items()},
        }
