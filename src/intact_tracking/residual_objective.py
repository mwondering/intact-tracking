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
    nominal_pair_weight: float = 1.0
    nominal_effect_weight: float = 1.0
    tracking_weight: float = 1.0
    residual_l2_weight: float = 0.2
    residual_smooth_weight: float = 1.0e-3
    root_position_weight: float = 5.0
    root_orientation_weight: float = 2.0
    joint_position_weight: float = 1.0
    action_clip: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "action_clip":
                if value is not None and value <= 0:
                    raise ValueError("action_clip must be positive when configured")
            elif value < 0:
                raise ValueError(f"{name} must be non-negative")


_STATE_POSE_SLICES = {
    "root_position": slice(0, 3),
    "root_orientation": slice(3, 7),
    "joint_position": slice(13, 42),
}
_POSE_DELTA_SLICES = {
    "root_position": slice(0, 3),
    "root_rotation_vector": slice(3, 6),
    "joint_position": slice(6, 35),
}


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for scalar-first (w, x, y, z) quaternions."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _rotation_vector_to_quaternion(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Map an axis-angle rotation vector to a unit scalar-first quaternion."""
    angle = rotation_vector.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    regular_scale = torch.sin(half_angle) / angle.clamp_min(1.0e-8)
    small_scale = 0.5 - angle.square() / 48.0
    vector_scale = torch.where(angle > 1.0e-4, regular_scale, small_scale)
    quaternion = torch.cat((torch.cos(half_angle), rotation_vector * vector_scale), dim=-1)
    return torch.nn.functional.normalize(quaternion, dim=-1, eps=1.0e-8)


def _physical_quaternion(
    normalized_state: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> torch.Tensor:
    section = _STATE_POSE_SLICES["root_orientation"]
    value = normalized_state[..., section] * state_std[section] + state_mean[section]
    return torch.nn.functional.normalize(value, dim=-1, eps=1.0e-8)


def _reconstruct_pose(
    pose_delta: torch.Tensor,
    current_state: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compose five endpoint deltas with the same current state, without chaining."""
    if pose_delta.size(-1) != 35 or current_state.size(-1) != 71:
        raise ValueError(
            "Pose delta/current state must end in 35/71 values, got "
            f"{tuple(pose_delta.shape)} and {tuple(current_state.shape)}"
        )
    if pose_delta.ndim != current_state.ndim + 1:
        raise ValueError(
            "Pose deltas must add one horizon dimension to current state, got "
            f"{tuple(pose_delta.shape)} and {tuple(current_state.shape)}"
        )
    current = current_state.unsqueeze(-2)
    root_position = (
        current[..., _STATE_POSE_SLICES["root_position"]]
        + pose_delta[..., _POSE_DELTA_SLICES["root_position"]]
    )
    joint_position = (
        current[..., _STATE_POSE_SLICES["joint_position"]]
        + pose_delta[..., _POSE_DELTA_SLICES["joint_position"]]
    )
    current_orientation = _physical_quaternion(current, state_mean, state_std)
    rotation_delta = _rotation_vector_to_quaternion(
        pose_delta[..., _POSE_DELTA_SLICES["root_rotation_vector"]]
    )
    # The delta is expressed in the world frame, matching target * current^-1.
    root_orientation = torch.nn.functional.normalize(
        _quaternion_multiply(rotation_delta, current_orientation),
        dim=-1,
        eps=1.0e-8,
    )
    return {
        "root_position": root_position,
        "root_orientation": root_orientation,
        "joint_position": joint_position,
    }


def _pose_losses(
    pose_delta: torch.Tensor,
    current_state: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    config: ResidualLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected = (*current_state.shape[:-1], pose_delta.size(-2), 71)
    if tuple(target.shape) != expected:
        raise ValueError(
            "Pose target must match the current-state batch and delta horizon: "
            f"expected {expected}, got {tuple(target.shape)}"
        )
    reconstructed = _reconstruct_pose(pose_delta, current_state, state_mean, state_std)
    target_quat = _physical_quaternion(target, state_mean, state_std)
    dot = (reconstructed["root_orientation"] * target_quat).sum(dim=-1).clamp(-1.0, 1.0)
    component = {
        "root_position": (
            reconstructed["root_position"] - target[..., _STATE_POSE_SLICES["root_position"]]
        )
        .square()
        .mean(),
        "root_orientation": (1.0 - dot.square()).mean(),
        "joint_position": (
            reconstructed["joint_position"] - target[..., _STATE_POSE_SLICES["joint_position"]]
        )
        .square()
        .mean(),
    }
    total = (
        config.root_position_weight * component["root_position"]
        + config.root_orientation_weight * component["root_orientation"]
        + config.joint_position_weight * component["joint_position"]
    )
    return total, component


def _quaternion_conjugate(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _pose_effect_losses(
    dr_pose_delta: torch.Tensor,
    nominal_pose_delta: torch.Tensor,
    current_state: torch.Tensor,
    dr_target: torch.Tensor,
    nominal_target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    config: ResidualLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Match the real pose difference caused only by DR under equal state/actions."""

    dr_pose = _reconstruct_pose(dr_pose_delta, current_state, state_mean, state_std)
    nominal_pose = _reconstruct_pose(
        nominal_pose_delta, current_state, state_mean, state_std
    )
    dr_target_quat = _physical_quaternion(dr_target, state_mean, state_std)
    nominal_target_quat = _physical_quaternion(nominal_target, state_mean, state_std)

    predicted_root_effect = dr_pose["root_position"] - nominal_pose["root_position"]
    target_root_effect = (
        dr_target[..., _STATE_POSE_SLICES["root_position"]]
        - nominal_target[..., _STATE_POSE_SLICES["root_position"]]
    )
    predicted_joint_effect = dr_pose["joint_position"] - nominal_pose["joint_position"]
    target_joint_effect = (
        dr_target[..., _STATE_POSE_SLICES["joint_position"]]
        - nominal_target[..., _STATE_POSE_SLICES["joint_position"]]
    )
    predicted_relative_quat = _quaternion_multiply(
        dr_pose["root_orientation"],
        _quaternion_conjugate(nominal_pose["root_orientation"]),
    )
    target_relative_quat = _quaternion_multiply(
        dr_target_quat,
        _quaternion_conjugate(nominal_target_quat),
    )
    orientation_dot = (
        predicted_relative_quat * target_relative_quat
    ).sum(dim=-1).clamp(-1.0, 1.0)
    target_identity_dot = target_relative_quat[..., 0].clamp(-1.0, 1.0)
    component = {
        "root_position": (predicted_root_effect - target_root_effect).square().mean(),
        "root_orientation": (1.0 - orientation_dot.square()).mean(),
        "joint_position": (predicted_joint_effect - target_joint_effect).square().mean(),
    }
    zero_effect_component = {
        "root_position": target_root_effect.square().mean(),
        "root_orientation": (1.0 - target_identity_dot.square()).mean(),
        "joint_position": target_joint_effect.square().mean(),
    }
    total = (
        config.root_position_weight * component["root_position"]
        + config.root_orientation_weight * component["root_orientation"]
        + config.joint_position_weight * component["joint_position"]
    )
    zero_effect = (
        config.root_position_weight * zero_effect_component["root_position"]
        + config.root_orientation_weight * zero_effect_component["root_orientation"]
        + config.joint_position_weight * zero_effect_component["joint_position"]
    )
    return total, zero_effect, component


class ResidualTrainingObjective(nn.Module):
    """Alternating model/policy objectives with explicit gradient routing."""

    def __init__(
        self,
        model: ResidualTrackingModel,
        loss_config: ResidualLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_config = loss_config or ResidualLossConfig()

    @staticmethod
    def _validate_batch(batch: dict[str, torch.Tensor]) -> None:
        required = {
            "context",
            "context_mask",
            "state",
            "reference_state",
            "action",
            "previous_action",
            "tracker_action",
            "policy_observation",
            "policy_world",
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

    def _model_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.loss_config
        model = self.model
        world = model.encode_context(batch["context"], batch["context_mask"])
        state = batch["state"]
        action = batch["action"]
        previous_action = batch["previous_action"]
        state_mean = batch["state_mean"]
        state_std = batch["state_std"]

        forward_pose_delta = model.predict_future(
            world,
            state[:, 0],
            previous_action,
            action,
        )
        forward_loss, forward_components = _pose_losses(
            forward_pose_delta,
            state[:, 0],
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

        weighted_forward = cfg.forward_weight * forward_loss
        weighted_backward = cfg.backward_weight * backward_loss
        model_loss = weighted_forward + weighted_backward

        nominal_pair_loss = forward_loss.new_zeros(())
        nominal_forward_loss = forward_loss.new_zeros(())
        nominal_effect_loss = forward_loss.new_zeros(())
        nominal_effect_zero_loss = forward_loss.new_zeros(())
        nominal_context_swap_ratio = forward_loss.new_ones(())
        nominal_pair_count = 0
        nominal_components: dict[str, torch.Tensor] = {}
        effect_components: dict[str, torch.Tensor] = {}
        if "nominal_state" in batch:
            nominal_target = batch["nominal_state"]
            if not torch.isfinite(nominal_target).all():
                raise ValueError("nominal_state contains non-finite values")
            if nominal_target.ndim != 3 or nominal_target.shape[1:] != state[:, 1:].shape[1:]:
                raise ValueError(
                    "nominal_state must be [pair_batch,5,71], got "
                    f"{tuple(nominal_target.shape)}"
                )
            nominal_pair_count = int(nominal_target.size(0))
            if nominal_pair_count < 1 or nominal_pair_count > state.size(0):
                raise ValueError("nominal_state pair batch must fit inside the model batch")
            pair_world = world[:nominal_pair_count]
            pair_state = state[:nominal_pair_count, 0]
            pair_previous_action = previous_action[:nominal_pair_count]
            pair_action = action[:nominal_pair_count]
            nominal_pose_delta = model.predict_future(
                torch.zeros_like(pair_world),
                pair_state,
                pair_previous_action,
                pair_action,
            )
            nominal_forward_loss, nominal_components = _pose_losses(
                nominal_pose_delta,
                pair_state,
                nominal_target,
                state_mean,
                state_std,
                cfg,
            )
            nominal_effect_loss, nominal_effect_zero_loss, effect_components = (
                _pose_effect_losses(
                    forward_pose_delta[:nominal_pair_count],
                    nominal_pose_delta,
                    pair_state,
                    state[:nominal_pair_count, 1:],
                    nominal_target,
                    state_mean,
                    state_std,
                    cfg,
                )
            )
            nominal_pair_loss = cfg.nominal_pair_weight * (
                nominal_forward_loss + cfg.nominal_effect_weight * nominal_effect_loss
            )
            model_loss = model_loss + nominal_pair_loss
            with torch.no_grad():
                dr_pair_forward_loss, _ = _pose_losses(
                    forward_pose_delta[:nominal_pair_count],
                    pair_state,
                    state[:nominal_pair_count, 1:],
                    state_mean,
                    state_std,
                    cfg,
                )
                zero_world_on_dr_loss, _ = _pose_losses(
                    nominal_pose_delta,
                    pair_state,
                    state[:nominal_pair_count, 1:],
                    state_mean,
                    state_std,
                    cfg,
                )
                dr_world_on_nominal_loss, _ = _pose_losses(
                    forward_pose_delta[:nominal_pair_count],
                    pair_state,
                    nominal_target,
                    state_mean,
                    state_std,
                    cfg,
                )
                correct_assignment_loss = dr_pair_forward_loss + nominal_forward_loss.detach()
                swapped_assignment_loss = zero_world_on_dr_loss + dr_world_on_nominal_loss
                nominal_context_swap_ratio = (
                    swapped_assignment_loss / correct_assignment_loss.clamp_min(1e-8)
                )

        with torch.no_grad():
            if world.size(0) > 1:
                shuffled_world = world.roll(1, dims=0)
                shuffled_forward_delta = model.predict_future(
                    shuffled_world,
                    state[:, 0],
                    previous_action,
                    action,
                )
                shuffled_forward_loss, _ = _pose_losses(
                    shuffled_forward_delta,
                    state[:, 0],
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
            no_change_loss, _ = _pose_losses(
                torch.zeros_like(forward_pose_delta),
                state[:, 0],
                state[:, 1:],
                state_mean,
                state_std,
                cfg,
            )

        output = {
            "loss": model_loss,
            "model_loss": model_loss.detach(),
            "forward_loss": forward_loss,
            "backward_loss": backward_loss,
            "weighted_forward_loss": weighted_forward.detach(),
            "weighted_backward_loss": weighted_backward.detach(),
            "nominal_pair_loss": nominal_pair_loss.detach(),
            "nominal_forward_loss": nominal_forward_loss.detach(),
            "nominal_effect_loss": nominal_effect_loss.detach(),
            "nominal_effect_nmse": (
                nominal_effect_loss.detach() / nominal_effect_zero_loss.clamp_min(1e-8)
            ),
            "nominal_true_effect_loss": nominal_effect_zero_loss.detach(),
            "nominal_context_swap_ratio": nominal_context_swap_ratio.detach(),
            "nominal_pair_count": forward_loss.new_tensor(float(nominal_pair_count)),
            "forward_no_change_loss": no_change_loss,
            "forward_nmse": forward_loss.detach() / no_change_loss.clamp_min(1e-8),
            "forward_context_shuffle_ratio": (
                shuffled_forward_loss / forward_loss.detach().clamp_min(1e-8)
            ),
            "backward_context_shuffle_ratio": (
                shuffled_backward_loss / backward_loss.detach().clamp_min(1e-8)
            ),
        }
        for name, value in forward_components.items():
            output[f"forward_{name}_loss"] = value.detach()
        for name, value in nominal_components.items():
            output[f"nominal_forward_{name}_loss"] = value.detach()
        for name, value in effect_components.items():
            output[f"nominal_effect_{name}_loss"] = value.detach()
        return output

    def _policy_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.loss_config
        model = self.model
        state = batch["state"]
        previous_action = batch["previous_action"]
        state_mean = batch["state_mean"]
        state_std = batch["state_std"]

        # Context and Forward parameters are constants on this branch. The
        # functional Forward call retains the derivative with respect to the
        # complete candidate trunk, which is the residual-policy signal.
        policy_world = batch["policy_world"].detach()
        with torch.no_grad():
            dynamics_world = model.encode_context(batch["context"], batch["context_mask"]).detach()
        residual = model.residual_action_trunk(policy_world, batch["policy_observation"])
        candidate_unclipped = batch["tracker_action"] + residual
        candidate = candidate_unclipped
        if cfg.action_clip is not None:
            candidate = candidate.clamp(-cfg.action_clip, cfg.action_clip)
        action_mean = batch["action_mean"]
        action_std = batch["action_std"]
        candidate_normalized = (candidate - action_mean) / action_std
        policy_pose_delta = model.predict_future_with_frozen_dynamics(
            dynamics_world,
            state[:, 0].detach(),
            previous_action.detach(),
            candidate_normalized,
        )
        tracking_loss, tracking_components = _pose_losses(
            policy_pose_delta,
            state[:, 0].detach(),
            batch["reference_state"],
            state_mean,
            state_std,
            cfg,
        )
        residual_l2 = residual.square().mean()
        residual_smooth = (residual[:, 1:] - residual[:, :-1]).square().mean()

        weighted_tracking = cfg.tracking_weight * tracking_loss
        weighted_residual_l2 = cfg.residual_l2_weight * residual_l2
        weighted_residual_smooth = cfg.residual_smooth_weight * residual_smooth
        policy_loss = weighted_tracking + weighted_residual_l2 + weighted_residual_smooth

        with torch.no_grad():
            clipped = (candidate_unclipped - candidate).abs() > 1e-7
            recompute_error = candidate_normalized - batch["action"]
            residual_detached = residual.detach()
            saturation_threshold = 0.95 * model.config.residual_scale
            residual_slot_rms = residual_detached.square().mean(dim=(0, 2)).sqrt()

        output = {
            "loss": policy_loss,
            "policy_loss": policy_loss.detach(),
            "tracking_loss": tracking_loss,
            "residual_l2": residual_l2,
            "residual_smooth": residual_smooth,
            "weighted_tracking_loss": weighted_tracking.detach(),
            "weighted_residual_l2": weighted_residual_l2.detach(),
            "weighted_residual_smooth": weighted_residual_smooth.detach(),
            "residual_abs_mean": residual_detached.abs().mean(),
            "residual_rms": residual_detached.square().mean().sqrt(),
            "residual_abs_max": residual_detached.abs().max(),
            "residual_saturation_fraction": (residual_detached.abs() >= saturation_threshold)
            .float()
            .mean(),
            "candidate_action_clipped_fraction": clipped.float().mean(),
            "candidate_action_change_abs_mean": (candidate.detach() - batch["tracker_action"])
            .abs()
            .mean(),
            "candidate_action_recompute_abs_mean": recompute_error.abs().mean(),
            "candidate_action_recompute_abs_max": recompute_error.abs().max(),
        }
        for index, value in enumerate(residual_slot_rms):
            output[f"residual_slot_{index}_rms"] = value
        for name, value in tracking_components.items():
            output[f"tracking_{name}_loss"] = value.detach()
        return output

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        phase: str = "joint",
    ) -> dict[str, torch.Tensor]:
        self._validate_batch(batch)
        if phase == "model":
            return self._model_forward(batch)
        if phase == "policy":
            return self._policy_forward(batch)
        if phase != "joint":
            raise ValueError(f"Unknown residual objective phase: {phase!r}")
        model_output = self._model_forward(batch)
        policy_output = self._policy_forward(batch)
        total = model_output["loss"] + policy_output["loss"]
        return {
            **{name: value for name, value in model_output.items() if name != "loss"},
            **{name: value for name, value in policy_output.items() if name != "loss"},
            "loss": total,
        }
