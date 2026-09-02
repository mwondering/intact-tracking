"""Losses and diagnostics for the context-conditioned Forward model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_model import ResidualTrackingModel


@dataclass(frozen=True)
class ResidualLossConfig:
    forward_weight: float = 2.0
    nominal_pair_weight: float = 1.0
    nominal_effect_weight: float = 1.0
    nominal_consistency_weight: float = 1.0
    root_position_weight: float = 5.0
    root_orientation_weight: float = 2.0
    joint_position_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0:
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
    nominal_pose = _reconstruct_pose(nominal_pose_delta, current_state, state_mean, state_std)
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
    orientation_dot = (predicted_relative_quat * target_relative_quat).sum(dim=-1).clamp(-1.0, 1.0)
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


def _pose_consistency_losses(
    first_pose_delta: torch.Tensor,
    second_pose_delta: torch.Tensor,
    current_state: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    config: ResidualLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require two contexts from the same nominal dynamics to predict equal poses."""
    first = _reconstruct_pose(first_pose_delta, current_state, state_mean, state_std)
    second = _reconstruct_pose(second_pose_delta, current_state, state_mean, state_std)
    orientation_dot = (
        (first["root_orientation"] * second["root_orientation"]).sum(dim=-1).clamp(-1.0, 1.0)
    )
    component = {
        "root_position": (first["root_position"] - second["root_position"]).square().mean(),
        "root_orientation": (1.0 - orientation_dot.square()).mean(),
        "joint_position": (first["joint_position"] - second["joint_position"]).square().mean(),
    }
    total = (
        config.root_position_weight * component["root_position"]
        + config.root_orientation_weight * component["root_orientation"]
        + config.joint_position_weight * component["joint_position"]
    )
    return total, component


class ResidualTrainingObjective(nn.Module):
    """Pure Forward objective; no Backward or policy branch is constructed."""

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
            "context_state",
            "context_action",
            "context_state_mask",
            "context_action_mask",
            "context_boundary",
            "state",
            "action",
            "previous_action",
            "is_nominal",
            "state_mean",
            "state_std",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Forward training batch is missing fields: {missing}")
        if any(not torch.isfinite(batch[name]).all() for name in required):
            raise ValueError("Forward training batch contains non-finite values")

    def _encode_context(
        self,
        batch: dict[str, torch.Tensor],
        *,
        prefix: str = "",
        count: int | None = None,
        zero_values: bool = False,
    ) -> torch.Tensor:
        names = {
            name: f"{prefix}context_{name}"
            for name in ("state", "action", "state_mask", "action_mask", "boundary")
        }
        values = {name: batch[key] for name, key in names.items()}
        if count is not None:
            values = {name: value[:count] for name, value in values.items()}
        if zero_values:
            values["state"] = torch.zeros_like(values["state"])
            values["action"] = torch.zeros_like(values["action"])
        return self.model.encode_context(
            values["state"],
            values["action"],
            values["state_mask"],
            values["action_mask"],
            values["boundary"],
        )

    def _model_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.loss_config
        model = self.model
        world = self._encode_context(batch)
        state = batch["state"]
        is_nominal = batch["is_nominal"]
        if is_nominal.dtype != torch.bool or is_nominal.shape != (state.size(0),):
            raise ValueError("is_nominal must be a [batch] boolean tensor")
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

        weighted_forward = cfg.forward_weight * forward_loss
        model_loss = weighted_forward

        nominal_pair_loss = forward_loss.new_zeros(())
        nominal_forward_loss = forward_loss.new_zeros(())
        nominal_effect_loss = forward_loss.new_zeros(())
        nominal_effect_zero_loss = forward_loss.new_zeros(())
        nominal_consistency_loss = forward_loss.new_zeros(())
        nominal_context_swap_ratio = forward_loss.new_ones(())
        nominal_pair_count = 0
        nominal_source_pair_count = 0
        dr_source_pair_count = 0
        nominal_context_latent_rms = forward_loss.new_zeros(())
        dr_to_nominal_context_latent_rms = forward_loss.new_zeros(())
        nominal_to_nominal_context_latent_rms = forward_loss.new_zeros(())
        nominal_components: dict[str, torch.Tensor] = {}
        effect_components: dict[str, torch.Tensor] = {}
        consistency_components: dict[str, torch.Tensor] = {}
        if "nominal_state" in batch:
            required_nominal = {
                "nominal_context_state",
                "nominal_context_action",
                "nominal_context_state_mask",
                "nominal_context_action_mask",
                "nominal_context_boundary",
            }
            missing_nominal = sorted(required_nominal.difference(batch))
            if missing_nominal:
                raise KeyError(
                    f"Nominal pair training requires real nominal histories: {missing_nominal}"
                )
            nominal_target = batch["nominal_state"]
            if not torch.isfinite(nominal_target).all():
                raise ValueError("nominal_state contains non-finite values")
            if nominal_target.ndim != 3 or nominal_target.shape[1:] != state[:, 1:].shape[1:]:
                raise ValueError(
                    f"nominal_state must be [pair_batch,5,71], got {tuple(nominal_target.shape)}"
                )
            nominal_pair_count = int(nominal_target.size(0))
            if nominal_pair_count < 1 or nominal_pair_count > state.size(0):
                raise ValueError("nominal_state pair batch must fit inside the model batch")
            pair_world = world[:nominal_pair_count]
            pair_state = state[:nominal_pair_count, 0]
            pair_previous_action = previous_action[:nominal_pair_count]
            pair_action = action[:nominal_pair_count]
            for suffix in ("state", "action", "state_mask", "action_mask", "boundary"):
                nominal_value = batch[f"nominal_context_{suffix}"][:nominal_pair_count]
                source_value = batch[f"context_{suffix}"][:nominal_pair_count]
                if nominal_value.shape != source_value.shape:
                    raise ValueError(
                        f"nominal_context_{suffix} must match source shape, got "
                        f"{tuple(nominal_value.shape)} vs {tuple(source_value.shape)}"
                    )
                if not torch.isfinite(nominal_value).all():
                    raise ValueError(f"nominal_context_{suffix} contains non-finite values")
            nominal_world = self._encode_context(batch, prefix="nominal_", count=nominal_pair_count)
            nominal_pose_delta = model.predict_future(
                nominal_world,
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
            source_is_nominal = batch["is_nominal"][:nominal_pair_count].bool()
            source_is_dr = ~source_is_nominal
            nominal_source_pair_count = int(source_is_nominal.sum().item())
            dr_source_pair_count = int(source_is_dr.sum().item())
            if dr_source_pair_count:
                nominal_effect_loss, nominal_effect_zero_loss, effect_components = (
                    _pose_effect_losses(
                        forward_pose_delta[:nominal_pair_count][source_is_dr],
                        nominal_pose_delta[source_is_dr],
                        pair_state[source_is_dr],
                        state[:nominal_pair_count, 1:][source_is_dr],
                        nominal_target[source_is_dr],
                        state_mean,
                        state_std,
                        cfg,
                    )
                )
            if nominal_source_pair_count:
                nominal_consistency_loss, consistency_components = _pose_consistency_losses(
                    forward_pose_delta[:nominal_pair_count][source_is_nominal],
                    nominal_pose_delta[source_is_nominal],
                    pair_state[source_is_nominal],
                    state_mean,
                    state_std,
                    cfg,
                )
            nominal_pair_loss = cfg.nominal_pair_weight * (
                nominal_forward_loss
                + cfg.nominal_effect_weight * nominal_effect_loss
                + cfg.nominal_consistency_weight * nominal_consistency_loss
            )
            model_loss = model_loss + nominal_pair_loss
            with torch.no_grad():
                context_delta = pair_world - nominal_world
                nominal_context_latent_rms = context_delta.square().mean().sqrt()
                if dr_source_pair_count:
                    dr_context_delta = context_delta[source_is_dr]
                    dr_to_nominal_context_latent_rms = dr_context_delta.square().mean().sqrt()
                    dr_pair_forward_loss, _ = _pose_losses(
                        forward_pose_delta[:nominal_pair_count][source_is_dr],
                        pair_state[source_is_dr],
                        state[:nominal_pair_count, 1:][source_is_dr],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    nominal_world_on_dr_loss, _ = _pose_losses(
                        nominal_pose_delta[source_is_dr],
                        pair_state[source_is_dr],
                        state[:nominal_pair_count, 1:][source_is_dr],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    dr_world_on_nominal_loss, _ = _pose_losses(
                        forward_pose_delta[:nominal_pair_count][source_is_dr],
                        pair_state[source_is_dr],
                        nominal_target[source_is_dr],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    nominal_forward_dr_loss, _ = _pose_losses(
                        nominal_pose_delta[source_is_dr],
                        pair_state[source_is_dr],
                        nominal_target[source_is_dr],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    correct_assignment_loss = dr_pair_forward_loss + nominal_forward_dr_loss
                    swapped_assignment_loss = nominal_world_on_dr_loss + dr_world_on_nominal_loss
                    nominal_context_swap_ratio = (
                        swapped_assignment_loss / correct_assignment_loss.clamp_min(1e-8)
                    )
                if nominal_source_pair_count:
                    nominal_context_delta = context_delta[source_is_nominal]
                    nominal_to_nominal_context_latent_rms = (
                        nominal_context_delta.square().mean().sqrt()
                    )

        with torch.no_grad():
            forward_same_domain_context_shuffle_ratio = forward_loss.new_ones(())
            forward_dr_context_shuffle_ratio = forward_loss.new_ones(())
            forward_nominal_context_shuffle_ratio = forward_loss.new_ones(())
            zero_context_world = self._encode_context(batch, zero_values=True)
            zero_context_delta = model.predict_future(
                zero_context_world,
                state[:, 0],
                previous_action,
                action,
            )
            zero_context_loss, _ = _pose_losses(
                zero_context_delta,
                state[:, 0],
                state[:, 1:],
                state_mean,
                state_std,
                cfg,
            )
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
                same_domain_world = world.clone()
                for group_mask in (is_nominal, ~is_nominal):
                    group_ids = group_mask.nonzero(as_tuple=False).flatten()
                    if group_ids.numel() > 1:
                        same_domain_world[group_ids] = world[group_ids.roll(1)]
                same_domain_delta = model.predict_future(
                    same_domain_world,
                    state[:, 0],
                    previous_action,
                    action,
                )
                same_domain_loss, _ = _pose_losses(
                    same_domain_delta,
                    state[:, 0],
                    state[:, 1:],
                    state_mean,
                    state_std,
                    cfg,
                )
                forward_same_domain_context_shuffle_ratio = (
                    same_domain_loss / forward_loss.detach().clamp_min(1e-8)
                )
                for group_mask, name in (
                    (~is_nominal, "dr"),
                    (is_nominal, "nominal"),
                ):
                    if bool(group_mask.any()):
                        group_shuffled_loss, _ = _pose_losses(
                            same_domain_delta[group_mask],
                            state[group_mask, 0],
                            state[group_mask, 1:],
                            state_mean,
                            state_std,
                            cfg,
                        )
                        group_correct_loss, _ = _pose_losses(
                            forward_pose_delta[group_mask],
                            state[group_mask, 0],
                            state[group_mask, 1:],
                            state_mean,
                            state_std,
                            cfg,
                        )
                        ratio = group_shuffled_loss / group_correct_loss.clamp_min(1e-8)
                        if name == "dr":
                            forward_dr_context_shuffle_ratio = ratio
                        else:
                            forward_nominal_context_shuffle_ratio = ratio
            else:
                shuffled_forward_loss = forward_loss.detach()
            no_change_loss, _ = _pose_losses(
                torch.zeros_like(forward_pose_delta),
                state[:, 0],
                state[:, 1:],
                state_mean,
                state_std,
                cfg,
            )

            source_group_metrics: dict[str, torch.Tensor] = {}
            for group_mask, group_name in (
                (is_nominal, "nominal"),
                (~is_nominal, "dr"),
            ):
                if bool(group_mask.any()):
                    group_loss, _ = _pose_losses(
                        forward_pose_delta[group_mask],
                        state[group_mask, 0],
                        state[group_mask, 1:],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    group_no_change, _ = _pose_losses(
                        torch.zeros_like(forward_pose_delta[group_mask]),
                        state[group_mask, 0],
                        state[group_mask, 1:],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    source_group_metrics[f"forward_source_{group_name}_loss"] = group_loss
                    source_group_metrics[f"forward_source_{group_name}_nmse"] = (
                        group_loss / group_no_change.clamp_min(1e-8)
                    )
                    group_zero_context_loss, _ = _pose_losses(
                        zero_context_delta[group_mask],
                        state[group_mask, 0],
                        state[group_mask, 1:],
                        state_mean,
                        state_std,
                        cfg,
                    )
                    source_group_metrics[f"forward_{group_name}_zero_context_ratio"] = (
                        group_zero_context_loss / group_loss.clamp_min(1e-8)
                    )

            horizon_metrics: dict[str, torch.Tensor] = {}
            for index in range(forward_pose_delta.size(1)):
                horizon_loss, _ = _pose_losses(
                    forward_pose_delta[:, index : index + 1],
                    state[:, 0],
                    state[:, index + 1 : index + 2],
                    state_mean,
                    state_std,
                    cfg,
                )
                horizon_no_change, _ = _pose_losses(
                    torch.zeros_like(forward_pose_delta[:, index : index + 1]),
                    state[:, 0],
                    state[:, index + 1 : index + 2],
                    state_mean,
                    state_std,
                    cfg,
                )
                step = index + 1
                horizon_metrics[f"forward_horizon_{step}_loss"] = horizon_loss
                horizon_metrics[f"forward_horizon_{step}_nmse"] = (
                    horizon_loss / horizon_no_change.clamp_min(1e-8)
                )

        output = {
            "loss": model_loss,
            "model_loss": model_loss.detach(),
            "forward_loss": forward_loss,
            "weighted_forward_loss": weighted_forward.detach(),
            "nominal_pair_loss": nominal_pair_loss.detach(),
            "nominal_forward_loss": nominal_forward_loss.detach(),
            "nominal_effect_loss": nominal_effect_loss.detach(),
            "nominal_consistency_loss": nominal_consistency_loss.detach(),
            "nominal_effect_nmse": (
                nominal_effect_loss.detach() / nominal_effect_zero_loss.clamp_min(1e-8)
            ),
            "nominal_true_effect_loss": nominal_effect_zero_loss.detach(),
            "nominal_context_swap_ratio": nominal_context_swap_ratio.detach(),
            "nominal_context_latent_rms": nominal_context_latent_rms.detach(),
            "dr_to_nominal_context_latent_rms": dr_to_nominal_context_latent_rms.detach(),
            "nominal_to_nominal_context_latent_rms": (
                nominal_to_nominal_context_latent_rms.detach()
            ),
            "nominal_pair_count": forward_loss.new_tensor(float(nominal_pair_count)),
            "nominal_source_pair_count": forward_loss.new_tensor(float(nominal_source_pair_count)),
            "dr_source_pair_count": forward_loss.new_tensor(float(dr_source_pair_count)),
            "forward_no_change_loss": no_change_loss,
            "forward_nmse": forward_loss.detach() / no_change_loss.clamp_min(1e-8),
            "forward_context_shuffle_ratio": (
                shuffled_forward_loss / forward_loss.detach().clamp_min(1e-8)
            ),
            "forward_zero_context_ratio": (
                zero_context_loss / forward_loss.detach().clamp_min(1e-8)
            ),
            "forward_same_domain_context_shuffle_ratio": (
                forward_same_domain_context_shuffle_ratio
            ),
            "forward_dr_context_shuffle_ratio": forward_dr_context_shuffle_ratio,
            "forward_nominal_context_shuffle_ratio": (forward_nominal_context_shuffle_ratio),
            **source_group_metrics,
            **horizon_metrics,
        }
        for name, value in forward_components.items():
            output[f"forward_{name}_loss"] = value.detach()
        for name, value in nominal_components.items():
            output[f"nominal_forward_{name}_loss"] = value.detach()
        for name, value in effect_components.items():
            output[f"nominal_effect_{name}_loss"] = value.detach()
        for name, value in consistency_components.items():
            output[f"nominal_consistency_{name}_loss"] = value.detach()
        return output

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        phase: str = "model",
    ) -> dict[str, torch.Tensor]:
        self._validate_batch(batch)
        if phase != "model":
            raise ValueError(f"Forward-only objective does not support phase {phase!r}")
        return self._model_forward(batch)
