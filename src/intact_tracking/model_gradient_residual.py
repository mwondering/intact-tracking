"""Frozen-model gradients for a dynamics-latent residual tracking policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP

from .forward_predictor import (
    ForwardDynamicsTransformer,
    ForwardPredictorConfig,
    physical_state_delta,
)
from .forward_predictor_inputs import (
    ACTION_DIM,
    CONTACT_BINARY_DIM,
    CONTACT_FORCE_DIM,
    FOOT_FEATURE_DIM,
    ROBOT_STATE_DIM,
    JointPositionTargetTransform,
)
from .residual_policy import _last_linear
from .rollout.mjlab_adapter import _sha256


@dataclass(frozen=True)
class FrozenForwardPredictorCheckpoint:
    """Inference-only Forward Predictor and all tensors in its input contract."""

    model: ForwardDynamicsTransformer
    config: ForwardPredictorConfig
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    foot_mean: torch.Tensor
    foot_std: torch.Tensor
    contact_force_mean: torch.Tensor
    contact_force_std: torch.Tensor
    delta_mean: torch.Tensor
    delta_std: torch.Tensor
    path: str
    sha256: str
    tracker_sha256: str | None


def _normalization_vector(
    normalization: Mapping[str, Any],
    name: str,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if name not in normalization:
        raise KeyError(f"Forward Predictor checkpoint normalization has no {name!r}")
    value = torch.as_tensor(normalization[name], dtype=torch.float32, device=device)
    if value.shape != (width,):
        raise ValueError(
            f"Normalization {name} must have shape [{width}], got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Normalization {name} contains NaN or Inf")
    if name.endswith("_std") and not bool((value > 0.0).all()):
        raise ValueError(f"Normalization {name} must be strictly positive")
    return value


def load_frozen_forward_predictor_checkpoint(
    checkpoint_file: str | Path,
    *,
    device: torch.device | str,
    expected_tracker_sha256: str | None = None,
) -> FrozenForwardPredictorCheckpoint:
    """Strictly load the complete v12 predictor for differentiation w.r.t. actions."""

    path = Path(checkpoint_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Forward Predictor checkpoint must be mapping-valued")
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise KeyError("Forward Predictor checkpoint has no mapping-valued model_config")
    config = ForwardPredictorConfig(**dict(raw_config))
    architecture = str(checkpoint.get("architecture_version", config.architecture_version))
    if architecture != config.architecture_version:
        raise ValueError(
            "Forward Predictor architecture metadata disagrees with model_config: "
            f"{architecture!r} != {config.architecture_version!r}"
        )
    expected_architecture = ForwardPredictorConfig().architecture_version
    if architecture != expected_architecture:
        raise ValueError(
            "Model-gradient training requires the complete v12 Forward Predictor, got "
            f"{architecture!r}"
        )
    expected_contract = {
        "state_dim": ROBOT_STATE_DIM,
        "action_dim": ACTION_DIM,
        "delta_dim": 70,
        "foot_feature_dim": FOOT_FEATURE_DIM,
        "contact_force_dim": CONTACT_FORCE_DIM,
        "contact_binary_dim": CONTACT_BINARY_DIM,
        "horizon": 5,
    }
    invalid_contract = {
        name: (getattr(config, name), expected)
        for name, expected in expected_contract.items()
        if getattr(config, name) != expected
    }
    if invalid_contract:
        raise ValueError(f"Incompatible Forward Predictor shape contract: {invalid_contract}")
    raw_model = checkpoint.get("model")
    if not isinstance(raw_model, Mapping):
        raise KeyError("Forward Predictor checkpoint has no mapping-valued model state")
    model = ForwardDynamicsTransformer(config)
    model.load_state_dict(raw_model, strict=True)
    model.to(target_device)
    model.requires_grad_(False)
    model.eval()

    tracker = checkpoint.get("tracker")
    tracker_sha256 = (
        str(tracker.get("checkpoint_sha256"))
        if isinstance(tracker, Mapping) and tracker.get("checkpoint_sha256")
        else None
    )
    if (
        expected_tracker_sha256 is not None
        and tracker_sha256 is not None
        and tracker_sha256 != expected_tracker_sha256
    ):
        raise ValueError(
            "Forward Predictor and frozen tracker checkpoints do not match: "
            f"predictor={tracker_sha256}, tracker={expected_tracker_sha256}"
        )
    raw_normalization = checkpoint.get("normalization")
    if not isinstance(raw_normalization, Mapping):
        raise KeyError("Forward Predictor checkpoint has no mapping-valued normalization")
    result = FrozenForwardPredictorCheckpoint(
        model=model,
        config=config,
        state_mean=_normalization_vector(
            raw_normalization, "state_mean", ROBOT_STATE_DIM, target_device
        ),
        state_std=_normalization_vector(
            raw_normalization, "state_std", ROBOT_STATE_DIM, target_device
        ),
        action_mean=_normalization_vector(
            raw_normalization, "action_mean", ACTION_DIM, target_device
        ),
        action_std=_normalization_vector(
            raw_normalization, "action_std", ACTION_DIM, target_device
        ),
        foot_mean=_normalization_vector(
            raw_normalization, "foot_mean", FOOT_FEATURE_DIM, target_device
        ),
        foot_std=_normalization_vector(
            raw_normalization, "foot_std", FOOT_FEATURE_DIM, target_device
        ),
        contact_force_mean=_normalization_vector(
            raw_normalization, "contact_force_mean", CONTACT_FORCE_DIM, target_device
        ),
        contact_force_std=_normalization_vector(
            raw_normalization, "contact_force_std", CONTACT_FORCE_DIM, target_device
        ),
        delta_mean=_normalization_vector(
            raw_normalization, "delta_mean", config.delta_dim, target_device
        ),
        delta_std=_normalization_vector(
            raw_normalization, "delta_std", config.delta_dim, target_device
        ),
        path=str(path),
        sha256=_sha256(path),
        tracker_sha256=tracker_sha256,
    )
    del checkpoint
    return result


class ModelGradientResidualPolicy(nn.Module):
    """A bounded residual head updated only through a frozen dynamics model."""

    def __init__(
        self,
        tracker_feature_dim: int,
        dynamics_latent_dim: int,
        *,
        action_dim: int = ACTION_DIM,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if tracker_feature_dim < 1 or dynamics_latent_dim < 1 or action_dim < 1:
            raise ValueError("Policy input/output dimensions must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        widths = tuple(int(width) for width in hidden_dims)
        if not widths or any(width < 1 for width in widths):
            raise ValueError("hidden_dims must contain positive widths")
        self.tracker_feature_dim = int(tracker_feature_dim)
        self.dynamics_latent_dim = int(dynamics_latent_dim)
        self.action_dim = int(action_dim)
        self.residual_scale = float(residual_scale)
        self.residual_mlp = MLP(
            self.tracker_feature_dim + self.dynamics_latent_dim,
            self.action_dim,
            list(widths),
            activation,
        )
        output = _last_linear(self.residual_mlp)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, tracker_features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if tracker_features.shape[:-1] != latent.shape[:-1]:
            raise ValueError("Tracker features and latent must have equal batch prefixes")
        if tracker_features.size(-1) != self.tracker_feature_dim:
            raise ValueError(
                f"Expected {self.tracker_feature_dim} tracker features, got "
                f"{tracker_features.size(-1)}"
            )
        if latent.size(-1) != self.dynamics_latent_dim:
            raise ValueError(
                f"Expected {self.dynamics_latent_dim} latent values, got {latent.size(-1)}"
            )
        value = torch.cat((tracker_features, latent.to(tracker_features.dtype)), dim=-1)
        return self.residual_scale * torch.tanh(self.residual_mlp(value))


class PredictorCausalHistory:
    """Causal simulator history shared by Context Encoder and Forward Predictor."""

    def __init__(
        self,
        checkpoint: FrozenForwardPredictorCheckpoint,
        *,
        num_envs: int,
        device: torch.device | str,
        use_bfloat16: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.context_steps = checkpoint.config.context_history_steps
        self.predictor_steps = checkpoint.config.history_steps
        self.use_bfloat16 = bool(
            use_bfloat16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.state = torch.zeros(
            (self.context_steps, num_envs, ROBOT_STATE_DIM), device=self.device
        )
        self.action = torch.zeros((self.context_steps, num_envs, ACTION_DIM), device=self.device)
        self.foot = torch.zeros(
            (self.context_steps, num_envs, FOOT_FEATURE_DIM), device=self.device
        )
        self.contact_force = torch.zeros(
            (self.context_steps, num_envs, CONTACT_FORCE_DIM), device=self.device
        )
        self.contact_binary = torch.zeros(
            (self.context_steps, num_envs, CONTACT_BINARY_DIM), device=self.device
        )
        self.valid = torch.zeros(
            (self.context_steps, num_envs), dtype=torch.bool, device=self.device
        )
        self.pointer = 0

    @torch.no_grad()
    def clear(self, boundary: torch.Tensor | None = None) -> None:
        if boundary is None:
            self.valid.zero_()
            self.pointer = 0
            return
        boundary = boundary.to(device=self.device, dtype=torch.bool)
        if boundary.shape != (self.num_envs,):
            raise ValueError(f"Boundary must have shape [{self.num_envs}]")
        self.valid[:, boundary] = False

    @torch.no_grad()
    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        foot: torch.Tensor,
        contact_force: torch.Tensor,
        contact_binary: torch.Tensor,
        boundary: torch.Tensor,
    ) -> None:
        expected = {
            "state": (self.num_envs, ROBOT_STATE_DIM),
            "action": (self.num_envs, ACTION_DIM),
            "foot": (self.num_envs, FOOT_FEATURE_DIM),
            "contact_force": (self.num_envs, CONTACT_FORCE_DIM),
            "contact_binary": (self.num_envs, CONTACT_BINARY_DIM),
        }
        values = {
            "state": state,
            "action": action,
            "foot": foot,
            "contact_force": contact_force,
            "contact_binary": contact_binary,
        }
        invalid = {
            name: (tuple(value.shape), expected[name])
            for name, value in values.items()
            if tuple(value.shape) != expected[name]
        }
        if invalid:
            raise ValueError(f"Invalid causal-history shapes: {invalid}")
        boundary = boundary.to(device=self.device, dtype=torch.bool)
        self.clear(boundary)
        for name, value in values.items():
            getattr(self, name)[self.pointer].copy_(value.detach())
        self.valid[self.pointer].copy_(~boundary)
        self.pointer = (self.pointer + 1) % self.context_steps

    def _ordered(self, value: torch.Tensor) -> torch.Tensor:
        order = torch.arange(self.context_steps, device=self.device)
        order = (order + self.pointer).remainder(self.context_steps)
        return value.index_select(0, order).transpose(0, 1)

    @staticmethod
    def _normalize_masked(
        value: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (value - mean) / std
        return torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))

    @torch.no_grad()
    def snapshot(
        self,
        current_state: torch.Tensor,
        current_foot: torch.Tensor,
        current_contact_force: torch.Tensor,
        current_contact_binary: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        checkpoint = self.checkpoint
        ordered_valid = self._ordered(self.valid)
        history_state = self._normalize_masked(
            self._ordered(self.state),
            checkpoint.state_mean,
            checkpoint.state_std,
            ordered_valid,
        )
        history_action = self._normalize_masked(
            self._ordered(self.action),
            checkpoint.action_mean,
            checkpoint.action_std,
            ordered_valid,
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bfloat16
            else nullcontext()
        )
        normalized_state = (current_state - checkpoint.state_mean) / checkpoint.state_std
        with context:
            latent = checkpoint.model.encode_context(
                history_state,
                history_action,
                normalized_state,
                ordered_valid,
            )
        suffix = slice(-self.predictor_steps, None)
        suffix_valid = ordered_valid[:, suffix]
        ordered_contact_binary = self._ordered(self.contact_binary)[:, suffix]
        return {
            "state": normalized_state,
            "foot": (current_foot - checkpoint.foot_mean) / checkpoint.foot_std,
            "contact_force": (current_contact_force - checkpoint.contact_force_mean)
            / checkpoint.contact_force_std,
            "contact_binary": current_contact_binary.float(),
            "history_state": history_state,
            "history_action": history_action,
            "history_foot": self._normalize_masked(
                self._ordered(self.foot)[:, suffix],
                checkpoint.foot_mean,
                checkpoint.foot_std,
                suffix_valid,
            ),
            "history_contact_force": self._normalize_masked(
                self._ordered(self.contact_force)[:, suffix],
                checkpoint.contact_force_mean,
                checkpoint.contact_force_std,
                suffix_valid,
            ),
            "history_contact_binary": torch.where(
                suffix_valid.unsqueeze(-1),
                ordered_contact_binary.float(),
                torch.zeros_like(ordered_contact_binary),
            ),
            "history_valid": ordered_valid,
            "latent": latent.float(),
        }

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "context_valid_fraction": float(self.valid.float().mean().item()),
            "context_full_fraction": float(self.valid.all(dim=0).float().mean().item()),
        }


@dataclass(frozen=True)
class ModelGradientLossConfig:
    """Trust-region tracking surrogate used to update the residual head."""

    horizon_discount: float = 0.9
    huber_delta: float = 1.0
    residual_weight: float = 0.01
    smoothness_weight: float = 0.01
    root_position_weight: float = 1.0
    root_orientation_weight: float = 1.0
    root_linear_velocity_weight: float = 0.25
    root_angular_velocity_weight: float = 0.25
    joint_position_weight: float = 1.0
    joint_velocity_weight: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.horizon_discount <= 1.0:
            raise ValueError("horizon_discount must be in (0,1]")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        names = (
            "residual_weight",
            "smoothness_weight",
            "root_position_weight",
            "root_orientation_weight",
            "root_linear_velocity_weight",
            "root_angular_velocity_weight",
            "joint_position_weight",
            "joint_velocity_weight",
        )
        invalid = {name: getattr(self, name) for name in names if getattr(self, name) < 0.0}
        if invalid:
            raise ValueError(f"Model-gradient loss weights must be non-negative: {invalid}")


def _component_huber(error: torch.Tensor, component: slice, delta: float) -> torch.Tensor:
    value = error[..., component]
    absolute = value.abs()
    elementwise = torch.where(
        absolute <= delta,
        0.5 * value.square(),
        delta * (absolute - 0.5 * delta),
    )
    return elementwise.mean(dim=-1)


def tracking_surrogate(
    normalized_prediction: torch.Tensor,
    physical_reference: torch.Tensor,
    valid: torch.Tensor,
    checkpoint: FrozenForwardPredictorCheckpoint,
    config: ModelGradientLossConfig,
) -> torch.Tensor:
    """Return a masked five-step cost in predictor-normalized physical error units."""

    if normalized_prediction.shape != physical_reference.shape:
        raise ValueError("Prediction and reference must have equal shapes")
    if normalized_prediction.ndim != 3 or normalized_prediction.shape[1:] != (5, 71):
        raise ValueError("Prediction and reference must be [batch,5,71]")
    if valid.shape != normalized_prediction.shape[:2]:
        raise ValueError("Tracking-valid mask must be [batch,5]")
    physical_prediction = normalized_prediction * checkpoint.state_std + checkpoint.state_mean
    error = physical_state_delta(physical_reference, physical_prediction) / checkpoint.delta_std
    components = (
        (slice(0, 3), config.root_position_weight),
        (slice(3, 6), config.root_orientation_weight),
        (slice(6, 9), config.root_linear_velocity_weight),
        (slice(9, 12), config.root_angular_velocity_weight),
        (slice(12, 41), config.joint_position_weight),
        (slice(41, 70), config.joint_velocity_weight),
    )
    per_step = error.new_zeros(error.shape[:2])
    for component, weight in components:
        per_step = per_step + weight * _component_huber(error, component, config.huber_delta)
    discount = config.horizon_discount ** torch.arange(
        per_step.size(1), device=per_step.device, dtype=per_step.dtype
    )
    weight = valid.to(per_step.dtype) * discount.unsqueeze(0)
    return (per_step * weight).sum() / weight.sum().clamp_min(1.0)


def policy_actions_to_normalized_targets(
    policy_action: torch.Tensor,
    env_ids: torch.Tensor,
    action_transform: JointPositionTargetTransform,
    checkpoint: FrozenForwardPredictorCheckpoint,
    *,
    action_clip: float | None,
) -> torch.Tensor:
    """Apply the live simulator action chain before predictor normalization."""

    if policy_action.ndim != 3 or policy_action.shape[1:] != (5, ACTION_DIM):
        raise ValueError("Policy action sequence must be [batch,5,29]")
    if env_ids.shape != (policy_action.size(0),):
        raise ValueError("env_ids must contain one live simulator slot per action sequence")
    action = policy_action
    if action_clip is not None:
        action = action.clamp(-float(action_clip), float(action_clip))
    flat_action = action.reshape(-1, ACTION_DIM)
    flat_env_ids = env_ids[:, None].expand(-1, 5).reshape(-1)
    target = action_transform(flat_action, env_ids=flat_env_ids).view_as(action)
    return (target - checkpoint.action_mean) / checkpoint.action_std


def model_gradient_loss(
    *,
    policy: ModelGradientResidualPolicy,
    predictor_inputs: Mapping[str, torch.Tensor],
    tracker_features: torch.Tensor,
    latent_sequence: torch.Tensor,
    tracker_actions: torch.Tensor,
    reference_states: torch.Tensor,
    valid: torch.Tensor,
    env_ids: torch.Tensor,
    action_transform: JointPositionTargetTransform,
    checkpoint: FrozenForwardPredictorCheckpoint,
    loss_config: ModelGradientLossConfig,
    action_clip: float | None,
) -> dict[str, torch.Tensor]:
    """Differentiate a recursive five-step tracking surrogate into the residual policy."""

    expected_prefix = (tracker_features.size(0), 5)
    if tracker_features.shape[:2] != expected_prefix:
        raise ValueError("Tracker features must be [batch,5,feature_dim]")
    if latent_sequence.shape[:2] != expected_prefix:
        raise ValueError("Latent sequence must be [batch,5,latent_dim]")
    if tracker_actions.shape != (*expected_prefix, ACTION_DIM):
        raise ValueError("Tracker actions must be [batch,5,29]")
    residual = policy(tracker_features, latent_sequence)
    policy_action = tracker_actions + residual
    normalized_action = policy_actions_to_normalized_targets(
        policy_action,
        env_ids,
        action_transform,
        checkpoint,
        action_clip=action_clip,
    )
    prediction = checkpoint.model.rollout(
        predictor_inputs["state"],
        normalized_action,
        predictor_inputs["foot"],
        predictor_inputs["contact_force"],
        predictor_inputs["contact_binary"],
        predictor_inputs["history_state"],
        predictor_inputs["history_action"],
        predictor_inputs["history_foot"],
        predictor_inputs["history_contact_force"],
        predictor_inputs["history_contact_binary"],
        predictor_inputs["history_valid"],
        checkpoint.state_mean,
        checkpoint.state_std,
        checkpoint.delta_mean,
        checkpoint.delta_std,
        dynamics_latent=latent_sequence[:, 0],
    )[0]
    tracking = tracking_surrogate(
        prediction,
        reference_states,
        valid,
        checkpoint,
        loss_config,
    )
    residual_penalty = residual.square().mean()
    smoothness = (residual[:, 1:] - residual[:, :-1]).square().mean()
    total = (
        tracking
        + loss_config.residual_weight * residual_penalty
        + loss_config.smoothness_weight * smoothness
    )
    return {
        "loss": total,
        "tracking_loss": tracking.detach(),
        "residual_penalty": residual_penalty.detach(),
        "smoothness_loss": smoothness.detach(),
        "residual_rms": residual.detach().square().mean().sqrt(),
        "prediction": prediction.detach(),
    }
