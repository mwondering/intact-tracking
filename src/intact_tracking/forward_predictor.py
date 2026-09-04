"""Recursive context-conditioned dynamics predictor with explicit state transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .forward_predictor_inputs import (
    ACTION_DIM,
    CONTACT_BINARY_DIM,
    CONTACT_FORCE_DIM,
    FOOT_FEATURE_DIM,
    ROBOT_STATE_DIM,
)

ROOT_POSITION = slice(0, 3)
ROOT_ORIENTATION = slice(3, 7)
ROOT_LINEAR_VELOCITY = slice(7, 10)
ROOT_ANGULAR_VELOCITY = slice(10, 13)
JOINT_POSITION = slice(13, 42)
JOINT_VELOCITY = slice(42, 71)

DELTA_ROOT_POSITION = slice(0, 3)
DELTA_ROOT_ROTATION_VECTOR = slice(3, 6)
DELTA_ROOT_LINEAR_VELOCITY = slice(6, 9)
DELTA_ROOT_ANGULAR_VELOCITY = slice(9, 12)
DELTA_JOINT_POSITION = slice(12, 41)
DELTA_JOINT_VELOCITY = slice(41, 70)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for scalar-first quaternions."""

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


def quaternion_conjugate(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def rotation_vector_to_quaternion(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = rotation_vector.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    regular_scale = torch.sin(half_angle) / angle.clamp_min(1.0e-8)
    small_scale = 0.5 - angle.square() / 48.0
    vector_scale = torch.where(angle > 1.0e-4, regular_scale, small_scale)
    quaternion = torch.cat((torch.cos(half_angle), rotation_vector * vector_scale), dim=-1)
    return torch.nn.functional.normalize(quaternion, dim=-1, eps=1.0e-8)


def quaternion_to_rotation_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the shortest axis-angle rotation vector for a scalar-first quaternion."""

    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=1.0e-8)
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[..., :1].clamp_min(1.0e-8))
    regular_scale = angle / vector_norm.clamp_min(1.0e-8)
    scale = torch.where(vector_norm > 1.0e-6, regular_scale, 2.0 * torch.ones_like(angle))
    return vector * scale


def physical_state_delta(current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    """Encode one physical 71-D state transition as a 70-D additive/rotational delta."""

    if current.shape != following.shape or current.size(-1) != 71:
        raise ValueError(
            "Current/following states must have equal [...,71] shapes, got "
            f"{tuple(current.shape)} and {tuple(following.shape)}"
        )
    current_quaternion = torch.nn.functional.normalize(
        current[..., ROOT_ORIENTATION], dim=-1, eps=1.0e-8
    )
    following_quaternion = torch.nn.functional.normalize(
        following[..., ROOT_ORIENTATION], dim=-1, eps=1.0e-8
    )
    relative_quaternion = quaternion_multiply(
        following_quaternion,
        quaternion_conjugate(current_quaternion),
    )
    return torch.cat(
        (
            following[..., ROOT_POSITION] - current[..., ROOT_POSITION],
            quaternion_to_rotation_vector(relative_quaternion),
            following[..., ROOT_LINEAR_VELOCITY] - current[..., ROOT_LINEAR_VELOCITY],
            following[..., ROOT_ANGULAR_VELOCITY] - current[..., ROOT_ANGULAR_VELOCITY],
            following[..., JOINT_POSITION] - current[..., JOINT_POSITION],
            following[..., JOINT_VELOCITY] - current[..., JOINT_VELOCITY],
        ),
        dim=-1,
    )


def apply_physical_state_delta(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply a 70-D delta to a physical 71-D state without mutating either input."""

    if state.size(-1) != 71 or delta.shape != (*state.shape[:-1], 70):
        raise ValueError(
            "State/delta must end in 71/70 values with matching prefixes, got "
            f"{tuple(state.shape)} and {tuple(delta.shape)}"
        )
    current_quaternion = torch.nn.functional.normalize(
        state[..., ROOT_ORIENTATION], dim=-1, eps=1.0e-8
    )
    rotation_delta = rotation_vector_to_quaternion(delta[..., DELTA_ROOT_ROTATION_VECTOR])
    next_quaternion = torch.nn.functional.normalize(
        quaternion_multiply(rotation_delta, current_quaternion),
        dim=-1,
        eps=1.0e-8,
    )
    return torch.cat(
        (
            state[..., ROOT_POSITION] + delta[..., DELTA_ROOT_POSITION],
            next_quaternion,
            state[..., ROOT_LINEAR_VELOCITY] + delta[..., DELTA_ROOT_LINEAR_VELOCITY],
            state[..., ROOT_ANGULAR_VELOCITY] + delta[..., DELTA_ROOT_ANGULAR_VELOCITY],
            state[..., JOINT_POSITION] + delta[..., DELTA_JOINT_POSITION],
            state[..., JOINT_VELOCITY] + delta[..., DELTA_JOINT_VELOCITY],
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class ForwardPredictorConfig:
    """Capacity and shape contract for the causal one-step transition model."""

    architecture_version: str = "local_contrastive_grouped_dynamics_causal_transformer_v11"
    state_dim: int = ROBOT_STATE_DIM
    action_dim: int = ACTION_DIM
    delta_dim: int = 70
    foot_feature_dim: int = FOOT_FEATURE_DIM
    contact_force_dim: int = CONTACT_FORCE_DIM
    contact_binary_dim: int = CONTACT_BINARY_DIM
    horizon: int = 5
    history_steps: int = 10
    context_history_steps: int = 100
    transformer_dim: int = 512
    transformer_depth: int = 6
    transformer_heads: int = 8
    context_dim: int = 128
    context_depth: int = 2
    context_heads: int = 4
    dynamics_latent_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"Forward Predictor dimensions must be positive: {invalid}")
        fixed_dimensions = (
            self.state_dim == ROBOT_STATE_DIM
            and self.action_dim == ACTION_DIM
            and self.delta_dim == 70
            and self.foot_feature_dim == FOOT_FEATURE_DIM
            and self.contact_force_dim == CONTACT_FORCE_DIM
            and self.contact_binary_dim == CONTACT_BINARY_DIM
        )
        if not fixed_dimensions:
            raise ValueError(
                "Forward Predictor is fixed to 71-D robot state, 8-D simulator foot state, "
                "6-D contact force, 2-D contact state, 29-D applied target and 70-D robot delta"
            )
        if self.horizon != 5:
            raise ValueError("Forward Predictor rollout is fixed to five steps")
        if self.history_steps != 10:
            raise ValueError("Forward Predictor history is fixed to ten completed interactions")
        if self.context_history_steps < self.history_steps:
            raise ValueError(
                "context_history_steps must be at least the ten-frame predictor history"
            )
        if self.transformer_dim % self.transformer_heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if self.context_dim % self.context_heads:
            raise ValueError("context_dim must be divisible by context_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def sequence_length(self) -> int:
        """Ten historical state-action tokens followed by the current token."""

        return self.history_steps + 1

    @property
    def token_state_dim(self) -> int:
        return (
            self.state_dim
            + self.foot_feature_dim
            + self.contact_force_dim
            + self.contact_binary_dim
        )


class DynamicsContextEncoder(nn.Module):
    """Infer dynamics from completed proprioceptive state-action outcomes only."""

    def __init__(self, config: ForwardPredictorConfig) -> None:
        super().__init__()
        self.config = config
        width = config.context_dim
        interaction_dim = 2 * config.state_dim + config.action_dim
        self.interaction_projection = nn.Sequential(
            nn.Linear(interaction_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, config.context_history_steps + 1, width))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.context_heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.context_depth,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.dynamics_latent_dim),
            nn.LayerNorm(config.dynamics_latent_dim),
        )

    def forward(
        self,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        current_state: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history_state.size(0)
        next_state = torch.cat(
            (history_state[:, 1:], current_state[:, None]),
            dim=1,
        )
        valid = history_valid.to(device=history_state.device, dtype=torch.bool)
        valid_scale = valid.unsqueeze(-1).to(dtype=history_state.dtype)
        interactions = torch.cat(
            (history_state.float(), history_action.float(), next_state.float()),
            dim=-1,
        )
        history_tokens = self.interaction_projection(interactions * valid_scale) * valid_scale
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls, history_tokens), dim=1) + self.position
        padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=valid.device),
                ~valid,
            ),
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return self.output(encoded[:, 0])


class ForwardDynamicsTransformer(nn.Module):
    """Causal transition model conditioned only on a history-inferred latent."""

    def __init__(self, config: ForwardPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ForwardPredictorConfig()
        width = self.config.transformer_dim
        self.state_projection = nn.Linear(self.config.token_state_dim, width)
        self.action_projection = nn.Linear(self.config.action_dim, width)
        self.validity_embedding = nn.Embedding(2, width)
        self.position = nn.Parameter(torch.empty(1, self.config.sequence_length, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.config.transformer_heads,
            dim_feedforward=4 * width,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.transformer_depth,
            enable_nested_tensor=False,
        )
        self.context_encoder = DynamicsContextEncoder(self.config)
        self.context_condition = nn.Linear(self.config.dynamics_latent_dim, width)
        self.output_norm = nn.LayerNorm(width)
        self.delta_head = nn.Linear(width, self.config.delta_dim)
        self.foot_head = nn.Linear(width, self.config.foot_feature_dim)
        self.contact_force_head = nn.Linear(width, self.config.contact_force_dim)
        self.contact_binary_head = nn.Linear(width, self.config.contact_binary_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    self.config.sequence_length,
                    self.config.sequence_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

    def encode_context(
        self,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        normalized_state: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the long proprioceptive interaction context exactly once."""

        return self.context_encoder(
            history_state,
            history_action,
            normalized_state,
            history_valid,
        )

    def forward(
        self,
        normalized_state: torch.Tensor,
        normalized_action: torch.Tensor,
        normalized_foot: torch.Tensor,
        normalized_contact_force: torch.Tensor,
        contact_binary: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_foot: torch.Tensor,
        history_contact_force: torch.Tensor,
        history_contact_binary: torch.Tensor,
        history_valid: torch.Tensor,
        *,
        dynamics_latent: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if tuple(normalized_state.shape) != (normalized_state.size(0), self.config.state_dim):
            raise ValueError(
                f"State must be [batch,{self.config.state_dim}], got {tuple(normalized_state.shape)}"
            )
        batch_size = normalized_state.size(0)
        expected_action = (batch_size, self.config.action_dim)
        if tuple(normalized_action.shape) != expected_action:
            raise ValueError(
                f"Action must have shape {expected_action}, got {tuple(normalized_action.shape)}"
            )
        state_history_steps = history_state.size(1)
        allowed_state_history_steps = (
            {self.config.context_history_steps}
            if dynamics_latent is None
            else {self.config.history_steps, self.config.context_history_steps}
        )
        if state_history_steps not in allowed_state_history_steps:
            raise ValueError(
                "State/action history must contain the full context when encoding z and may "
                "contain only the predictor suffix when z is supplied; got "
                f"{state_history_steps} frames"
            )
        expected_history = {
            "state": (
                batch_size,
                state_history_steps,
                self.config.state_dim,
            ),
            "action": (
                batch_size,
                state_history_steps,
                self.config.action_dim,
            ),
            "foot": (batch_size, self.config.history_steps, self.config.foot_feature_dim),
            "contact_force": (
                batch_size,
                self.config.history_steps,
                self.config.contact_force_dim,
            ),
            "contact_binary": (
                batch_size,
                self.config.history_steps,
                self.config.contact_binary_dim,
            ),
            "valid": (batch_size, state_history_steps),
        }
        actual_history = {
            "state": tuple(history_state.shape),
            "action": tuple(history_action.shape),
            "foot": tuple(history_foot.shape),
            "contact_force": tuple(history_contact_force.shape),
            "contact_binary": tuple(history_contact_binary.shape),
            "valid": tuple(history_valid.shape),
        }
        invalid = {
            name: (actual_history[name], shape)
            for name, shape in expected_history.items()
            if actual_history[name] != shape
        }
        if invalid:
            raise ValueError(f"Invalid Forward Predictor history shapes: {invalid}")
        expected_current = {
            "foot": (batch_size, self.config.foot_feature_dim),
            "contact_force": (batch_size, self.config.contact_force_dim),
            "contact_binary": (batch_size, self.config.contact_binary_dim),
        }
        actual_current = {
            "foot": tuple(normalized_foot.shape),
            "contact_force": tuple(normalized_contact_force.shape),
            "contact_binary": tuple(contact_binary.shape),
        }
        invalid_current = {
            name: (actual_current[name], shape)
            for name, shape in expected_current.items()
            if actual_current[name] != shape
        }
        if invalid_current:
            raise ValueError(f"Invalid Forward Predictor current/stat shapes: {invalid_current}")

        if dynamics_latent is None:
            dynamics_latent = self.encode_context(
                history_state,
                history_action,
                normalized_state,
                history_valid,
            )
        elif tuple(dynamics_latent.shape) != (
            batch_size,
            self.config.dynamics_latent_dim,
        ):
            raise ValueError(
                "Dynamics latent must have shape "
                f"[{batch_size},{self.config.dynamics_latent_dim}], got "
                f"{tuple(dynamics_latent.shape)}"
            )
        predictor_history_state = history_state[:, -self.config.history_steps :]
        predictor_history_action = history_action[:, -self.config.history_steps :]
        predictor_history_valid = history_valid[:, -self.config.history_steps :]
        history_features = self._state_features(
            predictor_history_state,
            history_foot,
            history_contact_force,
            history_contact_binary,
        )
        current_features = self._state_features(
            normalized_state,
            normalized_foot,
            normalized_contact_force,
            contact_binary,
        )
        valid = predictor_history_valid.to(device=history_state.device, dtype=torch.bool)
        valid_scale = valid.unsqueeze(-1).to(dtype=history_features.dtype)
        history_tokens = self.state_projection(
            history_features * valid_scale
        ) + self.action_projection(predictor_history_action.float() * valid_scale)
        history_tokens = history_tokens * valid_scale + self.validity_embedding(valid.long())
        current_token = (
            self.state_projection(current_features)
            + self.action_projection(normalized_action.float())
            + self.validity_embedding.weight[1]
        ).unsqueeze(1)
        tokens = (
            torch.cat((history_tokens, current_token), dim=1)
            + self.position
            + self.context_condition(dynamics_latent).unsqueeze(1)
        )
        encoded = self.transformer(tokens, mask=self.causal_mask)
        output = self.output_norm(encoded[:, -1])
        transition = (
            self.delta_head(output),
            self.foot_head(output),
            self.contact_force_head(output),
            self.contact_binary_head(output),
        )
        if not return_context:
            return transition
        return (*transition, dynamics_latent)

    def _state_features(
        self,
        normalized_state: torch.Tensor,
        normalized_foot: torch.Tensor,
        normalized_contact_force: torch.Tensor,
        contact_binary: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                normalized_state.float(),
                normalized_foot.float(),
                normalized_contact_force.float(),
                contact_binary.float(),
            ),
            dim=-1,
        )

    @staticmethod
    def _roll_history(
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_foot: torch.Tensor,
        history_contact_force: torch.Tensor,
        history_contact_binary: torch.Tensor,
        history_valid: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        foot: torch.Tensor,
        contact_force: torch.Tensor,
        contact_binary: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            torch.cat((history_state[:, 1:], state[:, None]), dim=1),
            torch.cat((history_action[:, 1:], action[:, None]), dim=1),
            torch.cat((history_foot[:, 1:], foot[:, None]), dim=1),
            torch.cat((history_contact_force[:, 1:], contact_force[:, None]), dim=1),
            torch.cat((history_contact_binary[:, 1:], contact_binary[:, None]), dim=1),
            torch.cat(
                (
                    history_valid[:, 1:],
                    torch.ones_like(history_valid[:, :1], dtype=torch.bool),
                ),
                dim=1,
            ),
        )

    @staticmethod
    def _apply_normalized_delta(
        normalized_state: torch.Tensor,
        normalized_delta: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
    ) -> torch.Tensor:
        physical_state = normalized_state * state_std + state_mean
        physical_delta = normalized_delta * delta_std + delta_mean
        next_physical_state = apply_physical_state_delta(physical_state, physical_delta)
        return (next_physical_state - state_mean) / state_std

    def _validate_rollout_inputs(
        self,
        initial_normalized_state: torch.Tensor,
        normalized_actions: torch.Tensor,
        initial_normalized_foot: torch.Tensor,
        initial_normalized_contact_force: torch.Tensor,
        initial_contact_binary: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_foot: torch.Tensor,
        history_contact_force: torch.Tensor,
        history_contact_binary: torch.Tensor,
        history_valid: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
    ) -> None:
        batch_size = initial_normalized_state.size(0)
        expected = {
            "initial_state": (batch_size, self.config.state_dim),
            "actions": (batch_size, self.config.horizon, self.config.action_dim),
            "initial_foot": (batch_size, self.config.foot_feature_dim),
            "initial_contact_force": (batch_size, self.config.contact_force_dim),
            "initial_contact_binary": (batch_size, self.config.contact_binary_dim),
            "history_state": (
                batch_size,
                self.config.context_history_steps,
                self.config.state_dim,
            ),
            "history_action": (
                batch_size,
                self.config.context_history_steps,
                self.config.action_dim,
            ),
            "history_foot": (
                batch_size,
                self.config.history_steps,
                self.config.foot_feature_dim,
            ),
            "history_contact_force": (
                batch_size,
                self.config.history_steps,
                self.config.contact_force_dim,
            ),
            "history_contact_binary": (
                batch_size,
                self.config.history_steps,
                self.config.contact_binary_dim,
            ),
            "history_valid": (batch_size, self.config.context_history_steps),
            "state_mean": (self.config.state_dim,),
            "state_std": (self.config.state_dim,),
            "delta_mean": (self.config.delta_dim,),
            "delta_std": (self.config.delta_dim,),
        }
        actual = {
            "initial_state": tuple(initial_normalized_state.shape),
            "actions": tuple(normalized_actions.shape),
            "initial_foot": tuple(initial_normalized_foot.shape),
            "initial_contact_force": tuple(initial_normalized_contact_force.shape),
            "initial_contact_binary": tuple(initial_contact_binary.shape),
            "history_state": tuple(history_state.shape),
            "history_action": tuple(history_action.shape),
            "history_foot": tuple(history_foot.shape),
            "history_contact_force": tuple(history_contact_force.shape),
            "history_contact_binary": tuple(history_contact_binary.shape),
            "history_valid": tuple(history_valid.shape),
            "state_mean": tuple(state_mean.shape),
            "state_std": tuple(state_std.shape),
            "delta_mean": tuple(delta_mean.shape),
            "delta_std": tuple(delta_std.shape),
        }
        invalid = {
            name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape
        }
        if invalid:
            raise ValueError(f"Invalid Forward Predictor rollout shapes: {invalid}")

    def rollout(
        self,
        initial_normalized_state: torch.Tensor,
        normalized_actions: torch.Tensor,
        initial_normalized_foot: torch.Tensor,
        initial_normalized_contact_force: torch.Tensor,
        initial_contact_binary: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_foot: torch.Tensor,
        history_contact_force: torch.Tensor,
        history_contact_binary: torch.Tensor,
        history_valid: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
        *,
        dynamics_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recursively predict five states and privileged features from prior outputs."""

        self._validate_rollout_inputs(
            initial_normalized_state,
            normalized_actions,
            initial_normalized_foot,
            initial_normalized_contact_force,
            initial_contact_binary,
            history_state,
            history_action,
            history_foot,
            history_contact_force,
            history_contact_binary,
            history_valid,
            state_mean,
            state_std,
            delta_mean,
            delta_std,
        )

        normalized_state = initial_normalized_state.float()
        normalized_foot = initial_normalized_foot.float()
        normalized_contact_force = initial_normalized_contact_force.float()
        contact_binary = initial_contact_binary.float()
        predicted_states: list[torch.Tensor] = []
        predicted_deltas: list[torch.Tensor] = []
        predicted_feet: list[torch.Tensor] = []
        predicted_contact_forces: list[torch.Tensor] = []
        predicted_contact_logits: list[torch.Tensor] = []
        if dynamics_latent is None:
            dynamics_latent = self.encode_context(
                history_state,
                history_action,
                normalized_state,
                history_valid,
            )
        for index in range(self.config.horizon):
            action = normalized_actions[:, index]
            normalized_delta, next_foot, next_contact_force, next_contact_logits = self(
                normalized_state,
                action,
                normalized_foot,
                normalized_contact_force,
                contact_binary,
                history_state,
                history_action,
                history_foot,
                history_contact_force,
                history_contact_binary,
                history_valid,
                dynamics_latent=dynamics_latent,
            )
            predicted_deltas.append(normalized_delta)
            predicted_feet.append(next_foot)
            predicted_contact_forces.append(next_contact_force)
            predicted_contact_logits.append(next_contact_logits)
            previous_state = normalized_state
            previous_foot = normalized_foot
            previous_contact_force = normalized_contact_force
            previous_contact_binary = contact_binary
            normalized_state = self._apply_normalized_delta(
                previous_state,
                normalized_delta,
                state_mean,
                state_std,
                delta_mean,
                delta_std,
            )
            predicted_states.append(normalized_state)
            (
                history_state,
                history_action,
                history_foot,
                history_contact_force,
                history_contact_binary,
                history_valid,
            ) = self._roll_history(
                history_state,
                history_action,
                history_foot,
                history_contact_force,
                history_contact_binary,
                history_valid,
                previous_state,
                action,
                previous_foot,
                previous_contact_force,
                previous_contact_binary,
            )
            normalized_foot = next_foot
            normalized_contact_force = next_contact_force
            contact_binary = torch.sigmoid(next_contact_logits)
        return (
            torch.stack(predicted_states, dim=1),
            torch.stack(predicted_deltas, dim=1),
            torch.stack(predicted_feet, dim=1),
            torch.stack(predicted_contact_forces, dim=1),
            torch.stack(predicted_contact_logits, dim=1),
        )

    def teacher_forced(
        self,
        normalized_states: torch.Tensor,
        normalized_actions: torch.Tensor,
        normalized_feet: torch.Tensor,
        normalized_contact_forces: torch.Tensor,
        contact_binaries: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_foot: torch.Tensor,
        history_contact_force: torch.Tensor,
        history_contact_binary: torch.Tensor,
        history_valid: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
        *,
        dynamics_latent: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Predict five independent one-step targets while advancing true history."""

        if tuple(normalized_states.shape[1:]) != (self.config.horizon + 1, self.config.state_dim):
            raise ValueError(
                f"Teacher-forced states must be [batch,6,71], got {tuple(normalized_states.shape)}"
            )
        expected_privileged = {
            "foot": (
                normalized_states.size(0),
                self.config.horizon + 1,
                self.config.foot_feature_dim,
            ),
            "contact_force": (
                normalized_states.size(0),
                self.config.horizon + 1,
                self.config.contact_force_dim,
            ),
            "contact_binary": (
                normalized_states.size(0),
                self.config.horizon + 1,
                self.config.contact_binary_dim,
            ),
        }
        actual_privileged = {
            "foot": tuple(normalized_feet.shape),
            "contact_force": tuple(normalized_contact_forces.shape),
            "contact_binary": tuple(contact_binaries.shape),
        }
        invalid_privileged = {
            name: (actual_privileged[name], shape)
            for name, shape in expected_privileged.items()
            if actual_privileged[name] != shape
        }
        if invalid_privileged:
            raise ValueError(f"Invalid teacher-forced privileged shapes: {invalid_privileged}")
        self._validate_rollout_inputs(
            normalized_states[:, 0],
            normalized_actions,
            normalized_feet[:, 0],
            normalized_contact_forces[:, 0],
            contact_binaries[:, 0],
            history_state,
            history_action,
            history_foot,
            history_contact_force,
            history_contact_binary,
            history_valid,
            state_mean,
            state_std,
            delta_mean,
            delta_std,
        )

        batch_size = normalized_states.size(0)
        horizon = self.config.horizon
        history_steps = self.config.history_steps

        def sliding_history(history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
            """Build every true-history window and fold horizon into the batch axis."""

            sequence = torch.cat((history, current), dim=1)
            windows = sequence.unfold(1, history_steps, 1)[:, :horizon]
            if sequence.ndim == 3:
                windows = windows.movedim(-1, 2)
            return windows.contiguous().view(
                batch_size * horizon, history_steps, *history.shape[2:]
            )

        flat_state = normalized_states[:, :horizon].reshape(batch_size * horizon, -1)
        flat_action = normalized_actions.reshape(batch_size * horizon, -1)
        flat_foot = normalized_feet[:, :horizon].reshape(batch_size * horizon, -1)
        flat_contact_force = normalized_contact_forces[:, :horizon].reshape(
            batch_size * horizon, -1
        )
        flat_contact_binary = contact_binaries[:, :horizon].reshape(batch_size * horizon, -1)
        if dynamics_latent is None:
            dynamics_latent = self.encode_context(
                history_state,
                history_action,
                normalized_states[:, 0],
                history_valid,
            )
        flat_latent = (
            dynamics_latent[:, None]
            .expand(-1, horizon, -1)
            .reshape(
                batch_size * horizon,
                -1,
            )
        )
        flat_outputs = self(
            flat_state,
            flat_action,
            flat_foot,
            flat_contact_force,
            flat_contact_binary,
            sliding_history(history_state[:, -history_steps:], normalized_states[:, :horizon]),
            sliding_history(history_action[:, -history_steps:], normalized_actions),
            sliding_history(history_foot, normalized_feet[:, :horizon]),
            sliding_history(history_contact_force, normalized_contact_forces[:, :horizon]),
            sliding_history(history_contact_binary, contact_binaries[:, :horizon]),
            sliding_history(
                history_valid[:, -history_steps:],
                torch.ones(
                    (batch_size, horizon),
                    dtype=torch.bool,
                    device=history_valid.device,
                ),
            ),
            dynamics_latent=flat_latent,
            return_context=return_context,
        )
        flat_delta, flat_next_foot, flat_next_force, flat_next_binary_logits = flat_outputs[:4]
        predicted_deltas = flat_delta.view(batch_size, horizon, -1)
        predicted_feet = flat_next_foot.view(batch_size, horizon, -1)
        predicted_contact_forces = flat_next_force.view(batch_size, horizon, -1)
        predicted_contact_logits = flat_next_binary_logits.view(batch_size, horizon, -1)
        predicted_states = self._apply_normalized_delta(
            normalized_states[:, :horizon],
            predicted_deltas,
            state_mean,
            state_std,
            delta_mean,
            delta_std,
        )
        transition = (
            predicted_states,
            predicted_deltas,
            predicted_feet,
            predicted_contact_forces,
            predicted_contact_logits,
        )
        if not return_context:
            return transition
        context_valid = history_valid.all(dim=-1, keepdim=True).expand(-1, horizon)
        return (
            *transition,
            dynamics_latent[:, None].expand(-1, horizon, -1),
            context_valid,
        )
