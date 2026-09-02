"""Recursive nominal dynamics predictor with an explicit full-state transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

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

    architecture_version: str = "nominal_recursive_causal_transformer_v3"
    state_dim: int = 71
    action_dim: int = 29
    delta_dim: int = 70
    horizon: int = 5
    history_steps: int = 10
    transformer_dim: int = 512
    transformer_depth: int = 6
    transformer_heads: int = 8
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
        if self.state_dim != 71 or self.action_dim != 29 or self.delta_dim != 70:
            raise ValueError("Forward Predictor is fixed to 71-D state, 29-D action, 70-D delta")
        if self.horizon != 5:
            raise ValueError("Forward Predictor rollout is fixed to five steps")
        if self.history_steps != 10:
            raise ValueError("Forward Predictor history is fixed to ten completed interactions")
        if self.transformer_dim % self.transformer_heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def sequence_length(self) -> int:
        """Ten historical state-action tokens followed by the current token."""

        return self.history_steps + 1


class ForwardDynamicsTransformer(nn.Module):
    """Causal Transformer over ten historical and one current state-action token."""

    def __init__(self, config: ForwardPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ForwardPredictorConfig()
        width = self.config.transformer_dim
        self.state_projection = nn.Linear(self.config.state_dim, width)
        self.action_projection = nn.Linear(self.config.action_dim, width)
        self.validity_embedding = nn.Embedding(2, width)
        self.position = nn.Parameter(
            torch.empty(1, self.config.sequence_length, width)
        )
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
        self.output_norm = nn.LayerNorm(width)
        self.delta_head = nn.Linear(width, self.config.delta_dim)
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

    def forward(
        self,
        normalized_state: torch.Tensor,
        normalized_action: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
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
        expected_history = {
            "state": (batch_size, self.config.history_steps, self.config.state_dim),
            "action": (batch_size, self.config.history_steps, self.config.action_dim),
            "valid": (batch_size, self.config.history_steps),
        }
        actual_history = {
            "state": tuple(history_state.shape),
            "action": tuple(history_action.shape),
            "valid": tuple(history_valid.shape),
        }
        invalid = {
            name: (actual_history[name], shape)
            for name, shape in expected_history.items()
            if actual_history[name] != shape
        }
        if invalid:
            raise ValueError(f"Invalid Forward Predictor history shapes: {invalid}")
        valid = history_valid.to(device=history_state.device, dtype=torch.bool)
        valid_scale = valid.unsqueeze(-1).to(dtype=torch.float32)
        history_tokens = (
            self.state_projection(history_state.float() * valid_scale)
            + self.action_projection(history_action.float() * valid_scale)
        )
        history_tokens = history_tokens * valid_scale + self.validity_embedding(valid.long())
        current_token = (
            self.state_projection(normalized_state.float())
            + self.action_projection(normalized_action.float())
            + self.validity_embedding.weight[1]
        ).unsqueeze(1)
        tokens = torch.cat((history_tokens, current_token), dim=1) + self.position
        encoded = self.transformer(tokens, mask=self.causal_mask)
        return self.delta_head(self.output_norm(encoded[:, -1]))

    @staticmethod
    def _roll_history(
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_valid: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.cat((history_state[:, 1:], state[:, None]), dim=1),
            torch.cat((history_action[:, 1:], action[:, None]), dim=1),
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
        history_state: torch.Tensor,
        history_action: torch.Tensor,
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
            "history_state": (
                batch_size,
                self.config.history_steps,
                self.config.state_dim,
            ),
            "history_action": (
                batch_size,
                self.config.history_steps,
                self.config.action_dim,
            ),
            "history_valid": (batch_size, self.config.history_steps),
            "state_mean": (self.config.state_dim,),
            "state_std": (self.config.state_dim,),
            "delta_mean": (self.config.delta_dim,),
            "delta_std": (self.config.delta_dim,),
        }
        actual = {
            "initial_state": tuple(initial_normalized_state.shape),
            "actions": tuple(normalized_actions.shape),
            "history_state": tuple(history_state.shape),
            "history_action": tuple(history_action.shape),
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
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_valid: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recursively predict five normalized states using the model's own prior outputs."""

        self._validate_rollout_inputs(
            initial_normalized_state,
            normalized_actions,
            history_state,
            history_action,
            history_valid,
            state_mean,
            state_std,
            delta_mean,
            delta_std,
        )

        normalized_state = initial_normalized_state.float()
        predicted_states: list[torch.Tensor] = []
        predicted_deltas: list[torch.Tensor] = []
        for index in range(self.config.horizon):
            action = normalized_actions[:, index]
            normalized_delta = self(
                normalized_state,
                action,
                history_state,
                history_action,
                history_valid,
            )
            predicted_deltas.append(normalized_delta)
            previous_state = normalized_state
            normalized_state = self._apply_normalized_delta(
                previous_state,
                normalized_delta,
                state_mean,
                state_std,
                delta_mean,
                delta_std,
            )
            predicted_states.append(normalized_state)
            history_state, history_action, history_valid = self._roll_history(
                history_state,
                history_action,
                history_valid,
                previous_state,
                action,
            )
        return torch.stack(predicted_states, dim=1), torch.stack(predicted_deltas, dim=1)

    def teacher_forced(
        self,
        normalized_states: torch.Tensor,
        normalized_actions: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_valid: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict five independent one-step targets while advancing true history."""

        if tuple(normalized_states.shape[1:]) != (self.config.horizon + 1, self.config.state_dim):
            raise ValueError(
                f"Teacher-forced states must be [batch,6,71], got {tuple(normalized_states.shape)}"
            )
        self._validate_rollout_inputs(
            normalized_states[:, 0],
            normalized_actions,
            history_state,
            history_action,
            history_valid,
            state_mean,
            state_std,
            delta_mean,
            delta_std,
        )

        predicted_states: list[torch.Tensor] = []
        predicted_deltas: list[torch.Tensor] = []
        for index in range(self.config.horizon):
            current_state = normalized_states[:, index]
            action = normalized_actions[:, index]
            normalized_delta = self(
                current_state,
                action,
                history_state,
                history_action,
                history_valid,
            )
            predicted_states.append(
                self._apply_normalized_delta(
                    current_state,
                    normalized_delta,
                    state_mean,
                    state_std,
                    delta_mean,
                    delta_std,
                )
            )
            predicted_deltas.append(normalized_delta)
            history_state, history_action, history_valid = self._roll_history(
                history_state,
                history_action,
                history_valid,
                current_state,
                action,
            )
        return torch.stack(predicted_states, dim=1), torch.stack(predicted_deltas, dim=1)
