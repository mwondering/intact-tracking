"""Temporally tokenized context-conditioned Forward dynamics model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be positive")
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ResidualTrackingConfig:
    """Shape and capacity contract for Forward-only world-model training."""

    architecture_version: str = "temporal_state_action_context_forward_pose_model_v7"
    proprio_dim: int = 122
    action_dim: int = 29
    state_dim: int = 71
    pose_delta_dim: int = 35
    horizon: int = 5
    context_steps: int = 160
    context_dim: int = 408
    context_depth: int = 5
    context_heads: int = 8
    hidden_dim: int = 512
    forward_depth: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"Residual model dimensions must be positive: {invalid}")
        if self.state_dim != 71:
            raise ValueError("Residual tracking state is fixed to the 71-D robot/reference state")
        if self.pose_delta_dim != 35:
            raise ValueError(
                "Residual Forward output is fixed to 35-D pose deltas: "
                "root translation, root rotation vector, and 29 joint displacements"
            )
        if self.horizon != 5:
            raise ValueError("Residual policy optimization is fixed to five control steps")
        if self.context_dim % self.context_heads:
            raise ValueError("context_dim must be divisible by context_heads")

    @property
    def context_sequence_length(self) -> int:
        """Interleaved states/actions followed by one learned environment token."""

        return 2 * self.context_steps + 2


class TemporalStateActionContextEncoder(nn.Module):
    """Encode an ordered ``S0,A0,...,S_T,[ENV]`` history into one world code."""

    def __init__(self, config: ResidualTrackingConfig) -> None:
        super().__init__()
        dim = config.context_dim
        self.proprio_dim = config.proprio_dim
        self.action_dim = config.action_dim
        self.context_steps = config.context_steps
        self.sequence_length = config.context_sequence_length
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.proprio_dim),
            nn.Linear(config.proprio_dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.position = nn.Parameter(torch.empty(1, self.sequence_length, dim))
        self.token_type = nn.Parameter(torch.empty(3, dim))
        self.boundary_embedding = nn.Parameter(torch.empty(dim))
        self.environment_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.token_type, std=0.02)
        nn.init.trunc_normal_(self.boundary_embedding, std=0.02)
        nn.init.trunc_normal_(self.environment_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.context_heads,
            dim_feedforward=4 * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.context_depth)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = states.size(0)
        expected_states = (batch_size, self.context_steps + 1, self.proprio_dim)
        expected_actions = (batch_size, self.context_steps, self.action_dim)
        if tuple(states.shape) != expected_states:
            raise ValueError(
                f"Context states must have shape {expected_states}, got {tuple(states.shape)}"
            )
        if tuple(actions.shape) != expected_actions:
            raise ValueError(
                f"Context actions must have shape {expected_actions}, got {tuple(actions.shape)}"
            )
        expected_state_mask = expected_states[:2]
        expected_action_mask = expected_actions[:2]
        if tuple(state_mask.shape) != expected_state_mask:
            raise ValueError(
                "Context state mask must have shape "
                f"{expected_state_mask}, got {tuple(state_mask.shape)}"
            )
        if tuple(action_mask.shape) != expected_action_mask:
            raise ValueError(
                "Context action mask must have shape "
                f"{expected_action_mask}, got {tuple(action_mask.shape)}"
            )
        if tuple(boundaries.shape) != expected_state_mask:
            raise ValueError(
                f"Context boundaries must have shape {expected_state_mask}, "
                f"got {tuple(boundaries.shape)}"
            )
        state_mask = state_mask.to(device=states.device, dtype=torch.bool)
        action_mask = action_mask.to(device=states.device, dtype=torch.bool)
        boundaries = boundaries.to(device=states.device, dtype=torch.bool)
        if not state_mask[:, 0].all() or not action_mask.any(dim=1).all():
            raise ValueError("Every context needs an initial state and at least one action")

        state_tokens = self.state_projection(states.float()) + self.token_type[0]
        state_tokens = state_tokens + boundaries.unsqueeze(-1) * self.boundary_embedding
        action_tokens = self.action_projection(actions.float()) + self.token_type[1]
        sequence = states.new_empty(
            (batch_size, self.sequence_length, self.position.size(-1)), dtype=torch.float32
        )
        sequence[:, 0 : 2 * self.context_steps + 1 : 2] = state_tokens
        sequence[:, 1 : 2 * self.context_steps : 2] = action_tokens
        sequence[:, -1:] = self.environment_token + self.token_type[2]
        sequence = sequence + self.position

        token_mask = torch.ones(
            (batch_size, self.sequence_length), dtype=torch.bool, device=states.device
        )
        token_mask[:, 0 : 2 * self.context_steps + 1 : 2] = state_mask
        token_mask[:, 1 : 2 * self.context_steps : 2] = action_mask
        causal_mask = torch.triu(
            torch.ones(
                self.sequence_length,
                self.sequence_length,
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )
        encoded = self.transformer(
            sequence,
            mask=causal_mask,
            src_key_padding_mask=~token_mask,
        )
        return self.output_norm(encoded[:, -1])


class CausalForwardPredictor(nn.Module):
    """Predict five future pose deltas while exposing only each action prefix."""

    def __init__(self, config: ResidualTrackingConfig) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.pose_delta_dim = config.pose_delta_dim
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.initial_encoder = _mlp(
            config.context_dim + config.state_dim + config.action_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.forward_depth,
        )
        self.action_encoder = _mlp(
            config.action_dim,
            config.hidden_dim,
            config.hidden_dim,
            1,
        )
        self.transition = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.delta_head = _mlp(
            config.hidden_dim,
            config.hidden_dim,
            config.pose_delta_dim,
            1,
        )

    def forward(
        self,
        world: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        expected = (*state.shape[:-1], self.horizon, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"Forward actions must have shape {expected}, got {tuple(actions.shape)}"
            )
        hidden = self.initial_encoder(
            torch.cat((world.float(), state.float(), previous_action.float()), dim=-1)
        )
        predictions = []
        for index in range(self.horizon):
            hidden = self.transition(self.action_encoder(actions[:, index].float()), hidden)
            predictions.append(self.delta_head(hidden))
        return torch.stack(predictions, dim=1)


class ResidualTrackingModel(nn.Module):
    """The active world model: Context Encoder followed by Forward Predictor."""

    def __init__(self, config: ResidualTrackingConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = TemporalStateActionContextEncoder(config)
        self.forward_predictor = CausalForwardPredictor(config)

    def encode_context(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        return self.context_encoder(states, actions, state_mask, action_mask, boundaries)

    def predict_future(
        self,
        world: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_predictor(world, state, previous_action, actions)
