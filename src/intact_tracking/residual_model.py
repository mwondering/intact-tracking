"""Unified temporal Transformer for history-conditioned Forward dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UnifiedForwardConfig:
    """Shape and capacity contract for the unified Forward world model."""

    architecture_version: str = "unified_history_action_forward_transformer_v9"
    proprio_dim: int = 122
    action_dim: int = 29
    state_dim: int = 71
    pose_delta_dim: int = 35
    horizon: int = 5
    context_steps: int = 160
    transformer_dim: int = 400
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
            raise ValueError(f"Forward model dimensions must be positive: {invalid}")
        if self.state_dim != 71:
            raise ValueError("Forward state is fixed to the 71-D robot/reference state")
        if self.pose_delta_dim != 35:
            raise ValueError(
                "Forward output is fixed to 35-D pose deltas: root translation, "
                "root rotation vector, and 29 joint displacements"
            )
        if self.horizon != 5:
            raise ValueError("Forward prediction is fixed to five control steps")
        if self.transformer_dim % self.transformer_heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")

    @property
    def sequence_length(self) -> int:
        """History states/actions, one current-condition token, and five query actions."""

        return 2 * self.context_steps + self.horizon + 2


class UnifiedForwardTransformer(nn.Module):
    """One causal Transformer from interaction history to five future pose deltas."""

    def __init__(self, config: UnifiedForwardConfig) -> None:
        super().__init__()
        self.config = config
        self.proprio_dim = config.proprio_dim
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.context_steps = config.context_steps
        self.horizon = config.horizon
        self.sequence_length = config.sequence_length
        dim = config.transformer_dim

        self.history_state_projection = nn.Sequential(
            nn.LayerNorm(config.proprio_dim),
            nn.Linear(config.proprio_dim, dim),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, dim),
        )
        condition_dim = config.state_dim + config.action_dim
        self.current_condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, dim),
        )
        self.position = nn.Parameter(torch.empty(1, self.sequence_length, dim))
        self.token_type = nn.Parameter(torch.empty(4, dim))
        self.boundary_embedding = nn.Parameter(torch.empty(dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.token_type, std=0.02)
        nn.init.trunc_normal_(self.boundary_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.transformer_heads,
            dim_feedforward=4 * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_depth)
        self.output_norm = nn.LayerNorm(dim)
        self.delta_head = nn.Linear(dim, config.pose_delta_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.sequence_length, self.sequence_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def predict_future(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        history_state_mask: torch.Tensor,
        history_action_mask: torch.Tensor,
        history_boundaries: torch.Tensor,
        current_state: torch.Tensor,
        previous_action: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history_states.size(0)
        expected_states = (batch_size, self.context_steps + 1, self.proprio_dim)
        expected_history_actions = (batch_size, self.context_steps, self.action_dim)
        expected_future_actions = (batch_size, self.horizon, self.action_dim)
        expected_current_state = (batch_size, self.state_dim)
        expected_previous_action = (batch_size, self.action_dim)
        if tuple(history_states.shape) != expected_states:
            raise ValueError(
                f"History states must have shape {expected_states}, "
                f"got {tuple(history_states.shape)}"
            )
        if tuple(history_actions.shape) != expected_history_actions:
            raise ValueError(
                f"History actions must have shape {expected_history_actions}, "
                f"got {tuple(history_actions.shape)}"
            )
        if tuple(future_actions.shape) != expected_future_actions:
            raise ValueError(
                f"Future actions must have shape {expected_future_actions}, "
                f"got {tuple(future_actions.shape)}"
            )
        if tuple(current_state.shape) != expected_current_state:
            raise ValueError(
                f"Current state must have shape {expected_current_state}, "
                f"got {tuple(current_state.shape)}"
            )
        if tuple(previous_action.shape) != expected_previous_action:
            raise ValueError(
                f"Previous action must have shape {expected_previous_action}, "
                f"got {tuple(previous_action.shape)}"
            )
        expected_state_mask = expected_states[:2]
        expected_action_mask = expected_history_actions[:2]
        if tuple(history_state_mask.shape) != expected_state_mask:
            raise ValueError(
                f"History state mask must have shape {expected_state_mask}, "
                f"got {tuple(history_state_mask.shape)}"
            )
        if tuple(history_action_mask.shape) != expected_action_mask:
            raise ValueError(
                f"History action mask must have shape {expected_action_mask}, "
                f"got {tuple(history_action_mask.shape)}"
            )
        if tuple(history_boundaries.shape) != expected_state_mask:
            raise ValueError(
                f"History boundaries must have shape {expected_state_mask}, "
                f"got {tuple(history_boundaries.shape)}"
            )

        state_mask = history_state_mask.to(device=history_states.device, dtype=torch.bool)
        action_mask = history_action_mask.to(device=history_states.device, dtype=torch.bool)
        boundaries = history_boundaries.to(device=history_states.device, dtype=torch.bool)
        if not state_mask[:, 0].all() or not action_mask.any(dim=1).all():
            raise ValueError("Every history needs an initial state and at least one action")

        state_tokens = self.history_state_projection(history_states.float()) + self.token_type[0]
        state_tokens = state_tokens + boundaries.unsqueeze(-1) * self.boundary_embedding
        history_action_tokens = self.action_projection(history_actions.float()) + self.token_type[1]
        condition = torch.cat((current_state.float(), previous_action.float()), dim=-1)
        condition_token = self.current_condition_projection(condition).unsqueeze(1)
        condition_token = condition_token + self.token_type[2]
        query_action_tokens = self.action_projection(future_actions.float()) + self.token_type[3]

        sequence = history_states.new_empty(
            (batch_size, self.sequence_length, self.position.size(-1)), dtype=torch.float32
        )
        history_end = 2 * self.context_steps + 1
        sequence[:, :history_end:2] = state_tokens
        sequence[:, 1:history_end:2] = history_action_tokens
        sequence[:, history_end : history_end + 1] = condition_token
        sequence[:, history_end + 1 :] = query_action_tokens
        sequence = sequence + self.position

        token_mask = torch.ones(
            (batch_size, self.sequence_length), dtype=torch.bool, device=history_states.device
        )
        token_mask[:, :history_end:2] = state_mask
        token_mask[:, 1:history_end:2] = action_mask
        encoded = self.transformer(
            sequence,
            mask=self.causal_mask,
            src_key_padding_mask=~token_mask,
        )
        query_encoded = encoded[:, -self.horizon :]
        return self.delta_head(self.output_norm(query_encoded))

    def forward(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        history_state_mask: torch.Tensor,
        history_action_mask: torch.Tensor,
        history_boundaries: torch.Tensor,
        current_state: torch.Tensor,
        previous_action: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_future(
            history_states,
            history_actions,
            history_state_mask,
            history_action_mask,
            history_boundaries,
            current_state,
            previous_action,
            future_actions,
        )


# Compatibility aliases for checkpoints and imports created before architecture v9.
ResidualTrackingConfig = UnifiedForwardConfig
ResidualTrackingModel = UnifiedForwardTransformer
