"""Context-conditioned Forward model for five-step dynamics prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import InteractionContextEncoder


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

    architecture_version: str = "context_forward_pose_model_v6"
    proprio_dim: int = 122
    action_dim: int = 29
    state_dim: int = 71
    pose_delta_dim: int = 35
    horizon: int = 5
    context_chunk_steps: int = 5
    context_tokens: int = 16
    context_dim: int = 192
    context_depth: int = 2
    context_heads: int = 4
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
        if self.context_tokens != 16:
            raise ValueError("Residual context is fixed to 16 interaction tokens")
        if self.context_dim % self.context_heads:
            raise ValueError("context_dim must be divisible by context_heads")

    @property
    def context_token_dim(self) -> int:
        return 2 * self.proprio_dim + self.context_chunk_steps * self.action_dim


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
        self.context_encoder = InteractionContextEncoder(
            token_dim=config.context_token_dim,
            embed_dim=config.context_dim,
            token_count=config.context_tokens,
            depth=config.context_depth,
            heads=config.context_heads,
            dropout=config.dropout,
        )
        self.forward_predictor = CausalForwardPredictor(config)

    def encode_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.context_encoder(context, context_mask)

    def predict_future(
        self,
        world: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_predictor(world, state, previous_action, actions)
