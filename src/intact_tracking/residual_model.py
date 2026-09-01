"""Context-conditioned residual control models for five-step tracking updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.func import functional_call

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
    """Shape and capacity contract for residual world-model training."""

    architecture_version: str = "context_residual_tracking_action_trunk_v3"
    policy_observation_dim: int = 1645
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
    backward_depth: int = 3
    policy_depth: int = 3
    dropout: float = 0.0
    residual_scale: float = 0.25

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
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

    @property
    def context_token_dim(self) -> int:
        return 2 * self.proprio_dim + self.context_chunk_steps * self.action_dim


class ResidualPolicy(nn.Module):
    """Emit one bounded five-action residual trunk from the current observation."""

    def __init__(self, config: ResidualTrackingConfig) -> None:
        super().__init__()
        self.policy_observation_dim = config.policy_observation_dim
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.residual_scale = float(config.residual_scale)
        self.net = _mlp(
            config.context_dim + config.policy_observation_dim,
            config.hidden_dim,
            config.horizon * config.action_dim,
            config.policy_depth,
        )
        # Start from the frozen tracker exactly. This also makes warm-start
        # rollouts stable before the learned dynamics gradients are useful.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, world: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if observation.size(-1) != self.policy_observation_dim:
            raise ValueError(
                f"Residual policy observation width must be {self.policy_observation_dim}, "
                f"got {observation.size(-1)}"
            )
        if world.shape[:-1] != observation.shape[:-1]:
            raise ValueError(
                "Residual policy world and observation batches must match: "
                f"{tuple(world.shape)} vs {tuple(observation.shape)}"
            )
        trunk = self.net(torch.cat((world.float(), observation.float()), dim=-1))
        return (
            trunk.reshape(*observation.shape[:-1], self.horizon, self.action_dim)
            .tanh()
            .mul(self.residual_scale)
        )


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


class BackwardPredictor(nn.Module):
    """Infer the command between adjacent true states under the inferred world."""

    def __init__(self, config: ResidualTrackingConfig) -> None:
        super().__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.net = _mlp(
            config.context_dim + 2 * config.state_dim + config.action_dim,
            config.hidden_dim,
            config.action_dim,
            config.backward_depth,
        )

    def forward(
        self,
        world: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        while world.ndim < state.ndim:
            world = world.unsqueeze(1)
        world = world.expand(*state.shape[:-1], world.size(-1))
        return self.net(
            torch.cat(
                (world.float(), state.float(), next_state.float(), previous_action.float()),
                dim=-1,
            )
        )


class ResidualTrackingModel(nn.Module):
    """Context encoder, dynamics pair, and residual policy with explicit routes."""

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
        self.backward_predictor = BackwardPredictor(config)
        self.residual_policy = ResidualPolicy(config)

    def encode_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.context_encoder(context, context_mask)

    def residual_action_trunk(self, world: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return self.residual_policy(world, observation)

    def residual_action(self, world: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Compatibility alias returning the complete five-action trunk."""
        return self.residual_action_trunk(world, observation)

    def predict_future(
        self,
        world: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_predictor(world, state, previous_action, actions)

    def predict_future_with_frozen_dynamics(
        self,
        world: torch.Tensor,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Keep input gradients while preventing tracking loss from updating Forward."""
        frozen_state = {
            name: value.detach()
            for name, value in (
                *self.forward_predictor.named_parameters(),
                *self.forward_predictor.named_buffers(),
            )
        }
        return functional_call(
            self.forward_predictor,
            frozen_state,
            (world, state, previous_action, actions),
        )
