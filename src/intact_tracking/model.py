"""Tracking adaptation of the INTACT Encoder/Forward/Intent-Action architecture.

The physical and goal branches deliberately share one four-slot INTACT Predictor:
``[z_t, m_t, z_t * m_t, A(a_{t-1})]``.  The fixed interaction context is folded
into ``z`` through a shared FiLM transform, so it does not create a fifth actor
slot or a second action head.

The attention, conditional-transformer, SIGReg, action embedding, and actor
structure follow the INTACT architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@dataclass(frozen=True)
class TrackingINTACTConfig:
    """Shape and capacity contract for the first tracking implementation."""

    architecture_version: str = "single_step_effect_v1"
    observation_dim: int = 64
    proprio_dim: int = 122
    action_dim: int = 29
    effect_steps: int = 5
    context_chunk_steps: int = 5
    context_tokens: int = 16
    embed_dim: int = 192
    encoder_hidden_dim: int = 512
    context_depth: int = 2
    context_heads: int = 4
    forward_history: int = 3
    forward_depth: int = 6
    forward_heads: int = 8
    forward_mlp_dim: int = 768
    actor_hidden_dim: int = 1024
    actor_depth: int = 3
    dropout: float = 0.0
    predict_residual: bool = False

    def __post_init__(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"All integer dimensions must be positive, got {invalid}")
        if self.context_tokens != 16:
            raise ValueError("Tracking INTACT context_tokens is fixed at 16")
        if self.embed_dim % self.context_heads:
            raise ValueError("embed_dim must be divisible by context_heads")

    @property
    def forward_action_dim(self) -> int:
        """Width of the action sequence conditioning one Forward transition."""
        return self.action_dim * self.effect_steps

    @property
    def context_token_dim(self) -> int:
        return 2 * self.proprio_dim + self.action_dim * self.context_chunk_steps


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by the INTACT reference."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("SIGReg requires knots >= 2 and num_proj >= 1")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projection: torch.Tensor) -> torch.Tensor:
        projection32 = projection.float()
        directions = torch.randn(
            projection32.size(-1),
            self.num_proj,
            device=projection.device,
            dtype=torch.float32,
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-8)
        x_t = (projection32 @ directions).unsqueeze(-1) * self.t
        error = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (error @ self.weights) * projection32.size(-2)
        return statistic.mean().to(dtype=projection.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if inner_dim != dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        batch, length, _ = q.shape
        q = q.reshape(batch, length, self.heads, -1).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, -1).transpose(1, 2)
        v = v.reshape(batch, length, self.heads, -1).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.to_out(output)


class ConditionalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(
            6, dim=-1
        )
        x = x + gate_a * self.attention(_modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.condition_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        condition = self.condition_proj(condition)
        for layer in self.layers:
            x = layer(x, condition)
        return self.output_proj(self.norm(x))


class ARPredictor(nn.Module):
    """LeWM-style autoregressive, action-conditioned latent predictor."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConditionalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.pos_embedding.size(1):
            raise ValueError(
                f"Predictor received {length} frames, max is {self.pos_embedding.size(1)}"
            )
        return self.transformer(self.dropout(x + self.pos_embedding[:, :length]), condition)


class ObservationEncoder(nn.Module):
    """Structured replacement for INTACT's visual encoder and projector."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation.float())


class ActionEncoder(nn.Module):
    """Embed a flattened action vector or finite action sequence."""

    def __init__(self, input_dim: int, embed_dim: int, mlp_scale: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * embed_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * embed_dim, embed_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected a flattened action input of width {self.input_dim}, "
                f"got {action.size(-1)}"
            )
        return self.net(action.float())


class InteractionContextEncoder(nn.Module):
    """Encode exactly 16 causal action-response tokens into one world code."""

    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
        token_count: int = 16,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.token_count = token_count
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.position = nn.Parameter(torch.empty(1, token_count, embed_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Context must be [B,16,D], got {tuple(tokens.shape)}")
        if tokens.size(1) != self.token_count or tokens.size(2) != self.token_dim:
            raise ValueError(
                f"Context contract is [B,{self.token_count},{self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"Context mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}"
            )
        if not mask.any(dim=1).all():
            raise ValueError("Every sample needs at least one valid context token")
        encoded = self.token_projection(tokens.float()) + self.position
        encoded = self.transformer(encoded, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output_norm(pooled)


class ContextFiLM(nn.Module):
    """Inject the inferred world while keeping the INTACT latent width unchanged."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.affine = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, latent: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(world).chunk(2, dim=-1)
        while scale.ndim < latent.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return latent * (1 + scale) + shift


class IntentActionActor(nn.Module):
    """One shared local/goal Gaussian action law with INTACT's four slots."""

    def __init__(
        self,
        *,
        embed_dim: int,
        action_emb_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("IntentActionActor depth must be positive")
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        layers: list[nn.Module] = [
            nn.Linear(3 * embed_dim + action_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 2 * action_dim))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def actor_features(
        z: torch.Tensor, intent: torch.Tensor, previous_action_embedding: torch.Tensor
    ) -> torch.Tensor:
        if z.shape != intent.shape:
            raise ValueError(
                f"z and intent must have the same shape, got {z.shape} and {intent.shape}"
            )
        return torch.cat([z, intent, z * intent, previous_action_embedding], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(self.actor_features(z, intent, previous_action_embedding))
        mean, log_std = params.chunk(2, dim=-1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    def action_mean(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self(z, intent, previous_action_embedding)[0]


class TrackingINTACT(nn.Module):
    """Context-conditioned tracking model with the original INTACT loss routes."""

    def __init__(self, config: TrackingINTACTConfig | None = None) -> None:
        super().__init__()
        self.config = config or TrackingINTACTConfig()
        cfg = self.config
        self.encoder = ObservationEncoder(
            cfg.observation_dim, cfg.encoder_hidden_dim, cfg.embed_dim
        )
        self.context_encoder = InteractionContextEncoder(
            token_dim=cfg.context_token_dim,
            embed_dim=cfg.embed_dim,
            token_count=cfg.context_tokens,
            depth=cfg.context_depth,
            heads=cfg.context_heads,
            dropout=cfg.dropout,
        )
        self.context_film = ContextFiLM(cfg.embed_dim)
        self.forward_action_encoder = ActionEncoder(cfg.forward_action_dim, cfg.embed_dim)
        self.previous_action_encoder = ActionEncoder(cfg.action_dim, cfg.embed_dim)
        self.predictor = ARPredictor(
            num_frames=cfg.forward_history,
            depth=cfg.forward_depth,
            heads=cfg.forward_heads,
            mlp_dim=cfg.forward_mlp_dim,
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.embed_dim,
            output_dim=cfg.embed_dim,
            dim_head=max(cfg.embed_dim // cfg.forward_heads, 1),
            dropout=cfg.dropout,
        )
        self.intent_actor = IntentActionActor(
            embed_dim=cfg.embed_dim,
            action_emb_dim=cfg.embed_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            depth=cfg.actor_depth,
            dropout=cfg.dropout,
        )

    def encode_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.context_encoder(context, context_mask)

    def encode_observation(self, observation: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        if observation.size(-1) != self.config.observation_dim:
            raise ValueError(
                f"Observation width must be {self.config.observation_dim}, "
                f"got {observation.size(-1)}"
            )
        return self.context_film(self.encoder(observation), world)

    def predict(self, embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        prediction = self.predictor(embedding, action_embedding)
        if self.config.predict_residual:
            return embedding + prediction
        return prediction

    def action_parameters(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_embedding = self.previous_action_encoder(previous_action)
        return self.intent_actor(z, intent, previous_embedding)

    def action_nll(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action: torch.Tensor,
        target_action: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.action_parameters(z, intent, previous_action)
        per_step = 0.5 * (
            (target_action - mean).square() * torch.exp(-2 * log_std) + 2 * log_std
        ).mean(dim=-1)
        if mask is None:
            loss = per_step.mean()
            denominator = per_step.new_tensor(per_step.numel())
        else:
            weights = mask.to(device=per_step.device, dtype=per_step.dtype)
            denominator = weights.sum().clamp_min(1)
            loss = (per_step * weights).sum() / denominator
        return {
            "loss": loss,
            "nll": per_step,
            "mean": mean,
            "log_std": log_std,
            "mae": (mean - target_action).abs().mean(dim=-1),
            "count": denominator.detach(),
        }

    @torch.inference_mode()
    def direct_action(
        self,
        observation: torch.Tensor,
        goal_observation: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the one deployable action for the current control step.

        ``goal_observation`` is the reference endpoint ``effect_steps`` into the
        future.  Forward is a training-only representation objective here; the
        deployed tracking path does not roll latent dynamics forward.
        """
        world = self.encode_context(context, context_mask)
        current = self.encode_observation(observation, world)
        goal = self.encode_observation(goal_observation, world)
        if current.ndim == 2:
            current = current[:, None]
        if goal.ndim == 3:
            goal = goal[:, -1]
        else:
            goal = goal.reshape(goal.size(0), -1)
        z = current[:, -1]
        intent = goal - z
        previous_embedding = self.previous_action_encoder(previous_action)
        return self.intent_actor.action_mean(z, intent, previous_embedding)

    @torch.inference_mode()
    def direct_plan(
        self,
        observation: torch.Tensor,
        goal_observation: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        horizon: int = 1,
    ) -> torch.Tensor:
        """Compatibility wrapper returning ``[B,1,action_dim]``.

        Tracking deployment intentionally supports one action only.  A longer
        open-loop plan would require future actions that this actor does not
        predict and would conflate policy horizon with the Forward effect span.
        """
        if horizon != 1:
            raise ValueError("Single-step tracking only supports horizon=1")
        return self.direct_action(
            observation=observation,
            goal_observation=goal_observation,
            previous_action=previous_action,
            context=context,
            context_mask=context_mask,
        )[:, None]
