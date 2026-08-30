"""Observation-conditioned residual mixture-of-experts policy core."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class RMSNorm(nn.Module):
    """ONNX-friendly RMS normalization with a learned per-feature scale."""

    def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(hidden_dim)))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return value * inverse_rms * self.weight


class LayerNormResidualBlock(nn.Module):
    """FlashSAC-shaped residual block using per-sample LayerNorm."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        *,
        activation: str = "relu",
        linear_bias: bool = False,
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        expanded_dim = hidden_dim * int(expansion)
        if hidden_dim <= 0 or expanded_dim <= 0:
            raise ValueError("hidden_dim and expansion must be positive")

        # Bias-free orthogonal Linears remain the legacy default used by SPV5-1.
        # SPV7 opts into biased, default-initialized Linears to isolate its
        # Residual+LayerNorm ablation from ordinary RSL-RL MLP details.
        self.linear1 = nn.Linear(hidden_dim, expanded_dim, bias=bool(linear_bias))
        self.norm1 = nn.LayerNorm(expanded_dim)
        self.activation1 = resolve_nn_activation(activation)
        self.linear2 = nn.Linear(expanded_dim, hidden_dim, bias=bool(linear_bias))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation2 = resolve_nn_activation(activation)
        if orthogonal_init:
            nn.init.orthogonal_(self.linear1.weight, gain=1.0)
            nn.init.orthogonal_(self.linear2.weight, gain=1.0)

    def residual(self, value: torch.Tensor) -> torch.Tensor:
        value = self.activation1(self.norm1(self.linear1(value)))
        return self.activation2(self.norm2(self.linear2(value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.residual(value)


class PreNormResidualBlock(nn.Module):
    """Pre-LN residual block whose residual branch starts exactly at zero."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        *,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        expanded_dim = hidden_dim * int(expansion)
        if hidden_dim <= 0 or expanded_dim <= 0:
            raise ValueError("hidden_dim and expansion must be positive")

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, expanded_dim)
        self.activation = resolve_nn_activation(activation)
        self.linear2 = nn.Linear(expanded_dim, hidden_dim)

        # This makes the complete block an exact identity at initialization,
        # independent of depth and activation statistics.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def residual(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(self.norm(value))))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.residual(value)


class PreNormResidualPolicy(nn.Module):
    """Stable Pre-LN residual network shared by SPV7-2 actor/critic ablations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        context_hidden_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
        activation: str = "elu",
        output_init_gain: float = 1.0e-2,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        context_hidden_dim = int(context_hidden_dim)
        hidden_dim = int(hidden_dim)
        num_blocks = int(num_blocks)
        expansion = int(expansion)
        output_init_gain = float(output_init_gain)
        if min(input_dim, output_dim, context_hidden_dim, hidden_dim) <= 0:
            raise ValueError("Residual policy dimensions must be positive")
        if num_blocks <= 0:
            raise ValueError("Residual policy num_blocks must be positive")
        if expansion <= 0:
            raise ValueError("Residual policy expansion must be positive")
        if output_init_gain <= 0.0:
            raise ValueError("Residual policy output_init_gain must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_hidden_dim = context_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, context_hidden_dim),
            resolve_nn_activation(activation),
            nn.Linear(context_hidden_dim, hidden_dim),
            resolve_nn_activation(activation),
        )
        self.blocks = nn.ModuleList(
            PreNormResidualBlock(
                hidden_dim,
                expansion,
                activation=activation,
            )
            for _ in range(num_blocks)
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.output.weight, gain=output_init_gain)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.context_encoder(value)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.post_norm(hidden))

    @property
    @torch.jit.unused
    def dense_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ResidualLayerNormPolicy(nn.Module):
    """Dense residual policy with post-linear block-local LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        context_hidden_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
        activation: str = "elu",
        output_init_gain: float | None = None,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        context_hidden_dim = int(context_hidden_dim)
        hidden_dim = int(hidden_dim)
        num_blocks = int(num_blocks)
        expansion = int(expansion)
        if min(input_dim, output_dim, context_hidden_dim, hidden_dim) <= 0:
            raise ValueError("Residual policy dimensions must be positive")
        if num_blocks <= 0:
            raise ValueError("Residual policy num_blocks must be positive")
        if expansion <= 0:
            raise ValueError("Residual policy expansion must be positive")
        if output_init_gain is not None and output_init_gain <= 0.0:
            raise ValueError("Residual policy output_init_gain must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_hidden_dim = context_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, context_hidden_dim),
            resolve_nn_activation(activation),
            nn.Linear(context_hidden_dim, hidden_dim),
            resolve_nn_activation(activation),
        )
        self.blocks = nn.ModuleList(
            LayerNormResidualBlock(
                hidden_dim,
                expansion,
                activation=activation,
                linear_bias=True,
                orthogonal_init=False,
            )
            for _ in range(num_blocks)
        )
        # This variant intentionally has no final normalization; its LayerNorm
        # modules remain local to the two projections inside each residual block.
        self.post_norm = nn.Identity()
        self.output = nn.Linear(hidden_dim, output_dim)
        if output_init_gain is not None:
            nn.init.orthogonal_(self.output.weight, gain=float(output_init_gain))
            nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.context_encoder(value)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.post_norm(hidden))

    @property
    @torch.jit.unused
    def dense_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MLPExpertBlock(nn.Module):
    """Two-layer biased MLP block used by the plain-MLP MoE ablations."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        *,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        expanded_dim = hidden_dim * int(expansion)
        if hidden_dim <= 0 or expanded_dim <= 0:
            raise ValueError("hidden_dim and expansion must be positive")
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            resolve_nn_activation(activation),
            nn.Linear(expanded_dim, hidden_dim),
            resolve_nn_activation(activation),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ObservationConditionedMLPMoE(nn.Module):
    """Dense-compute Top-k MoE whose shared and expert blocks are plain MLPs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        context_hidden_dim: int = 1480,
        hidden_dim: int = 608,
        num_experts: int = 8,
        top_k: int = 2,
        expansion: int = 4,
        activation: str = "elu",
        router_temperature: float = 1.5,
        router_init_std: float = 1.0e-2,
        output_init_gain: float | None = None,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        context_hidden_dim = int(context_hidden_dim)
        hidden_dim = int(hidden_dim)
        num_experts = int(num_experts)
        top_k = int(top_k)
        expansion = int(expansion)
        if min(input_dim, output_dim, context_hidden_dim, hidden_dim) <= 0:
            raise ValueError("MoE dimensions must be positive")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than one")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        if router_init_std <= 0.0:
            raise ValueError("router_init_std must be positive")
        if output_init_gain is not None and output_init_gain <= 0.0:
            raise ValueError("output_init_gain must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_hidden_dim = context_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expansion = expansion
        self.router_temperature = float(router_temperature)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, context_hidden_dim),
            resolve_nn_activation(activation),
            nn.Linear(context_hidden_dim, hidden_dim),
            resolve_nn_activation(activation),
        )
        self.shared_block = MLPExpertBlock(hidden_dim, expansion, activation=activation)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            MLPExpertBlock(hidden_dim, expansion, activation=activation) for _ in range(num_experts)
        )
        self.output = nn.Linear(hidden_dim, output_dim)

        nn.init.normal_(self.router.weight, mean=0.0, std=float(router_init_std))
        if output_init_gain is not None:
            nn.init.orthogonal_(self.output.weight, gain=float(output_init_gain))
            nn.init.zeros_(self.output.bias)

    def _shared_features(self, value: torch.Tensor) -> torch.Tensor:
        return self.shared_block(self.context_encoder(value))

    def routing_probabilities_from_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        logits = self.router(shared_features) / self.router_temperature
        return torch.softmax(logits, dim=-1)

    def routing_probabilities(self, value: torch.Tensor) -> torch.Tensor:
        return self.routing_probabilities_from_features(self._shared_features(value))

    def sparse_probabilities(self, dense_probabilities: torch.Tensor) -> torch.Tensor:
        top_values, top_indices = torch.topk(dense_probabilities, self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        return torch.zeros_like(dense_probabilities).scatter(-1, top_indices, top_values)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        leading_shape = value.shape[:-1]
        flat_value = value.reshape(-1, value.shape[-1])
        shared = self._shared_features(flat_value)
        dense_probabilities = self.routing_probabilities_from_features(shared)
        sparse_probabilities = self.sparse_probabilities(dense_probabilities)
        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=-2)
        mixed = torch.sum(sparse_probabilities.unsqueeze(-1) * expert_outputs, dim=-2)
        output = self.output(mixed)
        return output.reshape(*leading_shape, self.output_dim)

    @property
    def dense_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def maximum_router_entropy(self) -> float:
        return math.log(self.num_experts)


class ObservationConditionedResidualMoE(nn.Module):
    """Dense-compute top-k residual MoE with a shared context backbone.

    All experts are evaluated in the first implementation.  Top-k sparsity is
    applied to the mixture probabilities, keeping the forward path simple and
    exportable while preserving the intended routing semantics.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        context_hidden_dim: int = 1472,
        hidden_dim: int = 608,
        num_experts: int = 8,
        top_k: int = 2,
        expansion: int = 4,
        router_temperature: float = 1.5,
        router_init_std: float = 1.0e-2,
        output_init_gain: float = 5.0e-2,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        context_hidden_dim = int(context_hidden_dim)
        hidden_dim = int(hidden_dim)
        num_experts = int(num_experts)
        top_k = int(top_k)
        expansion = int(expansion)
        if min(input_dim, output_dim, context_hidden_dim, hidden_dim) <= 0:
            raise ValueError("MoE dimensions must be positive")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than one")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        if router_init_std <= 0.0:
            raise ValueError("router_init_std must be positive")
        if output_init_gain <= 0.0:
            raise ValueError("output_init_gain must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_hidden_dim = context_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expansion = expansion
        self.router_temperature = float(router_temperature)

        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Linear(context_hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shared_block = LayerNormResidualBlock(hidden_dim, expansion)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            LayerNormResidualBlock(hidden_dim, expansion) for _ in range(num_experts)
        )
        self.post_norm = RMSNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        # Exactly-zero logits make torch.topk select the same experts for every
        # initial state.  A small random router keeps q close to uniform while
        # allowing the selected set to vary with observation content.
        nn.init.normal_(self.router.weight, mean=0.0, std=float(router_init_std))
        # RMSNorm fixes the feature RMS near one.  Default Linear initialization
        # would therefore produce action means roughly ten times wider than the
        # original SPV5-1 MLP.  Match the baseline startup scale explicitly.
        nn.init.orthogonal_(self.output.weight, gain=float(output_init_gain))
        nn.init.zeros_(self.output.bias)

    def _shared_features(self, value: torch.Tensor) -> torch.Tensor:
        return self.shared_block(self.context_encoder(value))

    def routing_probabilities_from_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        logits = self.router(shared_features) / self.router_temperature
        return torch.softmax(logits, dim=-1)

    def routing_probabilities(self, value: torch.Tensor) -> torch.Tensor:
        return self.routing_probabilities_from_features(self._shared_features(value))

    def sparse_probabilities(self, dense_probabilities: torch.Tensor) -> torch.Tensor:
        top_values, top_indices = torch.topk(dense_probabilities, self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        return torch.zeros_like(dense_probabilities).scatter(-1, top_indices, top_values)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        leading_shape = value.shape[:-1]
        flat_value = value.reshape(-1, value.shape[-1])
        shared = self._shared_features(flat_value)
        dense_probabilities = self.routing_probabilities_from_features(shared)
        sparse_probabilities = self.sparse_probabilities(dense_probabilities)

        expert_residuals = torch.stack([expert.residual(shared) for expert in self.experts], dim=-2)
        mixed_residual = torch.sum(sparse_probabilities.unsqueeze(-1) * expert_residuals, dim=-2)
        mixed = self.post_norm(shared + mixed_residual)
        output = self.output(mixed)
        return output.reshape(*leading_shape, self.output_dim)

    @property
    def dense_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def maximum_router_entropy(self) -> float:
        return math.log(self.num_experts)
