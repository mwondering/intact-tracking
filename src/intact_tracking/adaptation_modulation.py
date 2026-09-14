"""Latent-conditioned hidden layers for a single shared residual controller."""

from __future__ import annotations

import copy

import torch
from torch import nn


class _ModulatedLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.condition = nn.Linear(latent_dim, 2 * output_dim)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, value: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.elu(self.linear(value))
        gain, bias = self.condition(latent).chunk(2, dim=-1)
        return hidden * (1 + 0.5 * gain.tanh()) + 0.5 * bias.tanh()


class LatentModulatedResidual(nn.Module):
    """Concatenation plus bounded feature-wise latent conditioning at each layer.

    Every environment uses the same weights; there are no per-world policies or
    lookup tables. Latent can come from privileges (teacher) or measured history
    (student). The zero output layer preserves the frozen tracker at creation.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims, latent_dim: int):
        super().__init__()
        if not hidden_dims or latent_dim < 1 or input_dim <= latent_dim:
            raise ValueError("Modulated residual requires features, a latent, and hidden layers")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.layers = nn.ModuleList()
        width = input_dim
        for hidden in hidden_dims:
            self.layers.append(_ModulatedLayer(width, int(hidden), latent_dim))
            width = int(hidden)
        # Register LAST: the existing zero-output initialization helper must
        # find the action output, not a hidden conditioning head.
        self.output = nn.Linear(width, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.input_dim:
            raise ValueError("Unexpected modulated residual input width")
        latent = value[..., -self.latent_dim:]
        for layer in self.layers:
            value = layer(value, latent)
        return self.output(value)


class PhysicsBilinearResidual(nn.Module):
    """Learn feature-dependent physical sensitivities, with zero nominal offset.

    This is one network, not a set of environment-specific controllers. For a
    true zero physics code the correction is structurally zero regardless of
    learned weights. That fact is NOT a no-regression guarantee in DR, nor for
    a student's imperfect predicted physics code.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims, latent_dim: int):
        super().__init__()
        from rsl_rl.modules import MLP

        self.feature_dim = int(input_dim - latent_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.coefficients = MLP(self.feature_dim, output_dim * latent_dim, hidden_dims, "elu")
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.feature_dim + self.latent_dim:
            raise ValueError("Unexpected bilinear residual input width")
        features, code = value[..., :self.feature_dim], value[..., self.feature_dim:]
        coefficients = self.coefficients(features).unflatten(-1, (self.output_dim, self.latent_dim))
        return (coefficients * code.unsqueeze(-2)).sum(-1)


class _PhysicsLowRankLayer(nn.Module):
    def __init__(self, linear, activation, latent_dim: int, rank_per_physics: int, shared_rank: int = 0):
        super().__init__()
        self.base = copy.deepcopy(linear).requires_grad_(False)
        self.activation = copy.deepcopy(activation)
        self.rank_per_physics = int(rank_per_physics)
        rank = latent_dim * rank_per_physics
        self.down = nn.Linear(linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.shared_down = nn.Linear(linear.in_features, shared_rank, bias=False) if shared_rank else None
        self.shared_up = nn.Linear(shared_rank, linear.out_features, bias=False) if shared_rank else None
        if self.shared_up is not None:
            nn.init.zeros_(self.shared_up.weight)

    def forward(self, value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gate = code.repeat_interleave(self.rank_per_physics, dim=-1)
        correction = self.up(self.down(value) * gate)
        if self.shared_down is not None and self.shared_up is not None:
            correction = correction + self.shared_up(self.shared_down(value))
        return self.activation(self.base(value) + correction)


class PhysicsLowRankResidual(nn.Module):
    """Physical code modulates low-rank updates throughout the frozen base MLP.

    Outputs the conditional controller's difference from the original MLP; the
    standard residual action bound still applies. All original weights remain
    frozen. A zero physical code disables conditional updates; an explicitly
    configured shared update remains active and can learn common DR behavior.
    """

    def __init__(self, base_mlp, latent_dim: int, rank_per_physics: int = 2, shared_rank: int = 0):
        super().__init__()
        if latent_dim < 1 or rank_per_physics < 1 or shared_rank < 0:
            raise ValueError("Positive code dimension and low-rank factor required")
        self.latent_dim = int(latent_dim)
        self.shared_rank = int(shared_rank)
        self.reference = copy.deepcopy(base_mlp).requires_grad_(False)
        # Sequential iteration preserves repeated shared activation modules;
        # children() de-duplicates them and would silently lose ELU layers.
        modules = list(base_mlp)
        self.feature_dim = modules[0].in_features
        self.layers = nn.ModuleList()
        index = 0
        while index < len(modules):
            linear = modules[index]
            if not isinstance(linear, nn.Linear):
                raise ValueError("Low-rank frontend requires alternating Linear/ELU base layers")
            activation = nn.Identity()
            if index + 1 < len(modules):
                if not isinstance(modules[index + 1], nn.ELU):
                    raise ValueError("Low-rank frontend currently supports ELU activation only")
                activation = modules[index + 1]
                index += 1
            self.layers.append(_PhysicsLowRankLayer(linear, activation, latent_dim, rank_per_physics, self.shared_rank))
            index += 1

    def freeze_reference(self):
        self.reference.requires_grad_(False)
        for layer in self.layers:
            layer.base.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.feature_dim + self.latent_dim:
            raise ValueError("Unexpected low-rank residual input width")
        features, code = value[..., :self.feature_dim], value[..., self.feature_dim:]
        # Parameters are frozen, but a deployable feature adapter may need
        # the derivative of BOTH branches with respect to its input. Detaching
        # this branch would give an incorrect input gradient for the difference.
        original = self.reference(features)
        result = features
        for layer in self.layers:
            result = layer(result, code)
        return result - original
