"""Same checkpoint inputs, with optional learned or constant context conditioning."""

from __future__ import annotations

import torch
from torch import nn
from rsl_rl.utils import unpad_trajectories
from rsl_rl.modules.distribution import GaussianDistribution

from intact_tracking.residual_policy import (
    DYNAMICS_LATENT_GROUP, TRACKER_FEATURES_INPUT,
    FrozenTrackerResidualActor, WarmStartedHeftCritic,
)

FUSIONS = ("baseline", "film", "concat", "constant")
SCRATCH_INITIALIZATION = "residual_actor_and_critic_from_scratch_v3"
SCRATCH_ACTION_STD = 0.25


class ConditionedMLP(nn.Module):
    def __init__(self, base, latent_dim, fusion, width=256, strength=0.5):
        super().__init__()
        if fusion not in ("film", "concat", "constant"):
            raise ValueError(fusion)
        self.base = base
        self.latent_dim = int(latent_dim)
        self.feature_dim = base[0].in_features
        self.fusion = fusion
        self.strength = float(strength)
        if fusion == "concat":
            original = base[0]
            # Exactly [W_x W_z] [x; z] + b, keeping the source GEMM shape so
            # zero W_z preserves the initial value even under TF32 kernels.
            self.latent_input = nn.Linear(latent_dim, original.out_features, bias=False,
                                         device=original.weight.device, dtype=original.weight.dtype)
            nn.init.zeros_(self.latent_input.weight)
        else:
            factory = {"device": base[0].weight.device, "dtype": base[0].weight.dtype}
            self.condition = nn.Sequential(nn.Linear(latent_dim, width, **factory), nn.ELU())
            hidden_widths = [layer.out_features for layer in list(base)[:-1] if isinstance(layer, nn.Linear)]
            self.heads = nn.ModuleList([nn.Linear(width, 2 * size, **factory) for size in hidden_widths])
            for head in self.heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, value):
        features, latent = value[..., :self.feature_dim], value[..., self.feature_dim:]
        if latent.shape[-1] != self.latent_dim:
            raise ValueError("Incorrect context input width")
        latent = latent.detach()
        if self.fusion == "constant":
            latent = torch.zeros_like(latent)
        if self.fusion == "concat":
            features = self.base[0](features) + self.latent_input(latent)
            for layer in list(self.base)[1:]:
                features = layer(features)
            return features
        condition = self.condition(latent)
        linear_index = 0
        for layer in self.base:
            if isinstance(layer, nn.Linear):
                if linear_index:
                    # The preceding hidden block has finished its activation/normalization.
                    gain, bias = self.heads[linear_index - 1](condition).chunk(2, -1)
                    features = features * (1 + self.strength * gain.tanh()) + self.strength * bias.tanh()
                linear_index += 1
            features = layer(features)
        return features


class LimbContextResidualActor(FrozenTrackerResidualActor):
    def __init__(self, *args, fusion_mode="baseline", dynamics_latent_dim=64,
                 initialization_seed=None, initial_action_std=None, **kwargs):
        if fusion_mode not in FUSIONS:
            raise ValueError(fusion_mode)
        kwargs.pop("use_dynamics_latent", None)
        kwargs["residual_input_mode"] = TRACKER_FEATURES_INPUT
        # Build the common trunk first: shared seeds give identical original weights.
        super().__init__(*args, use_dynamics_latent=False,
                         dynamics_latent_dim=dynamics_latent_dim, **kwargs)
        self.initialization_seed = initialization_seed
        self.initial_action_std = initial_action_std
        if initialization_seed is not None:
            # Dedicated CPU RNG keeps the common trunk independent of FiLM's
            # parameter count. No CUDA generator or environment RNG is changed.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(initialization_seed))
                layers = [m for m in self.residual_mlp.modules() if isinstance(m, nn.Linear)]
                for layer in layers:
                    layer.reset_parameters()
                # A fresh residual policy starts with the frozen tracker's mean.
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
        if initial_action_std is not None:
            cfg = dict(kwargs.get("distribution_cfg") or {})
            cfg.pop("class_name", None)
            cfg["init_std"] = float(initial_action_std)
            self.distribution = GaussianDistribution(self.distribution.output_dim, **cfg)
        self.fusion_mode = fusion_mode
        if fusion_mode != "baseline":
            self.residual_mlp = ConditionedMLP(self.residual_mlp, dynamics_latent_dim, fusion_mode)
            self.use_dynamics_latent = True
            self.residual_input_dim += dynamics_latent_dim

    @torch.no_grad()
    def policy_metrics(self, obs):
        result = super().policy_metrics(obs)
        result["residual_saturation_fraction"] = float(
            (self.last_residual_mean.abs() >= 0.95 * self.residual_scale).float().mean()
        ) if self.last_residual_mean is not None else 0.0
        return result


class LimbContextCritic(WarmStartedHeftCritic):
    def __init__(self, *args, fusion_mode="baseline", dynamics_latent_dim=64,
                 initialization_seed=None, **kwargs):
        if fusion_mode not in FUSIONS:
            raise ValueError(fusion_mode)
        if initialization_seed is None:
            super().__init__(*args, **kwargs)
        else:
            if kwargs.get("initial_checkpoint") is not None:
                raise ValueError("Scratch critic must not load pretrained weights or normalization")
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(initialization_seed))
                super().__init__(*args, **kwargs)
        self.initialization_seed = initialization_seed
        self.fusion_mode = fusion_mode
        if fusion_mode != "baseline":
            self.mlp = ConditionedMLP(self.mlp, dynamics_latent_dim, fusion_mode)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        features = self.obs_normalizer(self._flat_obs(obs))
        if self.fusion_mode != "baseline":
            features = torch.cat((features, obs[DYNAMICS_LATENT_GROUP].detach()), -1)
        return self.mlp(features)


def configure_context_models(train, fusion, *, scratch_seed=None):
    import copy

    result = copy.deepcopy(train)
    result["actor"].update(class_name="intact_tracking.limb_context_policy:LimbContextResidualActor",
                           fusion_mode=fusion, dynamics_latent_dim=64,
                           residual_input_mode=TRACKER_FEATURES_INPUT)
    result["critic"].update(class_name="intact_tracking.limb_context_policy:LimbContextCritic",
                            fusion_mode=fusion, dynamics_latent_dim=64)
    if scratch_seed is not None:
        result["actor"].update(initialization_seed=int(scratch_seed) + 10007,
                               initial_action_std=SCRATCH_ACTION_STD)
        result["critic"].update(initial_checkpoint=None,
                                initialization_seed=int(scratch_seed) + 20003)
    return result
