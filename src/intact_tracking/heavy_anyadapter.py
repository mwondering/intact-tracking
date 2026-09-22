"""Full-rank layer adapters in the frozen SPV5-2A action MLP."""

import torch
from torch import nn
from rsl_rl.utils import unpad_trajectories

from intact_tracking.anyadapter_world_model import HistoryEncoder, EMBEDDING_DIM
from intact_tracking.heavy_rma_teacher import CachedTrackerActor
from intact_tracking.limb_context_policy import LimbContextCritic
from intact_tracking.memory350_tracker_action_policy import ACTION_GROUP, _input
from intact_tracking.residual_policy import DYNAMICS_LATENT_GROUP


class LayerAdapters(nn.Module):
    def __init__(self, base, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        linears = [m for m in base if isinstance(m, nn.Linear)]
        if not linears or any(not isinstance(m, (nn.Linear, nn.ELU, nn.Mish, nn.ReLU, nn.SiLU, nn.Tanh)) for m in base):
            raise ValueError("Expected a feed-forward linear/activation tracker MLP")
        self.layers = nn.ModuleList([nn.Linear(embedding_dim if i == 0 else layer.in_features,
                                              layer.out_features)
                                     for i, layer in enumerate(linears)])
        for layer in self.layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, base, features, embedding):
        value, index = features, 0
        for layer in base:
            if isinstance(layer, nn.Linear):
                # Frozen W still propagates dL/dinput into earlier adapters.
                value = layer(value) + self.layers[index](embedding if index == 0 else value)
                index += 1
            else:
                value = layer(value)
        return value


class AnyAdapterActor(CachedTrackerActor):
    def __init__(self, obs, *args, initialization_seed=None, initial_action_std=1., **kwargs):
        kwargs.pop("fusion_mode", None)
        kwargs.pop("dynamics_latent_dim", None)
        super().__init__(obs, *args, fusion_mode="baseline", initialization_seed=initialization_seed,
                         initial_action_std=initial_action_std, **kwargs)
        del self.residual_mlp
        self.use_dynamics_latent = True
        self.dynamics_latent_dim = EMBEDDING_DIM
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed or 0) + 31)
            self.adapters = LayerAdapters(self.tracker.mlp)
            self.history_encoder = HistoryEncoder().requires_grad_(False)
        self.residual_input_dim = self.tracker.policy_input_dim + EMBEDDING_DIM

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        embedding = (_input(obs, DYNAMICS_LATENT_GROUP, EMBEDDING_DIM)
                     if latent_override is None else latent_override.detach())
        return torch.cat((tracker_features, embedding.to(tracker_features)), -1)

    def adapted_mean(self, features, embedding):
        return self.tracker.distribution.deterministic_output(self.adapters(self.tracker.mlp, features, embedding))

    def _residual(self, value):
        features, embedding = value[..., :-EMBEDDING_DIM], value[..., -EMBEDDING_DIM:]
        with torch.no_grad():
            base = self.tracker.distribution.deterministic_output(self.tracker.mlp(features))
        return self.adapted_mean(features, embedding) - base

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        features, base = self._base_features_and_action(obs)
        embedding = _input(obs, DYNAMICS_LATENT_GROUP, EMBEDDING_DIM).to(features)
        mean = self.adapted_mean(features, embedding)
        self.last_base_action, self.last_residual_mean = base, (mean - base).detach()
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def policy_metrics(self, obs):
        result = super().policy_metrics(obs)
        result["Adapter/effective_action_delta_rms"] = result["residual_action_rms"]
        result["Adapter/embedding_rms"] = float(obs[DYNAMICS_LATENT_GROUP].square().mean().sqrt())
        return result


class AnyAdapterCritic(LimbContextCritic):
    def __init__(self, obs, *args, initialization_seed=None, **kwargs):
        kwargs.pop("fusion_mode", None)
        kwargs.pop("dynamics_latent_dim", None)
        super().__init__(obs, *args, fusion_mode="baseline", initialization_seed=initialization_seed, **kwargs)
        self.value_input_dim = self.obs_dim + EMBEDDING_DIM + 29
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed or 0) + 7)
            self.mlp[0] = nn.Linear(self.value_input_dim, self.mlp[0].out_features)

    def value_input(self, obs):
        features = self.obs_normalizer(self._flat_obs(obs))
        return torch.cat((features, _input(obs, DYNAMICS_LATENT_GROUP, EMBEDDING_DIM).to(features),
                          _input(obs, ACTION_GROUP, 29).to(features)), -1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.mlp(self.value_input(obs))
