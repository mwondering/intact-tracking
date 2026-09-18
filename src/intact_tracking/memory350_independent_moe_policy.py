"""Hard-routed residual experts with independent encoders, heads and action noise."""

import copy

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_policy import SCRATCH_ACTION_STD
from intact_tracking.memory350_moe_policy import HardRoutedMLP
from intact_tracking.memory350_moe_critic_action import (
    TrackerActionMoECritic, attach_tracker_action,
)
from intact_tracking.residual_uniform_protocol import UnboundedResidualActor, configure_models

VERSION = "residual_uniform_independent_kmeans16_moe_v1"
ACTOR_CLASS = "intact_tracking.memory350_independent_moe_policy:IndependentMoEActor"
CRITIC_CLASS = "intact_tracking.memory350_independent_moe_policy:IndependentMoECritic"


class CompleteExpert(nn.Module):
    def __init__(self, observation, head, feature_dim, tracker_action_dim):
        super().__init__()
        self.observation, self.head = observation, head
        self.feature_dim, self.tracker_action_dim = feature_dim, tracker_action_dim

    def forward(self, value):
        encoded = self.observation(value[..., :self.feature_dim])
        action = value[..., self.feature_dim:self.feature_dim + self.tracker_action_dim].detach()
        return self.head(torch.cat((encoded, action), -1))


class IndependentHardRoutedMLP(nn.Module):
    def __init__(self, feature_dim, output_dim, *, compression_dims, fusion="concat",
                 tracker_action_dim=29, seed=0, zero_output=False, num_experts=16,
                 center_rate=0.01, max_switch_fraction=0.02):
        super().__init__()
        if fusion != "concat":
            raise ValueError("Independent-expert comparison requires latent hard routing")
        # Match every initial encoder and head value to the shared-encoder arm.
        # deepcopy supplies independent storage; different training subsets then
        # let encoders specialize without changing initial control behavior.
        source = HardRoutedMLP(feature_dim, output_dim, compression_dims=compression_dims,
            fusion=fusion, tracker_action_dim=tracker_action_dim, seed=seed,
            zero_output=zero_output, num_experts=num_experts, center_rate=center_rate,
            max_switch_fraction=max_switch_fraction)
        self.feature_dim, self.output_dim = feature_dim, output_dim
        self.tracker_action_dim, self.num_experts = tracker_action_dim, num_experts
        self.fusion, self.router = fusion, source.router
        self.experts = nn.ModuleList([
            CompleteExpert(copy.deepcopy(source.observation), head, feature_dim, tracker_action_dim)
            for head in source.heads
        ])

    def forward(self, value, *, return_routes=False):
        stop = self.feature_dim + self.tracker_action_dim
        if value.shape[-1] != stop + 64:
            raise ValueError(f"Expected input width {stop + 64}, got {value.shape[-1]}")
        shape = value.shape[:-1]
        ids = self.router(value[..., stop:]).reshape(-1)
        inputs = value[..., :stop].reshape(-1, stop)
        output = inputs.new_zeros(len(inputs), self.output_dim)
        for i, expert in enumerate(self.experts):
            selected = (ids == i).nonzero().flatten()
            # Route BEFORE the encoder, and retain zero gradient tensors for
            # empty expert batches so all ranks have the same collective layout.
            output = output.index_copy(0, selected, expert(inputs.index_select(0, selected)))
        output = output.reshape(*shape, self.output_dim)
        return (output, ids.reshape(shape)) if return_routes else output


class ExpertGaussianDistribution(GaussianDistribution):
    """The same Gaussian parameterization, with one separate std per expert."""

    def __init__(self, source, num_experts):
        super().__init__(source.output_dim, std_type=source.std_type, std_range=source.std_range)
        key = "std_param" if source.std_type == "scalar" else "log_std_param"
        original = getattr(source, key)
        delattr(self, key)
        self.expert_std_parameters = nn.ParameterList([
            nn.Parameter(original.detach().clone(), requires_grad=original.requires_grad)
            for _ in range(num_experts)
        ])

    def update(self, mlp_output, route_ids=None):
        if route_ids is None or route_ids.shape != mlp_output.shape[:-1]:
            raise ValueError("Gaussian routing must match the current mean observation batch")
        # Stack retains a zero grad tensor on unused expert std parameters too.
        parameters = torch.stack(tuple(self.expert_std_parameters))
        std = parameters.index_select(0, route_ids.reshape(-1)).reshape_as(mlp_output)
        if self.std_type == "scalar":
            std = std.clamp(*self.std_range)
        else:
            std = std.clamp(*self.log_std_range).exp()
        self._distribution = Normal(mlp_output, std)


class IndependentMoEActor(UnboundedResidualActor):
    def __init__(self, *args, fusion_mode="concat", num_experts=16,
                 center_rate=0.01, max_switch_fraction=0.02, **kwargs):
        super().__init__(*args, fusion_mode=fusion_mode, num_experts=num_experts,
                         center_rate=center_rate, max_switch_fraction=max_switch_fraction, **kwargs)
        self.residual_mlp = IndependentHardRoutedMLP(
            1645, self.distribution.output_dim, compression_dims=(512, 256, 128),
            fusion=fusion_mode, seed=self.initialization_seed, zero_output=True,
            num_experts=num_experts, center_rate=center_rate, max_switch_fraction=max_switch_fraction)
        self.distribution = ExpertGaussianDistribution(self.distribution, num_experts)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        features, action = self._base_features_and_action(obs)
        residual, ids = self.residual_mlp(self._residual_input(obs, features, action), return_routes=True)
        mean = action + residual
        self.last_base_action, self.last_residual_mean = action, residual.detach()
        if stochastic_output:
            self.distribution.update(mean, ids)
            return self.distribution.sample()
        return mean


class IndependentMoECritic(TrackerActionMoECritic):
    def __init__(self, *args, fusion_mode="concat", compression_dims=(1024, 512, 256, 128),
                 num_experts=16, center_rate=0.01, max_switch_fraction=0.02, **kwargs):
        super().__init__(*args, fusion_mode=fusion_mode, compression_dims=compression_dims,
                         num_experts=num_experts, center_rate=center_rate,
                         max_switch_fraction=max_switch_fraction, **kwargs)
        self.mlp = IndependentHardRoutedMLP(
            self.obs_dim, 1, compression_dims=compression_dims, fusion=fusion_mode,
            seed=self.initialization_seed, num_experts=num_experts,
            center_rate=center_rate, max_switch_fraction=max_switch_fraction)


def configure_independent_models(train, fusion, **kwargs):
    if fusion != "concat":
        raise ValueError("Use the existing baseline checkpoint for this comparison")
    result = configure_models(train, fusion, **kwargs)
    result["actor"]["class_name"] = ACTOR_CLASS
    result["critic"]["class_name"] = CRITIC_CLASS
    return result


def audit_independent_models(actor, critic, obs, fusion):
    if fusion != "concat":
        raise ValueError(fusion)
    with torch.no_grad():
        features, base = actor._base_features_and_action(obs)
        attach_tracker_action(obs, base)
        action = actor(obs)
        torch.testing.assert_close(action, base, atol=0, rtol=0)
        ids = actor.residual_mlp.router(obs[actor.dynamics_latent_group])
        actor.distribution.update(action, ids)
        torch.testing.assert_close(actor.output_std, torch.full_like(action, SCRATCH_ACTION_STD), atol=0, rtol=0)
        a, c = actor.residual_mlp.router, critic.mlp.router
        assert bool(a.initialized) and bool(c.initialized)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, c.state_dict()[key], atol=0, rtol=0)
        alternate = obs.clone()
        alternate["dynamics_latent"] = torch.randn_like(obs["dynamics_latent"])
        torch.testing.assert_close(actor(alternate), action, atol=0, rtol=0)
        torch.testing.assert_close(critic(alternate), critic(obs), atol=2e-6, rtol=2e-6)
        actor_input = actor._residual_input(obs, features, base)
        torch.testing.assert_close(actor_input[..., 1645:1674], base, atol=0, rtol=0)
        torch.testing.assert_close(critic.value_input(obs)[..., critic.obs_dim:critic.obs_dim + 29], base, atol=0, rtol=0)

    seen = set()
    for i in range(16):
        parameters = (list(actor.residual_mlp.experts[i].parameters()) +
                      [actor.distribution.expert_std_parameters[i]] + list(critic.mlp.experts[i].parameters()))
        for p in parameters:
            if id(p) in seen:
                raise AssertionError("Experts and actor/critic must have disjoint trainable parameters")
            seen.add(id(p))
    all_trainable = {id(p) for model in (actor, critic) for p in model.parameters() if p.requires_grad}
    if seen != all_trainable:
        raise AssertionError("A trainable parameter lives outside the independent experts")
    return {
        "actor_original_features": 1645, "critic_original_features": critic.obs_dim,
        "actor_compression": [1645, 512, 256, 128],
        "critic_compression": [critic.obs_dim, 1024, 512, 256, 128],
        "actor_head": [157, 256, 128, 29], "critic_head": [157, 256, 128, 1],
        "actor_experts": 16, "critic_experts": 16,
        "actor_critic_parameters_shared": False, "expert_trainable_parameters_shared": False,
        "each_side_shares_one_obs_encoder_across_its_experts": False,
        "exploration_std_shared_across_experts": False,
        "all_trainable_parameters_owned_by_one_expert": True,
        "critic_observation_normalization": "Common pooled running moments; buffers only, no trainable parameters",
        "latent_dimensions": 64, "latent_use": "online K-means hard routing only",
        "router_has_trainable_parameters": False, "dispatch_before_observation_encoding": True,
        "identical_initial_action": True, "initial_value_independent_of_latent": True,
        "tracker_action_input_dim": 29, "critic_tracker_action_input_dim": 29,
        "critic_tracker_action_matches_actor": True,
        "critic_bootstrap_tracker_action": "recomputed on final next-state observation",
        "action_std_initialization": SCRATCH_ACTION_STD,
        "actor_initialization_seed": actor.initialization_seed,
        "critic_initialization_seed": critic.initialization_seed,
        "actor_compressor_sha256_by_expert": [tensor_digest(e.observation.named_parameters()) for e in actor.residual_mlp.experts],
        "critic_compressor_sha256_by_expert": [tensor_digest(e.observation.named_parameters()) for e in critic.mlp.experts],
        "critic_initial_normalizer_sha256": tensor_digest(critic.obs_normalizer.state_dict().items()),
        "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
        "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad),
        "routing_bootstrap": copy.deepcopy(getattr(actor, "routing_bootstrap", None)),
    }
