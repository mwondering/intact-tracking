"""Independent actor/critic encoders with sixteen online-K-means expert heads."""

import copy

import torch
from torch import nn
from rsl_rl.modules import MLP

from intact_tracking.limb_context_policy import LimbContextCritic, SCRATCH_ACTION_STD
from intact_tracking.memory350_compressed_policy import CompressedContextActor, configure_compressed_models
from intact_tracking.memory350_online_kmeans import OnlineKMeansRouter
from intact_tracking.limb_context_distributed import tensor_digest

VERSION = "memory350_online_kmeans16_hard_moe_residual_v1"
ACTOR_CLASS = "intact_tracking.memory350_moe_policy:HardMoEActor"
CRITIC_CLASS = "intact_tracking.memory350_moe_policy:HardMoECritic"


class HardRoutedMLP(nn.Module):
    def __init__(self, feature_dim, output_dim, *, compression_dims, fusion,
                 tracker_action_dim=0, seed=0, zero_output=False, num_experts=16,
                 center_rate=0.01, max_switch_fraction=0.02):
        super().__init__()
        if fusion not in ("baseline", "concat") or num_experts != 16:
            raise ValueError("This experiment compares one baseline head with sixteen routed heads")
        self.feature_dim, self.tracker_action_dim = feature_dim, tracker_action_dim
        self.output_dim, self.fusion = output_dim, fusion
        self.num_experts = 1 if fusion == "baseline" else num_experts
        self.router = (OnlineKMeansRouter(num_experts, 64, center_rate, max_switch_fraction)
                       if fusion == "concat" else None)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.observation = nn.Sequential(
                MLP(feature_dim, 128, list(compression_dims[:-1]), "elu"), nn.ELU())
            self.heads = nn.ModuleList()
            for i in range(self.num_experts):
                torch.random.default_generator.manual_seed(seed + 1000 + i)
                head = MLP(128 + tracker_action_dim, output_dim, [256, 128], "elu")
                if zero_output:
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
                elif i:
                    # Identical initial V across routes and arms. Hard routing
                    # supplies different training subsets, breaking symmetry.
                    head.load_state_dict(self.heads[0].state_dict())
                self.heads.append(head)

    def forward(self, value):
        stop = self.feature_dim + self.tracker_action_dim
        expected = stop + (64 if self.router is not None else 0)
        if value.shape[-1] != expected:
            raise ValueError(f"Expected input width {expected}, got {value.shape[-1]}")
        h = self.observation(value[..., :self.feature_dim])
        if self.tracker_action_dim:
            h = torch.cat((h, value[..., self.feature_dim:stop].detach()), -1)
        if self.router is None:
            return self.heads[0](h)
        ids = self.router(value[..., stop:]).reshape(-1)
        shape = h.shape[:-1]
        h = h.reshape(-1, h.shape[-1])
        output = h.new_zeros(len(h), self.output_dim)
        for i, head in enumerate(self.heads):
            selected = (ids == i).nonzero().flatten()
            # Execute empty subbatches too: every head gets a grad tensor on
            # every rank, even if its local count is zero. All-reduce layouts
            # must never depend on a rank's assignment histogram.
            output = output.index_copy(0, selected, head(h.index_select(0, selected)))
        return output.reshape(*shape, self.output_dim)


class HardMoEActor(CompressedContextActor):
    def __init__(self, *args, fusion_mode="baseline", num_experts=16,
                 center_rate=0.01, max_switch_fraction=0.02, **kwargs):
        kwargs["tracker_action_input"] = True
        super().__init__(*args, fusion_mode=fusion_mode, **kwargs)
        self.residual_mlp = HardRoutedMLP(
            1645, self.distribution.output_dim, compression_dims=(512, 256, 128),
            fusion=fusion_mode, tracker_action_dim=29, seed=self.initialization_seed,
            zero_output=True, num_experts=num_experts, center_rate=center_rate,
            max_switch_fraction=max_switch_fraction)

    @torch.no_grad()
    def policy_metrics(self, obs):
        metrics = super().policy_metrics(obs)
        router = self.residual_mlp.router
        if router is not None:
            ids = router(obs[self.dynamics_latent_group])
            fractions = torch.bincount(ids, minlength=router.k).float() / len(ids)
            metrics.update({f"router_current_fraction_{i:02d}": float(v) for i, v in enumerate(fractions)})
            metrics["router_current_effective_experts"] = float(
                (-fractions * fractions.clamp_min(1e-12).log()).sum().exp())
        return metrics


class HardMoECritic(LimbContextCritic):
    def __init__(self, *args, fusion_mode="baseline", compression_dims=(1024, 512, 256, 128),
                 head_dims=(256, 128), num_experts=16, center_rate=0.01,
                 max_switch_fraction=0.02, **kwargs):
        del head_dims
        super().__init__(*args, fusion_mode="baseline", **kwargs)
        self.mlp = HardRoutedMLP(
            self.obs_dim, 1, compression_dims=compression_dims,
            fusion=fusion_mode, seed=self.initialization_seed, num_experts=num_experts,
            center_rate=center_rate, max_switch_fraction=max_switch_fraction)
        self.fusion_mode = fusion_mode


def configure_moe_models(train, fusion, *, scratch_seed=None, num_experts=16,
                         center_rate=0.01, max_switch_fraction=0.02):
    result = configure_compressed_models(train, fusion, scratch_seed=scratch_seed,
                                         tracker_action_input=True)
    for key, cls in (("actor", ACTOR_CLASS), ("critic", CRITIC_CLASS)):
        result[key].update(class_name=cls, num_experts=num_experts, center_rate=center_rate,
                           max_switch_fraction=max_switch_fraction)
    return result


def audit_initial_models(actor, critic, obs, fusion):
    with torch.no_grad():
        _, base = actor._base_features_and_action(obs)
        action = actor(obs)
        torch.testing.assert_close(action, base, atol=0, rtol=0)
        actor.distribution.update(action)
        torch.testing.assert_close(actor.output_std, torch.full_like(actor.output_std, SCRATCH_ACTION_STD),
                                   atol=0, rtol=0)
        actor_params = {id(p) for p in actor.parameters()}
        critic_params = {id(p) for p in critic.parameters()}
        if actor_params & critic_params:
            raise AssertionError("Actor and critic must not share parameters")
        if fusion == "concat":
            a, c = actor.residual_mlp.router, critic.mlp.router
            assert bool(a.initialized) and bool(c.initialized)
            torch.testing.assert_close(a.centers, c.centers, atol=0, rtol=0)
            torch.testing.assert_close(a(obs["dynamics_latent"]), c(obs["dynamics_latent"]), atol=0, rtol=0)
            alternate = obs.clone()
            alternate["dynamics_latent"] = torch.randn_like(obs["dynamics_latent"])
            torch.testing.assert_close(actor(alternate), action, atol=0, rtol=0)
            # Sparse subbatch GEMMs may round slightly differently in IEEE FP32.
            torch.testing.assert_close(critic(alternate), critic(obs), atol=2e-6, rtol=2e-6)
    return {"actor_original_features": 1645, "critic_original_features": critic.obs_dim,
            "actor_compression": [1645, 512, 256, 128],
            "critic_compression": [critic.obs_dim, 1024, 512, 256, 128],
            "actor_head": [157, 256, 128, 29], "critic_head": [128, 256, 128, 1],
            "actor_experts": actor.residual_mlp.num_experts,
            "critic_experts": critic.mlp.num_experts,
            "actor_critic_parameters_shared": False,
            "each_side_shares_one_obs_encoder_across_its_experts": True,
            "latent_dimensions": 64 if fusion == "concat" else 0,
            "latent_use": "online K-means hard routing only" if fusion == "concat" else "absent",
            "router_has_trainable_parameters": False,
            "identical_initial_action": True, "initial_value_independent_of_latent": True,
            "tracker_action_input_dim": 29, "action_std_initialization": SCRATCH_ACTION_STD,
            "actor_initialization_seed": actor.initialization_seed,
            "critic_initialization_seed": critic.initialization_seed,
            "actor_compressor_sha256": tensor_digest(actor.residual_mlp.observation.named_parameters()),
            "critic_compressor_sha256": tensor_digest(critic.mlp.observation.named_parameters()),
            "critic_initial_normalizer_sha256": tensor_digest(critic.obs_normalizer.state_dict().items()),
            "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
            "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad),
            "routing_bootstrap": copy.deepcopy(getattr(actor, "routing_bootstrap", None))}
