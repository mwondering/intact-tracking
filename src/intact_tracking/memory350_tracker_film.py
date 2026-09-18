"""Observation/context FiLM in a frozen tracker, with a trained concat critic."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_policy import SCRATCH_ACTION_STD
from intact_tracking.memory350_compressed_policy import CompressedContextCritic
from intact_tracking.residual_policy import FrozenTrackerResidualActor, DYNAMICS_LATENT_GROUP

VERSION = "memory350_frozen_tracker_obs_latent_film_v1"
INITIALIZATION = "pretrained_tracker_identity_film_scratch_concat_critic_v1"
ACTOR_CLASS = "intact_tracking.memory350_tracker_film:TrackerFiLMActor"
CRITIC_CLASS = "intact_tracking.memory350_tracker_film:TrackerFiLMCritic"
CONDITIONING = ("latent", "constant")
TRACKER_WIDTHS = (1645, 2048, 2048, 1024, 1024, 512, 256, 128, 29)


def unit_context(latent):
    """A frozen encoder's direction; never propagate actor/value gradients to it."""
    return F.normalize(latent.detach().float(), dim=-1, eps=1e-8)


class ObservationContextFiLM(nn.Module):
    """Owns only adapters, never a second copy/registration of the frozen MLP."""

    def __init__(self, hidden_widths, *, observation_dim=1645,
                 compression_dims=(512, 256, 128), latent_dim=64, condition_width=128,
                 layer_indices=(4, 5, 6), conditioning="latent", seed=10128):
        super().__init__()
        if conditioning not in CONDITIONING:
            raise ValueError(f"Unknown actor conditioning: {conditioning}")
        indices = tuple(int(i) for i in layer_indices)
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("FiLM layer indices must be nonempty, unique and increasing")
        if min(indices) < 0 or max(indices) >= len(hidden_widths):
            raise ValueError("FiLM may modulate hidden layers only")
        self.layer_indices, self.conditioning = indices, conditioning
        self.latent_dim = int(latent_dim)
        self.observation_dim = int(observation_dim)
        # Both arms learn observation-dependent modulation. Only access to the
        # real environment code differs; all weights and dimensions are matched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.observation = nn.Sequential(MLP(observation_dim, compression_dims[-1],
                list(compression_dims[:-1]), "elu"), nn.ELU())
            self.condition = nn.Sequential(
                nn.Linear(compression_dims[-1] + latent_dim, condition_width), nn.ELU())
            self.heads = nn.ModuleDict({str(i): nn.Linear(condition_width, 2 * hidden_widths[i])
                                        for i in indices})
            for head in self.heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.register_buffer("constant_context", torch.zeros(latent_dim))

    def coefficients(self, features, latent):
        if features.shape[-1] != self.observation_dim or features.shape[:-1] != latent.shape[:-1]:
            raise ValueError("FiLM observation and latent batches must match")
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"FiLM requires {self.latent_dim} latent coordinates")
        if self.conditioning == "constant":
            value = self.constant_context.expand_as(latent)
        else:
            value = unit_context(latent)
        value = value.to(self.condition[0].weight.dtype)
        compressed = self.observation(features.detach())
        hidden = self.condition(torch.cat((compressed, value), -1))
        return {index: self.heads[str(index)](hidden).chunk(2, -1) for index in self.layer_indices}

    def forward(self, frozen_mlp, features, latent):
        coefficients = self.coefficients(features, latent)
        hidden_index, adapted = -1, False
        for layer in frozen_mlp:
            if isinstance(layer, nn.Linear):
                # Apply after the preceding hidden layer's activation, before
                # the next Linear. The output/action layer is never modulated.
                if hidden_index in coefficients:
                    gain, bias = coefficients[hidden_index]
                    features = features * (1 + gain) + bias
                    adapted = True
                hidden_index += 1
            if adapted:
                # Frozen weights must still transmit gradients to earlier FiLM.
                features = layer(features)
            else:
                with torch.no_grad():
                    features = layer(features)
        return features


class TrackerFiLMActor(FrozenTrackerResidualActor):
    """Directly adapts the source tracker; there is no residual action network."""

    def __init__(self, *args, actor_conditioning="latent", initialization_seed=10128,
                 initial_action_std=SCRATCH_ACTION_STD, film_condition_width=128,
                 film_layer_indices=(4, 5, 6), **kwargs):
        kwargs["use_dynamics_latent"] = True
        kwargs["residual_input_mode"] = "tracker_features"
        super().__init__(*args, **kwargs)
        del self.residual_mlp
        widths = tuple(layer.out_features for layer in self.tracker.mlp if isinstance(layer, nn.Linear))
        if (self.tracker.policy_input_dim, *widths) != TRACKER_WIDTHS:
            raise ValueError("Tracker FiLM requires the audited SPV5-2A tracker architecture")
        if tuple(film_layer_indices) != (4, 5, 6):
            raise ValueError("This experiment modulates exactly the 512/256/128 hidden layers")
        self.actor_conditioning = actor_conditioning
        self.initialization_seed = int(initialization_seed)
        self.initial_action_std = float(initial_action_std)
        self.film = ObservationContextFiLM(widths[:-1], conditioning=actor_conditioning,
            condition_width=film_condition_width, layer_indices=film_layer_indices,
            seed=initialization_seed)
        cfg = copy.deepcopy(dict(kwargs.get("distribution_cfg") or {}))
        cfg.pop("class_name", None)
        cfg["init_std"] = self.initial_action_std
        self.distribution = GaussianDistribution(widths[-1], **cfg)
        self.frozen_tracker_digest = tensor_digest(self.tracker.state_dict().items())

    def action_mean(self, features, latent):
        output = self.film(self.tracker.mlp, features, latent)
        return self.tracker.distribution.deterministic_output(output)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        with torch.no_grad():
            features = self.tracker.get_latent(obs).detach()
        latent = obs[self.dynamics_latent_group]
        mean = self.action_mean(features, latent)
        self.last_dynamics_latent = latent.detach()
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def assert_tracker_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.tracker.parameters()):
            raise RuntimeError("Source tracker received trainable parameters or gradients")
        if tensor_digest(self.tracker.state_dict().items()) != self.frozen_tracker_digest:
            raise RuntimeError("Source tracker weights or normalization buffers changed")

    @torch.no_grad()
    def policy_metrics(self, obs):
        features, original = self._base_features_and_action(obs)
        latent = obs[self.dynamics_latent_group]
        adapted = self.action_mean(features, latent)
        shuffled = self.action_mean(features, latent.roll(1, dims=0))
        constant = self.action_mean(features, torch.zeros_like(latent))
        coefficients = self.film.coefficients(features, latent)
        result = {
            "adaptation_action_delta_rms": float((adapted - original).square().mean().sqrt()),
            "latent_shuffle_action_delta_rms": float((adapted - shuffled).square().mean().sqrt()),
            "latent_zero_action_delta_rms": float((adapted - constant).square().mean().sqrt()),
            "actor_uses_environment_latent": float(self.actor_conditioning == "latent"),
        }
        for index, (gain, bias) in coefficients.items():
            result[f"film_{index}_delta_gain_rms"] = float(gain.square().mean().sqrt())
            result[f"film_{index}_bias_rms"] = float(bias.square().mean().sqrt())
        return result


class TrackerFiLMCritic(CompressedContextCritic):
    """Trainable compressed MLP; the baseline masks the latent after compression."""

    def __init__(self, *args, fusion_mode="concat", **kwargs):
        if fusion_mode not in ("concat", "baseline"):
            raise ValueError("Critic uses concat latent or a zero latent slot")
        super().__init__(*args, fusion_mode=fusion_mode, **kwargs)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        features = self.obs_normalizer(self._flat_obs(obs))
        if self.fusion_mode == "baseline":
            return self.mlp(features)
        return self.mlp(torch.cat((features, unit_context(obs[DYNAMICS_LATENT_GROUP])), -1))


def configure_models(train, fusion, *, scratch_seed, actor_conditioning="latent"):
    if fusion != "film" or actor_conditioning not in CONDITIONING:
        raise ValueError("Use fusion=film and latent/constant actor conditioning")
    result = copy.deepcopy(train)
    result["actor"].update(class_name=ACTOR_CLASS, use_dynamics_latent=True,
        dynamics_latent_dim=64, actor_conditioning=actor_conditioning,
        initialization_seed=int(scratch_seed) + 10007,
        initial_action_std=SCRATCH_ACTION_STD, film_condition_width=128,
        film_layer_indices=[4, 5, 6])
    result["critic"].update(class_name=CRITIC_CLASS, initial_checkpoint=None,
        initialization_seed=int(scratch_seed) + 20003, dynamics_latent_dim=64,
        fusion_mode="concat" if actor_conditioning == "latent" else "baseline",
        compression_dims=[1024, 512, 256, 128], head_dims=[256, 128])
    return result


def audit_initial_models(actor, critic, obs, fusion):
    if fusion != "film":
        raise ValueError("Unexpected FiLM experiment fusion")
    with torch.no_grad():
        original = actor.tracker(obs)
        actual = actor(obs)
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
        alternate = obs.clone()
        alternate[DYNAMICS_LATENT_GROUP] = torch.randn_like(obs[DYNAMICS_LATENT_GROUP])
        torch.testing.assert_close(actor(alternate), actual, atol=0, rtol=0)
        torch.testing.assert_close(critic(alternate), critic(obs), atol=1e-6, rtol=1e-6)
        actor.distribution.update(actual)
        torch.testing.assert_close(actor.output_std, torch.full_like(actual, SCRATCH_ACTION_STD), atol=0, rtol=0)
    actor.assert_tracker_frozen()
    actor_parameters = {id(p) for p in actor.parameters() if p.requires_grad}
    expected = {id(p) for m in (actor.film, actor.distribution) for p in m.parameters() if p.requires_grad}
    if actor_parameters != expected or actor_parameters & {id(p) for p in critic.parameters()}:
        raise RuntimeError("Trainable actor parameters must be independent FiLM and exploration only")
    if any(not p.requires_grad for p in critic.parameters()):
        raise RuntimeError("The complete critic must be trainable")
    if (critic.fusion_mode == "concat") != (actor.actor_conditioning == "latent"):
        raise RuntimeError("Actor and critic must agree on whether this arm receives latent")
    return {"actor_conditioning": actor.actor_conditioning,
        "actor_original_features": 1645, "critic_original_features": critic.obs_dim,
        "actor_film_input_dim": 192, "actor_film_receives_observations": True,
        "actor_film_observation_compression": [1645, 512, 256, 128],
        "actor_film_latent_input_dim": 64,
        "tracker_mlp_widths": list(TRACKER_WIDTHS), "film_hidden_widths": [512, 256, 128],
        "film_gain_formula": "1 + delta_gamma(compressed_obs, z)",
        "film_bias_formula": "beta(compressed_obs, z)",
        "film_modulation_bounded": False, "residual_network": False,
        "critic_fusion": "6330 -> 1024 -> 512 -> 256 -> 128; then concat 64 latent/zeros -> 256 -> 128 -> 1",
        "critic_has_film": False, "critic_uses_real_latent": critic.fusion_mode == "concat",
        "actor_critic_parameters_shared": False, "latent_normalization": "detached L2 unit vector",
        "constant_actor_code": "compressed observations + 64 zeros; baseline critic also uses 64 zeros",
        "identical_initial_action": True, "identical_initial_value": True,
        "action_std_initialization": SCRATCH_ACTION_STD,
        "tracker_warmup": getattr(actor, "tracker_warmup", None),
        "frozen_tracker_sha256": actor.frozen_tracker_digest,
        "actor_initial_adapter_sha256": tensor_digest(actor.film.state_dict().items()),
        "critic_initial_parameters_sha256": tensor_digest(critic.named_parameters()),
        "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
        "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad)}
