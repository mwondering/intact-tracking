"""RMA teacher on the frozen SPV5-2A backbone (108 physical inputs)."""

import copy

import torch
from torch import nn
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories

from intact_tracking.limb_context_policy import LimbContextResidualActor, LimbContextCritic
from intact_tracking.memory350_tracker_action_policy import ACTION_GROUP, _input

VERSION = "heavy_oracle_and_anyadapter_v1"
PHYSICS_GROUP = "teacher_physics"
PHYSICS_DIM = 108


def embedding_residual_input(features, embedding, base_action):
    """Shared teacher/student interface; accepts encoded64, never raw theta108."""
    if embedding.shape != (*features.shape[:-1], 64):
        raise ValueError('RMA control requires a 64D actor embedding')
    return torch.cat((features, embedding, base_action.detach()), -1)


def parameter_encoder():
    return nn.Sequential(nn.Linear(PHYSICS_DIM, 256), nn.ELU(), nn.Linear(256, 128),
                         nn.ELU(), nn.Linear(128, 64), nn.ELU())


class CachedTrackerActor(LimbContextResidualActor):
    """Cache a real tracker command before RSL derives the rollout schema."""

    @torch.no_grad()
    def populate_tracker_cache(self, obs):
        super().populate_tracker_cache(obs)
        features, action = super()._base_features_and_action(obs)
        obs.set(ACTION_GROUP, action.detach().clone())

    @torch.no_grad()
    def _base_features_and_action(self, obs):
        if ACTION_GROUP not in obs:
            self.populate_tracker_cache(obs)
        features = self.tracker.get_latent(obs).detach()
        return features, _input(obs, ACTION_GROUP, self.distribution.output_dim).to(features)


class RMATeacherActor(CachedTrackerActor):
    def __init__(self, obs, *args, initialization_seed=None, initial_action_std=1.,
                 residual_hidden_dims=(512, 256, 128), **kwargs):
        kwargs.pop("fusion_mode", None)
        kwargs.pop("dynamics_latent_dim", None)
        _input(obs, PHYSICS_GROUP, PHYSICS_DIM)
        super().__init__(obs, *args, fusion_mode="baseline", initialization_seed=initialization_seed,
                         initial_action_std=initial_action_std, residual_hidden_dims=residual_hidden_dims, **kwargs)
        # Store theta in rollout; encode it anew inside EVERY actor minibatch.
        # Do not use ConditionedMLP, whose explicit detach freezes its condition.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed or 0))
            self.dr_encoder = parameter_encoder()
            self.residual_input_dim = self.tracker.policy_input_dim + 29 + 64
            self.residual_mlp = MLP(self.residual_input_dim, 29, list(residual_hidden_dims), "elu")
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        theta = _input(obs, PHYSICS_GROUP, PHYSICS_DIM).to(tracker_features)
        embedding = self.dr_encoder(theta if latent_override is None else latent_override)
        return embedding_residual_input(tracker_features, embedding, base_action)

    @torch.no_grad()
    def policy_metrics(self, obs):
        result = super().policy_metrics(obs)
        result.pop("latent_shuffle_action_delta_rms", None)
        result.pop("latent_zero_action_delta_rms", None)
        theta = _input(obs, PHYSICS_GROUP, PHYSICS_DIM)
        features, base = self._base_features_and_action(obs)
        normal = self._residual(self._residual_input(obs, features, base))
        altered = self._residual(self._residual_input(obs, features, base, latent_override=theta.roll(1, 0)))
        result.update({"Teacher/embedding_rms": float(self.dr_encoder(theta).square().mean().sqrt()),
                       "Teacher/physics_shuffle_action_delta_rms": float((normal-altered).square().mean().sqrt())})
        return result


class RMATeacherCritic(LimbContextCritic):
    def __init__(self, obs, *args, initialization_seed=None, **kwargs):
        kwargs.pop("fusion_mode", None)
        kwargs.pop("dynamics_latent_dim", None)
        super().__init__(obs, *args, fusion_mode="baseline", initialization_seed=initialization_seed, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed or 0) + 7)
            self.dr_encoder = parameter_encoder()
            old = self.mlp[0]
            self.mlp[0] = nn.Linear(self.obs_dim + 64 + 29, old.out_features)
        self.value_input_dim = self.obs_dim + 64 + 29

    def value_input(self, obs):
        features = self.obs_normalizer(self._flat_obs(obs))
        return torch.cat((features, self.dr_encoder(_input(obs, PHYSICS_GROUP, PHYSICS_DIM).to(features)),
                          _input(obs, ACTION_GROUP, 29).to(features)), -1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.mlp(self.value_input(obs))


def configure_models(train, fusion, *, method, scratch_seed):
    if fusion != "concat" or method not in ("rma_teacher", "any2track"):
        raise ValueError("Expected one of the two heavy baseline methods")
    result = copy.deepcopy(train)
    module, actor, critic = (("heavy_rma_teacher", "RMATeacherActor", "RMATeacherCritic")
                             if method == "rma_teacher" else
                             ("heavy_anyadapter", "AnyAdapterActor", "AnyAdapterCritic"))
    result["actor"].update(class_name=f"intact_tracking.{module}:{actor}",
                           initialization_seed=scratch_seed + 10007, initial_action_std=1.)
    result["critic"].update(class_name=f"intact_tracking.{module}:{critic}",
                            initialization_seed=scratch_seed + 20003, initial_checkpoint=None)
    result["release_cuda_cache_after_update"] = True
    result["record_cuda_peak_memory"] = True
    return result


@torch.no_grad()
def audit_initial_models(actor, critic, obs, fusion):
    features, base = actor._base_features_and_action(obs)
    action = actor(obs)
    torch.testing.assert_close(action, base, rtol=0, atol=0)
    actor.distribution.update(action)
    torch.testing.assert_close(actor.output_std, torch.ones_like(action), rtol=0, atol=0)
    assert not any(p.requires_grad for p in actor.tracker.parameters())
    assert torch.isfinite(critic(obs)).all()
    counts = {"actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
              "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad)}
    if hasattr(actor, "history_encoder"):
        counts["history_encoder_parameters"] = sum(p.numel() for p in actor.history_encoder.parameters())
    return {**counts, "identical_initial_action": True, "action_std_initialization": 1.,
            "actor_original_features": features.shape[-1], "critic_original_features": critic.obs_dim,
            "critic_input_dimensions": critic.value_input_dim, "tracker_frozen": True,
            "critic_initial_normalizer_count": float(critic.obs_normalizer.count),
            "dr_auxiliary": {"enabled": False}, "residual_output_mode": "unbounded"}
