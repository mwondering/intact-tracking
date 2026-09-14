"""Online K-means MoE with the frozen tracker's current mean in both heads."""

import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils import unpad_trajectories

from intact_tracking.memory350_moe_policy import (
    HardMoECritic, HardRoutedMLP, configure_moe_models, audit_initial_models,
)
from intact_tracking.memory350_moe_training import OnlineMoEPPO
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper

VERSION = "memory350_online_kmeans16_actor_critic_action_v2"
CRITIC_CLASS = "intact_tracking.memory350_moe_critic_action:TrackerActionMoECritic"
ACTION_GROUP = "frozen_tracker_action"


def reserve_tracker_action(obs):
    """Declare the extra storage field before the policy/rollout storage exist.

    This placeholder is filled by the collector before EVERY critic value
    forward. The 6330-D critic observation normalizer excludes this field.
    """
    sample = next(iter(obs.values()))
    obs.set(ACTION_GROUP, sample.new_zeros(*obs.batch_size, 29))
    return obs


class TrackerActionSlotMixin:
    def get_observations(self):
        return reserve_tracker_action(super().get_observations())

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        return reserve_tracker_action(obs), rewards, dones, extras

    def reset(self):
        obs, extras = super().reset()
        return reserve_tracker_action(obs), extras


class TrackerActionBaselineWrapper(TrackerActionSlotMixin, RslRlVecEnvWrapper):
    pass


class TrackerActionContextWrapper(TrackerActionSlotMixin, Memory350PolicyWrapper):
    pass


def attach_tracker_action(obs, action):
    if action.shape != (*obs.batch_size, 29):
        raise ValueError("Tracker mean must match the current observation batch")
    # An independent tensor preserves old transitions if a later forward or
    # environment step replaces the actor's diagnostic tensors.
    obs.set(ACTION_GROUP, action.detach().clone())


class TrackerActionMoECritic(HardMoECritic):
    def __init__(self, *args, fusion_mode="baseline", compression_dims=(1024, 512, 256, 128),
                 num_experts=16, center_rate=0.01, max_switch_fraction=0.02, **kwargs):
        super().__init__(*args, fusion_mode=fusion_mode, compression_dims=compression_dims,
                         num_experts=num_experts, center_rate=center_rate,
                         max_switch_fraction=max_switch_fraction, **kwargs)
        self.mlp = HardRoutedMLP(
            self.obs_dim, 1, compression_dims=compression_dims, fusion=fusion_mode,
            tracker_action_dim=29, seed=self.initialization_seed, num_experts=num_experts,
            center_rate=center_rate, max_switch_fraction=max_switch_fraction)

    def value_input(self, obs):
        features = self.obs_normalizer(self._flat_obs(obs))
        action = obs[ACTION_GROUP].to(features).detach()
        if action.shape != (*features.shape[:-1], 29):
            raise ValueError("Critic needs a current 29-D frozen tracker mean")
        value = torch.cat((features, action), -1)
        if self.fusion_mode != "baseline":
            value = torch.cat((value, obs["dynamics_latent"].detach()), -1)
        return value

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.mlp(self.value_input(obs))


class TrackerActionMoEPPO(OnlineMoEPPO):
    def act(self, obs):
        # Same ordering and transition fields as RSL PPO.act, with one extra
        # observation captured between the actor and critic. No extra tracker
        # forward, applied action, sampled residual, or previous-step action.
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        attach_tracker_action(obs, self.actor.last_base_action)
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def compute_returns(self, obs):
        # The last next-state observation has not gone through actor.act yet.
        # Bootstrap V on its own tracker mean, never the preceding state's.
        with torch.no_grad():
            _, mean = self.actor._base_features_and_action(obs)
            attach_tracker_action(obs, mean)
        return super().compute_returns(obs)


def configure_critic_action_models(train, fusion, **kwargs):
    result = configure_moe_models(train, fusion, **kwargs)
    result["critic"]["class_name"] = CRITIC_CLASS
    return result


def audit_critic_action_models(actor, critic, obs, fusion):
    with torch.no_grad():
        _, mean = actor._base_features_and_action(obs)
        attach_tracker_action(obs, mean)
        result = audit_initial_models(actor, critic, obs, fusion)
        value = critic.value_input(obs)
        torch.testing.assert_close(value[..., critic.obs_dim:critic.obs_dim+29], mean, atol=0, rtol=0)
        features, current_mean = actor._base_features_and_action(obs)
        actor_input = actor._residual_input(obs, features, current_mean)
        torch.testing.assert_close(actor_input[..., 1645:1674], mean, atol=0, rtol=0)
        assert all(head[0].in_features == 157 for head in critic.mlp.heads)
    result.update(critic_head=[157, 256, 128, 1],
                  critic_tracker_action_input_dim=29,
                  critic_tracker_action_source="same current raw deterministic tracker mean used by actor; stored in rollout observations",
                  critic_tracker_action_matches_actor=True,
                  critic_bootstrap_tracker_action="recomputed on final next-state observation",
                  critic_tracker_action_normalization="unmodified raw action; appended after obs compression",
                  actor_critic_parameters_shared=False)
    return result
