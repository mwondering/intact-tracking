"""Physics-aware value estimation; neither actor observations nor rewards change."""

from __future__ import annotations

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories

from intact_tracking.adaptation_policy import physics_observation
from intact_tracking.residual_policy import WarmStartedHeftCritic

CRITIC_PHYSICS = "adaptation_critic_physics"


class PhysicsCriticWrapper:
    """Attach actual static parameters to a separate, training-only critic group."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def attach(self, obs):
        return obs.set(CRITIC_PHYSICS, physics_observation(self.unwrapped))

    def get_observations(self):
        return self.attach(self.wrapped.get_observations())

    def step(self, actions):
        obs, reward, done, extras = self.wrapped.step(actions)
        return self.attach(obs), reward, done, extras

    def reset(self):
        obs, extras = self.wrapped.reset()
        return self.attach(obs), extras


class PhysicsAwareHeftCritic(WarmStartedHeftCritic):
    """Original warm-started value plus a zero-initialized state/physics branch.

    Actual physical values may reduce value aliasing across DR worlds. This is
    an input/architecture hypothesis, not a claim that the existing critic is
    necessarily the cause of poor tracking. Value targets remain ordinary PPO
    returns computed from the unchanged environment reward.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        groups = {key: list(value) for key, value in obs_groups.items()}
        if CRITIC_PHYSICS not in groups[obs_set]:
            raise ValueError("Physics-aware critic requires a dedicated critic input group")
        groups[obs_set].remove(CRITIC_PHYSICS)
        super().__init__(obs, groups, obs_set, output_dim, **kwargs)
        physics_dim = int(obs[CRITIC_PHYSICS].shape[-1])
        self.physics_normalizer = EmpiricalNormalization(physics_dim, until=20_000_000)
        self.physics_value = MLP(self.obs_dim + physics_dim, output_dim, (256, 128), "mish")
        torch.nn.init.zeros_(self.physics_value[-1].weight)
        torch.nn.init.zeros_(self.physics_value[-1].bias)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        state = self.obs_normalizer(self._flat_obs(obs))
        physics = self.physics_normalizer(obs[CRITIC_PHYSICS]).clamp(-10, 10)
        return self.mlp(state) + self.physics_value(torch.cat((state, physics), -1))

    def update_normalization(self, obs):
        super().update_normalization(obs)
        self.physics_normalizer.update(obs[CRITIC_PHYSICS])
