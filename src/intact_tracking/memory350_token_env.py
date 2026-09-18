"""Collect frozen features/actions and causal token windows before PPO sees them."""

import torch

from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.memory350_token_inputs import (
    BASE_ACTION, FUTURE, HISTORY, VALID, SPV52TokenSplitter, TokenHistory,
)


class TokenPolicyWrapper(Memory350PolicyWrapper):
    def __init__(self, env, clip_actions=None, checkpoint=None):
        super().__init__(env, clip_actions, checkpoint)
        self.token_history = TokenHistory(self.num_envs, self.device)
        self.splitter = SPV52TokenSplitter().to(self.device)
        self._policy = None

    def bind_policy(self, actor):
        if actor.architecture != "transformer":
            raise ValueError("Only the Transformer arm collects additional token history")
        self._policy = actor
        self.token_history.clear()

    @torch.no_grad()
    def _attach(self, obs):
        obs = super()._attach(obs)
        if self._policy is None:
            # Reserve the full schema before RSL constructs RolloutStorage.
            obs.set(HISTORY, self.token_history.frames)
            obs.set(VALID, self.token_history.valid)
            obs.set(FUTURE, torch.zeros(self.num_envs, 4, 77, device=self.device))
            obs.set(BASE_ACTION, torch.zeros(self.num_envs, 29, device=self.device))
            return obs
        self._policy.populate_tracker_cache(obs)
        features, base = self._policy._base_features_and_action(obs)
        frame, future = self.splitter(features, base, self._latent)
        stamp = torch.stack((self.episode_ids, self.motion_command.motion_idx,
                             self.motion_command.time_steps), -1)
        history, valid = self.token_history.observe(frame, stamp)
        obs.set(HISTORY, history)
        obs.set(VALID, valid)
        obs.set(FUTURE, future)
        obs.set(BASE_ACTION, base)
        return obs

    def reset(self):
        self.token_history.clear()
        return super().reset()

    def clear_policy_history(self, env_ids=None):
        """For explicit out-of-band simulator/DR resets in deployment."""
        self.token_history.clear(env_ids)


def evaluation_wrapper(env, clip_actions=None, checkpoint=None):
    if checkpoint is None:
        return Memory350PolicyWrapper(env, clip_actions, checkpoint)
    return TokenPolicyWrapper(env, clip_actions, checkpoint)
