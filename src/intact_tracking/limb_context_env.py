"""Causal frozen context, adding only z to the original tracker observations."""

import torch
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.residual_context import DynamicsContextInference
from intact_tracking.residual_policy import DYNAMICS_LATENT_GROUP
from intact_tracking.rollout.mjlab_adapter import _robot_raw_state
from intact_tracking.rollout.online import _read_motion_resample_boundary


class LimbContextWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, checkpoint=None):
        super().__init__(env, clip_actions=clip_actions)
        self.context = (DynamicsContextInference(checkpoint, num_envs=self.num_envs,
                                               device=self.device)
                        if checkpoint is not None else None)
        self.motion_command = self.unwrapped.command_manager.get_term("motion")
        self._state = _robot_raw_state(self.unwrapped).detach().clone()
        self._latent = self._encode()

    def _encode(self):
        if self.context is None:
            return torch.zeros(self.num_envs, 64, device=self.device)
        return self.context.encode(self._state)

    def _attach(self, obs):
        obs.set(DYNAMICS_LATENT_GROUP, self._latent)
        return obs

    def get_observations(self):
        return self._attach(super().get_observations())

    def reset(self):
        obs, extras = super().reset()
        if self.context is not None:
            self.context.clear()
        self._state = _robot_raw_state(self.unwrapped).detach().clone()
        self._latent = self._encode()
        return self._attach(obs), extras

    def step(self, actions):
        state = self._state
        obs, reward, dones, extras = super().step(actions)
        if self.context is not None:
            target = self.unwrapped.scene["robot"].data.joint_pos_target
            boundary = dones.bool() | _read_motion_resample_boundary(self.motion_command, dones.bool())
            self.context.append(state, target.detach(), boundary)
            self._state = _robot_raw_state(self.unwrapped).detach().clone()
            self._latent = self._encode()
        return self._attach(obs), reward, dones, extras

    @property
    def latent_metrics(self):
        return {**(self.context.metrics if self.context is not None else {}),
                "dynamics_latent_rms": float(self._latent.square().mean().sqrt())}
