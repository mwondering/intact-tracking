"""PPO wrapper collecting completed interactions with the stage1 reset protocol."""

import torch
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.memory350_policy_inference import CachedMemory350Inference
from intact_tracking.residual_policy import DYNAMICS_LATENT_GROUP
from intact_tracking.rollout.mjlab_adapter import _robot_raw_state
from intact_tracking.rollout.online import _read_motion_resample_boundary


class Memory350PolicyWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, checkpoint=None, *, latent_history_frames=1):
        super().__init__(env, clip_actions=clip_actions)
        self.context = (CachedMemory350Inference(checkpoint, self.num_envs)
                        if checkpoint is not None else None)
        self.motion_command = self.unwrapped.command_manager.get_term("motion")
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.encoding_enabled = True
        if latent_history_frames not in (1, 5):
            raise ValueError("Use one or five latent frames")
        from intact_tracking.latent_history import LatentHistory
        self.latent_history = (LatentHistory(self.num_envs, frames=latent_history_frames, device=self.device)
                               if latent_history_frames > 1 else None)
        self._latent = self._record_latent(self._encode())

    def _encode(self):
        if self.context is None:
            return torch.zeros(self.num_envs, 64, device=self.device)
        return self.context.encode()

    def _record_latent(self, latent, *, reset=None):
        return self.latent_history.append(latent, reset=reset) if self.latent_history is not None else latent

    def _attach(self, obs):
        obs.set(DYNAMICS_LATENT_GROUP, self._latent)
        return obs

    def get_observations(self):
        return self._attach(super().get_observations())

    def reset(self):
        if self.context is not None:
            self.context.episode_reset()
        obs, extras = super().reset()
        self.episode_ids += 1
        if self.latent_history is not None:
            self.latent_history.clear()
        self._latent = self._record_latent(self._encode())
        return self._attach(obs), extras

    def step(self, actions):
        if self.context is not None:
            # Read the actual pre-step state, including after an external partial
            # reset in evaluation. No reset edge is admitted as an interaction.
            before = {"robot_state": _robot_raw_state(self.unwrapped).detach().clone(),
                      "episode_id": self.episode_ids.clone(),
                      "episode_step": self.unwrapped.episode_length_buf.clone(),
                      "motion_id": self.motion_command.motion_idx.clone(),
                      "motion_step": self.motion_command.time_steps.clone()}
        obs, reward, dones, extras = super().step(actions)
        boundary = dones.bool() | _read_motion_resample_boundary(self.motion_command, dones.bool())
        if self.context is not None:
            before.update(next_robot_state=_robot_raw_state(self.unwrapped).detach(),
                          joint_target=self.unwrapped.scene["robot"].data.joint_pos_target.detach(),
                          reset_boundary=boundary)
            self.context.append(before)
            if self.encoding_enabled:
                self._latent = self._record_latent(self._encode(), reset=boundary)
        self.episode_ids += boundary.long()
        return self._attach(obs), reward, dones, extras

    @property
    def latent_metrics(self):
        return {**(self.context.metrics if self.context is not None else {}),
                "dynamics_latent_rms": float(self._latent.square().mean().sqrt())}
