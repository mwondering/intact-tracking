"""Causal current-frame history for RMA; physical labels live outside actor observations."""

import torch
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.heavy_rma_student import ProprioHistory, EMBEDDING_GROUP, HISTORY_STEPS
from intact_tracking.memory350_proprio_inputs import current_proprio, validate_proprio_observations
from intact_tracking.memory350_native_policy import audit_native_runtime
from intact_tracking.rollout.online import _read_motion_resample_boundary


class RMAStudentWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, checkpoint=None):
        if checkpoint is not None or clip_actions is not None:
            raise ValueError('RMA student requires its own history and unclipped actions')
        super().__init__(env, clip_actions=None)
        self.context = None
        self.history = ProprioHistory(env.num_envs, env.device)
        self.encoder, self._ready, self.native_steps = None, False, 0
        self.encoding_enabled = True
        self.motion_command = env.command_manager.get_term('motion')
        self.query_forces = []
        self.input_audit = validate_proprio_observations(env)

    def bind_policy(self, actor):
        self.encoder = actor.adaptation

    @torch.no_grad()
    def _attach(self, obs):
        embedding = (self.encoder(self.history.frames, self.history.count)
                     if self.encoder is not None and self.encoding_enabled
                     else torch.zeros(self.num_envs, 64, device=self.device))
        obs.set(EMBEDDING_GROUP, embedding)
        return obs

    def get_observations(self):
        obs = super().get_observations()
        if not self._ready:
            self.history.append(current_proprio(obs)); self._ready = True
        return self._attach(obs)

    def reset(self):
        self.history.clear()
        obs, extras = super().reset()
        self.history.append(current_proprio(obs)); self._ready = True
        return self._attach(obs), extras

    def step(self, actions):
        obs, reward, dones, extras = super().step(actions)
        boundary = _read_motion_resample_boundary(self.motion_command, dones.bool())
        self.history.append(current_proprio(obs), dones.bool() | boundary)
        self.native_steps += 1
        if self.native_steps == 1 or self.native_steps % 500 == 0:
            audit_native_runtime(self.unwrapped)
        if getattr(self.unwrapped, 'native_paired_query', False):
            self.query_forces.append(self.unwrapped.event_manager.get_term_cfg('push_robot').func.current_force_w.detach().clone())
        return self._attach(obs), reward, dones, {**extras, 'motion_resample_boundary': boundary}

    @property
    def latent_metrics(self):
        return {'nominal_physics_max_error': self.unwrapped.native_policy_runtime_audit['last_parameter_audit_max_error'],
                'Distill/history_valid_fraction': float(self.history.count.float().mean()/HISTORY_STEPS),
                'Distill/history_full_fraction': float((self.history.count == HISTORY_STEPS).float().mean())}

    @property
    def evaluation_diagnostics(self):
        import hashlib
        digest, prefixes = hashlib.sha256(), []
        for force in self.query_forces:
            digest.update(force.cpu().contiguous().numpy().tobytes()); prefixes.append(digest.hexdigest())
        return {'query_force_steps': len(prefixes), 'query_force_sha256': digest.hexdigest(),
                'query_force_prefix_sha256': prefixes, 'cached_proprio_preserved_after_partial_reset': True}
