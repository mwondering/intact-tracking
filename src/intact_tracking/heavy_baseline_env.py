"""Native heavy physics conditioning, independent of any Memory350 checkpoint."""

import torch
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.anyadapter_world_model import (
    CausalHistory, STATE_GROUP, WORLD_GROUP, COUNT_GROUP, VELOCITY_SCALE,
)
from intact_tracking.heavy_rma_teacher import PHYSICS_GROUP
from intact_tracking.memory350_native_policy import audit_native_runtime
from intact_tracking.memory350_proprio_inputs import current_proprio, control_action, validate_proprio_observations
from intact_tracking.residual_policy import DYNAMICS_LATENT_GROUP
from intact_tracking.rollout.online import _read_motion_resample_boundary


def physical_schema(env):
    from intact_tracking.memory350_native_dr import native_metric_schema
    from intact_tracking.memory350_heavy_policy import heavy_aux_schema
    from intact_tracking.rollout.online import (
        _capture_privileged_dynamics_targets, _entity_indices_and_names, _expanded_and_default_field,
    )
    physical = _capture_privileged_dynamics_targets(env)
    names = list(physical.names)
    params = {name.split('/')[0]: env.event_manager.get_term_cfg(name.split('/')[0]).params for name in names}
    defaults = {}
    for event, config in params.items():
        if any(name.startswith(event + '/relative_mass/') for name in names):
            ids, body_names = _entity_indices_and_names(env, config['asset_cfg'], 'body')
            _, masses = _expanded_and_default_field(env, 'body_mass')
            defaults.update(zip(body_names, masses[ids].cpu().tolist(), strict=True))
    return heavy_aux_schema(native_metric_schema(names, params, defaults))


def proprio_state(obs):
    raw = current_proprio(obs)
    return torch.cat((raw[:, 61:64] * VELOCITY_SCALE, raw[:, 58:61],
                      raw[:, :29], raw[:, 29:58] * VELOCITY_SCALE), -1)


def world_state(env):
    data = env.scene['robot'].data
    return torch.cat((env.scene['robot/imu_ang_vel'].data * VELOCITY_SCALE,
                      data.projected_gravity_b, data.joint_pos - data.default_joint_pos,
                      (data.joint_vel-data.default_joint_vel) * VELOCITY_SCALE,
                      (data.root_link_pos_w-env.scene.env_origins)[:, 2:3]), -1).detach().clone()


class HeavyBaselineWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, checkpoint=None, *, method):
        if checkpoint is not None or clip_actions is not None:
            raise ValueError('New baselines have no pretrained context or action clipping')
        super().__init__(env, clip_actions=clip_actions)
        self.method, self.context = method, None
        self.encoding_enabled, self.native_steps = True, 0
        self.motion_command = env.command_manager.get_term('motion')
        self.physics_schema = physical_schema(env)
        self.history, self.encoder = None, None
        self._last_proprio = None
        self.query_forces = []
        if method == 'rma_teacher':
            from intact_tracking.residual_dr_aux import capture_dr_aux_targets
            self.theta = capture_dr_aux_targets(env, self.physics_schema) * 2 - 1
        elif method == 'any2track':
            validate_proprio_observations(env)
            self.history = CausalHistory(env.num_envs, env.device)
        else:
            raise ValueError(method)

    def bind_policy(self, actor):
        if self.history is not None:
            self.encoder = actor.history_encoder

    @torch.no_grad()
    def _attach(self, obs):
        if self.method == 'rma_teacher':
            obs.set(PHYSICS_GROUP, self.theta)
        else:
            # Keep the same sensor realization even if inactive evaluation
            # worlds invalidate the environment's global observation cache.
            self._last_proprio = proprio_state(obs)
            obs.set(STATE_GROUP, self._last_proprio)
            obs.set(WORLD_GROUP, world_state(self.unwrapped))
            obs.set(COUNT_GROUP, self.history.count.clone())
            embedding = (self.encoder(self.history.frames) if self.encoder is not None and self.encoding_enabled
                         else torch.zeros(self.num_envs, 128, device=self.device))
            obs.set(DYNAMICS_LATENT_GROUP, embedding.detach())
        return obs

    def get_observations(self):
        return self._attach(super().get_observations())

    def reset(self):
        if self.history is not None:
            self.history.clear()
        obs, extras = super().reset()
        return self._attach(obs), extras

    def step(self, actions):
        before = self._last_proprio
        obs, reward, dones, extras = super().step(actions)
        boundary = _read_motion_resample_boundary(self.motion_command, dones.bool())
        if self.history is not None:
            if before is None:
                raise RuntimeError('Observe before acting')
            self.history.append(before, control_action(self.unwrapped), dones.bool() | boundary)
        self.native_steps += 1
        if self.native_steps == 1 or self.native_steps % 500 == 0:
            audit_native_runtime(self.unwrapped)
        if getattr(self.unwrapped, 'native_paired_query', False):
            self.query_forces.append(self.unwrapped.event_manager.get_term_cfg('push_robot').func.current_force_w.detach().clone())
        return self._attach(obs), reward, dones, {**extras, 'motion_resample_boundary': boundary}

    @property
    def latent_metrics(self):
        result = {'nominal_physics_max_error': self.unwrapped.native_policy_runtime_audit['last_parameter_audit_max_error']}
        if self.history is not None:
            result['Adapter/history_valid_fraction'] = float(self.history.count.float().mean() / 79)
        return result

    @property
    def evaluation_diagnostics(self):
        import hashlib
        digest, prefixes = hashlib.sha256(), []
        for force in self.query_forces:
            digest.update(force.cpu().contiguous().numpy().tobytes())
            prefixes.append(digest.hexdigest())
        return {'query_force_steps': len(prefixes), 'query_force_sha256': digest.hexdigest(),
                'query_force_prefix_sha256': prefixes, 'cached_proprio_preserved_after_partial_reset': True}
