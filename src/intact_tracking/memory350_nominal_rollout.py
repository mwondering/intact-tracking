"""Memory350's 50/50 nominal and tracker-DR-plus-payload training worlds."""

from dataclasses import asdict, dataclass
import hashlib

import torch

from intact_tracking.limb_context_dr import audit_limb_dr
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT
from intact_tracking.memory350_rollout import Memory350TrackerRollout
from intact_tracking.rollout.online import (
    FixedDRRolloutConfig, JointPositionTargetTransform,
    _capture_privileged_dynamics_targets, _capture_randomized_model_fields,
    _restore_nominal_physics,
)


@dataclass(frozen=True)
class NominalMemory350RolloutConfig(FixedDRRolloutConfig):
    nominal_fraction: float = 0.5

    def __post_init__(self):
        if self.nominal_fraction != 0.5 or self.num_envs % 2:
            raise ValueError('Nominal Memory350 requires exactly half nominal worlds')
        if not self.tracker_dr_plus_limb_payload or self.limb_payload_only:
            raise ValueError('The DR half must retain tracker DR plus four-limb payloads')
        # Reuse all original validation except its all-DR sampling restriction.
        FixedDRRolloutConfig(**{**asdict(self), 'nominal_fraction': 0.0})


class NominalExcludedForcePulse:
    """Keep the original DR pulse process; nominal slots can never trigger it.

    The original pulse ignores env_ids, so filtering that argument is insufficient.
    Its timers must also remain disabled after every EventManager reset.
    """

    def __init__(self, source, nominal_ids):
        self.source = source
        self.nominal_ids = nominal_ids
        self._disable_nominal()

    def _disable_nominal(self):
        source, ids = self.source, self.nominal_ids
        source._clear_wrench(ids)
        source.time_to_next_pulse_s[ids] = float('inf')
        source.active_time_left_s[ids] = 0
        source.active[ids] = False
        source.current_force_w[ids] = 0

    def __call__(self, env, env_ids, **kwargs):
        # reset() already maintains this mask; setting timers also guards direct calls.
        self.source.time_to_next_pulse_s[self.nominal_ids] = float('inf')
        self.source(env, env_ids, **kwargs)

    def reset(self, env_ids=None):
        self.source.reset(env_ids=env_ids)
        self._disable_nominal()

    def observe(self, **kwargs):
        return self.source.observe(**kwargs)

    def __getattr__(self, name):
        return getattr(self.source, name)


class NominalMemory350TrackerRollout(Memory350TrackerRollout):
    def __init__(self, config):
        if not isinstance(config, NominalMemory350RolloutConfig):
            raise TypeError('Use NominalMemory350RolloutConfig for the nominal mixture')
        # Construct exactly the original DR population, then restore selected slots.
        # This preserves the original sampled physics of every remaining DR world.
        super().__init__(FixedDRRolloutConfig(**{**asdict(config), 'nominal_fraction': 0.0}))
        try:
            self.config = config
            self.nominal_env_ids = torch.arange(0, config.num_envs, 2, device=self.env.device)
            self.nominal_count = len(self.nominal_env_ids)
            self.is_nominal.zero_()
            self.is_nominal[self.nominal_env_ids] = True
            dr_ids = torch.arange(1, config.num_envs, 2, device=self.env.device)
            before = self._fixed_dr_model_fields
            before_bias = self.env.scene['robot'].data.encoder_bias[dr_ids].clone()
            self.nominal_restore_metrics = _restore_nominal_physics(self.env, self.nominal_env_ids)
            payload = self.env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
            payload.mass[self.nominal_env_ids] = 0
            # The original limb audit still applies: zero nominal loads, unchanged DR loads.
            physics = audit_limb_dr(self.env, self.payload_configuration)
            self._fixed_dr_model_fields = _capture_randomized_model_fields(self.env)
            for name, expected in before.items():
                actual = self._fixed_dr_model_fields[name]
                if actual.ndim > 0 and actual.shape[0] == config.num_envs:
                    torch.testing.assert_close(actual[dr_ids], expected[dr_ids], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(self.env.scene['robot'].data.encoder_bias[dr_ids],
                                       before_bias, rtol=0, atol=0)
            pulse_cfg = self.env.event_manager.get_term_cfg('push_robot')
            if type(pulse_cfg.func).__name__ != 'body_force_pulse':
                raise ValueError('Unexpected tracker force-pulse implementation')
            self.force_pulse = NominalExcludedForcePulse(pulse_cfg.func, self.nominal_env_ids)
            pulse_cfg.func = self.force_pulse
            # Both step execution and class-reset callbacks must reference this same cfg.
            callbacks = self.env.event_manager._mode_class_term_cfgs['step']
            if not any(term is pulse_cfg for term in callbacks):
                raise RuntimeError('Force-pulse reset callback does not use the masked event')
            self.predictor_action_transform = JointPositionTargetTransform.from_mjlab(
                self.env, self.env.action_manager.get_term('joint_pos'))
            privileged = _capture_privileged_dynamics_targets(self.env)
            self.privileged_dynamics_names = privileged.names
            self.privileged_dynamics = privileged.values
            self.ignored_privileged_startup_events = privileged.ignored_startup_events
            self.dynamics_prototype_sha256 = hashlib.sha256(
                privileged.values.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            self.observations = self.wrapped.get_observations()
            self.assert_nominal_clean()
            self.payload_configuration.update(
                sampling='even slots compiled nominal; odd slots original tracker DR plus independent U(0,4)^4',
                nominal_fraction=0.5,
                force_pulse_scope='DR slots only; nominal timers disabled after every reset',
            )
            physics['nominal_mixture'] = {
                'nominal_count': self.nominal_count,
                'dr_count': config.num_envs - self.nominal_count,
                'layout': 'even local IDs nominal; odd local IDs DR',
                'nominal_restore': self.nominal_restore_metrics,
                'dr_physics_unchanged_from_original_sampling': True,
                'nominal_payload_max_abs_kg': float(payload.observe()[self.nominal_env_ids].abs().max()),
                'dr_payload_min_kg': payload.observe()[dr_ids].amin(0).tolist(),
                'dr_payload_max_kg': payload.observe()[dr_ids].amax(0).tolist(),
                'nominal_pulses_disabled': True,
            }
            self.payload_configuration['runtime_audit'] = physics
        except BaseException:
            self.close()
            raise

    def assert_nominal_clean(self):
        ids = self.nominal_env_ids
        bias = self.env.scene['robot'].data.encoder_bias[ids]
        if bool(bias.ne(0).any()) or bool(self.force_pulse.active[ids].any()):
            raise RuntimeError('Nominal encoder bias or force pulse became active')
        if bool(self.force_pulse.current_force_w[ids].ne(0).any()):
            raise RuntimeError('Nominal world received an external force pulse')
        if not bool(torch.isinf(self.force_pulse.time_to_next_pulse_s[ids]).all()):
            raise RuntimeError('Nominal force-pulse timers were re-enabled')

    def _assert_fixed_dr(self, env_ids=None):
        super()._assert_fixed_dr(env_ids)
        if hasattr(self, 'force_pulse'):
            self.assert_nominal_clean()

    @property
    def metadata(self):
        return {**super().metadata,
                'nominal_world_local_ids': self.nominal_env_ids.cpu().tolist(),
                'domain_randomization_contract':
                    '50% compiled nominal without payload or pulses; 50% original tracker DR plus four-limb payloads',
                'nominal_mixture_version': 'memory350_nominal50_v1'}
