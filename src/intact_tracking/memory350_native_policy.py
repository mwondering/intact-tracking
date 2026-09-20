"""Frozen Memory350 residual PPO using the 144000 flat-world physics contract."""
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import torch

from intact_tracking.environment.mdp.actions import SpTrackingJointPositionAction, SpTrackingJointPositionActionCfg
from intact_tracking.limb_context_dr import _config_value, _event_contract, _resolved_event_contract
from intact_tracking.limb_context_sampling import configure_motion_sampling as original_motion_sampling
from intact_tracking.memory350_native_dr import CapturedAction, CapturedActionCfg, native_nominal_ids
from intact_tracking.memory350_nominal_rollout import NominalExcludedForcePulse
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.rollout.online import _capture_privileged_dynamics_targets, _restore_nominal_physics

VERSION = 'memory350_native_flat_direct_concat_residual_v1'
PROFILE = 'checkpoint_native_flat_v1'
TRACKER = '/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/2026-09-10_16-19-06_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct/checkpoint_144000.pt'
TRACKER_SHA256 = '717fa6e627f368f880bf71ecbc4021da02d8af9e069d5bf907fe45fdcfb79e49'
FULL_DATASET = '/data_zcy/wxy/motion_data_correct'
CONTEXT = str(Path(__file__).resolve().parents[2] / 'runs/144000_exp/stage1_8192/update_014500.pt')
CONTEXT_SHA256 = 'f0a69d8a83310e3fa544adb8a83ccd579f46d8a2b7ac0e65db6a32d684231d32'


@dataclass(kw_only=True)
class NativePolicyActionCfg(CapturedActionCfg):
    def build(self, env):
        return NativePolicyAction(self, env)


class NativePolicyAction(CapturedAction):
    def process_actions(self, actions):
        # PPO's runner separately records the distribution mean. The sampled
        # action drives physics and must never overwrite mean-action history.
        SpTrackingJointPositionAction.process_actions(self, actions)


class FlatTerrainHeightOffset:
    model_fields = ()

    def __init__(self, cfg, env):
        if getattr(env.cfg.commands['motion'], 'terrain_motion_plan', None) is not None:
            raise ValueError('Native residual PPO is explicitly flat-only')

    def __call__(self, env, env_ids, **kwargs):
        pass


def configure_physics(cfg, seed, *, profile):
    if profile != PROFILE:
        raise ValueError('Use the encoder native DR profile')
    if cfg.scene.terrain.terrain_type != 'plane' or getattr(cfg.commands['motion'], 'terrain_motion_plan', None):
        raise ValueError('This experiment uses only flat motions and terrain')
    source = cfg.actions['joint_pos']
    if not isinstance(source, SpTrackingJointPositionActionCfg):
        raise TypeError('Expected the 144000 stateful SP action chain')
    original = _config_value(source)
    cfg.actions['joint_pos'] = NativePolicyActionCfg(
        **{f.name: getattr(source, f.name) for f in fields(SpTrackingJointPositionActionCfg)}, nominal_fraction=.1)
    return {'profile': PROFILE, 'seed': seed, 'extra_payload': False,
            'original_events': {name: _event_contract(term) for name, term in cfg.events.items()},
            'original_action': original, 'nominal_fraction_requested': .1,
            'per_coordinate_nominal_mixture_probability': 0.,
            'sampling': 'ceil(10% per rank) compiled nominal; all other worlds original random DR ranges',
            'physics_lifetime': 'physical prototype fixed per world across motion and episode resets',
            'controller_lifetime': 'DR delay and smoothing resampled by original reset; nominal delay=0, alpha=1',
            'force_pulse_scope': 'DR only', 'terrain': 'flat',
            'encoder_action_input': 'mean of all four actual physical substep PD targets after residual action',
            'policy_mean_history': 'recorded separately by PPO runner; not overwritten by sampled action'}


def environment_factory(factory, **kwargs):
    env = factory(**kwargs)
    try:
        action = env.action_manager.get_term('joint_pos')
        if not isinstance(action, NativePolicyAction) or env.cfg.decimation != 4:
            raise TypeError('Native PPO needs four captured physical substeps')
        ids = native_nominal_ids(env.num_envs, .1, device=env.device)
        restoration = _restore_nominal_physics(env, ids)
        motor = env.event_manager.get_term_cfg('motor_params_implicit').func
        # Compiled fields were restored; keep critic's cached parameter
        # observations consistent with the same nominal physical values.
        for name in ('_obs_kp_scale', '_obs_kd_scale', '_obs_arm_scale'):
            getattr(motor, name)[ids] = 1
        motor.reset = lambda env_ids=None: None
        action.delay[ids] = 0
        action.alpha[ids] = 1
        action.joint_offset[ids] = 0
        action.boot_delay[ids] = 0
        pulse_cfg = env.event_manager.get_term_cfg('push_robot')
        pulse_cfg.func = NominalExcludedForcePulse(pulse_cfg.func, ids)
        if not any(term is pulse_cfg for term in env.event_manager._mode_class_term_cfgs['step']):
            raise RuntimeError('Pulse reset callback must retain nominal exclusion')
        physical = _capture_privileged_dynamics_targets(env)
        env.native_policy_reference_parameters = physical.values.clone()
        env.native_policy_nominal_ids = ids
        env.native_policy_runtime_audit = {
            'nominal_count': len(ids), 'dr_count': env.num_envs-len(ids),
            'nominal_fraction_actual': len(ids)/env.num_envs, 'restoration': restoration,
            'parameter_dimensions': physical.values.shape[-1],
            'physical_prototype_sha256': hashlib.sha256(physical.values.float().cpu().numpy().tobytes()).hexdigest(),
            'nominal_motor_observations_restored': True, 'motor_gains_fixed_across_resets': True,
            'nominal_pulses_disabled': True, 'physical_target_substeps': env.cfg.decimation,
            'physical_targets_per_control_step': 29, 'last_parameter_audit_max_error': 0.}
        audit_native_runtime(env)
        return env
    except BaseException:
        env.close()
        raise


def audit_native_runtime(env):
    ids = env.native_policy_nominal_ids
    action = env.action_manager.get_term('joint_pos')
    pulse = env.event_manager.get_term_cfg('push_robot').func
    if (action.delay[ids].ne(0).any() or action.alpha[ids].ne(1).any()
            or action.joint_offset[ids].ne(0).any() or env.scene['robot'].data.encoder_bias[ids].ne(0).any()
            or pulse.active[ids].any() or pulse.current_force_w[ids].ne(0).any()
            or not torch.isinf(pulse.time_to_next_pulse_s[ids]).all()):
        raise RuntimeError('Nominal physics/controller/pulse contract changed')
    physical = _capture_privileged_dynamics_targets(env).values
    error = float((physical-env.native_policy_reference_parameters).abs().max())
    if error != 0.:
        raise RuntimeError(f'Physical parameters changed across PPO episodes: {error}')
    env.native_policy_runtime_audit['last_parameter_audit_max_error'] = error
    return dict(env.native_policy_runtime_audit)


def audit_physics(env, configuration):
    if set(env.cfg.events) != set(configuration['original_events']):
        raise ValueError('Original DR event set changed')
    for name, expected in configuration['original_events'].items():
        term = env.event_manager.get_term_cfg(name)
        wrapped = term.func
        if isinstance(wrapped, NominalExcludedForcePulse):
            term.func = wrapped.source
        try:
            if _event_contract(term) != _resolved_event_contract(expected, env.scene):
                raise ValueError(f'Original event changed: {name}')
        finally:
            term.func = wrapped
    return audit_native_runtime(env)


def configure_sampling(cfg, requested, after_update, completed_updates):
    if requested != 'adaptive' or after_update != 0:
        raise ValueError('This stage starts adaptive sampling at update zero')
    result = original_motion_sampling(cfg, requested, after_update, completed_updates)
    result['scope'] = 'motion/bin sampling only; 10% nominal and random native DR physics are allocated independently'
    return result


def resolve_profile(profile, previous=None):
    if profile != PROFILE or (previous and previous['dr_profile'] != PROFILE):
        raise ValueError('Native DR profile mismatch')
    return PROFILE


def validate_context(state, profile):
    if (profile != PROFILE or state.get('native_dr_version') != 1
            or state['tracker']['checkpoint_sha256'] != TRACKER_SHA256
            or state.get('predictor_action_contract') != {
                'available': True, 'mode': 'captured_substep_physical_targets',
                'predictor_input': 'mean over one control step'}
            or state.get('episode_length_control_steps') != 500):
        raise ValueError('Context checkpoint does not match the native physical target and episode contract')


class NativePolicyWrapper(Memory350PolicyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_steps = 0

    def _applied_joint_target(self):
        action = self.unwrapped.action_manager.get_term('joint_pos')
        return action.physical_target_trace.detach().mean(dim=1)

    def step(self, actions):
        result = super().step(actions)
        self.native_steps += 1
        if self.native_steps == 1 or self.native_steps % 500 == 0:
            audit_native_runtime(self.unwrapped)
        return result

    @property
    def latent_metrics(self):
        action = self.unwrapped.action_manager.get_term('joint_pos')
        trace = action.physical_target_trace
        return {**super().latent_metrics,
                'context_mean_vs_last_target_rms': float((trace.mean(1)-trace[:, -1]).square().mean().sqrt()),
                'nominal_physics_max_error': self.unwrapped.native_policy_runtime_audit['last_parameter_audit_max_error']}
