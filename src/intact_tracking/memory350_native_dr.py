"""Memory350 collection with checkpoint-native flat-world DR."""
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from intact_tracking.environment.mdp.actions import (
    SpTrackingJointPositionAction, SpTrackingJointPositionActionCfg,
)
from intact_tracking.environment.runtime import _load_saved_config
from intact_tracking.memory350_dr_center_rollout import normalize_dr_metric
from intact_tracking.memory350_nominal_rollout import NominalExcludedForcePulse
from intact_tracking.rollout.online import (
    FixedDRTrackerRollout, _entity_indices_and_names, _expanded_and_default_field,
)


def native_nominal_ids(num_envs, fraction, *, device='cpu', allow_endpoints=False):
    if allow_endpoints and fraction in (0., 1.):
        return torch.arange(num_envs if fraction == 1. else 0, device=device, dtype=torch.long)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Native context needs both nominal and DR worlds')
    count = math.ceil(num_envs * fraction)
    if count >= num_envs:
        raise ValueError('No DR worlds remain')
    return ((2 * torch.arange(count, device=device) + 1) * num_envs) // (2 * count)


@dataclass(kw_only=True)
class CapturedActionCfg(SpTrackingJointPositionActionCfg):
    nominal_fraction: float = .1

    def build(self, env):
        return CapturedAction(self, env)


class CapturedAction(SpTrackingJointPositionAction):
    """Observe the unchanged SP action chain after each physical target is set."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.nominal_ids = native_nominal_ids(env.num_envs, cfg.nominal_fraction, device=env.device,
                                              allow_endpoints=True)
        self.physical_target_trace = torch.zeros(env.num_envs, self._decimation, self.action_dim,
                                                device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # Reset can resample controller nuisance variables, but nominal stays clean.
        ids = self.nominal_ids
        self.delay[ids] = 0
        self.alpha[ids] = 1
        self.joint_offset[ids] = 0
        self.boot_delay[ids] = 0

    def process_actions(self, actions):
        # The frozen tracker is deterministic, so its action is its policy mean.
        self.record_policy_mean(actions)
        super().process_actions(actions)

    def apply_actions(self):
        index = self._substep
        super().apply_actions()
        self.physical_target_trace[:, index].copy_(self._entity.data.joint_pos_target)


def configure_native_environment(env_cfg, config):
    from intact_tracking.limb_context_dr import _event_contract, _config_value
    from intact_tracking.environment.mdp.multi_commands import _normalize_motion_exclude_files
    command = env_cfg.commands['motion']
    if getattr(command, 'terrain_motion_plan', None) is not None:
        raise ValueError('This explicitly flat-only collector cannot consume terrain motions')
    for value in _normalize_motion_exclude_files(command.motion_exclude_files):
        if not Path(value).is_file():
            raise FileNotFoundError(f'Checkpoint motion exclusion file is missing: {value}')
    original_action = _config_value(env_cfg.actions['joint_pos'])
    source = env_cfg.actions['joint_pos']
    if not isinstance(source, SpTrackingJointPositionActionCfg) or config.stochastic_policy:
        raise ValueError('Native collection currently requires the deterministic SP tracker')
    env_cfg.actions['joint_pos'] = CapturedActionCfg(
        **{field.name: getattr(source, field.name) for field in fields(source)},
        nominal_fraction=config.nominal_fraction)
    command.sampling_mode = 'uniform'
    command.rewind.enabled = False
    command.adaptive_bin_snapshot_interval_iterations = 0
    result = {'dr_profile': 'checkpoint_native_flat_v1', 'extra_payload': False,
            'nominal_fraction_requested': config.nominal_fraction,
            'dr_parameter_nominal_probability': 0.,
            'original_events': {name: _event_contract(term) for name, term in env_cfg.events.items()},
            'original_action': original_action,
            'sampling': 'original checkpoint random ranges; no per-coordinate nominal mixture',
            'physics_lifetime': 'one random physical prototype per world; motor gains fixed across motion resets',
            'motion_sampling': 'uniform', 'motion_exclusions_preserved': True,
            'terrain': 'flat only, explicitly selected by user'}
    if getattr(config, 'heavy_payload', False):
        from intact_tracking.memory350_heavy_dr import PROFILE, configure_heavy_payload
        result.update(dr_profile=PROFILE, extra_payload=True,
                      heavy_payload=configure_heavy_payload(env_cfg, config))
    return result


def install_native_runtime(rollout):
    env, ids = rollout.env, rollout.nominal_env_ids
    action = env.action_manager.get_term('joint_pos')
    if not isinstance(action, CapturedAction):
        raise TypeError('Native physical target capture was not installed')
    action.delay[ids] = 0
    action.alpha[ids] = 1
    # Cross-motion positives require the same physical parameters after a reset.
    motor = env.event_manager.get_term_cfg('motor_params_implicit').func
    motor.reset = lambda env_ids=None: None
    pulse_cfg = env.event_manager.get_term_cfg('push_robot')
    rollout.force_pulse = NominalExcludedForcePulse(pulse_cfg.func, ids)
    pulse_cfg.func = rollout.force_pulse
    rollout.captures_physical_targets = True
    if getattr(rollout.config, 'heavy_payload', False):
        from intact_tracking.memory350_heavy_dr import EVENT
        rollout.payload_configuration['heavy_payload']['runtime_audit'] = (
            env.event_manager.get_term_cfg(EVENT).func.audit())


def native_metric_schema(names, event_params, default_body_mass):
    lower, upper, groups = [], [], {}
    for i, name in enumerate(names):
        event, kind, *labels = name.split('/')
        params = event_params[event]
        group = name
        if kind == 'com_offset' and labels[0] == 'torso_link':
            ranges = params['ranges']; axis = 'xyz'.index(labels[1])
            low, high = ranges.get(axis, ranges.get(str(axis)))
        elif kind == 'relative_mass' and labels == ['torso_link']:
            low, high = (float(x) / default_body_mass['torso_link'] for x in params['ranges'])
        elif kind == 'friction' and labels == ['shared', '0']:
            low, high = params['ranges']
        elif kind in ('armature_scale', 'kp_scale', 'kd_scale'):
            field = {'armature_scale': 'armature_range', 'kp_scale': 'stiffness_range',
                     'kd_scale': 'damping_range'}[kind]
            matches = [v for pattern, v in params[field].items() if re.fullmatch(pattern, labels[0])]
            if len(matches) != 1 or params.get('mode', 'uniform') != 'uniform':
                raise ValueError(f'Unsupported native motor DR range: {name}')
            low, high = matches[0]
            group = 'joint_' + kind + '_rms'
        elif kind in ('added_mass_kg', 'payload_com_offset') and event == 'stratified_limb_payload':
            from intact_tracking.preview_protocol import LIMBS
            limb = list(LIMBS).index(labels[0])
            if kind == 'added_mass_kg':
                low, high = 0., params['max_masses_kg'][limb]
            else:
                low, high = -params['com_half_width_m'], params['com_half_width_m']
                group = f'payload_com_offset/{labels[0]}/xyz_rms'
        else:
            raise ValueError(f'Unknown causal DR coordinate: {name}')
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError(f'Invalid native DR range: {name}')
        lower.append(float(low)); upper.append(float(high))
        groups.setdefault(group, []).append(i)
    weights = [0.] * len(names)
    for columns in groups.values():
        for i in columns:
            weights[i] = 1 / (len(groups) * len(columns))
    heavy = any(n.startswith('stratified_limb_payload/') for n in names)
    return {'version': 3 if heavy else 2,
            'profile': 'checkpoint_native_flat_heavy_v1' if heavy else 'checkpoint_native_flat_v1', 'names': list(names),
            'lower': lower, 'upper': upper, 'coordinate_weights': weights, 'groups': groups,
            'normalization': 'range normalization, then sqrt(factor-balanced coordinate weight)',
            'distance': 'RMS across physical factors; each motor block contributes its own RMS',
            'excluded': ['encoder bias/delay/smoothing: actual physical target supplied to predictor',
                         'time-varying force pulse: disturbance, not a persistent physical factor']}


class NativeDRTrackerRollout(FixedDRTrackerRollout):
    def __init__(self, config):
        if not config.checkpoint_native_dr:
            raise ValueError('Native collector requires checkpoint_native_dr')
        super().__init__(config)
        try:
            events = {n.split('/', 1)[0] for n in self.privileged_dynamics_names}
            params = {n: self.env.event_manager.get_term_cfg(n).params for n in events}
            defaults = {}
            for name, p in params.items():
                if any(n.startswith(name + '/relative_mass/') for n in self.privileged_dynamics_names):
                    ids, names = _entity_indices_and_names(self.env, p['asset_cfg'], 'body')
                    _, mass = _expanded_and_default_field(self.env, 'body_mass')
                    defaults.update(zip(names, mass[ids].cpu().tolist(), strict=True))
            self.dr_metric_schema = native_metric_schema(self.privileged_dynamics_names, params, defaults)
            self.dr_metric = normalize_dr_metric(self.privileged_dynamics, self.dr_metric_schema)
            self.replay_kwargs = {'dr_metric_dim': len(self.privileged_dynamics_names)}
            self.payload_configuration['runtime_audit'] = self.audit_native()
        except BaseException:
            self.close()
            raise

    def audit_native(self):
        action, ids = self.env.action_manager.get_term('joint_pos'), self.nominal_env_ids
        assert not bool(action.delay[ids].any()) and bool(action.alpha[ids].eq(1).all())
        assert not bool(self.env.scene['robot'].data.encoder_bias[ids].any())
        assert not bool(self.force_pulse.active[ids].any())
        return {'nominal_count': len(ids), 'dr_count': self.num_envs-len(ids),
                'nominal_fraction_actual': len(ids)/self.num_envs,
                'nominal_restore': self.nominal_restore_metrics,
                'nominal_delay_zero': True, 'nominal_alpha_one': True, 'nominal_pulses_disabled': True,
                'extra_payload': self.payload_configuration['extra_payload'],
                'heavy_payload': self.payload_configuration.get('heavy_payload'),
                'dr_metric_dimensions': self.dr_metric.shape[1],
                'dr_metric_factors': list(self.dr_metric_schema['groups']),
                'physical_targets': 'all four applied substeps recorded; mean target is the 29D predictor input',
                'counterfactual': 'nominal B replays every recorded physical substep target',
                'motion_sampling': self.motion_command.cfg.sampling_mode,
                'termination_terms': sorted(self.env.cfg.terminations),
                'episode_length_control_steps': self.env.max_episode_length}

    def _assert_fixed_dr(self, env_ids=None):
        super()._assert_fixed_dr(env_ids)
        if hasattr(self, 'dr_metric'):
            self.audit_native()

    def step(self, **kwargs):
        batch = super().step(**kwargs)
        batch['dr_metric'] = self.dr_metric
        return batch

    @property
    def metadata(self):
        return {**super().metadata, 'dr_metric_schema': self.dr_metric_schema,
                'nominal_world_local_ids': self.nominal_env_ids.cpu().tolist(),
                'predictor_action_transform': {'available': True, 'mode': 'captured_substep_physical_targets',
                                              'predictor_input': 'mean over one control step'},
                'memory_protocol': 'short50_disjoint_raw_chunk10_long30_v1',
                'episode_length_control_steps': self.env.max_episode_length}


def native_dataset_identity(checkpoint, motion_file=None, motion_path=None):
    from intact_tracking.environment.mdp.multi_commands import filter_excluded_motion_files
    cfg = _load_saved_config(Path(checkpoint))
    command = cfg.task.command.command
    root = Path(motion_path).resolve() if motion_path else Path(motion_file).resolve().parent
    files = sorted(root.rglob('*.npz')) if motion_path else [Path(motion_file).resolve()]
    paths = filter_excluded_motion_files([str(p) for p in files], motion_path=str(root),
        excluded_motion_files=tuple(command.get('excluded_motion_files', ())),
        motion_exclude_files=command.get('motion_exclude_files', ()),
        motion_exclude_file=command.get('motion_exclude_file', ''))
    selected = [Path(p) for p in paths]
    rows = [str(p.relative_to(root)) for p in selected]
    digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return selected, {'root': str(root), 'motion_count': len(selected), 'scanned_motion_count': len(files),
                      'excluded_motion_count': len(files)-len(selected), 'manifest_sha256': digest,
                      'manifest_hash_kind': 'ordered relative paths after checkpoint exclusions; not content hashes',
                      'coverage': 'complete checkpoint flat-data catalog with original exclusions; no terrain data'}
