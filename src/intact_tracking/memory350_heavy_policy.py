"""Heavy context physics reused by matched learned/zero-latent residual PPO."""

from copy import deepcopy
from types import SimpleNamespace

from intact_tracking import memory350_native_policy as native
from intact_tracking.memory350_heavy_dr import PROFILE, EVENT, MASS_LIMITS, configure_heavy_payload
from intact_tracking.memory350_proprio_inputs import INPUT_CONTRACT, validate_input_checkpoint
from intact_tracking.limb_context_dr import _event_contract
from intact_tracking.residual_dr_aux import dr_aux_layout


def validate_resume_models(old_train, train, args):
    """Allow only the explicitly requested all-payload -> mass-only transition."""
    previous, current = deepcopy(old_train), deepcopy(train)
    change = None
    if args.allow_dr_aux_change:
        if not args.resume or not (args.reset_adaptive_sampling or getattr(args, 'resume_uniform_sampling', False)):
            raise ValueError('Auxiliary transition requires resume and reset-adaptive-sampling or resume-uniform-sampling')
        old_actor, actor = previous['actor'], current['actor']
        mode = old_actor.get('latent_input_mode')
        old_coef, new_coef = previous['algorithm']['dr_aux_coef'], current['algorithm']['dr_aux_coef']
        if (mode not in ('learned','zero') or actor.get('latent_input_mode') != mode
                or len(old_actor.get('dr_aux_schema',{}).get('names',())) != 108
                or not old_actor.get('dr_aux_payload_com_enabled',True)
                or actor.get('dr_aux_payload_com_enabled') is not False
                or old_actor.get('dr_aux_motor_weight',0.) != 0.
                or (old_coef,new_coef) != ((.1,.5) if mode == 'learned' else (0.,0.))):
            raise ValueError('Only heavy COM removal with coefficient 0.1 -> 0.5 (baseline 0 -> 0) is allowed')
        old_actor.pop('dr_aux_payload_com_enabled',None)
        actor.pop('dr_aux_payload_com_enabled',None)
        current['algorithm']['dr_aux_coef'] = old_coef
        change = {'version':'heavy_payload_mass_only_keep_normalization_v1',
                  'coefficient_from':old_coef,'coefficient_to':new_coef,
                  'active_coordinates_from':20 if mode == 'learned' else 0,
                  'active_coordinates_to':8 if mode == 'learned' else 0,
                  'group_weight_denominator':8.,'retained_coordinate_weight_multiplier':5. if mode == 'learned' else None,
                  'model_optimizer_normalizers_and_std':'restored without modification'}
    for key in ('actor','critic','algorithm','obs_groups'):
        if previous[key] != current[key]:
            raise ValueError(f'Resume changed {key}')
    return change


def resolve_profile(profile, previous=None):
    if profile != PROFILE or (previous and previous['dr_profile'] != PROFILE):
        raise ValueError('Heavy residual physics profile mismatch')
    return PROFILE


def validate_context(state, profile):
    resolve_profile(profile)
    native.validate_context(state, native.PROFILE)
    validate_input_checkpoint(state)
    payload = state.get('heavy_payload_contract', {})
    if (state.get('dr_profile') != PROFILE
            or tuple(payload.get('max_masses_kg', ())) != MASS_LIMITS
            or payload.get('com_half_width_m') != .05
            or payload.get('mass_bins_per_limb') != 4
            or payload.get('mass_groups') != 256
            or payload.get('nominal_fraction') != .1):
        raise ValueError('Encoder must use the matched heavy mass/COM/nominal contract')
    heavy_aux_schema(state['dr_metric_schema'])


def native_aux_schema(schema):
    """Legacy 92-output schema for checkpoints trained before payload supervision."""
    names = schema['names']
    indices = [i for i, name in enumerate(names) if not name.startswith(EVENT + '/')]
    if len(names) != 108 or len(set(names)) != 108 or len(indices) != 92:
        raise ValueError('Heavy encoder schema must have 92 native and 16 payload coordinates')
    result = {key: [schema[key][i] for i in indices] for key in ('names', 'lower', 'upper')}
    result.update(source_profile=PROFILE, source_dimensions=108,
                  normalization='per-coordinate physical range; no encoder distance weights')
    dr_aux_layout(result)
    return result


def heavy_aux_schema(schema):
    """All physical ranges, without the context encoder's distance weights."""
    native_aux_schema(schema)
    result = {key: list(schema[key]) for key in ('names', 'lower', 'upper')}
    result.update(source_profile=PROFILE, source_dimensions=108,
                  normalization='per-coordinate physical range; no encoder distance weights')
    dr_aux_layout(result)
    return result


def configure_physics(cfg, seed, *, profile, rank):
    resolve_profile(profile)
    result = native.configure_physics(cfg, seed, profile=native.PROFILE)
    payload = configure_heavy_payload(cfg, SimpleNamespace(
        world_id_offset=rank * cfg.scene.num_envs, num_envs=cfg.scene.num_envs,
        nominal_fraction=.1, seed=seed, dynamics_seed=None,
        heavy_max_masses_kg=MASS_LIMITS, heavy_com_half_width_m=.05))
    result.update(profile=PROFILE, extra_payload=True, heavy_payload=payload,
                  encoder_action_input=INPUT_CONTRACT['action_source'],
                  encoder_state_input=INPUT_CONTRACT['state_sampling'],
                  sampling='10% compiled nominal; native independent DR plus 256 balanced Cartesian payload mass bins with independent continuous mass/xyz draws')
    result['original_events'][EVENT] = _event_contract(cfg.events[EVENT])
    return result


def configure_evaluation_physics(cfg, seed, *, profile, fixed_masses=None, physics='mixed'):
    """Select complete nominal/HDR worlds without altering the training profile."""
    if fixed_masses is not None:
        raise ValueError('Heavy evaluation uses the 256-bin continuous payload distribution')
    fraction = {'nominal': 1., 'hdr': 0., 'mixed': .1}[physics]
    result = configure_physics(cfg, seed, profile=profile, rank=0)
    cfg.actions['joint_pos'].nominal_fraction = fraction
    cfg.events[EVENT].params['nominal_fraction'] = fraction
    result['original_events'][EVENT] = _event_contract(cfg.events[EVENT])
    result['evaluation_physics'] = physics
    result['nominal_fraction_requested'] = fraction
    result['heavy_payload']['nominal_fraction'] = fraction
    result['sampling'] = (
        f'{fraction:.0%} compiled nominal; remaining worlds use native independent DR '
        'plus 256 balanced Cartesian payload mass bins with continuous mass/xyz draws')
    cfg.commands['motion'].sampling_mode = 'uniform'
    cfg.commands['motion'].rewind.enabled = False
    result['evaluation_sampling'] = 'fixed paired query starts; uniform frozen-tracker warmup'
    return result


def environment_factory(factory, **kwargs):
    env = native.environment_factory(factory, **kwargs)
    try:
        env.heavy_policy_payload = env.event_manager.get_term_cfg(EVENT).func
        native.audit_native_runtime(env)
        return env
    except BaseException:
        env.close()
        raise


def audit_physics(env, configuration):
    result = native.audit_physics(env, configuration)
    if result['parameter_dimensions'] != 108:
        raise ValueError('Heavy environment must expose all 108 physical parameters')
    if not result['heavy_payload']['actual_composite_inertial_fields_verified']:
        raise ValueError('Composite payload physics was not verified')
    return result
