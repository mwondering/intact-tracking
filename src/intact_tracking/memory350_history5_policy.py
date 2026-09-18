"""Five-frame compressed-concat PPO with the soft encoder's physical DR mixture."""
from functools import partial
import math
import torch

from intact_tracking.memory350_compressed_policy import configure_compressed_models, audit_initial_models
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.limb_context_dr import configure_limb_dr, audit_limb_dr
from intact_tracking.limb_context_sampling import configure_motion_sampling as original_motion_sampling
from intact_tracking.memory350_nominal_rollout import NominalExcludedForcePulse
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT
from intact_tracking.rollout.online import _restore_nominal_physics

VERSION = 'memory350_compressed_concat_history5_nominal10_mix50_v1'


def configure_models(train, fusion, *, scratch_seed=None):
    return configure_compressed_models(train, fusion, scratch_seed=scratch_seed, latent_history_frames=5)


def configure_motion_sampling(env_cfg, requested, after_update, completed_updates):
    result = original_motion_sampling(env_cfg, requested, after_update, completed_updates)
    result['scope'] = 'motion/bin sampling only; physical DR is specified separately in physics metadata'
    return result


def configure_physics(cfg, seed, *, profile, fixed_masses=None):
    # Fixed-load diagnostics retain background DR and respect the training caps.
    if fixed_masses is not None:
        if (len(fixed_masses) != 4 or any(not math.isfinite(mass) or not 0 <= mass <= limit
                for mass, limit in zip(fixed_masses, (2.5, 2.5, 4., 4.)))):
            raise ValueError("History5 payload limits are 2.5 kg per hand and 4 kg per shin")
        return configure_limb_dr(cfg, seed, profile=profile, fixed_masses=fixed_masses)
    result = configure_limb_dr(cfg, seed, profile=profile, nominal_probability=.5,
                               max_masses_kg=(2.5, 2.5, 4., 4.))
    result['nominal_population_fraction'] = .1
    result['nominal_population_rounding'] = 'ceil; at least 10% in each local population'
    result['force_pulse_scope'] = 'DR only; full nominal worlds excluded'
    return result


def minimum_nominal_ids(num_envs, fraction=.1, *, device="cpu"):
    if num_envs < 2 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Need a positive minimum fraction and at least two worlds")
    count = math.ceil(num_envs * fraction)
    return ((2 * torch.arange(count, device=device) + 1) * num_envs) // (2 * count)


def restore_nominal_population(env):
    ids = minimum_nominal_ids(env.num_envs, .1, device=env.device)
    metrics = _restore_nominal_physics(env, ids)
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    payload.mass[ids] = 0
    term = env.event_manager.get_term_cfg('push_robot')
    if type(term.func).__name__ != 'body_force_pulse':
        raise ValueError('Unexpected force pulse implementation')
    term.func = NominalExcludedForcePulse(term.func, ids)
    if not any(c is term for c in env.event_manager._mode_class_term_cfgs['step']):
        raise RuntimeError('Nominal pulse exclusion must cover reset callbacks')
    env.history5_nominal_audit = {'count':len(ids),'num_envs':env.num_envs,
        'fraction':len(ids)/env.num_envs,'restoration':metrics,'pulse_exclusion':True}
    return env


def environment_factory(factory, *, restore_nominal=True, **kwargs):
    env = factory(**kwargs)
    if restore_nominal:
        try:
            restore_nominal_population(env)
        except BaseException:
            env.close()
            raise
    return env


def audit_physics(env, config):
    term = env.event_manager.get_term_cfg('push_robot')
    wrapped = term.func
    if isinstance(wrapped, NominalExcludedForcePulse):
        # Audit the unchanged source DR contract; separately verify exclusion.
        term.func = wrapped.source
        try:
            result = audit_limb_dr(env, config)
        finally:
            term.func = wrapped
        ids = wrapped.nominal_ids
        assert not wrapped.source.active[ids].any()
        assert torch.isinf(wrapped.source.time_to_next_pulse_s[ids]).all()
        assert not env.event_manager.get_term_cfg(PAYLOAD_EVENT).func.mass[ids].any()
        result['nominal_population'] = env.history5_nominal_audit
        return result
    return audit_limb_dr(env, config)


History5Wrapper = partial(Memory350PolicyWrapper, latent_history_frames=5)
