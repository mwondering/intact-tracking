"""Differential checks against the exact source recorded by tracker 144000."""
import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.environment.mdp import multi_commands as current
from intact_tracking.limb_context_sampling import PARAMETERS, SamplingCheckpoint, validate_sampling_resume
from intact_tracking.memory350_native_policy import configure_sampling

REFERENCE = Path('/data_zcy/wxy/SP_Tracking/src/sp_tracking/tasks/tracking/mdp/multi_commands.py')
REFERENCE_SHA = '447a11cab33329867a2ef72cde042e3c56f79e8eb5059e3e85655f72e9815a35'
HELPERS = (
    '_resolve_adaptive_prior_counts', '_resolve_adaptive_ema_iterations', '_adaptive_ema_decay',
    '_temperature_scale_adaptive_signal', '_resolve_probability_cap', '_allocate_capped_mass',
    '_allocate_capped_mass_by_group', '_apply_final_probability_caps', '_sample_adaptive_uniform_branches',
)
METHODS = (
    '_compute_failure_rate', '_compute_motion_bin_indices', '_compute_adaptive_pair_sampling_probabilities',
    '_adaptive_random_probability', '_adaptive_sampling', '_init_adaptive_sampling_metrics',
    '_record_adaptive_sampling_distribution_metrics', '_restore_rewind_sampling_metrics',
    '_failure_mask', '_failure_rewind_env_ids', '_prepare_reset_sampling', '_resample_command',
    '_init_adaptive_sampling_ema', '_init_adaptive_visit_tracking', '_record_completed_adaptive_visits',
    '_sync_adaptive_bin_visits', '_finalize_adaptive_bin_visits', '_stage_pre_resample_adaptive_stats',
    '_accumulate_current_adaptive_sampling_stats', '_apply_adaptive_sampling_ema_update',
    'begin_adaptive_sampling_iteration',
)


def source_cfg():
    return SimpleNamespace(
        sampling_mode='adaptive', adaptive_sampling=current.AdaptiveSamplingCfg(
            strategy='branch', random_probability=.5, temperature=.25),
        rewind=current.RewindCfg(enabled=True, failure_probability=1/3, min_steps=0, max_steps=0),
        adaptive_uniform_ratio=.1, adaptive_bin_width_s=1., adaptive_bin_width_steps=None,
        adaptive_prior_visit_count=1., adaptive_prior_failure_count=0.,
        adaptive_failure_rate_ema_iterations=1000, adaptive_failure_rate_window_iterations=None,
        adaptive_probability_max_over_mean=200., adaptive_sequence_length_agnostic=False,
        adaptive_max_prob_per_bin='auto', adaptive_max_prob_per_motion='auto',
        adaptive_pre_failure_sample_window_steps=100, adaptive_bin_snapshot_interval_iterations=1,
        if_log_metrics=True, gradient_test_mode=None, synchronized_group_size=1,
    )


@pytest.fixture(scope='module')
def reference_class():
    if not REFERENCE.exists():
        pytest.skip('The pinned external tracker checkout is required for differential tests')
    raw = REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA, 'Tracker reference changed'
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiMotionCommand')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    functions += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_ADAPTIVE_SAMPLING_DISTRIBUTION_METRICS'
                         for t in n.targets)]
    namespace = {'torch': torch, 'math': math, 'sample_uniform': current.sample_uniform}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *constants, *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(REFERENCE), 'exec'), namespace)
    return type('Tracker144000Sampler', (), {name: namespace[name] for name in METHODS})


def make_sampler(cls, case):
    sampler = cls()
    sampler.cfg = source_cfg()
    sampler.device, sampler.num_envs = 'cpu', 96
    sampler.motion = SimpleNamespace(num_files=512, file_lengths=torch.arange(512) * 3 + 1)
    sampler._flat_motion_count = sampler.motion.num_files
    sampler._terrain_enabled = False
    sampler.bin_width_steps = 50
    sampler.bin_count = int(sampler.motion.file_lengths.max()) // 50 + 1
    sampler.motion_bin_counts = (sampler.motion.file_lengths + 49) // 50
    bin_ids = torch.arange(sampler.bin_count)
    sampler.bin_valid_mask = bin_ids[None] < sampler.motion_bin_counts[:, None]
    sampler.valid_motion_ids, sampler.valid_bin_ids = torch.where(sampler.bin_valid_mask)
    sampler.bin_lengths = (sampler.motion.file_lengths[:, None] - bin_ids[None] * 50).clamp(0, 50)
    lengths = sampler.bin_lengths.float()
    sampler.bin_weights = lengths / lengths[sampler.bin_valid_mask].mean()
    sampler.adaptive_prior_visit_count, sampler.adaptive_prior_failure_count = 1., 0.
    sampler.bin_visit_count = sampler.bin_valid_mask.float()
    sampler.bin_failure_count = torch.zeros_like(sampler.bin_visit_count)
    torch.manual_seed(623)
    if case == 'sparse':
        sampler.bin_visit_count[-8:] += 100 * sampler.bin_valid_mask[-8:]
        sampler.bin_failure_count[-8:] = 90 * sampler.bin_valid_mask[-8:]
    elif case == 'broad':
        sampler.bin_visit_count += torch.rand_like(lengths) * 200 * sampler.bin_valid_mask
        sampler.bin_failure_count = torch.rand_like(lengths) * sampler.bin_visit_count * .9
    sampler.motion_idx = torch.arange(sampler.num_envs) * 5
    sampler.motion_length = sampler.motion.file_lengths[sampler.motion_idx]
    sampler.time_steps = torch.minimum(torch.arange(sampler.num_envs) * 11, sampler.motion_length - 1)
    sampler.metrics = {}
    sampler._env = SimpleNamespace(
        episode_length_buf=torch.full((sampler.num_envs,), 200),
        termination_manager=SimpleNamespace(terminated=torch.arange(sampler.num_envs) % 3 == 0))
    sampler._adaptive_sampling_phase = 'idle'
    sampler._init_adaptive_sampling_ema()
    sampler._init_adaptive_visit_tracking()
    sampler._init_adaptive_sampling_metrics()
    sampler._invalidate_reference_cache = lambda: None
    sampler._set_motion_origin_offset = lambda ids: None
    sampler._reset_robot_to_reference = lambda ids: None
    sampler._synchronized_full_groups = lambda ids: ids
    sampler._broadcast_synchronized_motion = lambda ids: None
    # A real command samples every world at construction, before any failure
    # can trigger rewind; establish that initial per-world distribution here.
    torch.manual_seed(624)
    sampler._adaptive_sampling(torch.arange(sampler.num_envs))
    return sampler


def assert_same(reference, candidate):
    for name in ('motion_idx', 'time_steps', 'motion_length', 'bin_visit_count', 'bin_failure_count',
                 '_adaptive_pending_visit_count', '_adaptive_pending_failure_count',
                 '_adaptive_visit_motion_ids', '_adaptive_visit_bin_ids', '_skip_current_adaptive_visit_update'):
        torch.testing.assert_close(getattr(reference, name), getattr(candidate, name), rtol=0, atol=0)
    assert reference._adaptive_ema_last_iteration == candidate._adaptive_ema_last_iteration
    assert set(reference.metrics) == set(candidate.metrics)
    for name in reference.metrics:
        torch.testing.assert_close(reference.metrics[name], candidate.metrics[name], rtol=0, atol=0)


@pytest.mark.parametrize('case', ['fresh', 'sparse', 'broad'])
def test_distribution_draws_rewind_visits_and_ema_match_original(reference_class, case):
    candidate_class = type('ResidualSampler', (), {name: getattr(current.MultiMotionCommand, name)
                                                   for name in METHODS})
    reference, candidate = make_sampler(reference_class, case), make_sampler(candidate_class, case)
    ids = torch.arange(reference.num_envs)
    for seed, iteration in [(831, 0), (832, 1), (833, 14)]:
        rng_states = []
        for sampler in (reference, candidate):
            torch.manual_seed(seed)
            sampler._sync_adaptive_bin_visits(ids)
            sampler.time_steps = torch.minimum(sampler.time_steps + 53, sampler.motion_length - 1)
            sampler._sync_adaptive_bin_visits(ids)
            sampler._resample_command(ids)
            sampler._accumulate_current_adaptive_sampling_stats()
            sampler.begin_adaptive_sampling_iteration(iteration)
            rng_states.append(torch.random.get_rng_state())
        assert torch.equal(*rng_states)
        assert_same(reference, candidate)
    # The source algorithm caps the adaptive branch BEFORE the 50% uniform mix.
    adaptive = candidate._adaptive_sampling_metric_state['sampling_adaptive_top1_over_uniform']
    final = candidate._adaptive_sampling_metric_state['sampling_final_top1_over_uniform']
    torch.testing.assert_close(final, .5 * adaptive + .5, rtol=1e-5, atol=1e-5)


def test_native_sampling_preserves_entire_source_rewind_and_probability_contract():
    command = source_cfg()
    before = copy.deepcopy(command)
    configuration = configure_sampling(SimpleNamespace(commands={'motion': command}), 'adaptive', 0, 7001)
    assert command.rewind == before.rewind
    assert configuration['rewind'] == vars(before.rewind)
    assert configuration['failure_rewind_enabled']
    for name in PARAMETERS:
        assert getattr(command, name) == getattr(before, name)
    assert command.adaptive_sampling == before.adaptive_sampling
    assert command.adaptive_bin_snapshot_interval_iterations == 0


def test_alignment_keeps_aggregate_statistics_and_rejects_other_parameter_changes(tmp_path, monkeypatch):
    import intact_tracking.limb_context_protocol as protocol
    from intact_tracking.limb_context_sampling import STATE_FIELDS
    monkeypatch.setattr(protocol, 'PROJECT_ROOT', tmp_path)
    desired = configure_sampling(SimpleNamespace(commands={'motion': source_cfg()}), 'adaptive', 0, 7001)
    old = {k: copy.deepcopy(v) for k, v in desired.items()
           if k not in ('rewind', 'sampling_contract', 'reference_tracker_sha256')}
    old['failure_rewind_enabled'] = False
    previous = {'motion_sampling': old}
    with pytest.raises(ValueError, match='failure_rewind_enabled'):
        validate_sampling_resume(previous, desired, 7001)
    assert validate_sampling_resume(previous, desired, 7001, align_to_tracker=True) == 'adaptive'
    changed = copy.deepcopy(desired)
    changed['adaptive_sampling']['random_probability'] = .1
    with pytest.raises(ValueError, match='only restore'):
        validate_sampling_resume(previous, changed, 7001, align_to_tracker=True)
    command = SimpleNamespace(cfg=SimpleNamespace(sampling_mode='adaptive', rewind=SimpleNamespace(enabled=False)),
                              motion_files=('a.npz', 'b.npz'), _adaptive_ema_last_iteration=7000)
    for i, name in enumerate(STATE_FIELDS):
        setattr(command, name, torch.full((2, 3), float(i + 2)))
    runner = SimpleNamespace(completed_learning_updates=7001, env=SimpleNamespace(unwrapped=SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command))))
    dist = SimpleNamespace(rank=0, world_size=1, all_gather_object=lambda x: [x])
    SamplingCheckpoint(tmp_path/'old', dist, old).prepare(runner)
    runner.loaded_motion_sampling_state = copy.deepcopy(runner.motion_sampling_state)
    expected = {name: getattr(command, name).clone() for name in STATE_FIELDS}
    for name in STATE_FIELDS:
        getattr(command, name).zero_()
    command.cfg.rewind.enabled = True
    sampler = SamplingCheckpoint(tmp_path/'new', dist, desired)
    with pytest.raises(ValueError, match='different shard or sampling configuration'):
        sampler.restore(runner, 'adaptive')
    audit = sampler.restore(runner, 'adaptive', align_to_tracker=True)
    assert audit['restored'] and audit['tracker_sampling_aligned']
    assert command._adaptive_ema_last_iteration == 7000
    for name in STATE_FIELDS:
        torch.testing.assert_close(getattr(command, name), expected[name], rtol=0, atol=0)
    sampler.prepare(runner)
