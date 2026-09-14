import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from intact_tracking.memory350_model import Memory350Config
from intact_tracking.memory350_scale_model import Memory350ScaleConfig
from intact_tracking.memory350_scale_nominal_comparison import reference_contract


def test_nominal_scale_defaults_match_original_nominal_except_depths():
    from intact_tracking.cli.forward_memory_nominal_train import build_parser as original_parser
    from intact_tracking.cli.forward_memory_scale_nominal_train import build_parser, _validate_arguments
    flags = ['--checkpoint-file', 'tracker.pt', '--motion-path', 'motions', '--output-dir', 'out']
    original = vars(original_parser().parse_args(flags))
    scaled = build_parser().parse_args(flags)
    with pytest.raises(ValueError, match='reference directory'):
        _validate_arguments(scaled)
    scaled.comparison_reference_dir = 'reference_nominal50'
    _validate_arguments(scaled)
    actual = vars(scaled).copy()
    assert actual.pop('stop_after_updates') is None
    for name in ('comparison_reference_dir', 'chunk_depth', 'memory_depth'):
        actual.pop(name)
    actual['context_depth'] = 2
    assert actual == original
    scaled.nominal_fraction = 0
    with pytest.raises(ValueError, match='nominal-fraction 0.5'):
        _validate_arguments(scaled)


def test_nominal_scale_guard_rejects_all_dr_reference_and_restoration_drift(tmp_path):
    audit = dict(rank=0, motion_count=42, motion_file_list_sha256='motions', loaded_frames=100,
                 training_worlds=8064, validation_worlds=128, num_envs=8192,
                 episode_length_control_steps=1000, dr_profile='tracker_dr_plus_limb_payload',
                 nominal_training_worlds=4032, dr_training_worlds=4032,
                 nominal_validation_worlds=64, dr_validation_worlds=64)
    audit['physics'] = dict(sampled_mass_sha256='sample', actual_mass_sha256='actual',
        original_events={}, observation_corruption={}, nominal_mixture=dict(
            nominal_count=4096, dr_count=4096, nominal_payload_max_abs_kg=0,
            nominal_pulses_disabled=True, dr_physics_unchanged_from_original_sampling=True,
            nominal_restore=dict(model_field_max_abs_error=0, encoder_bias_max_abs_error=0)))
    source = dict(arguments=dict(nominal_fraction=.5, context_depth=2), model=asdict(Memory350Config()),
                  dataset=dict(runtime_audits_by_rank=[audit]))
    (tmp_path / 'run_config.json').write_text(json.dumps(source))
    (tmp_path / 'normalization.json').write_text('{}')
    actual = copy.deepcopy(source)
    actual['arguments']['context_depth'] = 4
    actual['model'] = asdict(Memory350ScaleConfig())
    assert reference_contract(tmp_path, actual)['nominal_mixture_matched']
    changed = copy.deepcopy(actual)
    changed['dataset']['runtime_audits_by_rank'][0]['physics']['nominal_mixture']['nominal_pulses_disabled'] = False
    with pytest.raises(ValueError, match='physics audit differs'):
        reference_contract(tmp_path, changed)
    source['arguments']['nominal_fraction'] = 0
    (tmp_path / 'run_config.json').write_text(json.dumps(source))
    with pytest.raises(ValueError, match='50% nominal'):
        reference_contract(tmp_path, actual)


def test_nominal_scale_launcher_targets_authorized_gpus_with_memory_headroom(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_memory350_scale_nominal_stage1 import available, command_for, REFERENCE
    cards = [dict(index=i, free_mib=70000, processes=[dict(pid=1)]) for i in range(4, 8)]
    assert available(cards)
    cards[0]['free_mib'] = 59000
    assert not available(cards)
    command = command_for(tmp_path / 'stage1_8192', False)
    assert command[command.index('--nominal-fraction') + 1] == '0.5'
    assert 'intact_tracking.cli.forward_memory_scale_nominal_train' in command
    assert 'memory350_nominal50' in str(REFERENCE)


def test_paired_evaluation_keeps_nominal_error_out_of_dr_result():
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluate_memory350_scale import summarize
    samples = dict(original_mse=np.array([[10.] * 5, [1.] * 5]),
                   encoder2x_mse=np.array([[1.] * 5, [2.] * 5]),
                   unchanged_mse=np.full((2, 5), 2.), is_nominal=np.array([True, False]),
                   short_steps=np.array([50, 50]), long_chunks=np.array([30, 30]),
                   world_id=np.array([8190, 8191]))
    result = summarize(samples)
    assert result['nominal']['encoder2x_to_original_error_ratio'] == pytest.approx(.1)
    assert result['dr']['encoder2x_to_original_error_ratio'] == pytest.approx(2.)
    assert result['dr']['error_reduction_percent'] == pytest.approx(-100.)
