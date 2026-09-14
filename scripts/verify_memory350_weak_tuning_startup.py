"""Verify the live u8000 tuning branch, restored state, supervision, and fixed validation."""

import argparse
import json
import math
from pathlib import Path
import time

from intact_tracking.memory350_weak_tuning import TUNED_LOSS, validate_tuning_arguments, validate_tuning_losses
from run_memory350_scale_nominal_stage1 import read_json, write_json
from resume_memory350_weak_pairs import sha256


def verify(root):
    plan = read_json(root / 'launch_contract.json')
    output = root / 'stage1_8192'
    parent = Path(plan['parent_checkpoint']).parent
    config = read_json(output / 'run_config.json')
    previous = read_json(parent / 'run_config.json')
    progress = read_json(output / 'progress.json')
    assert progress and progress['completed_updates'] >= 8010
    assert progress['optimizer_steps'] == progress['completed_updates'] * 4
    assert progress['stop_after_updates'] == 10000
    validate_tuning_arguments(previous['arguments'], config['arguments'], 8000)
    validate_tuning_losses(previous['loss'], config['loss'])
    assert config['model'] == previous['model'] and config['optimization'] == previous['optimization']
    assert config['parameter_counts'] == previous['parameter_counts']
    assert config['matched_control']['physics_and_training_controls_passed']
    assert config['matched_control']['nominal_mixture_matched']
    resume = config['resume_history'][-1]
    assert resume['completed_update'] == 8000 and resume['optimizer_steps'] == 32000
    assert resume['schedule_extension'] is None
    assert resume['checkpoint_sha256'] == plan['parent_sha256'] == sha256(Path(plan['parent_checkpoint']))
    assert config['tuning_stage']['loss_changes'] == plan['loss_changes']
    assert config['tuning_stage']['learning_rate_schedule_unchanged']
    ranks = config['dataset']['runtime_audits_by_rank']
    assert len(ranks) == 4 and {r['physical_gpu'] for r in ranks} == {'4', '5', '6', '7'}
    assert all(r['nominal_training_worlds'] == r['dr_training_worlds'] == 4032 for r in ranks)
    assert config['dataset']['loaded_global_motion_count'] == 129827
    for name, digest in plan['validation_sha256'].items():
        assert sha256(parent / name) == sha256(output / name) == digest
    assert read_json(output / 'normalization.json') == read_json(parent / 'normalization.json')
    rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    first = rows[0]
    assert first['update'] == 8001 and first['optimizer_steps'] == 32004
    assert first['learning_rate_model'] == 1e-5
    train = first['optimization_train']
    assert all(math.isfinite(v) for v in train.values())
    assert all(math.isfinite(v) for v in first['fixed_probe'].values())
    expected = .004 * (train['weak_positive_loss'] + train['weak_negative_loss'])
    assert math.isclose(train['weak_pair_weighted_loss'], expected, rel_tol=1e-5, abs_tol=1e-8)
    assert train['gradient_norm'] > 0
    assert all(first['collector_memory'][f'gradient_norm_{name}'] > 0 for name in
               ('chunk_encoder', 'long_encoder', 'final_encoder'))
    assert all(config['loss'][key] == value for key, value in TUNED_LOSS.items())
    job = read_json(root / 'training_process.json')
    stat = Path(f'/proc/{job["pid"]}/stat').read_text().split()
    assert int(stat[21]) == job['process_start_ticks'] and stat[2] not in ('Z', 'T')
    result = {'passed': True, 'progress': progress, 'parent_checkpoint': plan['parent_checkpoint'],
              'parent_sha256': plan['parent_sha256'], 'loss_changes': plan['loss_changes'],
              'first_update': first['update'], 'first_optimizer_steps': first['optimizer_steps'],
              'learning_rate': first['learning_rate_model'], 'first_weighted_weak_loss': train['weak_pair_weighted_loss'],
              'first_weak_positive_pairs': train['weak_positive_pairs'], 'first_weak_negative_pairs': train['weak_negative_pairs'],
              'same_eight_validation_files_and_normalization': True, 'same_model_and_data': True,
              'physical_gpus': [4, 5, 6, 7], 'nominal_fraction': .5,
              'wandb': read_json(output / 'wandb_run.json'), 'unix_time': time.time()}
    write_json(root / 'startup_verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().run_root.resolve()), indent=2))
