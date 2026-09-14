from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_resume_loss_config
from intact_tracking.memory350_weak_margin_tuning import (
    PARENT_LOSS, FIXED_LOSS, TUNED_LOSS, validate_tuning_arguments, validate_tuning_losses,
)


def arguments():
    previous = dict(PARENT_LOSS, **FIXED_LOSS, output_dir='/parent', resume='/parent/update_008000.pt',
                    comparison_reference_dir='/reference', stop_after_updates=12000,
                    updates=8000, until_user_stop=True, continuation_min_learning_rate=1e-5,
                    seed=717, batch_size=1024, weak_negative_margin=1.)
    actual = dict(previous, **TUNED_LOSS)
    actual.update(output_dir='/new', resume='/parent/update_010000.pt', comparison_reference_dir='/parent')
    return previous, actual


def test_explicit_tuning_changes_only_three_losses_without_weakening_ordinary_resume():
    previous = dict(PARENT_LOSS, **FIXED_LOSS, recursive_weight=.5, weak_negative_margin=1.)
    actual = dict(previous, **TUNED_LOSS)
    validate_tuning_losses(previous, actual)
    _validate_resume_loss_config(previous, previous)
    with pytest.raises(ValueError, match='configuration changed'):
        _validate_resume_loss_config(previous, actual)
    with pytest.raises(ValueError, match='only weak weights'):
        validate_tuning_losses(previous, dict(actual, recursive_weight=.6))
    before, after = arguments()
    assert set(validate_tuning_arguments(before, after, 10000)) == set(TUNED_LOSS)


@pytest.mark.parametrize('key,value', [('continuation_min_learning_rate', 3e-5), ('updates', 10000),
                                     ('seed', 718), ('batch_size', 512), ('weak_negative_margin', 1.2), ('response_distance_scale', .2),
                                     ('output_dir', '/parent'), ('comparison_reference_dir', '/other'),
                                     ('stop_after_updates', 11000)])
def test_tuning_rejects_other_training_changes_and_parent_overwrite(key, value):
    previous, actual = arguments()
    actual[key] = value
    with pytest.raises(ValueError):
        validate_tuning_arguments(previous, actual, 10000)


def states():
    parent = {'model_config': {'depth': 4}, 'normalization': {}, 'tracker': {'frozen': True},
              'nominal_a_fraction': .5, 'update': 10000, 'optimizer_steps': 40000,
              'loss_config': dict(PARENT_LOSS, **FIXED_LOSS),
              'scheduler': {'T_max': 32000, 'last_epoch': 40000, 'continuation_min_learning_rate': 1e-5,
                            'base_lrs': [.0003], 'eta_min': 0.},
              'optimizer': {'param_groups': [{'lr': 1e-5}]}}
    candidate = deepcopy(parent)
    candidate.update(update=11000, optimizer_steps=44000)
    candidate['scheduler']['last_epoch'] = 44000
    candidate['loss_config'].update(TUNED_LOSS)
    return {'baseline': parent, 'memory350': candidate}


def test_tuning_evaluation_enforces_parent_losses_lr_and_optimizer_age(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluate_memory350_weak_margin import validate_checkpoints
    validate_checkpoints(states(), 'tuning')
    for field, value in [('optimizer_steps', 4000), ('update', 8000),
                         ('loss_config', dict(PARENT_LOSS, **FIXED_LOSS)),
                         ('normalization', {'changed': True})]:
        invalid = states()
        invalid['memory350'][field] = value
        with pytest.raises(ValueError):
            validate_checkpoints(invalid, 'tuning')
    invalid = states()
    invalid['memory350']['optimizer']['param_groups'][0]['lr'] = 3e-5
    with pytest.raises(ValueError, match='learning-rate floor'):
        validate_checkpoints(invalid, 'tuning')


def test_tuning_cannot_prepare_before_parent_u10000_evaluation_is_complete(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_memory350_weak_margin_tuning import prepare
    parent = tmp_path / 'parent'
    (parent / 'comparison_010000').mkdir(parents=True)
    summary = parent / 'comparison_010000/summary.json'
    summary.write_text(json.dumps({'complete': False}))
    with pytest.raises(RuntimeError, match='evaluation must finish'):
        prepare(tmp_path / 'new', parent)


def test_tuning_milestone_waits_for_atomic_checkpoint_and_one_evaluation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_memory350_weak_margin_tuning as supervisor
    stage = tmp_path / 'stage1_8192'
    stage.mkdir()
    plan = {'parent_checkpoint': '/parent/update_010000.pt', 'cached_trajectories': '/fixed/cache'}
    starts = []
    def start(command, env, log_path):
        child = SimpleNamespace(returncode=None)
        child.poll = lambda: child.returncode
        starts.append((command, env, child))
        return child, {'pid': 99999, 'command': command}
    monkeypatch.setattr(supervisor, 'child_start', start)
    monkeypatch.setattr(supervisor, 'gpu_status', lambda gpus: [{'free_mib': 20000}])
    monkeypatch.setattr(supervisor, 'refresh_report', lambda root, env: None)
    (stage / 'update_011000.pt.tmp').write_bytes(b'partial')
    assert supervisor.poll_evaluation(tmp_path, {}, plan, None) is None and not starts
    (stage / 'update_011000.pt.tmp').rename(stage / 'update_011000.pt')
    (stage / 'update_012000.pt').write_bytes(b'complete')
    active = supervisor.poll_evaluation(tmp_path, {}, plan, None)
    assert active[2] == 11000 and len(starts) == 1
    assert starts[0][1]['CUDA_VISIBLE_DEVICES'] == '7'
    assert starts[0][0][-2:] == ['--comparison-kind', 'tuning']
    assert supervisor.poll_evaluation(tmp_path, {}, plan, active) is active
    output = tmp_path / 'comparison_011000'
    output.mkdir()
    (output / 'summary.json').write_text(json.dumps({'complete': True}))
    active[0].returncode = 0
    following = supervisor.poll_evaluation(tmp_path, {}, plan, active)
    assert following[2] == 12000 and len(starts) == 2


def test_margin_11_adds_repulsion_for_a_pair_outside_the_old_margin():
    import math
    import torch
    from intact_tracking.memory350_weak_pairs import weak_pair_losses
    angle = torch.tensor(2 * math.asin(1.05 / 2), requires_grad=True)
    first = torch.stack((torch.ones_like(angle), torch.zeros_like(angle)))
    second = torch.stack((angle.cos(), angle.sin()))
    latent = torch.stack((first, second))
    valid = torch.ones(2, dtype=torch.bool)
    world = torch.tensor([1, 2])
    _, old_loss, old_stats = weak_pair_losses(latent, latent, ~valid, valid, world, margin=1.)
    _, new_loss, new_stats = weak_pair_losses(latent, latent, ~valid, valid, world, margin=1.1)
    assert old_loss == 0 and old_stats['weak_negative_active_fraction'] == 0
    assert new_loss > 0 and new_stats['weak_negative_active_fraction'] == 1
    (.008 * new_loss).backward()
    assert torch.isfinite(angle.grad) and angle.grad < 0  # Descent increases separation.


def test_margin_parser_preserves_response_scale_schedule_and_nominal50():
    from intact_tracking.cli.forward_memory_weak_margin_train import build_parser
    args = build_parser().parse_args(['--checkpoint-file', 'tracker.pt', '--motion-path', 'motions', '--output-dir', 'out'])
    assert all(getattr(args, key) == value for key, value in (TUNED_LOSS | FIXED_LOSS).items())
    assert args.nominal_fraction == .5 and args.updates == 8000
    assert args.stop_after_updates == 12000 and args.continuation_min_learning_rate == 1e-5


def test_margin_stage_does_not_accept_old_parent_or_a_shortened_training_budget():
    previous, actual = arguments()
    for update in (8000, 9000, 10001):
        with pytest.raises(ValueError):
            validate_tuning_arguments(previous, actual, update)
    with pytest.raises(ValueError):
        validate_tuning_losses(previous, dict(actual, response_distance_scale=.2))


@pytest.mark.parametrize('accuracy,met', [(.79, False), (.8, True), (.81, True)])
def test_margin_report_records_actual_threshold_without_reusing_prior_stage_text(tmp_path, monkeypatch, accuracy, met):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import evaluate_memory350_weak_margin as evaluator
    def original_report(summary, folder):
        (folder / 'README.md').write_text('弱正、弱负权重各从 0.002 翻倍至 0.004；response scale 从 0.5 改为 0.3；margin 保持 1.0。\n')
    monkeypatch.setattr(evaluator, '_base_report', original_report)
    summary = {'readout': {'memory_training': {'disjoint': {
        'models': {'memory350': {'top1_accuracy': accuracy}}, 'worlds': 292, 'test_samples': 2761}}}}
    evaluator.report(summary, tmp_path)
    text = (tmp_path / 'README.md').read_text()
    assert '0.008' in text and 'margin 从 1.0 改为 1.1' in text and 'scale 保持 0.3' in text
    assert '0.002' not in text
    assert summary['target']['met_on_fixed_diagnostic'] is met
