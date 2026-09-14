from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_resume_loss_config
from intact_tracking.memory350_weak_tuning import (
    PARENT_LOSS, TUNED_LOSS, validate_tuning_arguments, validate_tuning_losses,
)


def arguments():
    previous = dict(PARENT_LOSS, output_dir='/parent', resume='/parent/update_005000.pt',
                    comparison_reference_dir='/reference', stop_after_updates=10000,
                    updates=8000, until_user_stop=True, continuation_min_learning_rate=1e-5,
                    seed=717, batch_size=1024, weak_negative_margin=1.)
    actual = dict(previous, **TUNED_LOSS)
    actual.update(output_dir='/new', resume='/parent/update_008000.pt', comparison_reference_dir='/parent')
    return previous, actual


def test_explicit_tuning_changes_only_three_losses_without_weakening_ordinary_resume():
    previous = dict(PARENT_LOSS, recursive_weight=.5, weak_negative_margin=1.)
    actual = dict(previous, **TUNED_LOSS)
    validate_tuning_losses(previous, actual)
    _validate_resume_loss_config(previous, previous)
    with pytest.raises(ValueError, match='configuration changed'):
        _validate_resume_loss_config(previous, actual)
    with pytest.raises(ValueError, match='only weak weights'):
        validate_tuning_losses(previous, dict(actual, recursive_weight=.6))
    before, after = arguments()
    assert set(validate_tuning_arguments(before, after, 8000)) == set(TUNED_LOSS)


@pytest.mark.parametrize('key,value', [('continuation_min_learning_rate', 3e-5), ('updates', 10000),
                                     ('seed', 718), ('batch_size', 512), ('weak_negative_margin', 1.2),
                                     ('output_dir', '/parent'), ('comparison_reference_dir', '/other'),
                                     ('stop_after_updates', 11000)])
def test_tuning_rejects_other_training_changes_and_parent_overwrite(key, value):
    previous, actual = arguments()
    actual[key] = value
    with pytest.raises(ValueError):
        validate_tuning_arguments(previous, actual, 8000)


def states():
    parent = {'model_config': {'depth': 4}, 'normalization': {}, 'tracker': {'frozen': True},
              'nominal_a_fraction': .5, 'update': 8000, 'optimizer_steps': 32000,
              'loss_config': dict(PARENT_LOSS, weak_negative_margin=1.),
              'scheduler': {'T_max': 32000, 'last_epoch': 32000, 'continuation_min_learning_rate': 1e-5,
                            'base_lrs': [.0003], 'eta_min': 0.},
              'optimizer': {'param_groups': [{'lr': 1e-5}]}}
    candidate = deepcopy(parent)
    candidate.update(update=9000, optimizer_steps=36000)
    candidate['scheduler']['last_epoch'] = 36000
    candidate['loss_config'].update(TUNED_LOSS)
    return {'baseline': parent, 'memory350': candidate}


def test_tuning_evaluation_enforces_parent_losses_lr_and_optimizer_age(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluate_memory350_weak_pairs import validate_checkpoints
    validate_checkpoints(states(), 'tuning')
    for field, value in [('optimizer_steps', 4000), ('update', 8000),
                         ('loss_config', dict(PARENT_LOSS, weak_negative_margin=1.)),
                         ('normalization', {'changed': True})]:
        invalid = states()
        invalid['memory350'][field] = value
        with pytest.raises(ValueError):
            validate_checkpoints(invalid, 'tuning')
    invalid = states()
    invalid['memory350']['optimizer']['param_groups'][0]['lr'] = 3e-5
    with pytest.raises(ValueError, match='learning-rate floor'):
        validate_checkpoints(invalid, 'tuning')


def test_tuning_cannot_prepare_before_original_u8000_evaluation_is_complete(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_memory350_weak_tuning import prepare
    parent = tmp_path / 'parent'
    (parent / 'continuation_005000_010000/comparison_008000').mkdir(parents=True)
    summary = parent / 'continuation_005000_010000/comparison_008000/summary.json'
    summary.write_text(json.dumps({'complete': False}))
    with pytest.raises(RuntimeError, match='evaluation must finish'):
        prepare(tmp_path / 'new', parent)


def test_tuning_milestone_waits_for_atomic_checkpoint_and_one_evaluation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_memory350_weak_tuning as supervisor
    stage = tmp_path / 'stage1_8192'
    stage.mkdir()
    plan = {'parent_checkpoint': '/parent/update_008000.pt', 'cached_trajectories': '/fixed/cache'}
    starts = []
    def start(command, env, log_path):
        child = SimpleNamespace(returncode=None)
        child.poll = lambda: child.returncode
        starts.append((command, env, child))
        return child, {'pid': 99999, 'command': command}
    monkeypatch.setattr(supervisor, 'child_start', start)
    monkeypatch.setattr(supervisor, 'gpu_status', lambda gpus: [{'free_mib': 20000}])
    monkeypatch.setattr(supervisor, 'refresh_report', lambda root, env: None)
    (stage / 'update_009000.pt.tmp').write_bytes(b'partial')
    assert supervisor.poll_evaluation(tmp_path, {}, plan, None) is None and not starts
    (stage / 'update_009000.pt.tmp').rename(stage / 'update_009000.pt')
    (stage / 'update_010000.pt').write_bytes(b'complete')
    active = supervisor.poll_evaluation(tmp_path, {}, plan, None)
    assert active[2] == 9000 and len(starts) == 1
    assert starts[0][1]['CUDA_VISIBLE_DEVICES'] == '7'
    assert starts[0][0][-2:] == ['--comparison-kind', 'tuning']
    assert supervisor.poll_evaluation(tmp_path, {}, plan, active) is active
    output = tmp_path / 'comparison_009000'
    output.mkdir()
    (output / 'summary.json').write_text(json.dumps({'complete': True}))
    active[0].returncode = 0
    following = supervisor.poll_evaluation(tmp_path, {}, plan, active)
    assert following[2] == 10000 and len(starts) == 2
