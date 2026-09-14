from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.cli.forward_memory_weak_pairs_train import validate_continuation
from intact_tracking.forward_predictor_schedule import advance_predictor_schedule, extend_cosine_schedule


def _previous_config():
    return {
        'arguments': {'resume': None, 'stop_after_updates': 5000, 'updates': 8000,
                      'until_user_stop': True, 'continuation_min_learning_rate': 1e-5,
                      'weak_positive_weight': .002, 'weak_negative_weight': .002,
                      'response_distance_scale': .5, 'seed': 717, 'batch_size': 1024},
        'model': {'context_depth': 4}, 'loss': {'weak_positive_weight': .002},
        'objective_weights': {'weak_positive_weight': .002},
        'optimization': {'learning_rate': .0003}, 'distributed': {'world_size': 4},
    }


def test_continuation_allows_more_updates_but_preserves_training_settings():
    previous = _previous_config()
    actual = deepcopy(previous)
    actual['arguments'].update(resume='update_005000.pt', stop_after_updates=10000)
    result = validate_continuation(previous, actual, 5000)
    assert result['from_update'] == 5000 and result['stop_after_updates'] == 10000
    assert result['cosine_horizon_unchanged'] and result['losses_unchanged']


@pytest.mark.parametrize('key,value', [('weak_positive_weight', .004), ('weak_negative_weight', .004),
                                     ('response_distance_scale', .25), ('updates', 10000),
                                     ('continuation_min_learning_rate', 3e-5), ('seed', 718)])
def test_continuation_rejects_unrequested_hyperparameter_changes(key, value):
    previous = _previous_config()
    actual = deepcopy(previous)
    actual['arguments'].update(resume='update_005000.pt', stop_after_updates=10000)
    actual['arguments'][key] = value
    with pytest.raises(ValueError, match='preserve training settings'):
        validate_continuation(previous, actual, 5000)


def test_resume_keeps_the_original_lr_trajectory_through_and_after_cosine_endpoint():
    old_parameter = torch.nn.Parameter(torch.ones(()))
    old_optimizer = torch.optim.SGD([old_parameter], lr=.0003)
    old_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_optimizer, T_max=32000)
    for _ in range(20000):
        old_optimizer.step()
        advance_predictor_schedule(old_scheduler, 1e-5)
    restored_parameter = torch.nn.Parameter(torch.ones(()))
    restored_optimizer = torch.optim.SGD([restored_parameter], lr=.0003)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=32000)
    restored_optimizer.load_state_dict(old_optimizer.state_dict())
    restored_scheduler.load_state_dict(old_scheduler.state_dict())
    assert extend_cosine_schedule(restored_scheduler, 32000) is None
    assert restored_optimizer.param_groups[0]['lr'] == old_optimizer.param_groups[0]['lr']
    previous_lr = restored_optimizer.param_groups[0]['lr']
    for _ in range(20000):
        for optimizer, scheduler in [(old_optimizer, old_scheduler), (restored_optimizer, restored_scheduler)]:
            optimizer.step()
            advance_predictor_schedule(scheduler, 1e-5)
        lr = restored_optimizer.param_groups[0]['lr']
        assert lr == old_optimizer.param_groups[0]['lr'] and lr <= previous_lr
        previous_lr = lr
    assert restored_scheduler.last_epoch == 40000 and previous_lr == 1e-5


def _comparison_states():
    a = {'update': 5000, 'optimizer_steps': 20000, 'model_config': {'depth': 4},
         'normalization': {}, 'tracker': {'sha256': 'unchanged'}, 'nominal_a_fraction': .5,
         'loss_config': {'response_distance_scale': .5, 'weak_positive_weight': .002,
                         'weak_negative_weight': .002, 'weak_negative_margin': 1.},
         'scheduler': {'T_max': 32000, 'continuation_min_learning_rate': 1e-5,
                       'base_lrs': [.0003], 'eta_min': 0.}}
    b = deepcopy(a)
    b.update(update=6000, optimizer_steps=24000)
    return {'baseline': a, 'memory350': b}


def test_milestone_evaluation_rejects_changed_loss_or_unmatched_optimizer_budget(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluate_memory350_weak_pairs import validate_checkpoints
    states = _comparison_states()
    validate_checkpoints(states, 'continuation')
    for field, value in [('loss_config', {**states['memory350']['loss_config'], 'weak_positive_weight': .004}),
                         ('optimizer_steps', 4000), ('update', 5000),
                         ('scheduler', {**states['memory350']['scheduler'], 'T_max': 40000})]:
        changed = deepcopy(states)
        changed['memory350'][field] = value
        with pytest.raises(ValueError):
            validate_checkpoints(changed, 'continuation')


def test_milestone_checks_wait_for_atomic_checkpoint_and_run_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import resume_memory350_weak_pairs as supervisor
    journal = tmp_path / supervisor.JOURNAL
    journal.mkdir()
    stage = tmp_path / 'stage1_8192'
    stage.mkdir()
    (tmp_path / 'launch_contract.json').write_text(json.dumps({'cached_trajectories': '/fixed/cache'}))
    starts, reports = [], []
    def start(command, env, log_path):
        child = SimpleNamespace(returncode=None)
        child.poll = lambda: child.returncode
        starts.append((command, env, child))
        return child, {'pid': 99999, 'command': command}
    monkeypatch.setattr(supervisor, 'child_start', start)
    monkeypatch.setattr(supervisor, 'gpu_status', lambda gpus: [{'free_mib': 20000}])
    monkeypatch.setattr(supervisor, 'refresh_report', lambda root, env: reports.append(root))
    # An in-progress save cannot trigger evaluation.
    (stage / 'update_006000.pt.tmp').write_bytes(b'partial')
    assert supervisor.poll_evaluation(tmp_path, {}, None) is None and not starts
    (stage / 'update_006000.pt.tmp').rename(stage / 'update_006000.pt')
    (stage / 'update_007000.pt').write_bytes(b'complete')
    active = supervisor.poll_evaluation(tmp_path, {}, None)
    assert active[2] == 6000 and len(starts) == 1
    assert starts[0][1]['CUDA_VISIBLE_DEVICES'] == '7'
    assert str(stage / 'update_005000.pt') in starts[0][0]
    assert supervisor.poll_evaluation(tmp_path, {}, active) is active and len(starts) == 1
    folder = journal / 'comparison_006000'
    folder.mkdir()
    (folder / 'summary.json').write_text(json.dumps({'complete': True, 'assessment': {'text': 'checked'}}))
    active[0].returncode = 0
    following = supervisor.poll_evaluation(tmp_path, {}, active)
    assert following[2] == 7000 and len(starts) == 2 and len(reports) == 1
    record = json.loads((journal / 'evaluation_006000_result.json').read_text())
    assert record['complete'] and record['exit_code'] == 0
