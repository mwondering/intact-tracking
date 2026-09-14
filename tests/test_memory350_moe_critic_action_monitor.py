import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('run_memory350_moe_critic_action_ppo')
    monkeypatch.setattr(module, 'original_comparison', lambda root: None)
    return module


def write_arm(root, arm, *, action_dim=29, shared=False, updates=3, center_updates=3, loss=0.2):
    folder = root / 'ppo' / f'{arm}_121'
    folder.mkdir(parents=True)
    (folder / 'run_config.json').write_text(json.dumps({'input_audit': {
        'critic_tracker_action_input_dim': action_dim,
        'actor_critic_parameters_shared': shared}}))
    (folder / 'metrics.jsonl').write_text(json.dumps({'completed_updates': updates,
        'unix_time': 100, 'loss': {'value_function': loss, 'router_center_updates': center_updates}})+'\n')


def test_monitor_accepts_pending_startup_and_writes_durable_history(tmp_path, monitor):
    monitor.monitor_comparison(tmp_path)
    report = json.loads((tmp_path / 'health/latest.json').read_text())
    assert report['ok'] and report['arms']['concat']['latest_update'] is None
    assert (tmp_path / 'health/history.jsonl').exists()


@pytest.mark.parametrize('change', [
    {'action_dim': 0}, {'shared': True}, {'center_updates': 2}, {'loss': float('nan')}])
def test_monitor_rejects_broken_runtime_invariants(tmp_path, monitor, change):
    write_arm(tmp_path, 'baseline')
    write_arm(tmp_path, 'concat', **change)
    with pytest.raises(RuntimeError):
        monitor.monitor_comparison(tmp_path)
    assert not json.loads((tmp_path / 'health/latest.json').read_text())['ok']


def test_monitor_uses_last_complete_metric_and_retries_partial_comparison(tmp_path, monitor, monkeypatch):
    write_arm(tmp_path, 'baseline')
    write_arm(tmp_path, 'concat')
    with (tmp_path / 'ppo/concat_121/metrics.jsonl').open('a') as stream:
        stream.write('{"completed_updates":4')
    def partial(root):
        json.loads('{')
    monkeypatch.setattr(monitor, 'original_comparison', partial)
    monitor.monitor_comparison(tmp_path)
    report = json.loads((tmp_path / 'health/latest.json').read_text())
    assert report['ok'] and not report['comparison_updated']
    assert report['arms']['concat']['latest_update'] == 3
