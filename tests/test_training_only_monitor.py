import importlib
from pathlib import Path

import pytest


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('resume_soft20000_training_only')
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    return module


def test_progress_replacement_race_recovers(monitor, monkeypatch):
    calls = []

    def read(path):
        calls.append(path)
        if len(calls) < 3:
            raise FileNotFoundError(path)
        return {'completed_updates': 1501}

    monkeypatch.setattr(monitor, 'read', read)
    row, error = monitor.read_progress(Path('progress.json'), {'completed_updates': 1500})
    assert row['completed_updates'] == 1501 and error is None


def test_unavailable_progress_is_explicitly_stale_and_keeps_last_update(monitor, monkeypatch):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(monitor, 'read', missing)
    previous = {'completed_updates': 1500, 'unix_time': 10}
    row, error = monitor.read_progress(Path('progress.json'), previous)
    assert row == previous and row is not previous
    assert error['stale'] and 'FileNotFoundError' in error['error']


def test_attach_monitor_does_not_launch_trainers(monitor, monkeypatch, tmp_path):
    writes = []

    def read(path):
        if path.name.endswith('_process.json'):
            return {'pid': 100, 'start_ticks': 7}
        return {'completed_updates': 1501}

    monkeypatch.setattr(monitor, 'read', read)
    monkeypatch.setattr(monitor, 'process_identity', lambda *args: {'live': True})
    monkeypatch.setattr(monitor, 'verify_started', lambda *args: True)
    monkeypatch.setattr(monitor, 'write', lambda path, value: writes.append((path, value)))
    monkeypatch.setattr(monitor.subprocess, 'Popen', lambda *a, **k: pytest.fail('Monitor launched a process'))

    def finish(_):
        raise InterruptedError('End one observation cycle')

    monkeypatch.setattr(monitor.time, 'sleep', finish)
    with pytest.raises(InterruptedError):
        monitor.monitor(tmp_path, {arm: {'completed_updates': 800} for arm in monitor.ARMS})
    state = next(value for path, value in writes if path.name == 'state.json')
    assert not state['evaluations_enabled'] and not state['progress_read_errors']
    assert all(row['completed_updates'] == 1501 for row in state['progress'].values())


def test_monitor_single_arm_ignores_stopped_baseline(monitor, monkeypatch, tmp_path):
    writes = []

    def read(path):
        assert 'baseline' not in str(path)
        if path.name.endswith('_process.json'):
            return {'pid': 100, 'start_ticks': 7}
        return {'completed_updates': 1501}

    monkeypatch.setattr(monitor, 'read', read)
    monkeypatch.setattr(monitor, 'process_identity', lambda *args: {'live': True})
    monkeypatch.setattr(monitor, 'verify_started', lambda *args: True)
    monkeypatch.setattr(monitor, 'write', lambda path, value: writes.append((path, value)))
    monkeypatch.setattr(monitor.subprocess, 'Popen', lambda *a, **k: pytest.fail('Monitor launched a process'))

    def finish(_):
        raise InterruptedError('End one observation cycle')

    monkeypatch.setattr(monitor.time, 'sleep', finish)
    with pytest.raises(InterruptedError):
        monitor.monitor(tmp_path, {'concat': {'completed_updates': 800}}, arms=['concat'])
    state = next(value for path, value in writes if path.name == 'state.json')
    assert state['status'] == 'ppo_training_only'
    assert state['assignments'] == {'concat': [0, 1, 2, 3]}
    assert state['inactive_arms'] == ['baseline']
    assert set(state['progress']) == {'concat'} and not state['evaluations_enabled']
