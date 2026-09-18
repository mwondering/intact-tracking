import importlib
import json
from pathlib import Path
import signal

import pytest


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('run_soft20000_history5_ppo')


def test_encoder_stop_requires_known_live_workers_and_rechecks_pid(scheduler, monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler, 'verified_leader', lambda root: {'live': True, 'pid': 100})
    workers = [{'live': True, 'pid': i, 'start_ticks': i + 20} for i in range(1, 5)]
    monkeypatch.setattr(scheduler, 'worker_processes', lambda leader: workers)
    signals = []
    monkeypatch.setattr(scheduler.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(scheduler, 'process_identity', lambda pid, ticks: {'live': False})
    with pytest.raises(RuntimeError, match='identity changed'):
        scheduler.stop_encoder(tmp_path)
    assert signals == []
    monkeypatch.setattr(scheduler, 'process_identity', lambda pid, ticks: {'live': True})
    scheduler.stop_encoder(tmp_path)
    assert signals == [(1, signal.SIGTERM)]
    workers.pop()
    with pytest.raises(RuntimeError, match='four live'):
        scheduler.stop_encoder(tmp_path)
    assert len(signals) == 1


def test_comparison_ignores_partial_append_and_rejects_mismatched_protocol(scheduler, tmp_path):
    row = {'completed_updates': 100, 'protocol_sha256': 'protocol', 'policy_precision': 'fp32',
           'metrics': {'mixture_cold': {'error_body_pos': .1}}}
    for arm in ('baseline', 'concat'):
        path = tmp_path / 'ppo' / f'{arm}_121' / 'endpoint_eval_metrics.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row) + '\n{"completed_updates":200')
    assert scheduler.update_comparison(tmp_path) == [100]
    result = json.loads((tmp_path / 'ppo_comparison.json').read_text())
    assert result['maximum_updates'] is None
    row['protocol_sha256'] = 'changed'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='protocol or precision differs'):
        scheduler.update_comparison(tmp_path)


def test_source_or_artifact_edit_invalidates_preflight(scheduler, monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler, 'ROOT', tmp_path)
    source = tmp_path / 'source.py'
    artifact = tmp_path / 'protocol.json'
    source.write_text('validated')
    artifact.write_text('{}')
    ready = {'passed': True, 'source_sha256': {'source.py': scheduler.digest(source)},
             'artifact_sha256': {'protocol.json': scheduler.digest(artifact)}}
    (tmp_path / 'PPO_READY.json').write_text(json.dumps(ready))
    artifact.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match='Validated file changed: protocol.json'):
        scheduler.validate_ready(tmp_path)
