"""The specialist monitor tolerates concurrent status writes and can reattach."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def supervisor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("uniform_supervisor_test", scripts / "run_residual_uniform_specialists.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_replacement_keeps_last_complete_update(supervisor, tmp_path):
    path = tmp_path / "progress.json"
    previous = {"completed_updates": 4}
    assert supervisor.read_optional_json(path, previous) == previous
    path.write_text('{"completed_updates":')
    assert supervisor.read_optional_json(path, previous) == previous
    path.write_text('{"completed_updates": 5}')
    assert supervisor.read_optional_json(path, previous) == {"completed_updates": 5}


def test_status_permission_errors_are_not_hidden(supervisor, monkeypatch, tmp_path):
    def denied(_self):
        raise PermissionError("not a transient rewrite")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        supervisor.read_optional_json(tmp_path / "progress.json")


def test_jsonl_creation_and_partial_final_record(supervisor, tmp_path):
    path = tmp_path / "eval.jsonl"
    assert supervisor.evaluation_history(path) == []
    path.write_text('{"completed_updates": 1000}\n{"completed_updates":')
    assert supervisor.evaluation_history(path) == [{"completed_updates": 1000}]


def test_last_record_disappearing_during_open(supervisor, monkeypatch, tmp_path):
    def disappeared(_path):
        raise FileNotFoundError("replaced between exists and open")

    monkeypatch.setattr(supervisor, "_last_record", disappeared)
    assert supervisor.last_record(tmp_path / "metrics.jsonl") is None


def test_attach_checks_identity_and_signals_only_recorded_process(supervisor):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    child = subprocess.Popen(command)
    attached = None
    try:
        start = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
        record = {"pid": child.pid, "command": command, "process_start_ticks": start}
        with pytest.raises(ValueError, match="identity"):
            supervisor.AttachedProcess({**record, "process_start_ticks": str(int(start) + 1)})
        with pytest.raises(ValueError, match="identity"):
            supervisor.AttachedProcess({**record, "command": [sys.executable, "-c", "pass"]})
        assert child.poll() is None
        attached = supervisor.AttachedProcess(record)
        assert attached.poll() is None
        attached.start_ticks = str(int(start) + 1)
        attached.terminate()
        assert child.poll() is None
        attached.start_ticks = start
        attached.terminate()
        child.wait(timeout=5)
        assert attached.poll() is not None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
