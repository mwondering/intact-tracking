import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("limb_wandb_sync", SCRIPTS / "sync_limb_context_wandb.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class FakeRun:
    def __init__(self, step=0):
        self.step, self.summary, self.rows = step, {}, []
        self.url = "https://example.invalid/run"

    def define_metric(self, *args, **kwargs):
        self.metric_definitions = getattr(self, "metric_definitions", []) + [(args, kwargs)]

    def log(self, payload, step, commit=None):
        assert commit is True, "Each source record must commit without waiting for the next update"
        self.rows.append((step, payload))
        self.step = step + 1

    def save(self, *args, **kwargs):
        pass

    def _console_callback(self, name, text):
        self.console = getattr(self, "console", "") + text


def test_backfill_retries_partial_line_without_duplicate_updates(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = json.dumps({"completed_updates": 1, "loss": {"value": 0.5}}) + "\n"
    second = json.dumps({"completed_updates": 2, "mean_reward": 8})
    path.write_text(first + second[:12])
    tail = object.__new__(sync.Tail)
    tail.directory, tail.state = tmp_path, {}
    tail.offset, tail.last_update, tail.run = 0, -1, FakeRun()
    assert tail.read() == 1
    assert tail.read() == 0
    with path.open("a") as handle:
        handle.write(second[12:] + "\n")
    assert tail.read() == 1
    assert [step for step, _ in tail.run.rows] == [1, 2]
    assert tail.run.rows[0][1]["loss/value"] == 0.5
    tail.offset = 0
    assert tail.read() == 0


def test_resume_uses_remote_step_to_recover_unsent_local_rows(tmp_path, monkeypatch):
    remote = FakeRun(step=2)  # Server committed update 1; local cursor got ahead.
    monkeypatch.setattr(sync.wandb, "init", lambda **kwargs: remote)
    state = {"last_update": 2}
    args = SimpleNamespace(entity="team", project="project", stage1_project="predictor", root=tmp_path, state_dir=tmp_path)
    job = {"name": "baseline_121", "phase": "ppo", "output": str(tmp_path)}
    tail = sync.Tail(job, {"fusion": "baseline"}, state, args)
    (tmp_path / "metrics.jsonl").write_text(
        '\n'.join(json.dumps({"update": update, "fixed_probe": {"nmse": 1 / update}})
                  for update in (1, 2)) + '\n')
    assert tail.read() == 1
    assert remote.rows == [(2, {"update": 2, "fixed_probe/nmse": 0.5})]


def test_console_backfill_keeps_worker_isolation_and_retries_partial_line(tmp_path, monkeypatch):
    remote = FakeRun()
    monkeypatch.setattr(sync.wandb, "init", lambda **kwargs: remote)
    args = SimpleNamespace(entity="team", project="project", stage1_project="predictor", root=tmp_path, state_dir=tmp_path)
    job = {"name": "stage1", "phase": "encoder", "output": str(tmp_path / "stage1")}
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "baseline_121.log").write_text("other worker\n")
    log = logs / "stage1.log"
    log.write_text("initialization\nvalidation upd")
    tail = sync.Tail(job, {}, {}, args)
    tail.sync_log_files()
    assert remote.console == "initialization\n"
    tail.sync_log_files()
    assert remote.console == "initialization\n"
    with log.open("a") as handle:
        handle.write("ate 100\n")
    tail.sync_log_files()
    assert remote.console == "initialization\nvalidation update 100\n"


def test_stages_route_to_separate_projects_and_groups(tmp_path):
    args = SimpleNamespace(project="intact-preview-v2", stage1_project="intact-forward-predictor", root=tmp_path)
    stage1 = sync.job_destination({"name": "stage1", "phase": "encoder"}, args)
    stage2 = sync.job_destination({"name": "film_121", "phase": "ppo"}, args)
    smoke = sync.job_destination({"name": "smoke_stage1", "phase": "encoder"}, args)
    assert stage1 == ("intact-forward-predictor", f"{tmp_path.name}-stage1")
    assert stage2 == ("intact-preview-v2", f"{tmp_path.name}-stage2-2gpu8192-scratch")
    assert smoke == ("intact-forward-predictor", f"{tmp_path.name}-stage1-smoke")


def test_four_gpu_training_has_separate_group(tmp_path):
    (tmp_path / 'experiment_layout.json').write_text(json.dumps({
        'gpus_per_policy': 4, 'allowed_gpus': list(range(8)), 'controls': [], 'reuse_stage1': True}))
    args = SimpleNamespace(project='intact-preview-v2', stage1_project='intact-forward-predictor', root=tmp_path)
    assert sync.job_destination({'name': 'film_121', 'phase': 'ppo'}, args) == (
        'intact-preview-v2', f'{tmp_path.name}-stage2-4gpu8192-scratch')


def test_endpoint_chart_uses_checkpoint_update_even_when_resume_logs_later(tmp_path, monkeypatch):
    remote = FakeRun(step=204)
    monkeypatch.setattr(sync.wandb, "init", lambda **kwargs: remote)
    args = SimpleNamespace(entity="team", project="project", stage1_project="predictor", root=tmp_path, state_dir=tmp_path)
    tail = sync.Tail({"name": "baseline_121", "phase": "ppo", "output": str(tmp_path)}, {}, {}, args)
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"completed_updates": 204,
        "endpoint_eval": {"checkpoint_update": 203, "all_0": {"failure_rate": .01}, "all_4": {"failure_rate": .1}}}) + '\n')
    assert tail.read() == 1
    step, row = remote.rows[0]
    assert step == 204 and row["endpoint_eval/checkpoint_update"] == 203
    assert row["endpoint_eval/all_0/failure_rate"] == .01
    assert row["endpoint_eval/all_4/failure_rate"] == .1
    assert (("endpoint_eval/*",), {"step_metric": "endpoint_eval/checkpoint_update"}) in remote.metric_definitions
