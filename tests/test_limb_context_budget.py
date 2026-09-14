import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

from intact_tracking.forward_predictor_schedule import (
    extend_cosine_schedule, advance_predictor_schedule, training_update_indices,
)


def test_unbounded_updates_and_lr_tail_continue_past_original_horizon():
    from itertools import islice
    assert list(islice(training_update_indices(8,10,True),6)) == [8,9,10,11,12,13]
    assert list(training_update_indices(8,10,False)) == [8,9,10]
    weight=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([weight],lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    rates=[]
    for _ in range(17):
        weight.grad=torch.ones_like(weight)
        optimizer.step()
        advance_predictor_schedule(scheduler,.0001)
        rates.append(optimizer.param_groups[0]['lr'])
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    assert rates[-5:]==[.0001]*5 and scheduler.last_epoch==17
    other=torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.]))],lr=.001)
    restored=torch.optim.lr_scheduler.CosineAnnealingLR(other,T_max=10)
    other.load_state_dict(optimizer.state_dict())
    restored.load_state_dict(scheduler.state_dict())
    for _ in range(5):
        other.step()
        advance_predictor_schedule(restored,.0001)
        assert other.param_groups[0]['lr']==.0001
    assert restored.last_epoch==22

spec = importlib.util.spec_from_file_location("limb_scheduler_budget", Path(__file__).resolve().parents[1] / "scripts/run_limb_context_experiment.py")
scheduler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler_module)


def test_extended_cosine_preserves_lr_and_optimizer_until_new_endpoint():
    weight = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([weight], lr=.0003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    for _ in range(23):
        weight.grad = torch.ones_like(weight)
        optimizer.step()
        schedule.step()
    previous = copy.deepcopy(optimizer.state_dict())
    lr = optimizer.param_groups[0]["lr"]
    record = extend_cosine_schedule(schedule, 80)
    assert record["anchor_optimizer_step"] == 23
    assert optimizer.param_groups[0]["lr"] == lr
    assert schedule.last_epoch == 23
    for name, value in previous["state"][0].items():
        assert torch.equal(value, optimizer.state_dict()["state"][0][name])
    rates = [lr]
    for step in range(24, 81):
        optimizer.step()
        schedule.step()
        rates.append(optimizer.param_groups[0]["lr"])
        if step == 60:
            assert rates[-1] > 0
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert rates[-1] == 0


def test_extension_survives_checkpoint_resume_and_rejects_shortening():
    weight = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([weight], lr=.0003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    for _ in range(23):
        optimizer.step()
        schedule.step()
    extend_cosine_schedule(schedule, 80)
    other = torch.optim.SGD([torch.nn.Parameter(torch.tensor([2.]))], lr=.0003)
    restored = torch.optim.lr_scheduler.CosineAnnealingLR(other, T_max=80)
    other.load_state_dict(optimizer.state_dict())
    restored.load_state_dict(schedule.state_dict())
    assert extend_cosine_schedule(restored, 80) is None
    with pytest.raises(ValueError):
        extend_cosine_schedule(restored, 60)
    for _ in range(57):
        optimizer.step()
        schedule.step()
        other.step()
        restored.step()
        assert other.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_old_cap_does_not_release_latent_ppo_dependency(tmp_path):
    job = {"phase": "encoder", "output": str(tmp_path), "command": ["--updates", "8000"]}
    path = tmp_path / "completion.json"
    path.write_text(json.dumps({"completed_updates": 6000, "hit_cap": True, "converged": False}))
    assert scheduler_module.job_completion(job)[0] is False
    path.write_text(json.dumps({"completed_updates": 8000, "hit_cap": True, "converged": False}))
    assert scheduler_module.job_completion(job)[0] is True


def test_adoption_observes_original_process_and_validates_identity(tmp_path):
    command = ["-c", "import sys; sys.stdin.read()"]
    child = subprocess.Popen([scheduler_module.PYTHON, "-u", *command], stdin=subprocess.PIPE)
    job = {"phase": "ppo", "output": str(tmp_path), "command": ["--iterations", "5000"]}
    previous = {"pid": child.pid, "command": command, "status": "running"}
    try:
        assert scheduler_module.saved_worker_is_running(previous)
        assert not scheduler_module.saved_worker_is_running({**previous, "command": ["old command from a reused PID"]})
        adopted = scheduler_module.AdoptedProcess(previous, job)
        assert adopted.pid == child.pid and adopted.poll() is None
        with pytest.raises(RuntimeError):
            scheduler_module.AdoptedProcess({**previous, "command": ["wrong"]}, job)
        (tmp_path / "completion.json").write_text(json.dumps({"complete": True, "completed_updates": 5000,
            "initialization_protocol": scheduler_module.PPO_INITIALIZATION,
            "distributed": {"world_size": 2, "num_envs_per_rank": 8192, "global_num_envs": 16384},
            "distributed_parameter_agreement": {"passed": True, "world_size": 2}}))
        child.stdin.close()
        child.wait(timeout=10)
        assert adopted.poll() == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


def test_context200_stage1_queue_preserves_training_settings_and_has_no_ppo(tmp_path):
    before = {job["name"]: job for job in scheduler_module.jobs(tmp_path)}
    (tmp_path / "experiment_layout.json").write_text(json.dumps({
        "gpus_per_policy": 4, "allowed_gpus": list(range(8)), "controls": [],
        "stage1_only": True, "context_history_steps": 200, "encoder_gpu_pool": [0, 1, 2, 3],
    }))
    after = {job["name"]: job for job in scheduler_module.jobs(tmp_path)}
    assert set(after) == {"nominal_audit", "smoke_stage1", "stage1"}
    assert scheduler_module.formal_policies(tmp_path) == []
    for name in ("smoke_stage1", "stage1"):
        assert after[name]["command"] == before[name]["command"] + ["--context-history-steps", "200"]
        assert after[name]["gpus_needed"] == 4 and after[name]["gpu_pool"] == [0, 1, 2, 3]
        assert "--resume" not in after[name]["command"]


def test_launch_hold_waits_for_exact_completion_and_updates_without_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_module, "ROOT", tmp_path)
    job = {"name": "stage1"}
    assert scheduler_module.pending_job_hold(tmp_path, job) is None
    (tmp_path / "launch_holds.json").write_text(json.dumps({"jobs": {"stage1": {
        "completion_file": "predecessor.json", "required_values": {"complete": True, "completed_updates": 5000},
        "reason": "Wait for the active PPO to finish",
    }}}))
    assert scheduler_module.pending_job_hold(tmp_path, job)
    path = tmp_path / "predecessor.json"
    path.write_text(json.dumps({"complete": False, "completed_updates": 1000}))
    assert scheduler_module.pending_job_hold(tmp_path, job)
    path.write_text(json.dumps({"complete": True, "completed_updates": 5000}))
    assert scheduler_module.pending_job_hold(tmp_path, job) is None
    assert scheduler_module.pending_job_hold(tmp_path, {"name": "unrelated"}) is None
