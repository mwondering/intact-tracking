import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.limb_context_checkpoint_eval import (
    evaluation_due, evaluation_environment, validate_result,
    audit_saved_evaluations,
)
from intact_tracking.residual_runner import ResidualOnPolicyRunner


def test_endpoint_child_is_independent_of_parent_distributed_rank():
    inherited = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2",
                 "MASTER_ADDR": "host", "MASTER_PORT": "12345", "TORCHELASTIC_RUN_ID": "run",
                 "CUDA_VISIBLE_DEVICES": "2,3", "TMPDIR": "/project/tmp", "WANDB_DIR": "/project/wandb"}
    child = evaluation_environment(3, inherited)
    assert child["CUDA_VISIBLE_DEVICES"] == "3" and child["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert child["TMPDIR"] == "/project/tmp" and child["WANDB_DIR"] == "/project/wandb"
    assert not any(key in child for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_PORT", "TORCHELASTIC_RUN_ID"))
    assert inherited["WORLD_SIZE"] == "2"


def test_endpoint_result_rejects_wrong_checkpoint_update_or_payload():
    protocol = {"seed": 4, "steps": 500, "endpoints": {"all_4": [4] * 4}}
    row = {"checkpoint_sha256": "hash", "completed_training_updates": 200, "seed": 4,
           "max_steps": 500, "motion_files": ["motion"], "episodes": 1,
           "latent_intervention": "correct", "arguments": {"fixed_masses": [4] * 4},
           "actual_limb_masses_kg": [[4] * 4], "reference_timeline_audited": True,
           "partial_reset_survivor_state_audited": True, "partial_reset_survivor_history_audited": True}
    validate_result(row, protocol, ["motion"], "hash", 200, "all_4")
    for change in ({"completed_training_updates": 201}, {"checkpoint_sha256": "other"},
                   {"actual_limb_masses_kg": [[0] * 4]}):
        with pytest.raises(ValueError, match="contract mismatch"):
            validate_result({**copy.deepcopy(row), **change}, protocol, ["motion"], "hash", 200, "all_4")


def test_runner_evaluates_exact_completed_update_before_logging(tmp_path):
    events = []
    runner = object.__new__(ResidualOnPolicyRunner)
    runner.cfg = {"num_steps_per_env": 1, "check_for_nan": False, "save_interval": 100}
    runner.device, runner.is_distributed, runner.gpu_global_rank = "cpu", False, 0
    runner.stop_requested, runner.current_learning_iteration, runner.completed_learning_updates = False, 99, 99
    obs = torch.zeros(1, 1)
    runner.env = SimpleNamespace(device="cpu", get_observations=lambda: obs,
        step=lambda action: (obs, torch.zeros(1), torch.zeros(1), {}))
    runner.alg = SimpleNamespace(train_mode=lambda: None, act=lambda obs: obs,
        process_env_step=lambda *args: None, compute_returns=lambda obs: None, update=lambda: {},
        learning_rate=.001, get_policy=lambda: SimpleNamespace(output_std=torch.ones(1)))
    runner.logger = SimpleNamespace(writer=object(), log_dir=str(tmp_path),
        init_logging_writer=lambda: None, stop_logging_writer=lambda: None,
        process_env_step=lambda *args: None, log=lambda **kw: events.append(("log", kw["it"] + 1)))
    runner._begin_adaptive_sampling_iteration = lambda update: None
    runner._record_policy_action_mean = lambda: None
    runner._policy_diagnostics = lambda obs: {}
    runner.save = lambda path, **kw: events.append(("save", path))

    def evaluate(current):
        if evaluation_due(current.completed_learning_updates, 100):
            events.append(("evaluate", current.completed_learning_updates))

    runner.checkpoint_evaluator = evaluate
    runner.learn(2)
    assert events[:3] == [("evaluate", 100), ("log", 100), ("log", 101)]
    assert events[3:] == [("save", str(tmp_path / "checkpoint_final.pt"))]
    assert runner.completed_learning_updates == 101


def test_final_audit_rejects_a_missing_required_hundred_update_test(tmp_path):
    manifest = tmp_path / "motions.txt"
    manifest.write_text("motion\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"version": "limb_context_periodic_endpoints_v1",
        "interval_updates": 100, "motion_manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "motions": 1, "steps": 500, "endpoints": {"all_0": [0] * 4, "all_4": [4] * 4}}))
    with pytest.raises(ValueError, match="300"):
        audit_saved_evaluations(tmp_path, 300, protocol, 203)
