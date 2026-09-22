import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.limb_context_sampling import (
    STATE_FIELDS, VERSION, SamplingCheckpoint, active_mode, audit_sampling_curriculum, phase_budget,
    state_digest, validate_sampling_resume,
)


def config(mode="adaptive", boundary=1000):
    return dict(version=VERSION, requested_mode="adaptive", adaptive_after_update=boundary,
                active_mode=mode, adaptive_sampling={"random_probability": .5},
                parameters={"adaptive_bin_width_s": 1.}, failure_rewind_enabled=False)


def test_uniform_prefix_stops_exactly_at_boundary_and_resume_uses_adaptive():
    assert active_mode("adaptive", 1000, 999) == "uniform"
    assert phase_budget("adaptive", 1000, 999, 5000) == 1
    assert active_mode("adaptive", 1000, 1000) == "adaptive"
    assert phase_budget("adaptive", 1000, 1000, 5000) == 4000
    assert phase_budget("adaptive", 1000, 0, 600) == 600
    assert validate_sampling_resume({}, config(), 1000) == "uniform"
    for update in (999, 1001, 1046):
        with pytest.raises(ValueError, match="exact curriculum"):
            validate_sampling_resume({}, config(), update)
    old = {"motion_sampling": config()}
    assert validate_sampling_resume(old, config(), 1100) == "adaptive"
    changed = config()
    changed["adaptive_sampling"]["random_probability"] = .1
    with pytest.raises(ValueError, match="adaptive_sampling"):
        validate_sampling_resume(old, changed, 1100)


def test_sampler_restores_learned_counts_and_rejects_changed_shards_or_files(tmp_path, monkeypatch):
    import intact_tracking.limb_context_protocol as protocol
    monkeypatch.setattr(protocol, "PROJECT_ROOT", tmp_path)
    command = SimpleNamespace(cfg=SimpleNamespace(sampling_mode="adaptive", rewind=SimpleNamespace(enabled=False)),
                              motion_files=("a.npz", "b.npz"), _adaptive_ema_last_iteration=1002)
    for index, key in enumerate(STATE_FIELDS):
        setattr(command, key, torch.full((2, 3), float(index + 1)))
    dist = SimpleNamespace(rank=0, world_size=1, all_gather_object=lambda value: [value])
    runner = SimpleNamespace(completed_learning_updates=1003,
        env=SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))))
    sampler = SamplingCheckpoint(tmp_path, dist, config())
    expected = {key: getattr(command, key).clone() for key in STATE_FIELDS}
    sampler.prepare(runner)
    record = copy.deepcopy(runner.motion_sampling_state)
    sampler.prepare(runner)  # A repeated final save is idempotent.
    runner.loaded_motion_sampling_state = record
    for key in STATE_FIELDS:
        getattr(command, key).zero_()
    command._adaptive_ema_last_iteration = None
    assert sampler.restore(runner, "adaptive")["restored"]
    assert command._adaptive_ema_last_iteration == 1002
    for key in STATE_FIELDS:
        assert torch.equal(getattr(command, key), expected[key])
    command.motion_files = ("b.npz", "a.npz")
    with pytest.raises(ValueError, match="different shard"):
        sampler.restore(runner, "adaptive")
    command.motion_files = ("a.npz", "b.npz")
    path = Path(record["ranks"][0]["path"])
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        sampler.restore(runner, "adaptive")


def test_state_digest_detects_optimizer_and_normalizer_changes():
    value = {"optimizer": {0: {"step": torch.tensor(10.), "exp_avg": torch.ones(4)}},
             "normalizer": torch.tensor([1., 2.])}
    duplicate = copy.deepcopy(value)
    assert state_digest(value) == state_digest(duplicate)
    duplicate["optimizer"][0]["exp_avg"][0] += .01
    assert state_digest(value) != state_digest(duplicate)


def test_explicit_sampler_reset_ignores_old_files_and_restarts_ema(tmp_path, monkeypatch):
    from intact_tracking.environment.mdp.multi_commands import MultiMotionCommand

    command = SimpleNamespace(
        cfg=SimpleNamespace(sampling_mode="adaptive", rewind=SimpleNamespace(enabled=False)),
        motion_files=("a.npz", "b.npz"), _adaptive_ema_last_iteration=6007,
        adaptive_prior_visit_count=1., adaptive_prior_failure_count=0.,
        bin_valid_mask=torch.tensor([[True, True, False], [True, False, False]]),
        _adaptive_visit_motion_ids=torch.tensor([0, 1]),
        _adaptive_visit_bin_ids=torch.tensor([1, 0]),
    )
    for index, key in enumerate(STATE_FIELDS):
        setattr(command, key, torch.full((2, 3), float(index + 3)))
    model_state = {"weights": torch.tensor([3., 4.]),
                   "optimizer": {"step": torch.tensor(6008.), "exp_avg": torch.tensor([.2, .3])}}
    digest_before = state_digest(model_state)
    runner = SimpleNamespace(completed_learning_updates=6008, alg=model_state,
        loaded_motion_sampling_state={"ranks": [{"path": "/missing/old_statistics.pt"}]},
        env=SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))))
    dist = SimpleNamespace(rank=0, world_size=1, all_gather_object=lambda value: [value])
    sampler = SamplingCheckpoint(tmp_path, dist, config(boundary=0))
    with monkeypatch.context() as patch:
        def reject_load(*args, **kwargs):
            pytest.fail("An explicit sampler reset must not read old statistics")
        patch.setattr(torch, "load", reject_load)
        report = sampler.restore(runner, "adaptive", reset=True)
    assert report["reset_to_priors"] and not report["restored"]
    assert report["completed_updates"] == runner.completed_learning_updates == 6008
    assert report["visit_prior_max_error"] == report["failure_prior_max_error"] == 0
    assert runner.loaded_motion_sampling_state is None and runner.motion_sampling_state is None
    assert state_digest(model_state) == digest_before
    torch.testing.assert_close(command.bin_visit_count, command.bin_valid_mask.float())
    for key in STATE_FIELDS[1:]:
        assert not getattr(command, key).any()
    assert command._adaptive_ema_last_iteration is None
    assert command._adaptive_visit_motion_ids.tolist() == [0, 1]
    assert command._adaptive_visit_bin_ids.tolist() == [1, 0]
    # The next global update establishes a fresh EMA clock without applying
    # thousands of iterations of decay or skipping the next new statistics.
    MultiMotionCommand.begin_adaptive_sampling_iteration(command, 6008)
    assert command._adaptive_ema_last_iteration == 6008
    sampler.prepare(runner)
    header = runner.motion_sampling_state["ranks"][0]
    saved = torch.load(header["path"], weights_only=False)
    assert saved["completed_updates"] == 6008
    assert saved["ema_last_iteration"] == 6008
    torch.testing.assert_close(saved["statistics"]["bin_visit_count"], command.bin_valid_mask.float())


def test_sampler_reset_rejects_inactive_adaptive_phase(tmp_path):
    sampler = SamplingCheckpoint(tmp_path, None, config(mode="uniform"))
    with pytest.raises(ValueError, match="active adaptive"):
        sampler.restore(None, "uniform", reset=True)


def test_scheduler_requires_exact_checkpoint_for_planned_transition(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_limb_context_experiment.py"
    spec = importlib.util.spec_from_file_location("sampling_scheduler_test", path)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    job = {"command": ["--motion-sampling", "adaptive", "--adaptive-after-update", "1000", "--iterations", "5000"],
           "output": str(tmp_path), "status": "running"}
    completion = {"planned_sampling_transition": True, "completed_updates": 1000,
                  "stopped": False, "motion_sampling": config("uniform")}
    with pytest.raises(ValueError, match="exact transition checkpoint"):
        scheduler.advance_sampling_phase(job, 0, completion)
    checkpoint = tmp_path / "checkpoint_update_001000.pt"
    checkpoint.touch()
    assert scheduler.advance_sampling_phase(job, 0, completion)
    assert job["status"] == "pending" and job["command"][-2:] == ["--resume", str(checkpoint)]
    assert not scheduler.advance_sampling_phase(job, 1, completion)
    with pytest.raises(ValueError, match="Unexpected"):
        scheduler.advance_sampling_phase(job, 0, {**completion, "completed_updates": 1046})


def test_final_audit_checks_actual_phase_history_and_transition_tensor_audit(tmp_path):
    import json
    final = {"completed_updates": 3, "residual_policy": {"motion_sampling": config(boundary=2)},
             "cfg": SimpleNamespace(task=SimpleNamespace(command=SimpleNamespace(command=SimpleNamespace(
                 sampling_mode="adaptive", rewind=SimpleNamespace(enabled=False)))))}
    rows = [{"completed_updates": 1}, {"completed_updates": 2}, {"completed_updates": 3}]
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="wrong sampling phase"):
        audit_sampling_curriculum(tmp_path, final, boundary=2)
    rows[-1]["motion_sampling"] = dict(adaptive_enabled=True, failure_rewind_enabled=False, adaptive_after_update=2)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="restoration audit"):
        audit_sampling_curriculum(tmp_path, final, boundary=2)
