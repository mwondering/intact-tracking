import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from intact_tracking.limb_context_sampling import checkpoint_configuration
from intact_tracking.limb_context_terminations import (
    NO_EE_BODY_POS, ORIGINAL, PROFILE_FILE, audit_runtime_terminations,
    checkpoint_contract, configure_training_terminations, experiment_profile,
    resolve_profile, validate_termination_resume,
)


def source_configuration():
    return OmegaConf.create({"task": {"terminations": [
        {"name": "time_out", "term": "time_out", "time_out": True},
        {"name": "anchor_pos", "term": "bad_anchor_pos_z_only", "params": {"threshold": .5}},
        {"name": "anchor_ori", "term": "bad_anchor_ori", "params": {"threshold": .8}},
        {"name": "ee_body_pos", "term": "bad_motion_body_pos_z_only", "params": {"threshold": .5}},
    ], "command": {"command": {}}}, "agent": {}})


def configured(profile=NO_EE_BODY_POS):
    source = source_configuration()
    env = SimpleNamespace(terminations={item.name: object() for item in source.task.terminations})
    return source, env, configure_training_terminations(env, source, profile)


def test_training_removal_preserves_other_terms_and_original_evaluation_config():
    source = source_configuration()
    original_source = copy.deepcopy(source)
    original_terms = {item.name: object() for item in source.task.terminations}
    env = SimpleNamespace(terminations=original_terms)
    contract = configure_training_terminations(env, source, NO_EE_BODY_POS)
    assert list(env.terminations) == ["time_out", "anchor_pos", "anchor_ori"]
    assert all(env.terminations[name] is original_terms[name] for name in env.terminations)
    assert "ee_body_pos" in original_terms
    assert source == original_source
    assert contract["evaluation_profile"] == ORIGINAL
    assert not audit_runtime_terminations(SimpleNamespace(active_terms=list(env.terminations)), contract)["ee_body_pos_enabled"]
    with pytest.raises(ValueError, match="Runtime termination manager"):
        audit_runtime_terminations(SimpleNamespace(active_terms=list(original_terms)), contract)


def test_actual_manager_allows_ee_deviation_but_still_terminates_anchor_failure():
    import torch
    from mjlab.managers import TerminationManager, TerminationTermCfg
    from intact_tracking.environment.mdp.terminations import (
        bad_anchor_pos_z_only, bad_motion_body_pos_z_only,
    )

    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=("hand",)),
        anchor_pos_w=torch.zeros(2, 3), robot_anchor_pos_w=torch.zeros(2, 3),
        body_pos_relative_w=torch.zeros(2, 1, 3), body_quat_relative_w=torch.zeros(2, 1, 4),
        robot_body_pos_w=torch.zeros(2, 1, 3),
    )
    # Both worlds exceed EE height tolerance; only world 1 also violates the anchor rule.
    command.robot_body_pos_w[:, 0, 2] = .6
    command.robot_anchor_pos_w[1, 2] = .6
    env = SimpleNamespace(num_envs=2, device="cpu", timeouts=torch.zeros(2, dtype=torch.bool),
                          command_manager=SimpleNamespace(get_term=lambda _: command))
    terms = {
        "time_out": TerminationTermCfg(func=lambda env: env.timeouts, time_out=True),
        "anchor_pos": TerminationTermCfg(func=bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": .5}),
        "anchor_ori": TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool)),
        "ee_body_pos": TerminationTermCfg(func=bad_motion_body_pos_z_only,
            params={"command_name": "motion", "threshold": .5}),
    }
    original_manager = TerminationManager(terms, env)
    assert original_manager.compute().tolist() == [True, True]
    config = SimpleNamespace(terminations=terms)
    contract = configure_training_terminations(config, source_configuration(), NO_EE_BODY_POS)
    training_manager = TerminationManager(config.terminations, env)
    audit_runtime_terminations(training_manager, contract)
    assert training_manager.compute().tolist() == [False, True]
    assert original_manager.compute().tolist() == [True, True]
    env.timeouts[0] = True
    assert training_manager.compute().tolist() == [True, True]
    assert training_manager.time_outs.tolist() == [True, False]


def test_saved_configuration_and_resume_keep_actual_training_task(tmp_path):
    source, env, contract = configured()
    sampling = dict(active_mode="adaptive", adaptive_sampling={}, parameters={})
    metadata = dict(training_terminations=contract, motion_sampling=sampling)
    cfg = checkpoint_configuration(source, {}, metadata)
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    saved = OmegaConf.load(path)
    assert [item.name for item in saved.task.terminations] == list(env.terminations)
    assert source.task.terminations[-1].name == "ee_body_pos"
    checkpoint = {"cfg": saved, "residual_policy": metadata}
    assert checkpoint_contract(checkpoint) == contract
    validate_termination_resume(checkpoint, contract)
    original_contract = configured(ORIGINAL)[2]
    with pytest.raises(ValueError, match="Resume changed training terminations"):
        validate_termination_resume(checkpoint, original_contract)
    saved.task.terminations[1].params.threshold = 100
    with pytest.raises(ValueError, match="Saved training termination configuration"):
        checkpoint_contract(checkpoint)


def test_legacy_checkpoints_resume_original_profile_and_reject_silent_change(tmp_path):
    previous = {"cfg": source_configuration(), "residual_policy": {}}
    assert resolve_profile(None, tmp_path / "ppo", tmp_path, previous) == ORIGINAL
    validate_termination_resume(previous, configured(ORIGINAL)[2])
    with pytest.raises(ValueError, match="Resume changed training terminations"):
        validate_termination_resume(previous, configured()[2])
    assert resolve_profile(None, tmp_path / "new", tmp_path) == NO_EE_BODY_POS


def test_pinned_existing_matrix_survives_pending_jobs_and_sampler_restarts(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    (root / PROFILE_FILE).write_text(json.dumps({"profile": ORIGINAL}))
    for name in ("baseline_122", "film_122", "baseline_123", "film_123", "concat_121", "constant_121"):
        output = root / "ppo_2gpu8192_scratch" / name
        assert resolve_profile(None, output, tmp_path) == ORIGINAL
        with pytest.raises(ValueError, match="differs from this experiment"):
            resolve_profile(NO_EE_BODY_POS, output, tmp_path)
    assert experiment_profile(root) == ORIGINAL


def test_new_scheduler_disables_term_for_every_ppo_arm_without_changing_stage1(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_limb_context_experiment.py"
    spec = importlib.util.spec_from_file_location("termination_scheduler_test", path)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    fresh = scheduler.jobs(tmp_path)
    for job in fresh:
        if job["phase"] == "ppo":
            command = job["command"]
            assert command[command.index("--training-terminations") + 1] == NO_EE_BODY_POS
        else:
            assert "--training-terminations" not in job["command"]
    # Preserve old command identity so --adopt-running remains compatible.
    (tmp_path / PROFILE_FILE).write_text(json.dumps({"profile": ORIGINAL}))
    for old, new in zip(scheduler.jobs(tmp_path), fresh):
        command = new["command"][:-2] if new["phase"] == "ppo" else new["command"]
        assert old["command"] == command


def test_existing_unpinned_results_determine_profile_and_mixed_tasks_are_rejected(tmp_path):
    directory = tmp_path / "ppo_2gpu8192_scratch"
    baseline, film = directory / "baseline_121", directory / "film_121"
    baseline.mkdir(parents=True)
    film.mkdir()
    (baseline / "run_config.json").write_text("{}")
    assert experiment_profile(tmp_path) == ORIGINAL
    (film / "run_config.json").write_text(json.dumps({"training_terminations": configured()[2]}))
    with pytest.raises(ValueError, match="mixes training termination profiles"):
        experiment_profile(tmp_path)
