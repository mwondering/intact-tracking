from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from intact_tracking.limb_context_dr import (
    LOAD_ONLY, TRACKER_DR, configure_limb_dr, resolve_dr_profile, validate_context_dr,
)
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT
from intact_tracking.rollout.online import FixedDRRolloutConfig


def _event(env, env_ids, **kwargs):
    pass


def _original_cfg():
    events = {name: EventTermCfg(func=_event, mode="startup") for name in (
        "base_com", "base_mass", "encoder_bias", "foot_friction", "motor_params_implicit")}
    for name in ("base_com", "base_mass"):
        events[name].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("torso_link",))
    events["push_robot"] = EventTermCfg(func=_event, mode="step", params={"interval_range_s": (3, 6)})
    return SimpleNamespace(
        events=events,
        observations={"actor": SimpleNamespace(enable_corruption=True),
                      "critic": SimpleNamespace(enable_corruption=False)},
        commands={"motion": SimpleNamespace(rewind=SimpleNamespace(enabled=True),
                                            pose_range={"x": (-.1, .1)})},
        actions={"joint_pos": SimpleNamespace(observation_history_steps=8)},
    )


def test_original_events_noise_actions_and_reset_ranges_survive_payload_configuration():
    cfg = _original_cfg()
    original = copy.deepcopy(cfg)
    identities = {name: id(event) for name, event in cfg.events.items()}
    metadata = configure_limb_dr(cfg, 121, profile=TRACKER_DR)
    assert set(cfg.events) == set(original.events) | {PAYLOAD_EVENT}
    for name, event in original.events.items():
        assert cfg.events[name] == event
        assert id(cfg.events[name]) == identities[name]
    assert cfg.observations == original.observations
    assert cfg.actions == original.actions
    assert cfg.commands["motion"].pose_range == original.commands["motion"].pose_range
    assert cfg.commands["motion"].sampling_mode == "uniform"
    assert not cfg.commands["motion"].rewind.enabled
    assert metadata["removed_disturbances"] == []
    assert metadata["original_events"]["push_robot"]["mode"] == "step"
    assert json.loads(json.dumps(metadata)) == metadata


def test_partial_original_dr_is_rejected_instead_of_silently_training_load_only():
    cfg = _original_cfg()
    del cfg.events["push_robot"]
    with pytest.raises(ValueError, match="complete original DR"):
        configure_limb_dr(cfg, 121, profile=TRACKER_DR)


def test_runtime_audit_accepts_resolved_regex_but_rejects_changed_friction():
    from intact_tracking.limb_context_dr import _event_contract, _resolved_event_contract

    term = EventTermCfg(func=_event, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*foot"), "ranges": (.3, 2.)})
    original = _event_contract(term)
    scene = {"robot": SimpleNamespace(
        geom_names=("left_foot", "right_foot"), num_geoms=2,
        find_geoms=lambda names, **kwargs: ([0, 1], ["left_foot", "right_foot"]))}
    term.params["asset_cfg"].resolve(scene)
    expected = _resolved_event_contract(original, scene)
    assert _event_contract(term) == expected
    term.params["ranges"] = (.5, 1.)
    assert _event_contract(term) != expected


def test_evaluation_and_resume_inherit_dr_but_cannot_override_it():
    assert resolve_dr_profile(None, {}) == LOAD_ONLY
    assert resolve_dr_profile(None, {"dr_profile": TRACKER_DR}) == TRACKER_DR
    with pytest.raises(ValueError, match="differs"):
        resolve_dr_profile(LOAD_ONLY, {"dr_profile": TRACKER_DR})
    with pytest.raises(ValueError, match="differs"):
        resolve_dr_profile(TRACKER_DR, {})


def test_context_dr_must_match_ppo_including_legacy_checkpoints():
    legacy = {"nominal_counterfactual_representation_supervision": True,
              "load_only_experiment": True, "loss_config": {"representation_weight": .01}}
    validate_context_dr(legacy, LOAD_ONLY)
    with pytest.raises(ValueError, match="same DR"):
        validate_context_dr(legacy, TRACKER_DR)
    current = {**legacy, "load_only_experiment": False, "dr_profile": TRACKER_DR}
    validate_context_dr(current, TRACKER_DR)
    with pytest.raises(ValueError, match="same DR"):
        validate_context_dr(current, LOAD_ONLY)


def test_four_limb_dr_forbids_nominal_subsets_or_extra_payloads():
    kwargs = dict(checkpoint_file="tracker.pt", motion_file="motion.npz", tracker_dr_plus_limb_payload=True)
    assert FixedDRRolloutConfig(**kwargs).limb_payload_experiment
    for extra in ({"nominal_fraction": .5}, {"payload_enabled": True}, {"limb_payload_only": True}):
        with pytest.raises(ValueError):
            FixedDRRolloutConfig(**kwargs, **extra)


def test_new_preset_wires_both_stage1_jobs_and_every_residual_arm(tmp_path):
    from intact_tracking.cli.forward_predictor_train import build_parser, _validate_arguments
    from intact_tracking.cli.limb_context_train import build_parser as ppo_parser

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("scheduler_dr", root / "scripts/run_limb_context_experiment.py")
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    preset = json.loads((root / "configs/experiments/limb_context_memory350_representation.json").read_text())
    (tmp_path / "experiment_layout.json").write_text(json.dumps(preset))
    for job in scheduler.jobs(tmp_path):
        command = job["command"]
        if job["phase"] == "encoder":
            args = build_parser().parse_args(command[command.index("intact_tracking.cli.forward_predictor_train") + 1:])
            _validate_arguments(args)
            assert args.tracker_dr_plus_limb_payload and not args.limb_payload_only
        elif job["phase"] == "ppo":
            args = ppo_parser().parse_args(command[command.index("intact_tracking.cli.limb_context_train") + 1:])
            assert args.dr_profile == TRACKER_DR


def test_paired_comparisons_reject_mismatched_dr_even_with_equal_payload_fingerprints():
    from intact_tracking.limb_context_results import PAIRED_FIELDS, paired_rows

    reference = {name: [] for name in PAIRED_FIELDS}
    candidate = {**reference, "physics": {"dr_profile": TRACKER_DR}}
    with pytest.raises(ValueError, match="DR field"):
        paired_rows(reference, candidate, {}, {})


def test_final_evaluation_passes_original_dr_to_frozen_tracker_and_residual(tmp_path, monkeypatch):
    from intact_tracking.cli.limb_context_eval import build_parser

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluate_dr", root / "scripts/evaluate_limb_context_experiment.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    frozen = tmp_path / "frozen.pt"
    frozen.write_bytes(b"audit fixture: only the digest is used")
    monkeypatch.setattr(evaluator, "TRACKER", str(frozen))
    directory = evaluator.ppo_directory(tmp_path) / "baseline_121"
    directory.mkdir(parents=True)
    (directory / "checkpoint_final.pt").write_bytes(b"residual digest fixture")
    (tmp_path / "eval").mkdir()
    protocol = {"dr_profile": TRACKER_DR, "seed": 20001, "steps": 1000, "cases": [{
        "name": "all_0", "manifests": ["fixed_motions.txt"], "repeats": 1,
        "fixed_masses": [0, 0, 0, 0], "paired_starts": False}]}
    queue = evaluator.evaluation_jobs(tmp_path, protocol, selected_policies=("frozen", "baseline_121"))
    assert len(queue) == 2
    for job in queue:
        command = job["command"]
        args = build_parser().parse_args(command[command.index("intact_tracking.cli.limb_context_eval") + 1:])
        assert args.dr_profile == TRACKER_DR
        assert bool(args.checkpoint) == (job["policy"] != "frozen")
