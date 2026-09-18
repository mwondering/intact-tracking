from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "nominal_direction_monitor", Path(__file__).resolve().parents[1] / "scripts/monitor_memory350_nominal_direction.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def losses():
    return {"prediction_loss": .2, "representation_positive_loss": .1,
            "dr_nominal_relation_loss": .3, "weak_positive_loss": .4,
            "nominal_anchor_loss": .5, "loss": .2 + .01*.1 + .04*.3 + .008*.4 + .01*.5}


def test_loss_checks_detect_wrong_weight_negative_term_and_nonfinite():
    values = losses()
    result = monitor.check_loss(values)
    assert result["finite"] and result["six_term_sum_matches"] and result["negative_term_absent"]
    values["loss"] += .01
    assert not monitor.check_loss(values)["six_term_sum_matches"]
    values["weak_negative_loss"] = 0.
    assert not monitor.check_loss(values)["negative_term_absent"]
    values["loss"] = float("nan")
    assert not monitor.check_loss(values)["finite"]
    json.dumps(monitor.json_safe(values), allow_nan=False)


def test_seven_term_monitor_uses_the_retuned_weights():
    weights = dict(monitor.EXPECTED_WEIGHTS, dr_nominal_response_weight=.08,
                   dr_center_rank_weight=.02)
    values = dict(losses(), dr_center_rank_loss=.7)
    values['loss'] += .04 * values['dr_nominal_relation_loss'] + .02 * .7
    result = monitor.check_loss(values, weights)
    assert result['term_count'] == 7 and result['seven_term_sum_matches']
    assert result['term_sum_matches'] and result['finite']
    assert 'six_term_sum_matches' not in result
    assert not monitor.check_loss(values, dict(weights, dr_center_rank_weight=.002))['term_sum_matches']


def test_partially_appended_log_is_not_a_failed_training_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"update": 1}\n{"update": 2')
    assert monitor.complete_rows(path) == [{"update": 1}]
    with path.open("a") as stream:
        stream.write('}\n')
    assert monitor.complete_rows(path) == [{"update": 1}, {"update": 2}]


def test_live_process_identity_requires_the_original_start_time():
    actual = monitor.process_identity(os.getpid())
    assert actual["live"]
    assert monitor.process_identity(os.getpid(), actual["start_ticks"])["live"]
    assert not monitor.process_identity(os.getpid(), actual["start_ticks"] + 1)["live"]
    assert not monitor.process_identity(2_147_483_647)["live"]


def test_trends_do_not_present_zero_eligible_cross_motion_pairs_as_success():
    values = {key: 1. for key in monitor.PROBE_KEYS}
    rows = [{"update": 100, "unix_time": 200., "fixed_probe": dict(values, weak_positive_pairs=0.)},
            {"update": 200, "unix_time": 400., "fixed_probe": dict(values, dr_five_step_nmse=.8,
                                                                           weak_positive_pairs=0.)}]
    result = monitor.probe_trend(rows)
    assert result["seconds_per_update_recent"] == 2.
    assert result["metrics"]["dr_five_step_nmse"]["relative_change_percent"] == pytest.approx(-20.)
    assert result["fixed_probe_cross_motion_available"] is False


def test_checkpoint_audit_detects_changed_anchor_even_with_unchanged_claimed_hash(tmp_path):
    run = tmp_path / "stage1_8192"
    run.mkdir()
    direction = torch.eye(64)[0]
    digest = hashlib.sha256(direction.numpy().tobytes()).hexdigest()
    anchor = {"direction": direction.tolist(), "sha256": digest, "sha256_by_rank": [digest]*8}
    (run / "nominal_anchor.json").write_text(json.dumps(anchor))
    state = {"update": 200, "optimizer_steps": 800, "nominal_direction_anchor": deepcopy(anchor),
             "distributed_parameter_agreement": {"passed": True, "sha256_by_rank": ["same"]*8},
             "supervision_horizons": {"predictor": 5, "response_label": 10},
             "loss_config": {"representation_weight": .01, "representation_relation_weight": 4.,
                             "nominal_anchor_weight": .01, "weak_positive_weight": .008, "weak_negative_weight": 0.}}
    torch.save(state, run / "last.pt")
    checker = monitor.Monitor(tmp_path)
    assert checker.checkpoint_audit()["passed"]
    state["nominal_direction_anchor"]["direction"][0] = -1.
    torch.save(state, run / "replacement.pt")
    (run / "replacement.pt").replace(run / "last.pt")
    result = checker.checkpoint_audit()
    assert not result["passed"] and not result["checks"]["fixed_anchor_preserved"]


def test_soft_variant_monitor_accounts_for_replacement_term():
    weights = dict(monitor.EXPECTED_WEIGHTS, dr_soft_weight=.02)
    values = dict(losses(), dr_soft_loss=.7)
    values['loss'] += .02*.7
    result = monitor.check_loss(values, weights)
    assert result['term_count'] == 7 and result['seven_term_sum_matches']
    assert not monitor.check_loss(values, dict(weights, dr_soft_weight=.002))['term_sum_matches']
