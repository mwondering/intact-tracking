from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_adaptation_evals.py"
spec = importlib.util.spec_from_file_location("adaptation_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_same_dr_comparison_keeps_candidate_provenance_and_world_checks(monkeypatch):
    import copy
    import sys

    monkeypatch.setitem(sys.modules, "compare_adaptation_evals", comparison)
    script = SCRIPT.with_name("confirm_dr_residual_comparison.py")
    same_dr_spec = importlib.util.spec_from_file_location("same_dr_confirmation", script)
    same_dr = importlib.util.module_from_spec(same_dr_spec)
    same_dr_spec.loader.exec_module(same_dr)
    record = {"protocol": "balanced_fixed_starts_v2_isolated_resets", "seed": 93001,
              "motion_ids": [0], "start_frames": [5], "max_steps": 500,
              "metric_names": ["error_body_pos"], "motion_files": ["a"],
              "physics_world_fingerprints": ["world"], "physics": {"physics": "dr"},
              "checkpoint_sha256": "ref", "reference_timeline_audited": True,
              "partial_reset_survivor_state_audited": True,
              "partial_reset_survivor_history_audited": True,
              "training_reward_audit": {"eligible_fixed_reward_candidate": False}}
    candidate = copy.deepcopy(record)
    candidate["checkpoint_sha256"] = "cand"
    candidate["training_reward_audit"]["eligible_fixed_reward_candidate"] = True
    identities = {"reference": {"sha256": "ref"}, "candidate": {"sha256": "cand"}}
    audit = {"reference_checkpoint_sha256": "ref"}
    with pytest.raises(ValueError, match="provenance"):
        same_dr.validate_pair(record, candidate, identities)
    same_dr.validate_pair(record, candidate, identities, audit)
    candidate["training_reward_audit"]["eligible_fixed_reward_candidate"] = False
    with pytest.raises(ValueError, match="provenance"):
        same_dr.validate_pair(record, candidate, identities, audit)
    candidate["training_reward_audit"]["eligible_fixed_reward_candidate"] = True
    candidate["partial_reset_survivor_state_audited"] = False
    with pytest.raises(ValueError, match="reset"):
        same_dr.validate_pair(record, candidate, identities, audit)
    candidate["partial_reset_survivor_state_audited"] = True
    candidate["physics_world_fingerprints"] = ["different"]
    with pytest.raises(ValueError, match="fingerprints"):
        same_dr.validate_pair(record, candidate, identities, audit)


def evaluations(tmp_path, *, seeds=(40001, 40002, 40003)):
    references, candidates = [], []
    for seed in seeds:
        data = {
            "protocol": "balanced_fixed_starts_v2_isolated_resets",
            "seed": seed,
            "motion_ids": [0, 0, 1, 1],
            "start_frames": [10, 20, 30, 40],
            "metric_names": ["error_body_pos", "error_joint_pos"],
            "max_steps": 500,
            "motion_files": ["a", "b"],
            "per_episode_metrics": [[1.0, 1.0]] * 4,
            "failed": [False] * 4,
            "motions": 2,
            "mean": {"error_body_pos": 1.0, "error_joint_pos": 1.0},
            "failure_rate": 0.0,
            "reference_timeline_audited": True,
            "partial_reset_survivor_state_audited": True,
            "partial_reset_survivor_history_audited": True,
            "training_reward_audit": {"eligible_fixed_reward_candidate": True},
        }
        for kind, paths, physics in (
            ("reference", references, "nominal"),
            ("candidate", candidates, "dr"),
        ):
            path = tmp_path / f"{kind}_{seed}.json"
            path.write_text(
                json.dumps({**data, "checkpoint_sha256": kind, "physics": {"physics": physics}})
            )
            paths.append(path)
    return references, candidates


def test_comparison_requires_one_fixed_policy_and_distinct_seeds(tmp_path):
    references, candidates = evaluations(tmp_path)
    result = comparison.compare(references, candidates, samples=100)
    assert result["point_estimate_pass"]
    assert result["all_metrics_upper_95pct_within_margin"]
    assert result["confirmation_seed_set_complete"]
    with pytest.raises(ValueError, match="Duplicate evaluation seed"):
        comparison.compare(references[:1] * 2, candidates[:1] * 2, samples=100)
    data = json.loads(candidates[1].read_text())
    data["checkpoint_sha256"] = "different-policy"
    candidates[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="policy identity changed"):
        comparison.compare(references, candidates, samples=100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("partial_reset_survivor_history_audited", False),
        ("context_ablation", "zero"),
        ("teacher_feature_ablation", "estimated_all"),
        ("teacher_remove_privilege", "height_contact"),
        ("action_filter_strength", 0.2),
        ("orientation_filter_weight", 0.1),
        ("privileged_bias_compensation", 1.0),
        ("privileged_payload_gravity_compensation", 1.0),
        ("training_reward_audit", {"eligible_fixed_reward_candidate": False}),
    ],
)
def test_comparison_rejects_unaudited_and_diagnostic_acceptance(tmp_path, field, value):
    references, candidates = evaluations(tmp_path)
    data = json.loads(candidates[0].read_text())
    data[field] = value
    candidates[0].write_text(json.dumps(data))
    result = comparison.compare(references, candidates, samples=100)
    assert result["numerical_point_estimate_pass"]
    assert not result["point_estimate_pass"]
    assert not result["all_metrics_upper_95pct_within_margin"]


def test_any_observed_failure_increase_fails_even_below_old_one_percent_margin(tmp_path):
    references, candidates = evaluations(tmp_path)
    for path in references + candidates:
        data = json.loads(path.read_text())
        for key in ("motion_ids", "start_frames", "per_episode_metrics", "failed"):
            data[key] = data[key] * 100
        if path in candidates:
            data["failed"][0] = True
            data["failure_rate"] = 1 / 400
        path.write_text(json.dumps(data))
    result = comparison.compare(references, candidates, samples=100)
    assert 0 < result["failure_difference"] < 0.01
    assert not result["observed_failure_nonincrease"]
    assert not result["point_estimate_pass"]
    assert not result["all_metrics_point_estimate_pass"]
    assert result["allowed_observed_failure_increase"] == 0


def test_base_merge_preserves_other_modules_and_requires_matched_normalization():
    import torch

    path = SCRIPT.with_name("merge_adaptation_base.py")
    module_spec = importlib.util.spec_from_file_location("adaptation_merge", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    receiver = {
        "tracker.mlp.0.weight": torch.tensor([1.0, 2.0]),
        "tracker.policy_normalizer.mean": torch.zeros(2),
        "privilege_encoder.weight": torch.tensor([3.0]),
        "residual_mlp.weight": torch.tensor([4.0]),
    }
    donor = {key: value.clone() for key, value in receiver.items()}
    donor["tracker.mlp.0.weight"] += 4.0
    donor["privilege_encoder.weight"] += 10.0
    merged, keys = module.merge(receiver, donor, 0.25)
    assert keys == ["tracker.mlp.0.weight"]
    torch.testing.assert_close(merged[keys[0]], torch.tensor([2.0, 3.0]))
    for key in receiver:
        if key not in keys:
            torch.testing.assert_close(merged[key], receiver[key])
    torch.testing.assert_close(receiver[keys[0]], torch.tensor([1.0, 2.0]))
    receiver["tracker.reference_encoder.0.weight"] = torch.tensor([1.25, 2.5])
    donor["tracker.reference_encoder.0.weight"] = torch.tensor([3.75, -0.5])
    transferred, reference_keys = module.merge(
        receiver, donor, 1.0, "tracker.reference_encoder"
    )
    assert reference_keys == ["tracker.reference_encoder.0.weight"]
    assert torch.equal(transferred[reference_keys[0]], donor[reference_keys[0]])
    for key in receiver:
        if key not in reference_keys:
            assert torch.equal(transferred[key], receiver[key])
    with pytest.raises(ValueError, match="Only aligned"):
        module.merge(receiver, donor, 0.5, "privilege_encoder")
    donor["tracker.policy_normalizer.mean"] += 1
    with pytest.raises(ValueError, match="normalization differs"):
        module.merge(receiver, donor, 0.25)
