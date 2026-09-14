from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/aggregate_matched_controls.py"
spec = importlib.util.spec_from_file_location("matched_aggregation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_data(tmp_path):
    root = tmp_path / "goal/eval_v2/matched"
    root.mkdir(parents=True)
    for seed in (10, 11):
        (root.parents[1] / f"matched_seed{seed}_initial_audit.json").write_text(json.dumps({
            "all_initial_tensors_bitwise_equal": True, "agent_configuration_equal": True,
        }))
        summary = {}
        for physics in ("nominal", "dr"):
            summary[physics] = {"statistics": {"metrics": {"test": {"ratio": 1.2}}},
                                "paired_failures": {"reference": 0, "candidate": 3, "new_failures": 3, "recovered_failures": 0, "episodes": 24}}
            for es in (92001, 92002, 92003):
                for train in ("nominal", "dr"):
                    data = {"protocol": "balanced_fixed_starts_v2_isolated_resets", "seed": es,
                            "motion_ids": [0, 1] * 4, "start_frames": list(range(8)),
                            "max_steps": 500, "metric_names": ["test"], "motion_files": ["a", "b"],
                            "physics_world_fingerprints": ["one"] * 8, "physics": {"physics": physics},
                            "checkpoint_sha256": f"seed{seed}_{train}", "motions": 2,
                            "per_episode_metrics": [[1.2 if train == "dr" else 1.0]] * 8,
                            "failed": [train == "dr"] + [False] * 7,
                            "reference_timeline_audited": True,
                            "partial_reset_survivor_state_audited": True,
                            "partial_reset_survivor_history_audited": True}
                    (root / f"train{seed}_{train}_test{physics}_seed{es}.json").write_text(json.dumps(data))
        (root / f"training_seed{seed}_comparison.json").write_text(json.dumps(summary))
    return root


def test_training_pairs_not_evaluation_seeds_are_independent(tmp_path):
    root = fixture_data(tmp_path)
    result = module.aggregate(root, (10, 11), samples=300)
    assert result["independent_training_pairs"] == 2
    for physics in ("nominal", "dr"):
        condition = result["conditions"][physics]
        metric = condition["metrics"]["test"]
        assert metric["dr_over_nominal"] == pytest.approx(1.2)
        assert metric["paired_training_eval_motion_bootstrap_95pct"] == pytest.approx([1.2, 1.2])
        assert condition["nominal_trained_failure_rate"] == 0
        assert condition["dr_trained_failure_rate"] == 0.125
    with pytest.raises(ValueError, match="distinct"):
        module.aggregate(root, (10, 10), samples=10)


def test_mismatched_physics_never_pooled(tmp_path):
    root = fixture_data(tmp_path)
    path = root / "train10_dr_testnominal_seed92001.json"
    data = json.loads(path.read_text())
    data["physics_world_fingerprints"][1] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprints"):
        module.aggregate(root, (10, 11), samples=10)
def test_completed_pair_can_settle_after_controller_exit(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("matched_watcher_test", scripts / "watch_matched_control_evals.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a, b = tmp_path / "a.pt", tmp_path / "b.pt"
        a.touch()
        b.touch()
        models = {"nominal": a, "dr": b}
        now = max(a.stat().st_mtime, b.stat().st_mtime)
        assert not module.final_pair_ready(models, False, now + 1)
        assert module.final_pair_ready(models, False, now + 16)
        missing = {"nominal": a, "dr": tmp_path / "missing.pt"}
        assert not module.final_pair_ready(missing, True, now + 16)
        import pytest
        with pytest.raises(RuntimeError, match="missing final"):
            module.final_pair_ready(missing, False, now + 16)
    finally:
        sys.path.remove(str(scripts))

