import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_simulator_preview.py"
SPEC = importlib.util.spec_from_file_location("preview_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pair(tmp_path):
    value = {"protocol": "balanced_fixed_starts_v2_isolated_resets", "seed": 12,
             "motion_ids": [0, 0], "start_frames": [100, 200], "max_steps": 500,
             "metric_names": ["error_body_pos", "error_joint_pos"], "motion_files": ["one"],
             "physics_world_fingerprints": ["world_a", "world_b"], "physics": {"physics": "dr"},
             "motions": 1, "reference_timeline_audited": True,
             "partial_reset_survivor_state_audited": True, "partial_reset_survivor_history_audited": True,
             "training_reward_audit": {"eligible_fixed_reward_candidate": True},
             "per_episode_metrics": [[1.0, 2.0], [1.0, 2.0]], "failed": [True, False]}
    candidate = copy.deepcopy(value)
    candidate["per_episode_metrics"] = [[0.5, 1.0], [0.5, 1.0]]
    candidate["failed"] = [False, True]
    paths = [tmp_path / "reference.json", tmp_path / "candidate.json"]
    for path, record in zip(paths, (value, candidate), strict=True):
        path.write_text(json.dumps(record))
    return paths, candidate


def test_fixed_motion_ratios_and_new_failures(tmp_path):
    paths, _ = pair(tmp_path)
    result = MODULE.compare(*paths, samples=20)
    assert result["metrics"]["error_body_pos"]["ratio"] == 0.5
    assert result["failures"]["new"] == 1
    assert result["failures"]["recovered"] == 1
    assert result["failures"]["new_start_frames"] == [200]
    assert "ONE motion" in result["scope"]


@pytest.mark.parametrize("field,value", [
    ("physics_world_fingerprints", ["changed", "world_b"]),
    ("start_frames", [101, 200]), ("partial_reset_survivor_history_audited", False),
    ("training_reward_audit", {"eligible_fixed_reward_candidate": False}),
])
def test_comparison_fails_closed(tmp_path, field, value):
    paths, candidate = pair(tmp_path)
    candidate[field] = value
    paths[1].write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        MODULE.compare(*paths, samples=20)
