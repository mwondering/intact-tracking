import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_lafan_abc", Path(__file__).resolve().parents[1] / "scripts/run_lafan_abc.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def paired_rows():
    rows = {}
    for seed in (20001, 20002, 20003):
        row = {"protocol": "paired", "seed": seed, "motion_files": ["one", "two"],
               "motion_ids": [0, 0, 1, 1], "start_frames": [2, 4, 3, 5],
               "metric_names": ["error_body_pos", "error_joint_pos"],
               "max_steps": 500, "physics_world_fingerprints": ["same"] * 4,
               "motions": 2, "failed": [False, True, False, False],
               "per_episode_metrics": [[.1, .5], [.1, .5], [.2, 1.], [.2, 1.]]}
        rows[("A", "nominal", seed)] = row
        candidate = copy.deepcopy(row)
        candidate["per_episode_metrics"] = [[v * 1.2 for v in pair]
                                              for pair in row["per_episode_metrics"]]
        candidate["failed"] = [False, False, True, False]
        rows[("C", "nominal", seed)] = candidate
    return rows


def test_paired_comparison_ratios_and_failure_swaps():
    result = experiment.compare(paired_rows(), "A", "nominal")
    assert result["C_over_reference"] == pytest.approx([1.2, 1.2])
    assert result["ratio_ci95_low"] == pytest.approx([1.2, 1.2])
    assert result["both_errors_significantly_higher_this_training_seed"]
    assert result["C_new_failure_episodes"] == 3
    assert result["C_rescued_failure_episodes"] == 3
    assert result["reference_failure_rate"] == result["C_failure_rate"] == .25


@pytest.mark.parametrize("key", ["start_frames", "physics_world_fingerprints", "motion_ids"])
def test_comparison_refuses_unmatched_evaluations(key):
    rows = paired_rows()
    rows[("C", "nominal", 20001)][key] = ["different"]
    with pytest.raises(ValueError, match=key):
        experiment.compare(rows, "A", "nominal")
