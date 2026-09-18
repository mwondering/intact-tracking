import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("independent_moe_report", SCRIPTS / "report_residual_independent_moe.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
sys.path.pop(0)


def trials():
    row = dict(protocol="test", seed=2, motion_files=["a", "b"], motion_ids=[0, 1], start_frames=[0, 0],
               horizons=[3, 3], max_steps=3, metric_names=["body", "root"],
               reward_contract={}, memory_start="cold", physics_world_fingerprints=["p", "q"],
               actual_limb_masses_kg=[[0] * 4] * 2, query_initial_state_sha256="state",
               physics=dict(dr_profile="tracker", original_events={}, observation_corruption={}),
               warmup=dict(steps=0, policy="frozen", policy_sha256="frozen", seed=5), episodes=2,
               episode_lengths=[3, 3])
    rows = {arm: copy.deepcopy(row) for arm in report.ARMS}
    trace = np.array([[[1, 1], [5, 5], [9, 9]]] * 2, float)
    traces = {arm: trace.copy() for arm in report.ARMS}
    return rows, traces


def test_all_four_arms_use_same_window_when_different_policies_fail_first():
    rows, traces = trials()
    rows["independent_moe"]["episode_lengths"] = [1, 3]
    traces["independent_moe"][0, 1:] = 0
    rows["shared_moe"]["episode_lengths"] = [3, 2]
    traces["shared_moe"][1, 2:] = 0
    means, common = report.common_motion_means(rows, traces)
    np.testing.assert_array_equal(common, [1, 2])
    for arm in report.ARMS:
        np.testing.assert_array_equal(means[arm], [[1, 1], [3, 3]])


def test_architecture_comparison_rejects_changed_query_state():
    rows, traces = trials()
    rows["independent_moe"]["query_initial_state_sha256"] = "wrong"
    with pytest.raises(ValueError, match="query_initial_state_sha256"):
        report.common_motion_means(rows, traces)
