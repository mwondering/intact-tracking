import copy
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("uniform_moe_report", Path(__file__).parents[1] / "scripts/report_residual_uniform_moe.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_three_policy_window_does_not_reward_early_termination():
    row = dict(protocol="test", seed=2, motion_files=["a", "b"], motion_ids=[0, 1], start_frames=[0, 0],
               horizons=[2, 2], max_steps=2, metric_names=["body", "joint"],
               reward_contract={}, memory_start="cold", physics_world_fingerprints=["p", "q"],
               actual_limb_masses_kg=[[0] * 4] * 2, query_initial_state_sha256="state",
               physics=dict(dr_profile="tracker", original_events={}, observation_corruption={}),
               warmup=dict(steps=0, policy="frozen", policy_sha256="frozen", seed=5), episodes=2,
               episode_lengths=[2, 2])
    rows = {arm: copy.deepcopy(row) for arm in report.ARMS}
    rows["moe"]["episode_lengths"] = [1, 1]
    long = np.array([[[1, 1], [9, 9]]] * 2, float)
    short = np.array([[[1, 1], [0, 0]]] * 2, float)
    means, common = report.common_motion_means(rows, {"baseline": long, "specialist": long, "moe": short})
    np.testing.assert_array_equal(common, [1, 1])
    for arm in report.ARMS:
        np.testing.assert_array_equal(means[arm], np.ones((2, 2)))


def test_expert_gap_scale_and_absent_reference_advantage():
    baseline = np.array([10., 20.])
    expert = np.array([5., 10.])
    middle = (baseline + expert) / 2
    assert report.recovery(baseline, baseline, expert, repeats=100)["percent"] == 0.
    assert report.recovery(baseline, middle, expert, repeats=100)["percent"] == 50.
    assert report.recovery(baseline, expert, expert, repeats=100)["percent"] == 100.
    assert report.recovery(baseline, middle, baseline, repeats=100)["percent"] is None
