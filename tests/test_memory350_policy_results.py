import copy

import numpy as np
import pytest

from intact_tracking.memory350_policy_results import assert_paired, compare


def row():
    return dict(protocol="test", seed=2, motion_files=["a", "b"], motion_ids=[0, 1], start_frames=[0, 0],
                horizons=[2, 2], max_steps=2, metric_names=["error_body_pos", "error_joint_pos"],
                reward_contract={}, memory_start="cold", physics_world_fingerprints=["p", "q"],
                actual_limb_masses_kg=[[0] * 4] * 2, query_initial_state_sha256="state",
                physics=dict(dr_profile="tracker", original_events={}, observation_corruption={}),
                warmup=dict(steps=0, policy="frozen", policy_sha256="frozen", seed=5), episodes=2,
                episode_lengths=[2, 2], per_episode_metrics=[[5, 5], [5, 5]], failed=[False, False])


def test_common_survival_does_not_reward_early_failure():
    a, b = row(), row()
    b.update(episode_lengths=[1, 1], per_episode_metrics=[[1, 1], [1, 1]], failed=[True, True])
    trace_a = np.array([[[1, 1], [9, 9]]] * 2, float)
    trace_b = np.array([[[1, 1], [0, 0]]] * 2, float)
    result = compare(a, b, trace_a, trace_b, repeats=100)
    assert result["truncated_error_body_pos"]["reduction_percent"] == 80
    assert result["common_error_body_pos"]["reduction_percent"] == 0
    assert result["failure_rate"]["candidate_minus_reference"] == 1
    assert result["coverage"]["candidate_minus_reference"] == -.5


def test_pairing_rejects_different_worlds_or_warmup_policy():
    a = row()
    for key in ("physics_world_fingerprints", "query_initial_state_sha256"):
        b = copy.deepcopy(a)
        b[key] = "different"
        with pytest.raises(ValueError, match=key):
            assert_paired(a, b)
    b = copy.deepcopy(a)
    b["warmup"]["policy_sha256"] = "learned actor"
    with pytest.raises(ValueError, match="Warm-up"):
        assert_paired(a, b)


def test_full_global_trace_uses_common_survival_for_every_metric():
    a, b = row(), row()
    for item in (a, b):
        item["metric_names"] += ["error_anchor_pos", "error_body_pos_global"]
        item["global_metric_contract"] = {"version": "world_v1"}
    a["per_episode_metrics"] = [[5] * 4] * 2
    b.update(episode_lengths=[1, 1], per_episode_metrics=[[1] * 4] * 2, failed=[True, True])
    ta = np.array([[[1] * 4, [9] * 4]] * 2, float)
    tb = np.array([[[1] * 4, [0] * 4]] * 2, float)
    result = compare(a, b, ta, tb, repeats=100)
    assert result["truncated_error_body_pos_global"]["reduction_percent"] == 80
    assert result["common_error_body_pos_global"]["reduction_percent"] == 0
    assert result["common_error_anchor_pos"]["reduction_percent"] == 0
    b["global_metric_contract"] = {"version": "aligned_v1"}
    with pytest.raises(ValueError, match="global_metric_contract"):
        compare(a, b, ta, tb, repeats=100)
