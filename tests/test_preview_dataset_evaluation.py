import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate_preview_dataset", Path(__file__).resolve().parents[1] / "scripts/evaluate_preview_dataset.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def pair():
    reference = {
        "protocol": "fixed", "seed": 1, "motion_ids": [0, 1, 0, 1], "start_frames": [0, 1, 2, 3],
        "metric_names": ["error_body_pos", "error_joint_pos"], "max_steps": 10,
        "motion_files": ["a", "b"], "physics_world_fingerprints": ["1", "2", "3", "4"],
        "physics": {"physics": "dr", "dr_profile": "hands-shins-2-4kg"}, "motions": 2,
        "per_episode_metrics": [[1, 2], [2, 3], [3, 4], [4, 5]], "failed": [False, True, False, False],
    }
    candidate = copy.deepcopy(reference)
    candidate["per_episode_metrics"] = (np.array(reference["per_episode_metrics"]) * .8).tolist()
    candidate["failed"] = [True, False, False, False]
    return reference, candidate


def test_paired_full_dataset_summary_does_not_hide_new_failures():
    rows, added, rescued = evaluation.paired_motion_rows(*pair())
    assert rows.shape == (2, 6) and added == rescued == 1
    result = evaluation.summarize(rows, added, rescued, 100)
    assert result["clear_tracking_gain_this_seed"]
    assert not result["strict_no_regression_this_seed"]
    assert result["baseline_failure_rate"] == result["preview_failure_rate"] == .25
    np.testing.assert_allclose(result["preview_over_baseline_body_joint"], [.8, .8])


@pytest.mark.parametrize("key", ["start_frames", "motion_files", "physics_world_fingerprints"])
def test_pairing_rejects_mismatched_worlds_or_motion(key):
    reference, candidate = pair()
    candidate[key] = []
    with pytest.raises(ValueError, match=key):
        evaluation.paired_motion_rows(reference, candidate)


def test_training_pair_rejects_unequal_updates(monkeypatch):
    args = {key: 1 for key in ("num_envs", "seed", "rollout_steps", "actor_lr", "critic_lr",
                              "critic_warmup_updates", "epochs", "mini_batches", "entropy_coef",
                              "initial_action_std", "residual_scale", "hidden_dims")}
    meta = {"physics_mode": "dr", "reward_changes": {}, "reward_contract": {"sha256": "reward"},
            "physics": {"details": {}}, "arguments": args,
            **{key: "same" for key in ("version", "tracker_sha256", "motion_sha256", "dr_profile")}}
    states = {arm: {"completed_updates": 100, "residual_policy": {**meta, "variant": arm}}
              for arm in ("baseline", "preview")}
    monkeypatch.setattr(evaluation.torch, "load", lambda path, **kwargs: states[path])
    paths = {arm: arm for arm in states}
    assert evaluation.validate_training_pair(paths)["completed_updates"] == 100
    states["preview"]["completed_updates"] = 200
    with pytest.raises(ValueError, match="completed_updates"):
        evaluation.validate_training_pair(paths)
