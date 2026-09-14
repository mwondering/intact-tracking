import copy

import pytest

from intact_tracking.memory350_latent_usage_eval import validate_usage_result


def test_usage_validation_requires_same_policy_matched_starts_and_continuous_dr():
    spec = {"seed": 22001, "steps": 1000, "warmup_steps": 500}
    row = {"checkpoint_sha256": "checkpoint", "completed_training_updates": 1000,
        "seed": 22001, "max_steps": 1000, "motion_files": ["motion"], "episodes": 2,
        "repeats_per_motion": 2, "memory_start": "warm", "latent_intervention": "paired-swap",
        "arguments": {"fixed_masses": None, "paired_starts": True}, "warmup": {"steps": 500},
        "physics": {"dr_profile": "tracker_dr_plus_limb_payload"},
        "reference_timeline_audited": True, "partial_reset_survivor_state_audited": True,
        "partial_reset_survivor_history_audited": True}
    validate_usage_result(row, spec, ["motion"], "checkpoint", 1000, "paired-swap")
    for key, value in (("checkpoint_sha256", "different"), ("latent_intervention", "correct"),
                       ("seed", 22002), ("completed_training_updates", 999)):
        changed = {**row, key: value}
        with pytest.raises(ValueError, match="checkpoint/protocol"):
            validate_usage_result(changed, spec, ["motion"], "checkpoint", 1000, "paired-swap")
    changed = copy.deepcopy(row)
    changed["arguments"]["fixed_masses"] = [4] * 4
    with pytest.raises(ValueError, match="continuous DR"):
        validate_usage_result(changed, spec, ["motion"], "checkpoint", 1000, "paired-swap")
