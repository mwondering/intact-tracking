"""Authorized u10000->u12000 weak-pair weight and negative-margin tuning."""

import json
from pathlib import Path


PARENT_LOSS = {
    "weak_positive_weight": .004,
    "weak_negative_weight": .004,
    "weak_negative_margin": 1.,
}
TUNED_LOSS = {
    "weak_positive_weight": .008,
    "weak_negative_weight": .008,
    "weak_negative_margin": 1.1,
}
FIXED_LOSS = {"response_distance_scale": .3}
PARENT_UPDATE = 10000
MILESTONES = (11000, 12000)
TARGET_TOP1 = .8


def validate_tuning_losses(previous, actual):
    if any(previous.get(key) != value for key, value in (PARENT_LOSS | FIXED_LOSS).items()):
        raise ValueError("Margin tuning requires the evaluated 0.004/0.004/margin1.0/scale0.3 checkpoint")
    if actual != {**previous, **TUNED_LOSS}:
        raise ValueError("Margin tuning permits only weak weights 0.008/0.008 and margin 1.1; scale stays 0.3")


def validate_tuning_arguments(previous, actual, completed_update):
    if completed_update != PARENT_UPDATE:
        raise ValueError("This tuning branch must start from the evaluated u10000 checkpoint")
    operational = {"output_dir", "resume", "stop_after_updates", "comparison_reference_dir",
                   "wandb_group", "wandb_name", "wandb_tag"}
    canonical = lambda value: json.loads(json.dumps(value))
    mismatched = {key: (previous.get(key), actual.get(key))
                  for key in set(previous) | set(actual)
                  if key not in operational | TUNED_LOSS.keys()
                  and canonical(previous.get(key)) != canonical(actual.get(key))}
    if mismatched:
        raise ValueError(f"Tuning changed settings beyond the three losses: {mismatched}")
    validate_tuning_losses({key: previous.get(key) for key in PARENT_LOSS | FIXED_LOSS},
                          {key: actual.get(key) for key in TUNED_LOSS | FIXED_LOSS})
    if not actual.get("resume") or actual.get("stop_after_updates") != MILESTONES[-1]:
        raise ValueError("Tuning must resume u10000 and stop at u12000")
    parent = Path(actual["resume"]).resolve().parent
    if Path(actual["output_dir"]).resolve() == parent:
        raise ValueError("Tuning requires a separate output directory")
    if Path(actual["comparison_reference_dir"]).resolve() != parent:
        raise ValueError("Tuning must retain its parent's normalization and fixed validation")
    return {key: {"before": PARENT_LOSS[key], "after": value}
            for key, value in TUNED_LOSS.items()}
