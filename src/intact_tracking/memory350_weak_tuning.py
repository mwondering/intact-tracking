"""The three authorized loss changes for the u8000 Memory350 tuning branch."""

import json
from pathlib import Path


PARENT_LOSS = {
    "weak_positive_weight": .002,
    "weak_negative_weight": .002,
    "response_distance_scale": .5,
}
TUNED_LOSS = {
    "weak_positive_weight": .004,
    "weak_negative_weight": .004,
    "response_distance_scale": .3,
}
PARENT_UPDATE = 8000
MILESTONES = (9000, 10000)


def validate_tuning_losses(previous, actual):
    if any(previous.get(key) != value for key, value in PARENT_LOSS.items()):
        raise ValueError("Tuning requires the original 0.002/0.002/0.5 weak-pair checkpoint")
    if actual != {**previous, **TUNED_LOSS}:
        raise ValueError("Tuning permits only weak weights 0.004/0.004 and response scale 0.3")


def validate_tuning_arguments(previous, actual, completed_update):
    if completed_update != PARENT_UPDATE:
        raise ValueError("This tuning branch must start from the evaluated u8000 checkpoint")
    operational = {"output_dir", "resume", "stop_after_updates", "comparison_reference_dir",
                   "wandb_group", "wandb_name", "wandb_tag"}
    canonical = lambda value: json.loads(json.dumps(value))
    mismatched = {key: (previous.get(key), actual.get(key))
                  for key in set(previous) | set(actual)
                  if key not in operational | TUNED_LOSS.keys()
                  and canonical(previous.get(key)) != canonical(actual.get(key))}
    if mismatched:
        raise ValueError(f"Tuning changed settings beyond the three losses: {mismatched}")
    validate_tuning_losses({key: previous.get(key) for key in PARENT_LOSS},
                          {key: actual.get(key) for key in TUNED_LOSS})
    if not actual.get("resume") or actual.get("stop_after_updates") != MILESTONES[-1]:
        raise ValueError("Tuning must resume u8000 and stop at u10000")
    parent = Path(actual["resume"]).resolve().parent
    if Path(actual["output_dir"]).resolve() == parent:
        raise ValueError("Tuning requires a separate output directory")
    if Path(actual["comparison_reference_dir"]).resolve() != parent:
        raise ValueError("Tuning must retain its parent's normalization and fixed validation")
    return {key: {"before": PARENT_LOSS[key], "after": value}
            for key, value in TUNED_LOSS.items()}
