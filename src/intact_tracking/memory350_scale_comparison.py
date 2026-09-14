"""Matched physics and fixed probes for the Memory350 encoder-depth experiment."""

import json
from pathlib import Path

from intact_tracking.short50_comparison import (
    reference_normalization, reference_probes, sha256,
)


PAIRED_UPDATES = (100, 500, 1000, 3000, 5000, 7500, 10000, 15000, 20000, 22700)


def _json_value(value):
    """Compare runtime tuples with their equivalent serialized JSON arrays."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def reference_contract(reference, actual):
    root = Path(reference).resolve()
    source = json.loads((root / "run_config.json").read_text())
    actual = _json_value(actual)
    operational = {
        "output_dir", "resume", "wandb_entity", "wandb_group", "wandb_name",
        "wandb_project", "wandb_tag", "device", "distributed_backend",
    }
    mismatched = {}
    for name, expected in source["arguments"].items():
        if name in operational or name == "context_depth":
            continue
        observed = actual["arguments"].get(name)
        if observed != expected:
            mismatched[name] = (expected, observed)
    expected_model = {
        **source["model"], "chunk_depth": 2, "memory_depth": 4, "context_depth": 4,
    }
    if actual["model"] != expected_model:
        mismatched["model"] = "Only the three attention depths may change from 1/2/2 to 2/4/4"
    for field in ("loss", "objective_weights", "optimization", "distributed",
                  "tracker_checkpoint_sha256", "dr_profile", "episode_length_control_steps"):
        if source.get(field) != actual.get(field):
            mismatched[field] = "configuration differs"
    for a, b in zip(source["dataset"]["runtime_audits_by_rank"],
                    actual["dataset"]["runtime_audits_by_rank"], strict=True):
        for key in ("rank", "motion_count", "motion_file_list_sha256", "loaded_frames",
                    "training_worlds", "validation_worlds", "num_envs",
                    "episode_length_control_steps", "dr_profile"):
            if a[key] != b[key]:
                mismatched[f'rank{a["rank"]}/{key}'] = (a[key], b[key])
        for key in ("sampled_mass_sha256", "actual_mass_sha256", "original_events", "observation_corruption"):
            if a["physics"][key] != b["physics"][key]:
                mismatched[f'rank{a["rank"]}/physics/{key}'] = "physics differs"
    if mismatched:
        raise ValueError(f"Memory350 encoder scale comparison changed settings: {mismatched}")
    return {
        "reference_dir": str(root), "reference_config_sha256": sha256(root / "run_config.json"),
        "normalization_sha256": sha256(root / "normalization.json"),
        "shared_normalization_and_validation": True, "physics_and_training_controls_passed": True,
        "model_input": "identical short50 plus nonoverlapping cross-reset chunk10 x 30",
        "architecture_change": "chunk/memory/context attention depths 1/2/2 -> 2/4/4",
        "initialization": "fresh predictor and retained context weights plus subsequent RNG match original Memory350",
        "paired_checkpoint_updates": list(PAIRED_UPDATES),
        "primary_comparison_update": 22700,
        "conclusion_scope": "one training seed; larger encoder capacity with unchanged predictor capacity",
    }
