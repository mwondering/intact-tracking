"""Training-only start-state curricula; physical DR and sensors are untouched."""

from __future__ import annotations

from copy import deepcopy


def configure_training_starts(cfg, mode):
    if mode not in ("original", "reference"):
        raise ValueError("Training start mode must be original or reference")
    command = cfg.commands["motion"]
    fields = ("pose_range", "velocity_range", "joint_position_range", "init_noise")
    before = {name: deepcopy(getattr(command, name, None)) for name in fields}
    if mode == "reference":
        command.pose_range = {}
        command.velocity_range = {}
        command.joint_position_range = (0.0, 0.0)
        command.init_noise = {}
    return {
        "mode": mode,
        "before": before,
        "after": {name: deepcopy(getattr(command, name, None)) for name in fields},
        "contract": "Training reset curriculum only; no changes to physical DR, sensors, motions, terminations, or evaluation",
    }
