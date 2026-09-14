"""Fail-closed original-reward contract for the fixed-reward research phase.

This module audits rewards; it never computes, rescales, or adds rewards.
Historical reward-shaped checkpoints remain readable for evaluation but cannot
silently seed new fixed-reward training.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

VERSION = "original_task_reward_locked_v1"
REWARD_ARGUMENTS = {
    "tracking_multiplier": 1.0,
    "action_rate_multiplier": 1.0,
    "failure_penalty": 0.0,
    "metric_tracking_weight": 0.0,
    "auxiliary_tracking_weight": 0.0,
    "right_arm_angular_weight": 0.0,
    "right_arm_rotation_weight": 0.0,
    "metric_reward_shape": "exp",
    "auxiliary_reward_shape": "exp",
    "metric_reference_offset": 0,
    "auxiliary_metric_weights": None,
}


def assert_original_reward_arguments(args):
    changed = {
        name: getattr(args, name) for name, default in REWARD_ARGUMENTS.items()
        if hasattr(args, name) and getattr(args, name) != default
    }
    if changed:
        raise ValueError(f"Original reward is locked; forbidden reward overrides: {changed}")


def canonical(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": canonical(value.value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, Mapping):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, functools.partial):
        return {"partial": canonical(value.func), "args": canonical(value.args), "kwargs": canonical(value.keywords)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {field.name: canonical(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
    if callable(value):
        source = inspect.getsource(value)
        return {
            "function": f"{value.__module__}:{value.__qualname__}",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    raise TypeError(f"Cannot audit reward config object {type(value)}; refusing an incomplete signature")


def signature(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture_original_rewards(env_cfg):
    document = {
        "terms": canonical(env_cfg.rewards),
        "decimation": int(env_cfg.decimation),
        "simulation_dt": float(env_cfg.sim.mujoco.timestep),
    }
    return {"version": VERSION, "sha256": signature(document), "definition": document}


def assert_rewards_unchanged(contract, env_cfg):
    current = capture_original_rewards(env_cfg)
    if current != contract:
        raise ValueError("Original reward definition or dt changed after task preparation")


def assert_fixed_reward_checkpoint(checkpoint, contract):
    metadata = checkpoint.get("residual_policy", {})
    inherited = metadata.get("reward_contract", {})
    if (
        inherited.get("version") != VERSION
        or inherited.get("sha256") != contract["sha256"]
        or metadata.get("reward_changes")
    ):
        raise ValueError("Initialization/teacher lacks a matching fixed-reward lineage audit; historical shaped or unaudited checkpoints are evaluation-only")

