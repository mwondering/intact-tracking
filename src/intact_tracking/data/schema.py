"""Versioned transition schema shared by the SP collector and trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCHEMA_VERSION = "intact_tracking_transition_v1"


@dataclass(frozen=True)
class RolloutDimensions:
    proprio: int = 122
    observation: int = 64
    action: int = 29
    robot_state: int = 71
    reference_state: int = 71

    def __post_init__(self) -> None:
        invalid = {name: value for name, value in asdict(self).items() if value <= 0}
        if invalid:
            raise ValueError(f"Rollout dimensions must be positive, got {invalid}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FieldSpec:
    shape: tuple[int, ...]
    dtype: np.dtype


def core_field_specs(dimensions: RolloutDimensions) -> dict[str, FieldSpec]:
    """Fields required for causal context construction and INTACT training."""
    return {
        "proprio": FieldSpec((dimensions.proprio,), np.dtype(np.float32)),
        "next_proprio": FieldSpec((dimensions.proprio,), np.dtype(np.float32)),
        "observation": FieldSpec((dimensions.observation,), np.dtype(np.float32)),
        "next_observation": FieldSpec((dimensions.observation,), np.dtype(np.float32)),
        "reference_observation": FieldSpec((dimensions.observation,), np.dtype(np.float32)),
        "next_reference_observation": FieldSpec((dimensions.observation,), np.dtype(np.float32)),
        "action": FieldSpec((dimensions.action,), np.dtype(np.float32)),
        "reward": FieldSpec((), np.dtype(np.float32)),
        "terminated": FieldSpec((), np.dtype(np.bool_)),
        "truncated": FieldSpec((), np.dtype(np.bool_)),
        "reset_boundary": FieldSpec((), np.dtype(np.bool_)),
        "world_id": FieldSpec((), np.dtype(np.int64)),
        "episode_id": FieldSpec((), np.dtype(np.int64)),
        "episode_step": FieldSpec((), np.dtype(np.int64)),
        "collector_step": FieldSpec((), np.dtype(np.int64)),
        "env_id": FieldSpec((), np.dtype(np.int64)),
        "motion_id": FieldSpec((), np.dtype(np.int64)),
        "motion_step": FieldSpec((), np.dtype(np.int64)),
    }


def diagnostic_field_specs(dimensions: RolloutDimensions) -> dict[str, FieldSpec]:
    """Trace-only fields that never enter the deployed INTACT interface."""
    return {
        "applied_action": FieldSpec((dimensions.action,), np.dtype(np.float32)),
        "joint_target": FieldSpec((dimensions.action,), np.dtype(np.float32)),
        "joint_torque": FieldSpec((dimensions.action,), np.dtype(np.float32)),
        "robot_state": FieldSpec((dimensions.robot_state,), np.dtype(np.float32)),
        "next_robot_state": FieldSpec((dimensions.robot_state,), np.dtype(np.float32)),
        "reference_state": FieldSpec((dimensions.reference_state,), np.dtype(np.float32)),
        "next_reference_state": FieldSpec((dimensions.reference_state,), np.dtype(np.float32)),
    }
