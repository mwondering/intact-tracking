"""Scoped sharing of immutable motion arrays during shadow construction only."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_SOURCE = ContextVar("preview_motion_source", default=None)
_REPRESENTATION_FIELDS = (
    "motion_type", "fk_from_joint_pos", "recompute_joint_vel_from_joint_pos",
    "load_compact_qpos", "reference_storage_mode", "actor_reference_fps", "body_names",
)


@contextmanager
def share_motion_arrays(command):
    # Subset loaders mutate slot contents and require a different synchronization
    # contract. Never silently share their active subset.
    if hasattr(command, "motion_store"):
        raise ValueError("Preview sharing requires the full-catalog MultiMotionCommand")
    token = _SOURCE.set(command)
    try:
        yield
    finally:
        _SOURCE.reset(token)


def shared_motion_arrays(cfg, files, body_indexes, device):
    source = _SOURCE.get()
    if source is None:
        return None
    if tuple(files) != source.motion_files or str(device) != str(source.device):
        raise ValueError("Shadow motion catalog/order or device differs from real environment")
    if not torch.equal(body_indexes, source.body_indexes):
        raise ValueError("Shadow motion body ordering differs")
    for name in _REPRESENTATION_FIELDS:
        if getattr(cfg, name) != getattr(source.cfg, name):
            raise ValueError(f"Shadow motion representation differs: {name}")
    return source.motion
