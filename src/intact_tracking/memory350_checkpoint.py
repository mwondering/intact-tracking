"""Portable inference dependencies for Memory350 residual PPO checkpoints."""

from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf

from intact_tracking.limb_context_sampling import state_digest
from intact_tracking.memory350_inference import load_memory350_checkpoint, load_memory350_state
from intact_tracking.rollout.mjlab_adapter import _sha256

BUNDLE_VERSION = 1
BUNDLE_KEYS = ("inference_bundle_version", "frozen_context", "frozen_tracker")
CONTEXT_METADATA = (
    "architecture_version", "model_config", "normalization", "tracker",
    "context_input_contract", "memory_contract", "native_dr_version", "dr_profile",
    "predictor_action_contract", "episode_length_control_steps", "dr_metric_schema",
)


def make_context_bundle(state, *, source_path, source_sha256):
    """Keep encoder weights and input contracts, excluding the stage-one predictor/optimizer."""
    payload = {key: copy.deepcopy(state[key]) for key in CONTEXT_METADATA if key in state}
    payload["model"] = {
        name: tensor.detach().cpu().clone() for name, tensor in state["model"].items()
        if name.startswith("context_encoder.")
    }
    if not payload["model"]:
        raise ValueError("Context checkpoint contains no encoder weights")
    return {"format_version": BUNDLE_VERSION, "source_path": str(source_path),
            "source_sha256": source_sha256, "payload": payload,
            "payload_sha256": state_digest(payload)}


def make_tracker_bundle(config, actor_state, *, source_path, source_sha256):
    """Tracker tensors already live in actor_state_dict under the tracker.* prefix."""
    config = (OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config)
              else copy.deepcopy(dict(config)))
    return {"format_version": BUNDLE_VERSION, "source_path": str(source_path),
            "source_sha256": source_sha256, "cfg": config,
            "cfg_sha256": state_digest(config), "actor_state_sha256": state_digest(actor_state),
            "actor_state_prefix": "tracker."}


def _check_version(checkpoint, bundle):
    if (checkpoint.get("inference_bundle_version") != BUNDLE_VERSION
            or not isinstance(bundle, Mapping) or bundle.get("format_version") != BUNDLE_VERSION):
        raise ValueError("Unsupported frozen inference bundle version")


def embedded_context_state(checkpoint):
    """Validate embedded contents and their stage-one/tracker provenance."""
    bundle = checkpoint["frozen_context"]
    _check_version(checkpoint, bundle)
    meta = checkpoint["residual_policy"]
    if bundle["source_sha256"] != meta["context_sha256"]:
        raise ValueError("Embedded context source differs from the policy's frozen context")
    payload = bundle["payload"]
    if payload["tracker"]["checkpoint_sha256"] != meta["tracker_sha256"]:
        raise ValueError("Embedded context and frozen tracker checkpoints differ")
    if state_digest(payload) != bundle["payload_sha256"]:
        raise ValueError("Embedded context content checksum mismatch")
    return payload


def load_policy_context(checkpoint, *, device):
    """Prefer embedded dependencies; legacy checkpoints retain path-based loading."""
    meta = checkpoint["residual_policy"]
    if "frozen_context" in checkpoint:
        payload = embedded_context_state(checkpoint)
        bundle = checkpoint["frozen_context"]
        context = load_memory350_state(
            payload, device=device, source_path=bundle["source_path"],
            source_sha256=bundle["source_sha256"], expected_tracker_sha256=meta["tracker_sha256"])
    else:
        if "inference_bundle_version" in checkpoint:
            raise ValueError("Portable context policy is missing its frozen context bundle")
        context = load_memory350_checkpoint(meta["context_checkpoint"], device=device,
                                           expected_tracker_sha256=meta["tracker_sha256"])
    if context.sha256 != meta["context_sha256"]:
        raise ValueError("Frozen context checkpoint changed")
    return context


def embedded_tracker(checkpoint):
    """Return validated tracker config/weights without opening its original file."""
    if "frozen_tracker" not in checkpoint:
        if "inference_bundle_version" in checkpoint:
            raise ValueError("Portable policy is missing its frozen tracker bundle")
        return None
    bundle = checkpoint["frozen_tracker"]
    _check_version(checkpoint, bundle)
    if bundle["source_sha256"] != checkpoint["residual_policy"]["tracker_sha256"]:
        raise ValueError("Embedded tracker source differs from the policy's frozen tracker")
    if bundle["actor_state_prefix"] != "tracker.":
        raise ValueError("Unsupported embedded tracker weight prefix")
    weights = {name.removeprefix("tracker."): value
               for name, value in checkpoint["actor_state_dict"].items()
               if name.startswith("tracker.")}
    if not weights or state_digest(weights) != bundle["actor_state_sha256"]:
        raise ValueError("Embedded tracker weight checksum mismatch")
    if state_digest(bundle["cfg"]) != bundle["cfg_sha256"]:
        raise ValueError("Embedded tracker configuration checksum mismatch")
    return {"cfg": bundle["cfg"], "actor_state_dict": weights,
            "source_sha256": bundle["source_sha256"], "source_path": bundle["source_path"]}


def dependencies_from_legacy_checkpoint(checkpoint):
    """Resolve and verify the immutable sources once for a running-job exporter."""
    meta = checkpoint["residual_policy"]
    context_path = Path(meta["context_checkpoint"])
    tracker_path = Path(meta["tracker_checkpoint"])
    for path, expected in ((context_path, meta["context_sha256"]),
                           (tracker_path, meta["tracker_sha256"])):
        if _sha256(path) != expected:
            raise ValueError(f"Frozen source checksum changed: {path}")
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    tracker = torch.load(tracker_path, map_location="cpu", weights_only=False)
    result = {
        "inference_bundle_version": BUNDLE_VERSION,
        "frozen_context": make_context_bundle(context, source_path=context_path,
                                               source_sha256=meta["context_sha256"]),
        "frozen_tracker": make_tracker_bundle(
            tracker["cfg"], tracker.get("actor_state_dict", tracker.get("policy")),
            source_path=tracker_path, source_sha256=meta["tracker_sha256"]),
    }
    combined = {**checkpoint, **result}
    embedded_context_state(combined)
    embedded_tracker(combined)
    return result


def embed_checkpoint_file(path, dependencies=None):
    """Atomically add frozen dependencies, verifying all existing state is unchanged.

    The running trainer may publish another file while this one is read. Its
    original inode/mtime must still match before replacement; a conflicting
    write is retried by the watcher instead of overwriting a newer snapshot.
    """
    path = Path(path)
    before = path.stat()

    def signature(stat):
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "inference_bundle_version" in state:
        embedded_context_state(state)
        embedded_tracker(state)
        return {"path": str(path), "completed_updates": state["completed_updates"],
                "already_embedded": True, "bytes": before.st_size}
    dependencies = dependencies or dependencies_from_legacy_checkpoint(state)
    original_digest = state_digest(state)
    combined = {**state, **dependencies}
    embedded_context_state(combined)
    embedded_tracker(combined)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.bundle.", suffix=".tmp",
                                         dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(combined, stream)
            stream.flush()
            os.fsync(stream.fileno())
        restored = torch.load(temporary, map_location="cpu", weights_only=False)
        embedded_context_state(restored)
        embedded_tracker(restored)
        if state_digest({k: v for k, v in restored.items() if k not in BUNDLE_KEYS}) != original_digest:
            raise RuntimeError("Bundling changed the existing checkpoint state")
        if signature(path.stat()) != signature(before):
            raise RuntimeError("Checkpoint changed during packaging; retry the new file")
        os.chmod(temporary, before.st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "completed_updates": state["completed_updates"],
            "already_embedded": False, "bytes_before": before.st_size,
            "bytes": path.stat().st_size, "original_state_sha256": original_digest,
            "original_state_preserved": True,
            "context_payload_sha256": dependencies["frozen_context"]["payload_sha256"]}
