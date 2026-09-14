"""Dataset identity and load profile for the large-data preview comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from mjlab.managers.event_manager import EventTermCfg

from intact_tracking.environment.mdp.randomizations import rigid_body_payload

FULL_DATASET = "/data_zcy/wxy/motion_data_correct/motion_data_full"
LEGACY_PROFILE = "right-hand-1-3kg"
LIMB_PROFILE = "hands-shins-2-4kg"
DR_PROFILES = (LEGACY_PROFILE, LIMB_PROFILE)
LIMBS = {
    "left_hand": ("left_wrist_yaw_link", (0.12, 0.0, 0.0), (0.10, 0.08, 0.08)),
    "right_hand": ("right_wrist_yaw_link", (0.12, 0.0, 0.0), (0.10, 0.08, 0.08)),
    "left_shin": ("left_knee_link", (0.0, 0.0, -0.15), (0.10, 0.08, 0.10)),
    "right_shin": ("right_knee_link", (0.0, 0.0, -0.15), (0.10, 0.08, 0.10)),
}


def add_limb_payloads(cfg):
    """Four independent startup samples, never cumulative reset additions."""
    if any(term.func is rigid_body_payload for term in cfg.events.values()):
        raise ValueError("Refusing to stack four-limb payloads over an existing payload event")
    details = {}
    for name, (body, position, size) in LIMBS.items():
        event_name = "preview_payload_" + name
        if event_name in cfg.events:
            raise ValueError(f"Reserved event already exists: {event_name}")
        cfg.events[event_name] = EventTermCfg(
            mode="startup", func=rigid_body_payload,
            params={"body_name": body, "mass_range_kg": (2.0, 4.0),
                    "position_body_m": position, "size_m": size},
        )
        details[name] = {"event_name": event_name, "body_name": body,
                         "mass_range_kg": [2.0, 4.0], "position_body_frame_m": list(position),
                         "cuboid_size_m": list(size)}
    return {"limbs": details, "sampling": "independent per limb and world, fixed at startup",
            "total_added_mass_range_kg": [8.0, 16.0],
            "physics_model": "composite rigid cuboid mass/COM/inertia; no added collision geometry"}


def audit_limb_payloads(env):
    masses = []
    result = {}
    for name in LIMBS:
        term = env.event_manager.get_term_cfg("preview_payload_" + name).func
        mass = term.observe().flatten()
        if not torch.isfinite(mass).all() or not ((mass >= 2 - 1e-5) & (mass <= 4 + 1e-5)).all():
            raise RuntimeError(f"Incorrect actual added mass on {name}")
        masses.append(mass)
        result[name] = {"min_kg": float(mass.min()), "max_kg": float(mass.max()),
                        "mean_kg": float(mass.mean())}
    matrix = torch.stack(masses, -1)
    result["world_mass_sha256"] = hashlib.sha256(matrix.cpu().numpy().tobytes()).hexdigest()
    result["total_min_kg"] = float(matrix.sum(-1).min())
    result["total_max_kg"] = float(matrix.sum(-1).max())
    if env.num_envs > 1:
        result["sample_correlation"] = torch.corrcoef(matrix.T).cpu().tolist()
    return result


def dataset_identity(motion_file=None, motion_path=None):
    """Ordered file/stat manifest: detects ordinary dataset edits without rehashing TBs.

    This is explicitly NOT a hash of all NPZ contents. Single-file runs retain
    their historical content SHA in the caller for checkpoint compatibility.
    """
    root = Path(motion_path).resolve() if motion_path else Path(motion_file).resolve().parent
    files = sorted(root.rglob("*.npz")) if motion_path else [Path(motion_file).resolve()]
    if not files:
        raise ValueError(f"No NPZ motions in {root}")
    rows = []
    groups = {}
    for path in files:
        stat = path.stat()
        relative = str(path.relative_to(root))
        rows.append((relative, stat.st_size, stat.st_mtime_ns))
        group = relative.split("/")[0] if "/" in relative else "(root)"
        groups[group] = groups.get(group, 0) + 1
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return files, {"root": str(root), "motion_count": len(files),
                   "manifest_sha256": digest, "manifest_hash_kind": "relative-path,size,mtime_ns",
                   "archive_bytes": sum(row[1] for row in rows), "groups": groups,
                   "coverage": "all recursive *.npz files; no exclusions or active subset"}
