from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from intact_tracking.environment.assets.robots.g1_tracking_bfm import G1_TRACKING_BFM_XML

from .motion_fk import (
    MotionFKHelper,
    actor_body_velocity_from_compact_support_torch,
    actor_gather_from_support_torch,
    actor_smoothed_finite_difference_from_support_torch,
    joint_vel_from_joint_pos_torch,
    normalize,
)
from .motion_fk import (
    quat_apply as fk_quat_apply,
)
from .motion_fk import (
    quat_apply_inverse as fk_quat_apply_inverse,
)
from .motion_fk import (
    quat_mul as fk_quat_mul,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
_EXTRA_REFERENCE_GHOST_COLOR = (1.0, 0.45, 0.1, 0.45)
_ADAPTIVE_SAMPLING_DISTRIBUTION_METRICS = (
    "sampling_adaptive_temperature",
    "sampling_adaptive_probability_cap_over_mean",
    "sampling_adaptive_normalized_entropy",
    "sampling_adaptive_top1_prob",
    "sampling_adaptive_top1_over_uniform",
    "sampling_adaptive_effective_num_bins",
    "sampling_final_normalized_entropy",
    "sampling_uniform_mix_ratio_pre_cap",
    "sampling_uniform_branch_probability",
    "sampling_uniform_baseline_per_bin",
    "sampling_final_top1_prob",
    "sampling_final_top1_over_uniform",
    "sampling_failure_probability_mean",
    "sampling_failure_probability_max",
    "sampling_final_effective_num_bins",
)

_QPOS_ACTOR_SUPPORT_STEPS = tuple(range(-5, 6))
_QPOS_ACTOR_CURRENT_INDEX = 5
_QPOS_BODY_SMOOTH_STEPS = (-2, -1, 0, 1, 2)


def _validate_qpos_actor_fps(fps_values: list[float], expected_fps: float) -> None:
    """Fail fast when the scalar online derivative rate cannot match the NPZ."""
    if not math.isfinite(expected_fps) or expected_fps <= 0.0:
        raise ValueError(
            f"qpos-only Actor reference FPS must be finite and positive, got {expected_fps}"
        )
    tolerance = max(1.0e-4, abs(expected_fps) * 1.0e-6)
    mismatches = [
        (index, float(actual_fps))
        for index, actual_fps in enumerate(fps_values)
        if not math.isclose(float(actual_fps), expected_fps, rel_tol=0.0, abs_tol=tolerance)
    ]
    if mismatches:
        preview = ", ".join(
            f"motion[{index}]={actual_fps:g}" for index, actual_fps in mismatches[:5]
        )
        raise ValueError(
            "qpos-only Actor FK derives velocities online with one configured FPS, "
            f"but the motion archive FPS differs: expected={expected_fps:g}, {preview}"
        )


def _paths_from_motion_exclude_file(exclude_file: str) -> list[str]:
    """Read excluded motion paths from a benchmark JSON report or text file."""
    if not exclude_file:
        return []
    if not os.path.isfile(exclude_file):
        raise FileNotFoundError(f"Motion exclude file not found: {exclude_file}")

    if exclude_file.lower().endswith(".json"):
        with open(exclude_file, encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            if "failed_motions" not in payload:
                raise ValueError("Motion exclude JSON object must contain 'failed_motions'.")
            entries = payload["failed_motions"]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError(
                "Motion exclude JSON must contain a list or an object with 'failed_motions'."
            )
        if not isinstance(entries, list):
            raise ValueError("Motion exclude JSON entries must be a list.")
        paths = []
        for entry in entries:
            if isinstance(entry, str):
                paths.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
            else:
                raise ValueError(
                    "Each motion exclude JSON entry must be a path string or an object "
                    "containing a string 'path'."
                )
        return paths

    with open(exclude_file, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]


def _normalize_motion_exclude_files(value: object) -> tuple[str, ...]:
    """Accept a path sequence or a shell-quoted Hydra list string."""
    if value is None:
        return ()
    if isinstance(value, (str, os.PathLike)):
        candidates = (value,)
    else:
        try:
            candidates = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("motion_exclude_files must be a sequence of paths") from exc

    normalized: list[str] = []
    for candidate in candidates:
        path = os.fsdecode(os.fspath(candidate)).strip()
        if path.startswith("[") and path.endswith("]"):
            inner = path[1:-1].strip()
            if not inner:
                continue
            entries = next(csv.reader([inner], skipinitialspace=True))
            normalized.extend(entry.strip().strip("'\"") for entry in entries if entry.strip())
        elif path:
            normalized.append(path)
    return tuple(normalized)


def filter_excluded_motion_files(
    motion_files: list[str],
    *,
    motion_path: str = "",
    excluded_motion_files: tuple[str, ...] = (),
    motion_exclude_files: tuple[str, ...] = (),
    motion_exclude_file: str = "",
) -> list[str]:
    """Remove explicitly excluded motions while preserving input ordering.

    Relative paths are interpreted relative to ``motion_path`` when it is set.
    JSON benchmark reports can be passed directly via ``motion_exclude_files``;
    their ``failed_motions[*].path`` entries are merged. The legacy singular
    ``motion_exclude_file`` remains supported.
    """
    configured_paths = [
        *[os.fspath(path) for path in excluded_motion_files],
        *[
            path
            for exclude_file in _normalize_motion_exclude_files(motion_exclude_files)
            for path in _paths_from_motion_exclude_file(os.fspath(exclude_file))
        ],
        *_paths_from_motion_exclude_file(os.fspath(motion_exclude_file)),
    ]
    if not configured_paths:
        return list(motion_files)

    root = os.path.abspath(os.path.expanduser(motion_path)) if motion_path else ""
    real_root = os.path.realpath(root) if root else ""

    def key_from_absolute(path: str) -> str:
        absolute = os.path.normcase(os.path.abspath(path))
        for base in (root, real_root):
            if not base:
                continue
            try:
                if os.path.commonpath((absolute, base)) == base:
                    relative = os.path.relpath(absolute, base)
                    return f"relative:{os.path.normcase(os.path.normpath(relative))}"
            except ValueError:
                continue
        return f"absolute:{absolute}"

    def configured_key(path: str) -> str:
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded) and root:
            expanded = os.path.join(root, expanded)
        return key_from_absolute(expanded)

    def motion_key(path: str) -> str:
        # Scanner and manifest paths are already rooted at motion_path.  Resolve
        # those relative paths from the process working directory.  Bare relative
        # API inputs remain relative to motion_path for backward compatibility.
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded) and root:
            candidate_absolute = os.path.abspath(expanded)
            try:
                is_already_rooted = os.path.commonpath((candidate_absolute, root)) == root
            except ValueError:
                is_already_rooted = False
            if not is_already_rooted:
                expanded = os.path.join(root, expanded)
        return key_from_absolute(expanded)

    excluded = {configured_key(path) for path in configured_paths}
    filtered = [path for path in motion_files if motion_key(path) not in excluded]
    removed_count = len(motion_files) - len(filtered)
    if removed_count:
        print(
            f"[INFO] Excluded {removed_count} of {len(motion_files)} motion files.",
            flush=True,
        )
    return filtered


def _multimotion_bootstrap_log(message: str) -> None:
    rank = os.environ.get("RANK", "0")
    world_size = os.environ.get("WORLD_SIZE", "1")
    local_rank = os.environ.get("LOCAL_RANK", rank)
    print(
        f"[MULTIMOTION][{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"rank={rank}/{world_size} local_rank={local_rank}: {message}",
        flush=True,
    )


@dataclass(kw_only=True)
class AdaptiveSamplingCfg:
    """Optional structured controls layered on legacy adaptive sampling."""

    # ``None`` uses ``adaptive_uniform_ratio`` for backward compatibility.
    random_probability: float | None = None
    # ``branch`` first selects a pure uniform or pure adaptive source. ``mixture``
    # retains the legacy single-distribution marginal for older tasks.
    strategy: Literal["mixture", "branch"] = "mixture"
    # Raw failure-rate temperature. ``None`` preserves the legacy proportional
    # weighting; positive values apply a masked softmax to the raw failure rates.
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is None:
            return
        self.temperature = float(self.temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("adaptive sampling temperature must be finite and positive")


@dataclass(kw_only=True)
class RewindCfg:
    """Failure-only reset rewind policy."""

    enabled: bool = False
    failure_probability: float = 0.4
    min_steps: int = 0
    max_steps: int = 0


def _resolve_adaptive_prior_counts(cfg) -> tuple[float, float]:
    """Resolve the completed-visit and failed-visit priors."""
    visit_count = max(float(cfg.adaptive_prior_visit_count), 0.0)
    failure_count = max(float(cfg.adaptive_prior_failure_count), 0.0)
    if failure_count > visit_count:
        raise ValueError("adaptive_prior_failure_count cannot exceed visit prior")
    return visit_count, failure_count


def _resolve_adaptive_ema_iterations(cfg) -> int | None:
    """Resolve the EMA horizon, retaining the old window option as an alias."""
    ema_iterations = getattr(cfg, "adaptive_failure_rate_ema_iterations", None)
    if ema_iterations is None:
        ema_iterations = getattr(cfg, "adaptive_failure_rate_window_iterations", None)
    if ema_iterations is None or int(ema_iterations) <= 0:
        return None
    return max(int(ema_iterations), 1)


def _adaptive_ema_decay(cfg) -> float:
    """Return a decay whose steady-state effective mass is the configured horizon."""
    ema_iterations = _resolve_adaptive_ema_iterations(cfg)
    if ema_iterations is None:
        return 1.0
    return 1.0 - 1.0 / float(ema_iterations)


def _temperature_scale_adaptive_signal(
    signal: torch.Tensor,
    temperature: float | None,
) -> torch.Tensor:
    """Apply raw-value temperature scaling without giving zero-signal bins mass."""
    scaled = torch.clamp_min(signal, 0.0)
    if temperature is None or scaled.numel() == 0:
        return scaled
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("adaptive sampling temperature must be finite and positive")
    # Centering by the maximum is softmax-invariant and keeps exp() bounded.
    # Masking zeros retains the adaptive sampler's existing support semantics.
    logits = (scaled - scaled.max()) / temperature
    return logits.exp().masked_fill(scaled <= 0.0, 0.0)


def _resolve_probability_cap(
    value: float | Literal["auto"] | None,
    count: int,
    auto_cap_over_mean: float,
) -> float | None:
    if value is None:
        return None
    if value == "auto":
        if count <= 0:
            return 1.0
        return min(float(auto_cap_over_mean) / float(count), 1.0)
    resolved = float(value)
    if resolved <= 0.0:
        return None
    return min(resolved, 1.0)


def _allocate_capped_mass(
    weights: torch.Tensor,
    *,
    total_mass: float | torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    """Allocate mass proportionally without violating elementwise capacities."""
    if weights.ndim != 1 or capacities.shape != weights.shape:
        raise ValueError("weights and capacities must be matching rank-1 tensors")
    if weights.numel() == 0:
        return torch.zeros_like(weights)

    weights = torch.clamp_min(weights, 0.0)
    capacities = torch.clamp_min(capacities.to(dtype=weights.dtype), 0.0)
    target = torch.as_tensor(total_mass, dtype=weights.dtype, device=weights.device)
    tolerance = max(float(torch.finfo(weights.dtype).eps) * 32.0, 1e-7)
    if float(capacities.sum().item()) + tolerance < float(target.item()):
        raise ValueError("Probability caps are infeasible: total capacity is smaller than one")

    allocation = torch.zeros_like(weights)
    active = capacities > 0.0
    remaining_mass = target.clone()
    while bool(active.any()):
        active_weights = weights.masked_fill(~active, 0.0)
        active_weight_sum = active_weights.sum()
        if float(active_weight_sum.item()) <= tolerance:
            active_weights = capacities.masked_fill(~active, 0.0)
            active_weight_sum = active_weights.sum()
        proposal = remaining_mass * active_weights / torch.clamp(active_weight_sum, min=tolerance)
        over_cap = active & (proposal > capacities + tolerance)
        if not bool(over_cap.any()):
            allocation[active] = proposal[active]
            remaining_mass.zero_()
            break
        allocation[over_cap] = capacities[over_cap]
        remaining_mass -= capacities[over_cap].sum()
        active[over_cap] = False

    if float(remaining_mass.abs().item()) > tolerance:
        raise RuntimeError("Capped probability allocation did not conserve mass")
    return allocation


def _allocate_capped_mass_by_group(
    weights: torch.Tensor,
    group_ids: torch.Tensor,
    group_masses: torch.Tensor,
    *,
    per_item_cap: float,
) -> torch.Tensor:
    """Allocate each group's target mass while enforcing a per-item cap."""
    allocation = torch.zeros_like(weights)
    active = torch.ones_like(weights, dtype=torch.bool)
    remaining_group_mass = group_masses.clone()
    num_groups = int(group_masses.numel())
    tolerance = max(float(torch.finfo(weights.dtype).eps) * 32.0, 1e-7)

    while bool(active.any()):
        active_weights = weights.masked_fill(~active, 0.0)
        weight_sums = torch.zeros_like(group_masses)
        weight_sums.scatter_add_(0, group_ids, active_weights)
        active_counts = torch.zeros(num_groups, dtype=weights.dtype, device=weights.device)
        active_counts.scatter_add_(0, group_ids, active.to(weights.dtype))
        denominators = weight_sums[group_ids]
        weighted_proposal = (
            remaining_group_mass[group_ids]
            * active_weights
            / torch.clamp(denominators, min=tolerance)
        )
        uniform_proposal = remaining_group_mass[group_ids] / torch.clamp(
            active_counts[group_ids], min=1.0
        )
        proposal = torch.where(denominators > tolerance, weighted_proposal, uniform_proposal)
        proposal.masked_fill_(~active, 0.0)

        over_cap = active & (proposal > per_item_cap + tolerance)
        groups_with_over = torch.zeros(num_groups, dtype=torch.long, device=weights.device)
        groups_with_over.scatter_add_(0, group_ids, over_cap.to(torch.long))
        finalize = active & (groups_with_over[group_ids] == 0)
        allocation[finalize] = proposal[finalize]
        active[finalize] = False

        if bool(over_cap.any()):
            allocation[over_cap] = per_item_cap
            removed_mass = torch.zeros_like(group_masses)
            removed_mass.scatter_add_(
                0,
                group_ids[over_cap],
                torch.full_like(weights[over_cap], per_item_cap),
            )
            remaining_group_mass -= removed_mass
            active[over_cap] = False

    return allocation


def _apply_final_probability_caps(
    probabilities: torch.Tensor,
    motion_ids: torch.Tensor,
    *,
    num_motions: int,
    max_prob_per_bin: float | Literal["auto"] | None,
    max_prob_per_motion: float | Literal["auto"] | None,
    auto_cap_over_mean: float,
) -> torch.Tensor:
    """Apply exact bin and motion caps to a normalized sampling distribution."""
    probabilities = probabilities / torch.clamp(probabilities.sum(), min=1e-12)
    num_bins = int(probabilities.numel())
    if num_bins == 0:
        return probabilities

    bin_cap = _resolve_probability_cap(max_prob_per_bin, num_bins, auto_cap_over_mean)
    motion_cap = _resolve_probability_cap(max_prob_per_motion, num_motions, auto_cap_over_mean)
    if bin_cap is None and motion_cap is None:
        return probabilities

    resolved_bin_cap = 1.0 if bin_cap is None else float(bin_cap)
    resolved_motion_cap = 1.0 if motion_cap is None else float(motion_cap)
    motion_ids = motion_ids.to(dtype=torch.long, device=probabilities.device)
    bins_per_motion = torch.bincount(motion_ids, minlength=num_motions).to(probabilities.dtype)
    motion_capacities = torch.minimum(
        torch.full_like(bins_per_motion, resolved_motion_cap),
        bins_per_motion * resolved_bin_cap,
    )
    raw_motion_mass = torch.zeros(
        num_motions, dtype=probabilities.dtype, device=probabilities.device
    )
    raw_motion_mass.scatter_add_(0, motion_ids, probabilities)
    target_motion_mass = _allocate_capped_mass(
        raw_motion_mass,
        total_mass=1.0,
        capacities=motion_capacities,
    )
    constrained = _allocate_capped_mass_by_group(
        probabilities,
        motion_ids,
        target_motion_mass,
        per_item_cap=resolved_bin_cap,
    )
    return constrained / torch.clamp(constrained.sum(), min=1e-12)


def _sample_adaptive_uniform_branches(
    adaptive_probabilities: torch.Tensor,
    sample_count: int,
    uniform_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent uniform/adaptive branches.

    Returns the sampled pair indices and a mask identifying pure-uniform
    samples. The complementary mask therefore identifies pure-adaptive samples.
    """
    if adaptive_probabilities.ndim != 1 or adaptive_probabilities.numel() == 0:
        raise ValueError("adaptive_probabilities must be a non-empty rank-1 tensor")
    sample_count = int(sample_count)
    if sample_count < 0:
        raise ValueError("sample_count cannot be negative")
    uniform_probability = float(max(0.0, min(1.0, uniform_probability)))
    device = adaptive_probabilities.device
    uniform_mask = torch.rand(sample_count, device=device) < uniform_probability
    sampled_pair_indices = torch.empty(sample_count, dtype=torch.long, device=device)
    uniform_positions = torch.where(uniform_mask)[0]
    adaptive_positions = torch.where(~uniform_mask)[0]
    if uniform_positions.numel() > 0:
        sampled_pair_indices[uniform_positions] = torch.randint(
            adaptive_probabilities.numel(),
            (uniform_positions.numel(),),
            device=device,
        )
    if adaptive_positions.numel() > 0:
        sampled_pair_indices[adaptive_positions] = torch.multinomial(
            adaptive_probabilities,
            adaptive_positions.numel(),
            replacement=True,
        )
    return sampled_pair_indices, uniform_mask


def gradient_test_motion_assignment(
    mode: Literal["simple", "hard", "mixed"],
    env_ids: torch.Tensor,
    num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return loader indices and stable task labels for diagnostic runs."""
    if mode == "simple":
        return torch.zeros_like(env_ids), torch.zeros_like(env_ids)
    if mode == "hard":
        return torch.zeros_like(env_ids), torch.ones_like(env_ids)
    if mode != "mixed":
        raise ValueError(
            f"gradient_test_mode must be one of 'simple', 'hard', or 'mixed', got {mode!r}"
        )
    if num_envs % 2:
        raise ValueError(
            "mixed gradient diagnostics require an even number of environments "
            f"per rank, got {num_envs}"
        )
    assignment = (env_ids >= num_envs // 2).long()
    return assignment, assignment


def apply_reset_ground_clearance(
    root_pos: torch.Tensor,
    body_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    position_noise: torch.Tensor,
    orientation_delta: torch.Tensor,
    *,
    root_lift_height: float,
    min_body_z: float | None,
) -> torch.Tensor:
    """Apply reset noise and lift the root enough to keep all bodies above ground."""
    adjusted = root_pos + position_noise
    adjusted[:, 2] += float(root_lift_height)
    if min_body_z is None:
        return adjusted

    body_pos_relative = body_pos_w - root_pos.unsqueeze(1)
    body_orientation_delta = orientation_delta.unsqueeze(1).expand(
        -1, body_pos_relative.shape[1], -1
    )
    rotated_body_pos_relative = quat_apply(body_orientation_delta, body_pos_relative)
    predicted_body_z = adjusted[:, None, 2] + rotated_body_pos_relative[..., 2]
    ground_z = env_origins[:, 2] + float(min_body_z)
    correction = (ground_z - predicted_body_z.amin(dim=1)).clamp_min(0.0)
    adjusted[:, 2] += correction
    return adjusted


def clamp_reset_joint_velocity(joint_vel: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Clamp reset joint velocity when a task opts into a finite safety bound."""
    if limit is None:
        return joint_vel
    limit = float(limit)
    if limit <= 0.0:
        raise ValueError("reset_joint_vel_limit must be positive when configured")
    return joint_vel.clamp(min=-limit, max=limit)


_ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

_MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
MUJOCO_JOINT_NAMES = tuple(_MUJOCO_JOINT_NAMES)

_ISAACLAB_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

_MUJOCO_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

_ISAACLAB_TO_MUJOCO_JOINT_REINDEX = [
    _ISAACLAB_JOINT_NAMES.index(name) for name in _MUJOCO_JOINT_NAMES
]
_ISAACLAB_TO_MUJOCO_BODY_REINDEX = [_ISAACLAB_BODY_NAMES.index(name) for name in _MUJOCO_BODY_NAMES]
ISAACLAB_TO_MUJOCO_JOINT_REINDEX = tuple(_ISAACLAB_TO_MUJOCO_JOINT_REINDEX)
MUJOCO_BODY_NAMES = tuple(_MUJOCO_BODY_NAMES)

DEFAULT_MOTION_FPS = 50.0

REFERENCE_MOTION_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

REFERENCE_STORAGE_FULL = "full"
REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK = "qpos_only_actor_fk"
REFERENCE_STORAGE_MODES = (
    REFERENCE_STORAGE_FULL,
    REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK,
)


def extract_motion_fps(data: np.lib.npyio.NpzFile) -> tuple[float, bool, bool]:
    """Return ``(fps, is_non_scalar, used_default)`` for a motion archive."""
    if "fps" not in data.files:
        return DEFAULT_MOTION_FPS, False, True
    fps_array = np.asarray(data["fps"], dtype=np.float32)
    if fps_array.size == 0:
        return DEFAULT_MOTION_FPS, False, True
    return float(fps_array.reshape(-1)[0]), fps_array.size > 1, False


def _select_or_fk_body_fields(
    *,
    joint_pos: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    body_indexes: torch.Tensor,
    fps: float,
    fk_from_joint_pos: bool,
    fk_helper: MotionFKHelper | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=body_pos_w.device)
    if body_indexes.numel() == 0:
        raise ValueError("body_indexes cannot be empty")
    if fk_from_joint_pos:
        if fk_helper is None:
            raise ValueError("fk_helper is required when fk_from_joint_pos is enabled")
        # SP requests FK explicitly so body fields are derived from the same joint
        # reference used by the source dataset, even when a
        # legacy NPZ happens to carry compatible body arrays.
        fk = fk_helper.expand_motion(
            root_pos_w=body_pos_w[:, 0, :],
            root_quat_w=body_quat_w[:, 0, :],
            joint_pos=joint_pos,
            fps=fps,
        )
        return fk.body_pos_w, fk.body_quat_w, fk.body_lin_vel_w, fk.body_ang_vel_w
    if int(body_indexes.max().item()) < int(body_pos_w.shape[1]):
        return (
            body_pos_w[:, body_indexes, :],
            body_quat_w[:, body_indexes, :],
            body_lin_vel_w[:, body_indexes, :],
            body_ang_vel_w[:, body_indexes, :],
        )
    return (
        body_pos_w[:, body_indexes, :],
        body_quat_w[:, body_indexes, :],
        body_lin_vel_w[:, body_indexes, :],
        body_ang_vel_w[:, body_indexes, :],
    )


def _select_or_recompute_joint_vel(
    *,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    fps: float,
    recompute_joint_vel_from_joint_pos: bool,
) -> torch.Tensor:
    """Return raw joint velocity or the source-SP reconstruction, by config."""
    if not recompute_joint_vel_from_joint_pos:
        return joint_vel
    return joint_vel_from_joint_pos_torch(joint_pos, fps, dim=0)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: torch.Tensor,
        motion_type: Literal["isaaclab", "mujoco"] = "isaaclab",
        device: str = "cpu",
        fk_from_joint_pos: bool = False,
        recompute_joint_vel_from_joint_pos: bool = False,
        fk_helper: MotionFKHelper | None = None,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps, _, _ = extract_motion_fps(data)
        joint_reindex = None
        body_reindex = None
        if motion_type == "isaaclab":
            joint_reindex = _ISAACLAB_TO_MUJOCO_JOINT_REINDEX
            body_reindex = _ISAACLAB_TO_MUJOCO_BODY_REINDEX
        elif motion_type != "mujoco":
            raise ValueError(f"Unsupported motion_type: {motion_type}")
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        if joint_reindex is not None:
            self.joint_pos = self.joint_pos[:, joint_reindex]
            self.joint_vel = self.joint_vel[:, joint_reindex]
        if body_reindex is not None:
            self._body_pos_w = self._body_pos_w[:, body_reindex, :]
            self._body_quat_w = self._body_quat_w[:, body_reindex, :]
            self._body_lin_vel_w = self._body_lin_vel_w[:, body_reindex, :]
            self._body_ang_vel_w = self._body_ang_vel_w[:, body_reindex, :]
        self.joint_vel = _select_or_recompute_joint_vel(
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
            fps=self.fps,
            recompute_joint_vel_from_joint_pos=recompute_joint_vel_from_joint_pos,
        )
        (
            self._body_pos_w,
            self._body_quat_w,
            self._body_lin_vel_w,
            self._body_ang_vel_w,
        ) = _select_or_fk_body_fields(
            joint_pos=self.joint_pos,
            body_pos_w=self._body_pos_w,
            body_quat_w=self._body_quat_w,
            body_lin_vel_w=self._body_lin_vel_w,
            body_ang_vel_w=self._body_ang_vel_w,
            body_indexes=body_indexes,
            fps=self.fps,
            fk_from_joint_pos=fk_from_joint_pos,
            fk_helper=fk_helper,
        )
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w


class MultiMotionLoader:
    def __init__(
        self,
        motion_files: list[str],
        body_indexes: torch.Tensor,
        motion_type: Literal["isaaclab", "mujoco"] = "isaaclab",
        device: str = "cpu",
        fk_from_joint_pos: bool = False,
        recompute_joint_vel_from_joint_pos: bool = False,
        load_compact_qpos: bool = False,
        reference_storage_mode: Literal["full", "qpos_only_actor_fk"] = "full",
        fk_helper: MotionFKHelper | None = None,
        progress_log_interval_s: float = 10.0,
    ):
        assert len(motion_files) > 0, "motion_files cannot be empty"
        self.num_files = len(motion_files)
        self.device = device
        self._body_indexes = body_indexes
        if reference_storage_mode not in REFERENCE_STORAGE_MODES:
            raise ValueError(
                "reference_storage_mode must be one of "
                f"{REFERENCE_STORAGE_MODES}, got {reference_storage_mode!r}"
            )
        self.reference_storage_mode = reference_storage_mode
        self.qpos_only = reference_storage_mode == REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK
        self.fps_list = []
        self.file_lengths = []
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        qpos_list = []
        self.load_compact_qpos = bool(load_compact_qpos) or self.qpos_only

        joint_reindex = None
        body_reindex = None
        if motion_type == "isaaclab":
            joint_reindex = _ISAACLAB_TO_MUJOCO_JOINT_REINDEX
            body_reindex = _ISAACLAB_TO_MUJOCO_BODY_REINDEX
        elif motion_type != "mujoco":
            raise ValueError(f"Unsupported motion_type: {motion_type}")

        load_start = time.perf_counter()
        last_log_time = load_start
        progress_log_interval_s = max(float(progress_log_interval_s), 0.0)
        _multimotion_bootstrap_log(
            f"stage=motion_load start motions={len(motion_files)} device={device}"
        )
        for motion_index, motion_file in enumerate(motion_files, start=1):
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            fps_value, _, _ = extract_motion_fps(data)
            self.fps_list.append(fps_value)

            # Qpos-only A variants deliberately derive their compact state from the
            # same clean NPZ fields that feed the Actor today. An optional archive-level
            # ``qpos`` is ignored: accepting it could silently make task-side FK disagree
            # with the Actor whenever the two representations differ.
            if self.qpos_only:
                joint_pos_np = np.asarray(data["joint_pos"], dtype=np.float32)
                if joint_reindex is not None:
                    joint_pos_np = joint_pos_np[:, joint_reindex]
                root_source_index = 0 if body_reindex is None else body_reindex[0]
                root_pos_np = np.asarray(
                    data["body_pos_w"][:, root_source_index, :], dtype=np.float32
                )
                root_quat_np = np.asarray(
                    data["body_quat_w"][:, root_source_index, :], dtype=np.float32
                )
                if joint_pos_np.shape[1] != len(_MUJOCO_JOINT_NAMES):
                    raise ValueError(
                        f"Expected {len(_MUJOCO_JOINT_NAMES)} joints, got "
                        f"{joint_pos_np.shape[1]} for {motion_file}"
                    )
                compact_qpos = np.concatenate((root_pos_np, root_quat_np, joint_pos_np), axis=-1)
                if compact_qpos.shape != (joint_pos_np.shape[0], 36):
                    raise ValueError(
                        f"Expected compact qpos [frames,36], got {compact_qpos.shape} "
                        f"for {motion_file}"
                    )
                # Keep per-file staging on CPU and perform one final device transfer.
                # This avoids a second full qpos corpus on the GPU during concatenation.
                qpos_list.append(torch.from_numpy(compact_qpos))
                self.file_lengths.append(joint_pos_np.shape[0])
                data.close()
                now = time.perf_counter()
                if progress_log_interval_s > 0.0 and now - last_log_time >= progress_log_interval_s:
                    _multimotion_bootstrap_log(
                        "stage=motion_load progress "
                        f"loaded={motion_index}/{len(motion_files)} "
                        f"elapsed={now - load_start:.3f}s file={motion_file}"
                    )
                    last_log_time = now
                continue

            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
            bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
            bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
            blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
            bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)
            if joint_reindex is not None:
                jp = jp[:, joint_reindex]
                jv = jv[:, joint_reindex]
            if body_reindex is not None:
                bp = bp[:, body_reindex, :]
                bq = bq[:, body_reindex, :]
                blv = blv[:, body_reindex, :]
                bav = bav[:, body_reindex, :]

            if self.load_compact_qpos:
                if "qpos" in data.files:
                    compact_qpos = np.asarray(data["qpos"], dtype=np.float32)
                    if compact_qpos.shape != (jp.shape[0], 36):
                        raise ValueError(
                            f"Expected qpos [frames,36], got {compact_qpos.shape} for {motion_file}"
                        )
                    compact_qpos_t = torch.as_tensor(
                        compact_qpos, dtype=torch.float32, device=self.device
                    )
                else:
                    compact_qpos_t = torch.cat((bp[:, 0], bq[:, 0], jp), dim=-1)
                qpos_list.append(compact_qpos_t)
            jv = _select_or_recompute_joint_vel(
                joint_pos=jp,
                joint_vel=jv,
                fps=fps_value,
                recompute_joint_vel_from_joint_pos=recompute_joint_vel_from_joint_pos,
            )

            bp, bq, blv, bav = _select_or_fk_body_fields(
                joint_pos=jp,
                body_pos_w=bp,
                body_quat_w=bq,
                body_lin_vel_w=blv,
                body_ang_vel_w=bav,
                body_indexes=self._body_indexes,
                fps=fps_value,
                fk_from_joint_pos=fk_from_joint_pos,
                fk_helper=fk_helper,
            )

            joint_pos_list.append(jp)
            joint_vel_list.append(jv)
            body_pos_w_list.append(bp)
            body_quat_w_list.append(bq)
            body_lin_vel_w_list.append(blv)
            body_ang_vel_w_list.append(bav)
            self.file_lengths.append(jp.shape[0])
            data.close()

            now = time.perf_counter()
            if progress_log_interval_s > 0.0 and now - last_log_time >= progress_log_interval_s:
                _multimotion_bootstrap_log(
                    "stage=motion_load progress "
                    f"loaded={motion_index}/{len(motion_files)} "
                    f"elapsed={now - load_start:.3f}s file={motion_file}"
                )
                last_log_time = now

        _multimotion_bootstrap_log(
            "stage=motion_load read_done "
            f"motions={len(motion_files)} elapsed={time.perf_counter() - load_start:.3f}s"
        )
        _multimotion_bootstrap_log(
            f"stage=motion_load concatenate_start motions={len(motion_files)}"
        )
        self.file_lengths = torch.tensor(self.file_lengths, dtype=torch.long, device=self.device)
        self.fps = self.fps_list[0]  # 可以根据需求调整
        self.joint_dim = len(_MUJOCO_JOINT_NAMES) if self.qpos_only else joint_pos_list[0].shape[1]
        self.body_dim = int(body_indexes.numel()) if self.qpos_only else body_pos_w_list[0].shape[1]
        self.length_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),
                self.file_lengths[:-1].cumsum(dim=0),
            ]
        )
        if not self.qpos_only:
            self.joint_pos = torch.cat(joint_pos_list, dim=0)
            self.joint_vel = torch.cat(joint_vel_list, dim=0)
            self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
            self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
            self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
            self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
        if self.load_compact_qpos:
            compact_qpos = torch.cat(qpos_list, dim=0)
            self.qpos = (
                compact_qpos.to(device=self.device, dtype=torch.float32)
                if self.qpos_only
                else compact_qpos
            )

        self._amp_obs_flat: torch.Tensor | None = None
        _multimotion_bootstrap_log(
            "stage=motion_load done "
            f"motions={len(motion_files)} frames={int(self.file_lengths.sum().item())} "
            f"elapsed={time.perf_counter() - load_start:.3f}s"
        )

    # ------------------------------------------------------------------
    # AMP demo data sampling (reuses already-loaded GPU tensors)
    # ------------------------------------------------------------------

    # def build_amp_obs_buffer(self, anchor_body_idx: int) -> None:
    #   """Precompute a flat AMP obs tensor across all motion files.

    #   Feature layout per frame:
    #     [joint_pos (n_dof)]

    #   This is called once by the runner; subsequent ``sample_amp_obs`` calls are
    #   a single GPU randint + index, with no extra data loading.
    #   """
    #   obs_list = []
    #   for i in range(self.num_files):
    #     anchor_quat = self._body_quat_w_list[i][:, anchor_body_idx]  # (T, 4)
    #     lin_vel_w = self._body_lin_vel_w_list[i][:, anchor_body_idx]  # (T, 3)
    #     ang_vel_w = self._body_ang_vel_w_list[i][:, anchor_body_idx]  # (T, 3)

    #     quat_inv_anchor = quat_inv(anchor_quat)
    #     lin_vel_b = quat_apply(quat_inv_anchor, lin_vel_w)
    #     ang_vel_b = quat_apply(quat_inv_anchor, ang_vel_w)
    #     obs_list.append(
    #       torch.cat(
    #         [lin_vel_b, ang_vel_b, self.joint_pos_list[i], self.joint_vel_list[i]],
    #         dim=-1,
    #       )
    #     )

    #   self._amp_obs_flat = torch.cat(obs_list, dim=0)  # (total_frames, n_dof)
    #   self._amp_seq_starts: torch.Tensor | None = None
    #   self._amp_seq_steps: int = 0

    # @property
    # def amp_obs_dim(self) -> int:
    #   assert self._amp_obs_flat is not None, "Call build_amp_obs_buffer() first."
    #   return self._amp_obs_flat.shape[1]

    # def build_amp_seq_table(self, steps: int) -> None:
    #   """Precompute valid sequence start indices for ``sample_amp_obs_sequence``.

    #   Must be called once (after ``build_amp_obs_buffer``) before training starts.
    #   Builds a 1-D tensor of all absolute frame indices into ``_amp_obs_flat``
    #   that are valid starting positions for a ``steps``-length consecutive window
    #   within a single motion clip.

    #   Args:
    #     steps: Number of consecutive frames per sequence. Must match the value
    #       passed to every subsequent ``sample_amp_obs_sequence`` call.
    #   """
    #   assert self._amp_obs_flat is not None, "Call build_amp_obs_buffer() first."
    #   starts_list: list[torch.Tensor] = []
    #   offset = 0
    #   for length in self.file_lengths.tolist():
    #     n_valid = length - steps + 1
    #     if n_valid > 0:
    #       starts_list.append(
    #         torch.arange(offset, offset + n_valid, dtype=torch.long, device=self.device)
    #       )
    #     offset += length

    #   if not starts_list:
    #     raise RuntimeError(
    #       f"No motion file is long enough to provide sequences of {steps} frames."
    #     )
    #   self._amp_seq_starts = torch.cat(starts_list)  # (total_valid,)
    #   self._amp_seq_steps = steps

    # def sample_amp_obs(self, batch_size: int) -> torch.Tensor:
    #   """Return a random batch of AMP demo observations. Shape: (batch_size, amp_obs_dim)."""
    #   assert self._amp_obs_flat is not None, "Call build_amp_obs_buffer() first."
    #   idx = torch.randint(
    #     0, self._amp_obs_flat.shape[0], (batch_size,), device=self.device
    #   )
    #   return self._amp_obs_flat[idx]

    # def sample_amp_obs_sequence(self, batch_size: int, steps: int) -> torch.Tensor:
    #   """Return batches of *consecutive* AMP demo observations.

    #   Requires ``build_amp_seq_table(steps)`` to have been called first.
    #   Sampling is a single randint + two index operations — no Python loops,
    #   no CUDA synchronisation.

    #   Args:
    #     batch_size: Number of sequences to sample.
    #     steps: Number of consecutive frames per sequence. Must match the value
    #       passed to ``build_amp_seq_table``.

    #   Returns:
    #     Tensor of shape (batch_size, steps, amp_obs_dim).
    #   """
    #   assert self._amp_obs_flat is not None, "Call build_amp_obs_buffer() first."
    #   assert self._amp_seq_starts is not None, (
    #     "Call build_amp_seq_table(steps) before sample_amp_obs_sequence()."
    #   )
    #   assert steps == self._amp_seq_steps, (
    #     f"steps={steps} does not match precomputed table steps={self._amp_seq_steps}."
    #   )
    #   rand_idx = torch.randint(
    #     0, self._amp_seq_starts.shape[0], (batch_size,), device=self.device
    #   )
    #   start_frames = self._amp_seq_starts[rand_idx]  # (batch_size,)
    #   frame_idx = start_frames.unsqueeze(1) + torch.arange(
    #     steps, device=self.device
    #   ).unsqueeze(0)  # (batch_size, steps)
    #   return self._amp_obs_flat[frame_idx]  # (batch_size, steps, amp_obs_dim)

    def get_motion_data_batch(
        self, motion_idx: int, time_steps_start: torch.Tensor, time_steps_end: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        time_steps_tensor = torch.arange(
            time_steps_start.item(),
            time_steps_end.item(),
            device=self.device,
            dtype=torch.long,
        )
        time_steps_tensor = torch.clamp(
            time_steps_tensor,
            torch.tensor(0, device=self.device),
            self.file_lengths[motion_idx] - 1,
        )
        frame_indices = self.length_starts[motion_idx] + time_steps_tensor
        return {
            "joint_pos": self.joint_pos[frame_indices],
            "joint_vel": self.joint_vel[frame_indices],
            "body_pos_w": self.body_pos_w[frame_indices],
            "body_quat_w": self.body_quat_w[frame_indices],
            "body_lin_vel_w": self.body_lin_vel_w[frame_indices],
            "body_ang_vel_w": self.body_ang_vel_w[frame_indices],
        }


@dataclass(frozen=True)
class ActorRootReferenceState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass(frozen=True)
class ActorRootReferencePoseState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor


@dataclass(frozen=True)
class ActorRootReferenceVelocityState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor


@dataclass(frozen=True)
class ActorJointReferenceState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass(frozen=True)
class _ActorRootKinematicsSupport:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    root_ang_vel_b_body_support: torch.Tensor


@dataclass(frozen=True)
class ActorBodyReferenceState:
    pos_b: torch.Tensor
    quat_b: torch.Tensor
    lin_vel_b: torch.Tensor
    ang_vel_b: torch.Tensor


class MultiMotionCommand(CommandTerm):
    cfg: "MultiMotionCommandCfg"
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: "MultiMotionCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        motion_files = self._resolve_motion_files()
        self.motion_files = tuple(motion_files)
        fk_helper = self._build_fk_helper()
        self._reference_fk_helper = fk_helper
        self.motion = MultiMotionLoader(
            motion_files,
            self.body_indexes,
            motion_type=self.cfg.motion_type,
            device=self.device,
            fk_from_joint_pos=self.cfg.fk_from_joint_pos,
            recompute_joint_vel_from_joint_pos=self.cfg.recompute_joint_vel_from_joint_pos,
            load_compact_qpos=self.cfg.load_compact_qpos,
            reference_storage_mode=self.cfg.reference_storage_mode,
            fk_helper=fk_helper,
            progress_log_interval_s=float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0)),
        )
        if self.cfg.reference_storage_mode == REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK:
            _validate_qpos_actor_fps(self.motion.fps_list, float(self.cfg.actor_reference_fps))

        # 初始化状态变量
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gradient_test_motion_label = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # One-step pulse consumed by rollout-only observations. It marks a
        # mid-episode reference resample without turning it into an environment
        # termination.
        self.motion_resample_boundary = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._initialize_sp_tracking_state()

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling bins are tracked per-motion on a shared global bin axis.
        # Each motion only uses the prefix indicated by bin_valid_mask.
        max_motion_length = self.motion.file_lengths.max().item()
        if self.cfg.adaptive_bin_width_steps is not None:
            self.bin_width_steps = max(int(self.cfg.adaptive_bin_width_steps), 1)
        else:
            self.bin_width_steps = max(
                int(round(float(self.cfg.adaptive_bin_width_s) / env.step_dt)), 1
            )
        self.bin_count = int(max_motion_length // self.bin_width_steps) + 1
        self.motion_bin_counts = torch.clamp(
            torch.div(
                self.motion.file_lengths + self.bin_width_steps - 1,
                self.bin_width_steps,
                rounding_mode="floor",
            ),
            min=1,
        )
        bin_indices = torch.arange(self.bin_count, device=self.device)
        self.bin_valid_mask = bin_indices.unsqueeze(0) < self.motion_bin_counts.unsqueeze(1)
        self.valid_motion_ids, self.valid_bin_ids = torch.where(self.bin_valid_mask)
        self.num_valid_motion_bins = max(int(self.valid_motion_ids.numel()), 1)
        bin_starts = bin_indices.unsqueeze(0) * self.bin_width_steps
        remaining_lengths = (self.motion.file_lengths.unsqueeze(1) - bin_starts).clamp(min=0)
        self.bin_lengths = torch.minimum(
            remaining_lengths,
            torch.full_like(remaining_lengths, self.bin_width_steps),
        )
        self.bin_lengths.masked_fill_(~self.bin_valid_mask, 0)

        valid_bin_lengths = self.bin_lengths[self.bin_valid_mask].float()
        mean_bin_length = torch.clamp(valid_bin_lengths.mean(), min=1.0)
        self.bin_weights = self.bin_lengths.float() / mean_bin_length
        if self.cfg.adaptive_sequence_length_agnostic:
            self.bin_weights = self.bin_weights / self.motion_bin_counts.unsqueeze(1).float()
        self.bin_weights.masked_fill_(~self.bin_valid_mask, 0.0)

        (
            self.adaptive_prior_visit_count,
            self.adaptive_prior_failure_count,
        ) = _resolve_adaptive_prior_counts(self.cfg)
        self.bin_visit_count = torch.full(
            (self.motion.num_files, self.bin_count),
            self.adaptive_prior_visit_count,
            dtype=torch.float,
            device=self.device,
        )
        self.bin_failure_count = torch.full_like(
            self.bin_visit_count, self.adaptive_prior_failure_count
        )
        self.bin_visit_count.masked_fill_(~self.bin_valid_mask, 0.0)
        self.bin_failure_count.masked_fill_(~self.bin_valid_mask, 0.0)
        self._init_adaptive_sampling_ema()
        self._init_adaptive_visit_tracking()
        self._adaptive_sampling_phase = "idle"

        if self.cfg.if_log_metrics:
            self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
            self._init_adaptive_sampling_metrics()

        # Ghost model created lazily on first visualization
        self._ghost_model: mujoco.MjModel | None = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
        self._extra_reference_ghost_model: mujoco.MjModel | None = None
        self._extra_reference_ghost_color = np.array(_EXTRA_REFERENCE_GHOST_COLOR, dtype=np.float32)
        if (
            self.cfg.reference_storage_mode == REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK
            and self.cfg.extra_reference_motion_file
        ):
            raise ValueError("qpos_only_actor_fk does not support extra_reference_motion_file")
        self.extra_reference_motion = (
            MotionLoader(
                self.cfg.extra_reference_motion_file,
                self.body_indexes,
                motion_type=self.cfg.motion_type,
                device=self.device,
                fk_from_joint_pos=self.cfg.fk_from_joint_pos,
                recompute_joint_vel_from_joint_pos=self.cfg.recompute_joint_vel_from_joint_pos,
                fk_helper=fk_helper,
            )
            if self.cfg.extra_reference_motion_file
            else None
        )
        self._initialize_reference_cache()
        self._adaptive_bin_snapshot_writer = None
        self._adaptive_bin_snapshot_writer_key = None

    def _initialize_sp_tracking_state(self) -> None:
        """Allocate optional reference-frame and one-stage SP state.

        Both the in-memory and large-dataset commands call this helper, which keeps
        the SP preset decoupled from the data-loader implementation.
        """
        self.motion_origin_offset_w = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.boot_indicator = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )

        feet_names = tuple(self.cfg.feet_standing_body_names)
        if feet_names:
            missing = [name for name in feet_names if name not in self.cfg.body_names]
            if missing:
                raise ValueError(
                    "feet_standing_body_names must be included in command body_names; "
                    f"missing={missing}"
                )
            feet_ids = [self.cfg.body_names.index(name) for name in feet_names]
            self._feet_motion_ids = torch.as_tensor(feet_ids, dtype=torch.long, device=self.device)
        else:
            self._feet_motion_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.feet_standing = torch.zeros(
            (self.num_envs, int(self._feet_motion_ids.numel())),
            dtype=torch.bool,
            device=self.device,
        )

        self.reward_root_history_len = max(int(round(1.0 / float(self._env.step_dt))), 1)
        self.reward_root_ref_xy_history_w = torch.zeros(
            (self.num_envs, self.reward_root_history_len, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_root_actual_xy_history_w = torch.zeros_like(self.reward_root_ref_xy_history_w)
        self._reward_root_history_slot = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reward_root_pos_w = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.reward_root_quat_w = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.reward_root_quat_w[:, 0] = 1.0
        self._shared_joint_observation_cache: dict[tuple[str, float], torch.Tensor] = {}
        self._initialize_student_motion_randomization()

    def _initialize_student_motion_randomization(self) -> None:
        cfg = dict(self.cfg.student_motion_randomization)
        self._student_motion_randomization_cfg = cfg
        self._student_motion_randomization_enabled = bool(cfg.get("enable", False))
        if not self._student_motion_randomization_enabled:
            return
        shape = (self.num_envs,)
        self._student_root_z_offset = torch.zeros(shape, device=self.device)
        self._student_xy_direction = torch.zeros((self.num_envs, 2), device=self.device)
        self._student_xy_amplitude = torch.zeros(shape, device=self.device)
        self._student_xy_omega = torch.zeros(shape, device=self.device)
        self._student_xy_phase = torch.zeros(shape, device=self.device)
        self._student_z_amplitude = torch.zeros(shape, device=self.device)
        self._student_z_omega = torch.zeros(shape, device=self.device)
        self._student_z_phase = torch.zeros(shape, device=self.device)
        self._student_rot_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self._student_rot_amplitude = torch.zeros(shape, device=self.device)
        self._student_rot_omega = torch.zeros(shape, device=self.device)
        self._student_rot_phase = torch.zeros(shape, device=self.device)
        joint_names = tuple(getattr(self.motion, "joint_names", self.robot.joint_names))
        bias_std = torch.zeros(len(joint_names), device=self.device)
        for pattern, value in dict(cfg.get("joint_pos_bias_std", {})).items():
            for index, name in enumerate(joint_names):
                if re.fullmatch(str(pattern), name):
                    bias_std[index] = float(value)
        self._student_joint_bias_std = bias_std
        self._student_joint_bias = torch.zeros(
            (self.num_envs, len(joint_names)), device=self.device
        )
        self._resample_student_motion_randomization(torch.arange(self.num_envs, device=self.device))

    @staticmethod
    def _random_unit(shape: tuple[int, ...], device) -> torch.Tensor:
        value = torch.randn(shape, device=device)
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _sample_log_uniform(self, count: int, bounds) -> torch.Tensor:
        low, high = (float(bounds[0]), float(bounds[1]))
        return torch.empty(count, device=self.device).uniform_(math.log(low), math.log(high)).exp()

    def _resample_student_motion_randomization(self, env_ids: torch.Tensor) -> None:
        if (
            not getattr(self, "_student_motion_randomization_enabled", False)
            or env_ids.numel() == 0
        ):
            return
        cfg = self._student_motion_randomization_cfg
        pos = dict(cfg["root_pos_drift"])
        rot = dict(cfg["root_rot_drift"])
        count = env_ids.numel()
        self._student_root_z_offset[env_ids] = torch.empty(count, device=self.device).uniform_(
            *map(float, pos["root_z_offset_range_m"])
        )
        self._student_xy_direction[env_ids] = self._random_unit((count, 2), self.device)
        xy_frequency = self._sample_log_uniform(count, pos["xy_freq_range_hz"])
        xy_omega = 2.0 * math.pi * xy_frequency
        xy_speed = torch.empty(count, device=self.device).uniform_(
            *map(float, pos["xy_speed_range"])
        )
        self._student_xy_amplitude[env_ids] = xy_speed / xy_omega
        self._student_xy_omega[env_ids] = xy_omega
        self._student_xy_phase[env_ids] = torch.rand(count, device=self.device) * 2.0 * math.pi
        z_frequency = self._sample_log_uniform(count, pos["z_freq_range_hz"])
        self._student_z_omega[env_ids] = 2.0 * math.pi * z_frequency
        self._student_z_amplitude[env_ids] = torch.empty(count, device=self.device).uniform_(
            *map(float, pos["z_amplitude_range_m"])
        )
        self._student_z_phase[env_ids] = torch.rand(count, device=self.device) * 2.0 * math.pi
        self._student_rot_axis[env_ids] = self._random_unit((count, 3), self.device)
        rot_frequency = self._sample_log_uniform(count, rot["freq_range_hz"])
        self._student_rot_omega[env_ids] = 2.0 * math.pi * rot_frequency
        self._student_rot_amplitude[env_ids] = torch.empty(count, device=self.device).uniform_(
            *map(float, rot["amplitude_range_rad"])
        )
        self._student_rot_phase[env_ids] = torch.rand(count, device=self.device) * 2.0 * math.pi
        noise = (
            torch.rand((count, self._student_joint_bias.shape[1]), device=self.device) * 2.0 - 1.0
        )
        self._student_joint_bias[env_ids] = noise * self._student_joint_bias_std

    def _synchronize_student_motion_randomization(self, env_ids: torch.Tensor) -> None:
        """Share reference corruption inside a synchronized dynamics family."""

        group_size = int(getattr(self.cfg, "synchronized_group_size", 1))
        if (
            group_size <= 1
            or env_ids.numel() == 0
            or not getattr(self, "_student_motion_randomization_enabled", False)
        ):
            return
        selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        selected[env_ids] = True
        selected_by_group = selected.view(-1, group_size)
        group_ids = env_ids.div(group_size, rounding_mode="floor")
        full = selected_by_group.all(dim=1)
        first_unselected = (~selected_by_group).to(torch.int64).argmax(dim=1)
        source_by_group = torch.arange(
            selected_by_group.size(0), device=self.device
        ) * group_size + torch.where(full, torch.zeros_like(first_unselected), first_unselected)
        source_ids = source_by_group.index_select(0, group_ids)
        for name in (
            "_student_root_z_offset",
            "_student_xy_direction",
            "_student_xy_amplitude",
            "_student_xy_omega",
            "_student_xy_phase",
            "_student_z_amplitude",
            "_student_z_omega",
            "_student_z_phase",
            "_student_rot_axis",
            "_student_rot_amplitude",
            "_student_rot_omega",
            "_student_rot_phase",
            "_student_joint_bias",
        ):
            value = getattr(self, name)
            value[env_ids] = value[source_ids].clone()

    @staticmethod
    def _spherical_noise(value: torch.Tensor, radius: float) -> torch.Tensor:
        if radius <= 0.0:
            return value
        direction = torch.randn_like(value)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return value + direction * torch.rand_like(value[..., :1]) * radius

    @staticmethod
    def _quaternion_noise(value: torch.Tensor, radius: float) -> torch.Tensor:
        if radius <= 0.0:
            return value
        axis = torch.randn_like(value[..., 1:])
        axis /= axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        angle = (torch.rand_like(value[..., :1]) * 2.0 - 1.0) * radius
        delta = quat_from_angle_axis(angle.squeeze(-1), axis)
        return quat_mul(delta, value)

    def gather_student_reference(
        self, field_name: str, relative_steps: tuple[int, ...]
    ) -> torch.Tensor:
        value = self.gather_reference(field_name, relative_steps).clone()
        return self.apply_student_reference_randomization(field_name, relative_steps, value)

    def apply_student_reference_randomization(
        self,
        field_name: str,
        relative_steps: tuple[int, ...],
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply configured student corruption to an already-gathered window."""
        if not self._student_motion_randomization_enabled:
            return value
        cfg = self._student_motion_randomization_cfg
        steps = torch.as_tensor(relative_steps, device=self.device)
        time_s = (self.time_steps[:, None] + steps[None]).float() * float(self._env.step_dt)
        if field_name == "body_pos_w":
            xy = self._student_xy_amplitude[:, None] * torch.sin(
                self._student_xy_omega[:, None] * time_s + self._student_xy_phase[:, None]
            )
            z = (
                self._student_z_amplitude[:, None]
                * torch.sin(
                    self._student_z_omega[:, None] * time_s + self._student_z_phase[:, None]
                )
                + self._student_root_z_offset[:, None]
            )
            offset = value.new_zeros((self.num_envs, len(relative_steps), 3))
            offset[..., :2] = xy[..., None] * self._student_xy_direction[:, None]
            offset[..., 2] = z
            root = value[:, :, self.motion_anchor_body_index]
            value[:, :, self.motion_anchor_body_index] = self._spherical_noise(
                root + offset, float(cfg.get("root_pos_noise_std", 0.0))
            )
        elif field_name == "body_quat_w":
            angle = self._student_rot_amplitude[:, None] * torch.sin(
                self._student_rot_omega[:, None] * time_s + self._student_rot_phase[:, None]
            )
            axis = self._student_rot_axis[:, None].expand(-1, len(relative_steps), -1)
            delta = quat_from_angle_axis(angle, axis)
            root = quat_mul(delta, value[:, :, self.motion_anchor_body_index])
            value[:, :, self.motion_anchor_body_index] = self._quaternion_noise(
                root, float(cfg.get("root_ori_noise_std", 0.0))
            )
        elif field_name == "joint_pos":
            value += self._student_joint_bias[:, None].to(value.dtype)
            std = float(cfg.get("joint_pos_noise_std", 0.0))
            if std > 0.0:
                value += (torch.rand_like(value) * 2.0 - 1.0) * std
        return value

    def _uses_qpos_only_actor_fk(self) -> bool:
        return (
            getattr(self.cfg, "reference_storage_mode", REFERENCE_STORAGE_FULL)
            == REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK
        )

    def _gather_qpos_reference(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        relative_steps = tuple(int(step) for step in relative_steps)
        if env_ids is None and relative_steps == (0,):
            support_cache = getattr(self, "_qpos_actor_support_cache", None)
            if isinstance(support_cache, dict):
                support = support_cache.get(((0,), _QPOS_ACTOR_SUPPORT_STEPS))
                if support is not None:
                    # Reuse the exact current frame from the Actor support window.
                    return support[:, :, _QPOS_ACTOR_CURRENT_INDEX]
        return self._gather_qpos_reference_slice(relative_steps, slice(None), env_ids)

    def _gather_qpos_reference_slice(
        self,
        relative_steps: tuple[int, ...],
        field_slice: slice,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather only the requested resident qpos columns."""
        qpos = getattr(self.motion, "qpos", None)
        if not isinstance(qpos, torch.Tensor):
            raise RuntimeError("qpos-only reference storage has no resident qpos tensor")
        if env_ids is None:
            motion_ids = self.motion_idx
            base_steps = self.time_steps
        else:
            motion_ids = self.motion_idx[env_ids]
            base_steps = self.time_steps[env_ids]
        offsets = torch.as_tensor(relative_steps, device=self.device, dtype=torch.long)
        absolute_steps = base_steps.unsqueeze(1) + offsets.unsqueeze(0)
        frame_indices = self._get_frame_indices(motion_ids, absolute_steps)
        return qpos[:, field_slice][frame_indices]

    def _qpos_actor_target_context(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return motion ids, effective target frames, and expanded lengths."""
        if env_ids is None:
            motion_ids = self.motion_idx
            base_steps = self.time_steps
        else:
            motion_ids = self.motion_idx[env_ids]
            base_steps = self.time_steps[env_ids]
        offsets = torch.as_tensor(relative_steps, device=self.device, dtype=torch.long)
        target_steps = self._clamp_motion_time_steps(
            motion_ids, base_steps.unsqueeze(1) + offsets.unsqueeze(0)
        )
        lengths = self.motion.file_lengths[motion_ids].unsqueeze(1).expand_as(target_steps)
        return motion_ids, target_steps, lengths

    def _gather_qpos_actor_support(
        self,
        relative_steps: tuple[int, ...],
        support_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather and cache the Actor's symmetric 11-frame velocity support."""
        relative_steps = tuple(int(step) for step in relative_steps)
        support_steps = tuple(int(step) for step in support_steps)
        cache = getattr(self, "_qpos_actor_support_cache", None)
        cacheable = (
            env_ids is None
            and relative_steps == (0,)
            and support_steps == _QPOS_ACTOR_SUPPORT_STEPS
            and isinstance(cache, dict)
        )
        cache_key = (relative_steps, support_steps)
        if cacheable:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        motion_ids, target_steps, _ = self._qpos_actor_target_context(relative_steps, env_ids)
        actor_support = torch.as_tensor(support_steps, device=self.device, dtype=torch.long)
        absolute_steps = target_steps.unsqueeze(-1) + actor_support
        qpos = self.motion.qpos[self._get_frame_indices(motion_ids, absolute_steps)]
        if cacheable:
            cache[cache_key] = qpos
        return qpos

    def _gather_qpos_actor_component_support(
        self,
        relative_steps: tuple[int, ...],
        field_slice: slice,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather one qpos component over the exact 11-frame Actor support."""
        relative_steps = tuple(int(step) for step in relative_steps)
        support_steps = _QPOS_ACTOR_SUPPORT_STEPS
        if env_ids is None and relative_steps == (0,):
            # The current body state is required by both A variants. Sharing its full
            # support preserves the original strided arithmetic inputs and avoids a
            # second gather when metrics request root/joint velocity.
            return self._gather_qpos_actor_support(relative_steps, support_steps)[..., field_slice]

        qpos = getattr(self.motion, "qpos", None)
        if not isinstance(qpos, torch.Tensor):
            raise RuntimeError("qpos-only reference storage has no resident qpos tensor")
        motion_ids, target_steps, _ = self._qpos_actor_target_context(relative_steps, env_ids)
        actor_support = torch.as_tensor(support_steps, device=self.device, dtype=torch.long)
        absolute_steps = target_steps.unsqueeze(-1) + actor_support
        frame_indices = self._get_frame_indices(motion_ids, absolute_steps)
        # Slice before advanced indexing: root/joint-only callers gather the
        # eleven support values without materializing all 36 qpos columns.
        return qpos[:, field_slice][frame_indices]

    def _root_position_with_origin_offset(
        self,
        root_pos_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(self.cfg, "motion_origin_recenter", False):
            return root_pos_w
        offsets_w = (
            self.motion_origin_offset_w if env_ids is None else self.motion_origin_offset_w[env_ids]
        )
        return root_pos_w + offsets_w[:, None].to(root_pos_w)

    def _cache_qpos_actor_root_pose(
        self,
        relative_steps: tuple[int, ...],
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> ActorRootReferencePoseState:
        state = ActorRootReferencePoseState(
            root_pos_w=self._root_position_with_origin_offset(root_pos_w, env_ids),
            root_quat_w=root_quat_w,
        )
        cache = getattr(self, "_qpos_actor_root_pose_cache", None)
        if env_ids is None and isinstance(cache, dict):
            cache[relative_steps] = state
        return state

    def _qpos_actor_root_pose(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> ActorRootReferencePoseState:
        """Gather root pose without materializing joint qpos columns."""
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_root_pose_cache", None)
        if env_ids is None and isinstance(cache, dict):
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached
        root_qpos = self._gather_qpos_reference_slice(relative_steps, slice(0, 7), env_ids)
        return self._cache_qpos_actor_root_pose(
            relative_steps,
            root_qpos[..., :3],
            root_qpos[..., 3:7],
            env_ids,
        )

    def _qpos_actor_root_kinematics_support(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
        *,
        qpos_support: torch.Tensor | None = None,
    ) -> _ActorRootKinematicsSupport:
        """Compute exact current/root body-support kinematics from resident qpos."""
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_root_kinematics_support_cache", None)
        cacheable = env_ids is None and relative_steps == (0,) and isinstance(cache, dict)
        if cacheable:
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached
        if qpos_support is None:
            qpos_support = self._gather_qpos_actor_component_support(
                relative_steps, slice(0, 7), env_ids
            )
        root_pos_support = qpos_support[..., :3]
        # qpos-only storage takes this value from the archive's pelvis
        # ``body_quat_w`` field.  It is already the offline pipeline's normalized
        # output and must remain byte-identical for pose and root-angular-velocity
        # reconstruction.
        root_quat_support = qpos_support[..., 3:7]
        _, target_steps, motion_lengths = self._qpos_actor_target_context(relative_steps, env_ids)
        fps = float(getattr(self.cfg, "actor_reference_fps", 50.0))
        root_lin_vel_w = actor_smoothed_finite_difference_from_support_torch(
            root_pos_support,
            target_steps,
            motion_lengths,
            fps,
            (0,),
        )
        root_ang_vel_w_body_support = actor_smoothed_finite_difference_from_support_torch(
            root_quat_support,
            target_steps,
            motion_lengths,
            fps,
            _QPOS_BODY_SMOOTH_STEPS,
            quaternion=True,
            quaternion_is_pre_normalized=True,
        )
        root_quat_w_body_support = actor_gather_from_support_torch(
            root_quat_support,
            target_steps,
            motion_lengths,
            _QPOS_BODY_SMOOTH_STEPS,
        )
        root_ang_vel_b_body_support = fk_quat_apply_inverse(
            root_quat_w_body_support, root_ang_vel_w_body_support
        )
        current = _QPOS_ACTOR_CURRENT_INDEX
        support = _ActorRootKinematicsSupport(
            root_pos_w=root_pos_support[:, :, current],
            root_quat_w=root_quat_support[:, :, current],
            root_lin_vel_w=root_lin_vel_w[:, :, 0],
            root_ang_vel_w=root_ang_vel_w_body_support[:, :, 2],
            root_ang_vel_b_body_support=root_ang_vel_b_body_support,
        )
        if cacheable:
            cache[relative_steps] = support
        self._cache_qpos_actor_root_pose(
            relative_steps,
            support.root_pos_w,
            support.root_quat_w,
            env_ids,
        )
        return support

    def _qpos_actor_root_velocity_state(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> ActorRootReferenceVelocityState:
        """Return root pose/velocity without deriving joint or body-frame fields."""
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_root_velocity_state_cache", None)
        if env_ids is None and isinstance(cache, dict):
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached
        support = self._qpos_actor_root_kinematics_support(relative_steps, env_ids)
        state = ActorRootReferenceVelocityState(
            root_pos_w=self._root_position_with_origin_offset(support.root_pos_w, env_ids),
            root_quat_w=support.root_quat_w,
            root_lin_vel_w=support.root_lin_vel_w,
            root_ang_vel_w=support.root_ang_vel_w,
        )
        if env_ids is None and isinstance(cache, dict):
            cache[relative_steps] = state
        return state

    def _qpos_actor_joint_state(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
        *,
        qpos_support: torch.Tensor | None = None,
    ) -> ActorJointReferenceState:
        """Return joint pose/velocity without deriving any root velocity fields."""
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_joint_state_cache", None)
        if env_ids is None and isinstance(cache, dict):
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached
        if qpos_support is None:
            joint_pos_support = self._gather_qpos_actor_component_support(
                relative_steps, slice(7, None), env_ids
            )
        else:
            joint_pos_support = qpos_support[..., 7:]
        _, target_steps, motion_lengths = self._qpos_actor_target_context(relative_steps, env_ids)
        fps = float(getattr(self.cfg, "actor_reference_fps", 50.0))
        joint_vel = actor_smoothed_finite_difference_from_support_torch(
            joint_pos_support,
            target_steps,
            motion_lengths,
            fps,
            (0,),
        )
        current = _QPOS_ACTOR_CURRENT_INDEX
        state = ActorJointReferenceState(
            joint_pos=joint_pos_support[:, :, current],
            joint_vel=joint_vel[:, :, 0],
        )
        if env_ids is None and isinstance(cache, dict):
            cache[relative_steps] = state
        return state

    def _qpos_actor_root_state(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> ActorRootReferenceState:
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_root_state_cache", None)
        if env_ids is None and isinstance(cache, dict):
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached

        qpos = self._gather_qpos_actor_support(relative_steps, _QPOS_ACTOR_SUPPORT_STEPS, env_ids)
        root = self._qpos_actor_root_kinematics_support(relative_steps, env_ids, qpos_support=qpos)
        joint = self._qpos_actor_joint_state(relative_steps, env_ids, qpos_support=qpos)
        root_lin_vel_b = fk_quat_apply_inverse(root.root_quat_w, root.root_lin_vel_w)
        root_ang_vel_b = fk_quat_apply_inverse(root.root_quat_w, root.root_ang_vel_w)
        state = ActorRootReferenceState(
            root_pos_w=self._root_position_with_origin_offset(root.root_pos_w, env_ids),
            root_quat_w=root.root_quat_w,
            root_lin_vel_w=root.root_lin_vel_w,
            root_ang_vel_w=root.root_ang_vel_w,
            root_lin_vel_b=root_lin_vel_b,
            root_ang_vel_b=root_ang_vel_b,
            joint_pos=joint.joint_pos,
            joint_vel=joint.joint_vel,
        )
        if env_ids is None and isinstance(cache, dict):
            cache[relative_steps] = state
            velocity_cache = getattr(self, "_qpos_actor_root_velocity_state_cache", None)
            if isinstance(velocity_cache, dict):
                velocity_cache[relative_steps] = ActorRootReferenceVelocityState(
                    root_pos_w=state.root_pos_w,
                    root_quat_w=state.root_quat_w,
                    root_lin_vel_w=state.root_lin_vel_w,
                    root_ang_vel_w=state.root_ang_vel_w,
                )
            joint_cache = getattr(self, "_qpos_actor_joint_state_cache", None)
            if isinstance(joint_cache, dict):
                joint_cache[relative_steps] = ActorJointReferenceState(
                    joint_pos=state.joint_pos,
                    joint_vel=state.joint_vel,
                )
            self._cache_qpos_actor_root_pose(
                relative_steps,
                root.root_pos_w,
                state.root_quat_w,
            )
        return state

    def gather_reference_body_pose_b(
        self, relative_steps: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Actor-MJCF FK poses directly in the reference-root frame."""
        if not self._uses_qpos_only_actor_fk():
            raise RuntimeError("Root-frame FK is available only in qpos-only mode")
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_body_pose_cache", None)
        if isinstance(cache, dict):
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached
        state_cache = getattr(self, "_qpos_actor_body_state_cache", None)
        if isinstance(state_cache, dict):
            state = state_cache.get(relative_steps)
            if state is not None:
                return state.pos_b, state.quat_b
        tiled_pose = self._gather_qpos_body_pose_tile_cache(relative_steps)
        if tiled_pose is not None:
            if isinstance(cache, dict):
                cache[relative_steps] = tiled_pose
            return tiled_pose
        helper = self._reference_fk_helper
        if helper is None:
            raise RuntimeError("qpos-only reference FK helper is not initialized")
        root_pose_cache = getattr(self, "_qpos_actor_root_pose_cache", None)
        root_pose_cached = isinstance(root_pose_cache, dict) and relative_steps in root_pose_cache
        if root_pose_cached:
            joint_pos = self._gather_qpos_reference_slice(relative_steps, slice(7, None))
        else:
            qpos = self._gather_qpos_reference(relative_steps)
            joint_pos = qpos[..., 7:]
            self._cache_qpos_actor_root_pose(
                relative_steps,
                qpos[..., :3],
                qpos[..., 3:7],
            )
        pose = helper.body_pose(joint_pos)
        if isinstance(cache, dict):
            cache[relative_steps] = pose
        return pose

    def gather_reference_body_state_b(
        self, relative_steps: tuple[int, ...]
    ) -> ActorBodyReferenceState:
        """Return Actor-identical root-frame body state at requested frames."""
        if not self._uses_qpos_only_actor_fk():
            raise RuntimeError("Root-frame FK is available only in qpos-only mode")
        relative_steps = tuple(int(step) for step in relative_steps)
        cache = getattr(self, "_qpos_actor_body_state_cache", None)
        if isinstance(cache, dict):
            cached = cache.get(relative_steps)
            if cached is not None:
                return cached
        helper = self._reference_fk_helper
        if helper is None:
            raise RuntimeError("qpos-only reference FK helper is not initialized")
        qpos = self._gather_qpos_actor_support(relative_steps, _QPOS_ACTOR_SUPPORT_STEPS)
        root_support = self._qpos_actor_root_kinematics_support(relative_steps, qpos_support=qpos)
        fps = float(getattr(self.cfg, "actor_reference_fps", 50.0))
        tiled_pose = self._gather_qpos_body_pose_support_tile_cache(
            relative_steps, _QPOS_ACTOR_SUPPORT_STEPS
        )
        if tiled_pose is None:
            pos_b, quat_b = helper.body_pose(qpos[..., 7:])
        else:
            pos_b, quat_b = tiled_pose
        _, target_steps, motion_lengths = self._qpos_actor_target_context(relative_steps)
        lin_vel_b, ang_vel_b = actor_body_velocity_from_compact_support_torch(
            pos_b,
            quat_b,
            root_support.root_ang_vel_b_body_support,
            target_steps,
            motion_lengths,
            fps,
        )
        current = _QPOS_ACTOR_CURRENT_INDEX
        state = ActorBodyReferenceState(
            pos_b=pos_b[:, :, current],
            quat_b=quat_b[:, :, current],
            lin_vel_b=lin_vel_b,
            ang_vel_b=ang_vel_b,
        )
        if isinstance(cache, dict):
            cache[relative_steps] = state
        pose_cache = getattr(self, "_qpos_actor_body_pose_cache", None)
        if isinstance(pose_cache, dict):
            pose_cache[relative_steps] = (state.pos_b, state.quat_b)
        return state

    def gather_reference_body_pos_w(
        self,
        body_ids: torch.Tensor | list[int] | tuple[int, ...],
        relative_steps: tuple[int, ...] = (0,),
        *,
        include_env_origins: bool = True,
    ) -> torch.Tensor:
        """Project only selected FK bodies to world coordinates."""
        ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        if not self._uses_qpos_only_actor_fk():
            value = self.gather_reference("body_pos_w", relative_steps).index_select(2, ids)
        else:
            body_pos_b, _ = self.gather_reference_body_pose_b(relative_steps)
            root = self._qpos_actor_root_pose(relative_steps)
            value = root.root_pos_w.unsqueeze(2) + fk_quat_apply(
                root.root_quat_w.unsqueeze(2), body_pos_b.index_select(2, ids)
            )
        if include_env_origins:
            value = value + self._env.scene.env_origins[:, None, None, :].to(value)
        return value

    def gather_reference_body_quat_w(
        self,
        body_ids: torch.Tensor | list[int] | tuple[int, ...],
        relative_steps: tuple[int, ...] = (0,),
    ) -> torch.Tensor:
        ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        if not self._uses_qpos_only_actor_fk():
            return self.gather_reference("body_quat_w", relative_steps).index_select(2, ids)
        _, body_quat_b = self.gather_reference_body_pose_b(relative_steps)
        root_quat_w = self._qpos_actor_root_pose(relative_steps).root_quat_w
        return normalize(fk_quat_mul(root_quat_w.unsqueeze(2), body_quat_b.index_select(2, ids)))

    def gather_reference_body_lin_vel_w(
        self,
        body_ids: torch.Tensor | list[int] | tuple[int, ...],
        relative_steps: tuple[int, ...] = (0,),
    ) -> torch.Tensor:
        ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        if not self._uses_qpos_only_actor_fk():
            return self.gather_reference("body_lin_vel_w", relative_steps).index_select(2, ids)
        body = self.gather_reference_body_state_b(relative_steps)
        root = self._qpos_actor_root_velocity_state(relative_steps)
        return root.root_lin_vel_w.unsqueeze(2) + fk_quat_apply(
            root.root_quat_w.unsqueeze(2), body.lin_vel_b.index_select(2, ids)
        )

    def gather_reference_body_ang_vel_w(
        self,
        body_ids: torch.Tensor | list[int] | tuple[int, ...],
        relative_steps: tuple[int, ...] = (0,),
    ) -> torch.Tensor:
        ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        if not self._uses_qpos_only_actor_fk():
            return self.gather_reference("body_ang_vel_w", relative_steps).index_select(2, ids)
        body = self.gather_reference_body_state_b(relative_steps)
        root = self._qpos_actor_root_velocity_state(relative_steps)
        return root.root_ang_vel_w.unsqueeze(2) + fk_quat_apply(
            root.root_quat_w.unsqueeze(2), body.ang_vel_b.index_select(2, ids)
        )

    def gather_root_reference(
        self, field_name: str, relative_steps: tuple[int, ...]
    ) -> torch.Tensor:
        """Gather only the motion root instead of materializing every body.

        SPV5 consumes a 50-frame root window.  Gathering the complete body cache
        for that window would allocate substantially more memory even though only
        the anchor body is used.
        """
        if field_name not in (
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            raise ValueError(f"Unsupported root reference field: {field_name}")
        if self._uses_qpos_only_actor_fk():
            if field_name in ("body_pos_w", "body_quat_w"):
                pose = self._qpos_actor_root_pose(relative_steps)
                return pose.root_pos_w if field_name == "body_pos_w" else pose.root_quat_w
            state = self._qpos_actor_root_velocity_state(relative_steps)
            return state.root_lin_vel_w if field_name == "body_lin_vel_w" else state.root_ang_vel_w
        steps = torch.as_tensor(
            tuple(int(step) for step in relative_steps),
            device=self.device,
            dtype=torch.long,
        )
        absolute_steps = self.time_steps.unsqueeze(1) + steps.unsqueeze(0)
        frame_indices = self._get_frame_indices(self.motion_idx, absolute_steps)
        value = getattr(self.motion, field_name)[frame_indices, self.motion_anchor_body_index]
        if field_name == "body_pos_w" and self.cfg.motion_origin_recenter:
            value = value + self.motion_origin_offset_w[:, None].to(value)
        return value

    def gather_student_root_reference(
        self, field_name: str, relative_steps: tuple[int, ...]
    ) -> torch.Tensor:
        """Gather the noisy student root window without a full-body tensor."""
        value = self.gather_root_reference(field_name, relative_steps).clone()
        return self.apply_student_root_reference_randomization(field_name, relative_steps, value)

    def gather_compact_qpos_reference(self, relative_steps: tuple[int, ...]) -> torch.Tensor:
        """Gather the exact compact-qpos window used by SPV8-1."""
        qpos = getattr(self.motion, "qpos", None)
        if not isinstance(qpos, torch.Tensor):
            raise RuntimeError("Compact qpos is not resident; enable command.load_compact_qpos")
        steps = torch.as_tensor(
            tuple(int(step) for step in relative_steps),
            device=self.device,
            dtype=torch.long,
        )
        absolute_steps = self.time_steps.unsqueeze(1) + steps.unsqueeze(0)
        frame_indices = self._get_frame_indices(self.motion_idx, absolute_steps)
        return qpos[frame_indices]

    def apply_student_root_reference_randomization(
        self,
        field_name: str,
        relative_steps: tuple[int, ...],
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply student root corruption to an already-gathered root window."""
        if not self._student_motion_randomization_enabled:
            return value
        cfg = self._student_motion_randomization_cfg
        steps = torch.as_tensor(relative_steps, device=self.device)
        time_s = (self.time_steps[:, None] + steps[None]).float() * float(self._env.step_dt)
        if field_name == "body_pos_w":
            xy = self._student_xy_amplitude[:, None] * torch.sin(
                self._student_xy_omega[:, None] * time_s + self._student_xy_phase[:, None]
            )
            z = (
                self._student_z_amplitude[:, None]
                * torch.sin(
                    self._student_z_omega[:, None] * time_s + self._student_z_phase[:, None]
                )
                + self._student_root_z_offset[:, None]
            )
            offset = value.new_zeros((self.num_envs, len(relative_steps), 3))
            offset[..., :2] = xy[..., None] * self._student_xy_direction[:, None]
            offset[..., 2] = z
            return self._spherical_noise(value + offset, float(cfg.get("root_pos_noise_std", 0.0)))
        if field_name == "body_quat_w":
            angle = self._student_rot_amplitude[:, None] * torch.sin(
                self._student_rot_omega[:, None] * time_s + self._student_rot_phase[:, None]
            )
            axis = self._student_rot_axis[:, None].expand(-1, len(relative_steps), -1)
            value = quat_mul(quat_from_angle_axis(angle, axis), value)
            return self._quaternion_noise(value, float(cfg.get("root_ori_noise_std", 0.0)))
        if field_name in ("body_lin_vel_w", "body_ang_vel_w"):
            return value
        raise ValueError(f"Unsupported student root reference field: {field_name}")

    def _clear_shared_joint_observation_cache(self) -> None:
        cache = getattr(self, "_shared_joint_observation_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        spv1_cache = getattr(self, "_shared_spv1_observation_cache", None)
        if isinstance(spv1_cache, dict):
            spv1_cache.clear()
        spv4_cache = getattr(self, "_shared_spv4_key_body_cache", None)
        if isinstance(spv4_cache, dict):
            spv4_cache.clear()
        spv5_cache = getattr(self, "_shared_spv5_reference_cache", None)
        if isinstance(spv5_cache, dict):
            spv5_cache.clear()

    def _shared_noisy_joint_observation(
        self, field_name: Literal["joint_pos", "joint_vel"], noise_std: float
    ) -> torch.Tensor:
        source = getattr(self.robot.data, field_name)
        std = max(float(noise_std), 0.0)
        if std <= 0.0:
            return source
        key = (field_name, std)
        cached = self._shared_joint_observation_cache.get(key)
        if cached is None:
            cached = source + (torch.rand_like(source) * 2.0 - 1.0) * std
            self._shared_joint_observation_cache[key] = cached
        return cached

    def get_shared_noisy_joint_pos(self, noise_std: float) -> torch.Tensor:
        return self._shared_noisy_joint_observation("joint_pos", noise_std)

    def get_shared_noisy_joint_vel(self, noise_std: float) -> torch.Tensor:
        return self._shared_noisy_joint_observation("joint_vel", noise_std)

    def _set_motion_origin_offset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.motion_origin_offset_w[env_ids] = 0.0
        if not self.cfg.motion_origin_recenter:
            return
        if self._uses_qpos_only_actor_fk():
            frame_indices = self._get_frame_indices(
                self.motion_idx[env_ids], self.time_steps[env_ids]
            )
            root_xy = self.motion.qpos[frame_indices, :2]
        else:
            raw_body_pos = self._gather_motion_field(
                "body_pos_w", self.motion_idx[env_ids], self.time_steps[env_ids]
            )
            root_xy = raw_body_pos[:, self.motion_anchor_body_index, :2]
        self.motion_origin_offset_w[env_ids, :2] = -root_xy

    def _apply_motion_origin_offset(self, field_name: str, reference: torch.Tensor) -> torch.Tensor:
        if field_name != "body_pos_w" or not getattr(self.cfg, "motion_origin_recenter", False):
            return reference
        return reference + self.motion_origin_offset_w[:, None, None, :].to(dtype=reference.dtype)

    def _reference_root_pos_w(self) -> torch.Tensor:
        root_pos = self.gather_root_reference("body_pos_w", (0,))[:, 0]
        return root_pos + self._env.scene.env_origins

    def _reference_root_quat_w(self) -> torch.Tensor:
        return self.gather_root_reference("body_quat_w", (0,))[:, 0]

    def _reset_sp_tracking_state(
        self, env_ids: torch.Tensor, actual_root_pos_w: torch.Tensor
    ) -> None:
        if env_ids.numel() == 0:
            return
        self._resample_student_motion_randomization(env_ids)
        self._synchronize_student_motion_randomization(env_ids)
        self.boot_indicator[env_ids] = float(max(int(self.cfg.boot_indicator_max), 0))
        self.feet_standing[env_ids] = False
        # These buffers implement source-style consecutive-frame termination.
        # They live on the command because the functional termination terms share
        # it; clear them on every episode reset just as the source mixin does.
        for name in (
            "_body_z_termination_buffer",
            "_gravity_dir_termination_buffer",
            "_global_key_body_pos_termination_buffer",
        ):
            buffer = getattr(self, name, None)
            if isinstance(buffer, torch.Tensor):
                buffer[env_ids] = 0
        reference_root_pos_w = self._reference_root_pos_w()[env_ids]
        reference_root_quat_w = self._reference_root_quat_w()[env_ids]
        self.reward_root_ref_xy_history_w[env_ids] = reference_root_pos_w[:, :2].unsqueeze(1)
        self.reward_root_actual_xy_history_w[env_ids] = actual_root_pos_w[:, :2].unsqueeze(1)
        self._reward_root_history_slot[env_ids] = 0
        self.reward_root_pos_w[env_ids] = reference_root_pos_w
        self.reward_root_quat_w[env_ids] = reference_root_quat_w
        if self.cfg.sliding_root_xy_reward:
            self.reward_root_pos_w[env_ids, :2] = actual_root_pos_w[:, :2]

    def _update_feet_standing(self) -> None:
        feet_motion_ids = getattr(self, "_feet_motion_ids", None)
        if not isinstance(feet_motion_ids, torch.Tensor):
            return
        feet_cfg = getattr(self.cfg, "feet_standing", {})
        if feet_motion_ids.numel() == 0 or not feet_cfg:
            return
        cfg = feet_cfg
        required = ("z_enter", "z_exit", "vxy_enter", "vxy_exit", "vz_enter", "vz_exit")
        missing = [name for name in required if name not in cfg]
        if missing:
            raise ValueError(f"feet_standing is missing required values: {missing}")
        feet_pos = self.gather_reference_body_pos_w(
            feet_motion_ids, (0,), include_env_origins=False
        )[:, 0]
        feet_vel = self.gather_reference_body_lin_vel_w(feet_motion_ids, (0,))[:, 0]
        root_vel = self.gather_root_reference("body_lin_vel_w", (0,))[:, 0]
        root_vxy = root_vel[:, :2].norm(dim=-1, keepdim=True).clamp_min(1.0)
        feet_vxy = feet_vel[..., :2].norm(dim=-1)
        feet_vz = feet_vel[..., 2].abs()
        feet_z = feet_pos[..., 2]
        enter_contact = (
            (feet_z < float(cfg["z_enter"]))
            & (feet_vxy < float(cfg["vxy_enter"]) * root_vxy)
            & (feet_vz < float(cfg["vz_enter"]) * root_vxy)
        )
        exit_contact = (
            (feet_z > float(cfg["z_exit"]))
            | (feet_vxy > float(cfg["vxy_exit"]) * root_vxy)
            | (feet_vz > float(cfg["vz_exit"]) * root_vxy)
        )
        self.feet_standing[:] = (self.feet_standing & (~exit_contact)) | enter_contact

    def _update_reward_root_target(self) -> None:
        if not hasattr(self, "reward_root_pos_w"):
            return
        reference_root_pos_w = self._reference_root_pos_w()
        self.reward_root_pos_w[:] = reference_root_pos_w
        self.reward_root_quat_w[:] = self._reference_root_quat_w()
        if not getattr(self.cfg, "sliding_root_xy_reward", False):
            return
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        history_slot = self._reward_root_history_slot
        history_ref_xy_w = self.reward_root_ref_xy_history_w[env_ids, history_slot]
        history_actual_xy_w = self.reward_root_actual_xy_history_w[env_ids, history_slot]
        expected_xy_w = history_actual_xy_w + (reference_root_pos_w[:, :2] - history_ref_xy_w)
        self.reward_root_pos_w[:, :2] = expected_xy_w
        self.reward_root_ref_xy_history_w[env_ids, history_slot] = reference_root_pos_w[:, :2]
        self.reward_root_actual_xy_history_w[env_ids, history_slot] = (
            self.robot.data.root_link_pos_w[:, :2]
        )
        self._reward_root_history_slot.add_(1)
        self._reward_root_history_slot.remainder_(self.reward_root_history_len)

    def _configured_reference_steps(self) -> tuple[int, ...]:
        offsets = []
        if self.cfg.history_steps > 0:
            offsets.extend(range(-self.cfg.history_steps, 0))
        offsets.append(0)
        if self.cfg.future_steps > 1:
            offsets.extend(range(1, self.cfg.future_steps))
        return tuple(offsets)

    def _reference_cache_step_groups(self) -> dict[str, tuple[int, ...]]:
        configured = self.cfg.reference_cache_steps
        if configured is None:
            steps = self._configured_reference_steps()
            return {field_name: steps for field_name in REFERENCE_MOTION_FIELDS}
        fallback = self._configured_reference_steps()
        return {
            field_name: tuple(int(step) for step in configured.get(field_name, fallback))
            for field_name in REFERENCE_MOTION_FIELDS
        }

    def _initialize_reference_cache(self) -> None:
        self._reference_cache_field_steps = self._reference_cache_step_groups()
        self._reference_cache: dict[str, torch.Tensor] = {}
        self._reference_cache_slices: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self._reference_cache_valid = False
        self._reference_cache_build_count = 0
        self._qpos_actor_root_state_cache: dict[tuple[int, ...], ActorRootReferenceState] = {}
        self._qpos_actor_root_pose_cache: dict[tuple[int, ...], ActorRootReferencePoseState] = {}
        self._qpos_actor_root_velocity_state_cache: dict[
            tuple[int, ...], ActorRootReferenceVelocityState
        ] = {}
        self._qpos_actor_joint_state_cache: dict[tuple[int, ...], ActorJointReferenceState] = {}
        self._qpos_actor_root_kinematics_support_cache: dict[
            tuple[int, ...], _ActorRootKinematicsSupport
        ] = {}
        self._qpos_actor_body_pose_cache: dict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._qpos_actor_body_state_cache: dict[tuple[int, ...], ActorBodyReferenceState] = {}
        self._qpos_actor_support_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor
        ] = {}
        self._initialize_qpos_body_pose_tile_cache()

    def _qpos_body_pose_tile_cache_enabled(self) -> bool:
        return (
            self._uses_qpos_only_actor_fk()
            and int(getattr(self.cfg, "qpos_body_pose_cache_tile_steps", 0)) > 0
        )

    def _initialize_qpos_body_pose_tile_cache(self) -> None:
        """Allocate the small per-environment pose tile used by qpos-only mode.

        All environments share a host-side phase clock. A reset fills its tile from
        an epoch-aligned start, so every valid tile advances by exactly ``K`` frames
        at the next phase boundary even when resets occur asynchronously.
        """
        self._qpos_body_pose_cache_tick = 0
        self._qpos_body_pose_cache_initialized = False
        if not self._qpos_body_pose_tile_cache_enabled():
            self._qpos_body_pose_cache_pos_b = None
            self._qpos_body_pose_cache_quat_b = None
            return

        tile_steps = int(self.cfg.qpos_body_pose_cache_tile_steps)
        raw_range = tuple(int(step) for step in self.cfg.qpos_body_pose_cache_range)
        if len(raw_range) != 2:
            raise ValueError("qpos_body_pose_cache_range must contain [minimum, maximum]")
        minimum, maximum = raw_range
        if minimum > -5 or maximum < 5:
            raise ValueError(
                "qpos_body_pose_cache_range must cover the Actor velocity support [-5, 5]"
            )
        window_len = tile_steps + maximum - minimum
        if window_len <= 0:
            raise ValueError("qpos body-pose cache window must be positive")

        self._qpos_body_pose_cache_tile_steps = tile_steps
        self._qpos_body_pose_cache_min_offset = minimum
        self._qpos_body_pose_cache_max_offset = maximum
        self._qpos_body_pose_cache_window_len = window_len
        self._qpos_body_pose_cache_all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._qpos_body_pose_cache_fill_offsets = torch.arange(
            window_len, dtype=torch.long, device=self.device
        )
        self._qpos_body_pose_cache_append_offsets = torch.arange(
            tile_steps, dtype=torch.long, device=self.device
        )
        self._qpos_body_pose_cache_logical_offsets: dict[tuple[int, ...], torch.Tensor] = {}
        body_count = len(self.cfg.body_names)
        self._qpos_body_pose_cache_pos_b = torch.empty(
            (self.num_envs, window_len, body_count, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._qpos_body_pose_cache_quat_b = torch.empty(
            (self.num_envs, window_len, body_count, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self._qpos_body_pose_cache_head = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._qpos_body_pose_cache_start_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._qpos_body_pose_cache_motion_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._qpos_body_pose_cache_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _refill_qpos_body_pose_tile_cache(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill complete epoch-aligned pose tiles for reset or cold environments."""
        if not self._qpos_body_pose_tile_cache_enabled():
            raise RuntimeError("qpos body-pose tile cache is disabled")
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        helper = self._reference_fk_helper
        if helper is None:
            raise RuntimeError("qpos-only reference FK helper is not initialized")
        phase = self._qpos_body_pose_cache_tick % self._qpos_body_pose_cache_tile_steps
        starts = self.time_steps[env_ids] + self._qpos_body_pose_cache_min_offset - phase
        local_steps = starts.unsqueeze(1) + self._qpos_body_pose_cache_fill_offsets.unsqueeze(0)
        motion_ids = self.motion_idx[env_ids]
        frame_indices = self._get_frame_indices(motion_ids, local_steps)
        joint_pos = self.motion.qpos[:, 7:][frame_indices]
        pos_b, quat_b = helper.body_pose(joint_pos)

        assert isinstance(self._qpos_body_pose_cache_pos_b, torch.Tensor)
        assert isinstance(self._qpos_body_pose_cache_quat_b, torch.Tensor)
        self._qpos_body_pose_cache_pos_b[env_ids] = pos_b
        self._qpos_body_pose_cache_quat_b[env_ids] = quat_b
        self._qpos_body_pose_cache_head[env_ids] = 0
        self._qpos_body_pose_cache_start_step[env_ids] = starts
        self._qpos_body_pose_cache_motion_id[env_ids] = motion_ids
        self._qpos_body_pose_cache_valid[env_ids] = True
        if env_ids.numel() == self.num_envs:
            self._qpos_body_pose_cache_initialized = True
        return pos_b, quat_b

    def _ensure_qpos_body_pose_tile_cache(self) -> None:
        if self._qpos_body_pose_tile_cache_enabled() and not self._qpos_body_pose_cache_initialized:
            self._refill_qpos_body_pose_tile_cache(self._qpos_body_pose_cache_all_env_ids)
            self._qpos_body_pose_cache_initialized = True

    def _advance_qpos_body_pose_tile_cache(self) -> None:
        """Append one tile at a shared phase boundary without moving overlap."""
        if (
            not self._qpos_body_pose_tile_cache_enabled()
            or not self._qpos_body_pose_cache_initialized
        ):
            return
        tile_steps = self._qpos_body_pose_cache_tile_steps
        if self._qpos_body_pose_cache_tick % tile_steps != 0:
            return
        helper = self._reference_fk_helper
        if helper is None:
            raise RuntimeError("qpos-only reference FK helper is not initialized")

        env_ids = self._qpos_body_pose_cache_all_env_ids
        new_local_steps = (
            self._qpos_body_pose_cache_start_step.unsqueeze(1)
            + self._qpos_body_pose_cache_window_len
            + self._qpos_body_pose_cache_append_offsets.unsqueeze(0)
        )
        frame_indices = self._get_frame_indices(self.motion_idx, new_local_steps)
        joint_pos = self.motion.qpos[:, 7:][frame_indices]
        pos_b, quat_b = helper.body_pose(joint_pos)
        slots = (
            self._qpos_body_pose_cache_head.unsqueeze(1)
            + self._qpos_body_pose_cache_append_offsets.unsqueeze(0)
        ).remainder(self._qpos_body_pose_cache_window_len)

        assert isinstance(self._qpos_body_pose_cache_pos_b, torch.Tensor)
        assert isinstance(self._qpos_body_pose_cache_quat_b, torch.Tensor)
        self._qpos_body_pose_cache_pos_b[env_ids.unsqueeze(1), slots] = pos_b
        self._qpos_body_pose_cache_quat_b[env_ids.unsqueeze(1), slots] = quat_b
        self._qpos_body_pose_cache_head.add_(tile_steps).remainder_(
            self._qpos_body_pose_cache_window_len
        )
        self._qpos_body_pose_cache_start_step.add_(tile_steps)

    def _tick_qpos_body_pose_tile_cache(self) -> None:
        if not self._qpos_body_pose_tile_cache_enabled():
            return
        self._qpos_body_pose_cache_tick += 1
        self._advance_qpos_body_pose_tile_cache()

    def _gather_qpos_body_pose_tile_cache(
        self,
        relative_steps: tuple[int, ...],
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Gather cached poses, returning ``None`` outside the configured range."""
        if not self._qpos_body_pose_tile_cache_enabled():
            return None
        relative_steps = tuple(int(step) for step in relative_steps)
        if not relative_steps:
            return None
        if (
            min(relative_steps) < self._qpos_body_pose_cache_min_offset
            or max(relative_steps) > self._qpos_body_pose_cache_max_offset
        ):
            return None
        self._ensure_qpos_body_pose_tile_cache()
        ids = (
            self._qpos_body_pose_cache_all_env_ids
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        phase = self._qpos_body_pose_cache_tick % self._qpos_body_pose_cache_tile_steps
        logical_by_phase = self._qpos_body_pose_cache_logical_offsets.get(relative_steps)
        if logical_by_phase is None:
            base_offsets = torch.as_tensor(
                tuple(step - self._qpos_body_pose_cache_min_offset for step in relative_steps),
                dtype=torch.long,
                device=self.device,
            )
            logical_by_phase = self._qpos_body_pose_cache_append_offsets.unsqueeze(
                1
            ) + base_offsets.unsqueeze(0)
            self._qpos_body_pose_cache_logical_offsets[relative_steps] = logical_by_phase
        logical_offsets = logical_by_phase[phase]
        slots = (
            self._qpos_body_pose_cache_head[ids].unsqueeze(1) + logical_offsets.unsqueeze(0)
        ).remainder(self._qpos_body_pose_cache_window_len)
        assert isinstance(self._qpos_body_pose_cache_pos_b, torch.Tensor)
        assert isinstance(self._qpos_body_pose_cache_quat_b, torch.Tensor)
        return (
            self._qpos_body_pose_cache_pos_b[ids.unsqueeze(1), slots],
            self._qpos_body_pose_cache_quat_b[ids.unsqueeze(1), slots],
        )

    def _gather_qpos_body_pose_support_tile_cache(
        self,
        relative_steps: tuple[int, ...],
        support_steps: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._qpos_body_pose_tile_cache_enabled():
            return None
        flat_steps = tuple(
            target + support for target in relative_steps for support in support_steps
        )
        if not flat_steps or (
            min(flat_steps) < self._qpos_body_pose_cache_min_offset
            or max(flat_steps) > self._qpos_body_pose_cache_max_offset
        ):
            return None
        self._ensure_qpos_body_pose_tile_cache()

        _, target_steps, _ = self._qpos_actor_target_context(relative_steps)
        support = torch.as_tensor(support_steps, device=self.device, dtype=torch.long)
        requested_steps = self._clamp_motion_time_steps(
            self.motion_idx, target_steps.unsqueeze(-1) + support
        )
        logical_offsets = requested_steps - self._qpos_body_pose_cache_start_step[:, None, None]
        slots = (self._qpos_body_pose_cache_head[:, None, None] + logical_offsets).remainder(
            self._qpos_body_pose_cache_window_len
        )
        env_ids = self._qpos_body_pose_cache_all_env_ids[:, None, None]
        assert isinstance(self._qpos_body_pose_cache_pos_b, torch.Tensor)
        assert isinstance(self._qpos_body_pose_cache_quat_b, torch.Tensor)
        return (
            self._qpos_body_pose_cache_pos_b[env_ids, slots],
            self._qpos_body_pose_cache_quat_b[env_ids, slots],
        )

    def _invalidate_reference_cache(self) -> None:
        self._clear_shared_joint_observation_cache()
        for name in (
            "_qpos_actor_root_state_cache",
            "_qpos_actor_root_pose_cache",
            "_qpos_actor_root_velocity_state_cache",
            "_qpos_actor_joint_state_cache",
            "_qpos_actor_root_kinematics_support_cache",
            "_qpos_actor_body_pose_cache",
            "_qpos_actor_body_state_cache",
            "_qpos_actor_support_cache",
        ):
            cache = getattr(self, name, None)
            if isinstance(cache, dict):
                cache.clear()
        if not hasattr(self, "_reference_cache"):
            return
        self._reference_cache.clear()
        self._reference_cache_slices.clear()
        self._reference_cache_valid = False

    def _gather_motion_fields(
        self,
        field_names: tuple[str, ...],
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        frame_indices = self._get_frame_indices(motion_ids, time_steps)
        return {
            field_name: getattr(self.motion, field_name)[frame_indices]
            for field_name in field_names
        }

    def _build_reference_cache(self) -> None:
        if self._uses_qpos_only_actor_fk():
            # Qpos mode owns small step-local derived caches.  Building the legacy
            # six-field horizon cache would defeat its memory contract.
            self._reference_cache.clear()
            self._reference_cache_slices.clear()
            self._reference_cache_valid = True
            return
        fields_by_steps: dict[tuple[int, ...], list[str]] = {}
        for field_name, steps in self._reference_cache_field_steps.items():
            fields_by_steps.setdefault(steps, []).append(field_name)

        cache: dict[str, torch.Tensor] = {}
        for steps, field_names in fields_by_steps.items():
            offset_tensor = torch.as_tensor(steps, device=self.device, dtype=torch.long)
            absolute_steps = self.time_steps.unsqueeze(1) + offset_tensor.unsqueeze(0)
            cache.update(
                self._gather_motion_fields(tuple(field_names), self.motion_idx, absolute_steps)
            )
        self._reference_cache = cache
        self._reference_cache_slices.clear()
        self._reference_cache_valid = True
        self._reference_cache_build_count += 1

    def gather_reference(self, field_name: str, relative_steps: tuple[int, ...]) -> torch.Tensor:
        """Gather reference motion data, reusing one cache per command state."""
        relative_steps = tuple(int(step) for step in relative_steps)
        if self._uses_qpos_only_actor_fk():
            if field_name == "joint_pos":
                if relative_steps == (0,):
                    return self._gather_qpos_reference(relative_steps)[..., 7:]
                return self._gather_qpos_reference_slice(relative_steps, slice(7, None))
            if field_name == "joint_vel":
                return self._qpos_actor_joint_state(relative_steps).joint_vel
            body_ids = torch.arange(len(self.cfg.body_names), device=self.device, dtype=torch.long)
            if field_name == "body_pos_w":
                return self.gather_reference_body_pos_w(
                    body_ids, relative_steps, include_env_origins=False
                )
            if field_name == "body_quat_w":
                return self.gather_reference_body_quat_w(body_ids, relative_steps)
            if field_name == "body_lin_vel_w":
                return self.gather_reference_body_lin_vel_w(body_ids, relative_steps)
            if field_name == "body_ang_vel_w":
                return self.gather_reference_body_ang_vel_w(body_ids, relative_steps)
            raise ValueError(f"Unsupported reference field: {field_name}")
        if not self.cfg.reference_cache_enabled:
            offset_tensor = torch.as_tensor(relative_steps, device=self.device, dtype=torch.long)
            absolute_steps = self.time_steps.unsqueeze(1) + offset_tensor.unsqueeze(0)
            return self._apply_motion_origin_offset(
                field_name,
                self._gather_motion_field(field_name, self.motion_idx, absolute_steps),
            )

        cached_steps = self._reference_cache_field_steps.get(field_name, ())
        cached_indices = {step: index for index, step in enumerate(cached_steps)}
        if any(step not in cached_indices for step in relative_steps):
            offset_tensor = torch.as_tensor(relative_steps, device=self.device, dtype=torch.long)
            absolute_steps = self.time_steps.unsqueeze(1) + offset_tensor.unsqueeze(0)
            return self._apply_motion_origin_offset(
                field_name,
                self._gather_motion_field(field_name, self.motion_idx, absolute_steps),
            )

        if not self._reference_cache_valid:
            self._build_reference_cache()
        slice_key = (field_name, relative_steps)
        cached_slice = self._reference_cache_slices.get(slice_key)
        if cached_slice is not None:
            return self._apply_motion_origin_offset(field_name, cached_slice)

        field = self._reference_cache[field_name]
        if relative_steps == cached_steps:
            result = field
        elif len(relative_steps) == 1:
            index = cached_indices[relative_steps[0]]
            result = field[:, index : index + 1]
        else:
            indices = torch.as_tensor(
                [cached_indices[step] for step in relative_steps],
                device=self.device,
                dtype=torch.long,
            )
            result = field.index_select(1, indices)
        self._reference_cache_slices[slice_key] = result
        return self._apply_motion_origin_offset(field_name, result)

    def _build_fk_helper(self) -> MotionFKHelper | None:
        if self.cfg.reference_storage_mode == REFERENCE_STORAGE_QPOS_ONLY_ACTOR_FK:
            # This is intentionally the same MJCF, joint order, body order, and base
            # convention used by SPV5ReferenceKinematics inside the Actor.
            return MotionFKHelper.from_mjcf_path(
                xml_path=G1_TRACKING_BFM_XML,
                dataset_joint_names=_MUJOCO_JOINT_NAMES,
                output_body_names=self.cfg.body_names,
                base_body_name="pelvis",
                device=self.device,
            )
        if not self.cfg.fk_from_joint_pos:
            return None
        return MotionFKHelper.from_mjlab_asset(
            asset=self.robot,
            dataset_joint_names=_MUJOCO_JOINT_NAMES,
            output_body_names=self.cfg.body_names,
        )

    def _resolve_motion_files(self) -> list[str]:
        """Resolve multi-motion inputs from ``motion_path`` or a single ``motion_file``."""
        gradient_test_mode = self.cfg.gradient_test_mode
        if gradient_test_mode is not None:
            simple_file = os.fspath(self.cfg.gradient_test_simple_motion_file)
            hard_file = os.fspath(self.cfg.gradient_test_hard_motion_file)
            if gradient_test_mode not in {"simple", "hard", "mixed"}:
                raise ValueError(
                    "gradient_test_mode must be one of 'simple', 'hard', or 'mixed', "
                    f"got {gradient_test_mode!r}"
                )
            required_files = {
                "simple": (("simple", simple_file),),
                "hard": (("hard", hard_file),),
                "mixed": (("simple", simple_file), ("hard", hard_file)),
            }[gradient_test_mode]
            for label, path in required_files:
                if not path:
                    raise ValueError(f"gradient-test {label} motion file is required")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Invalid gradient-test {label} motion file: {path}")
                if not path.lower().endswith(".npz"):
                    raise ValueError(f"gradient-test {label} motion must be a .npz file: {path}")
            if gradient_test_mode == "mixed" and os.path.realpath(simple_file) == os.path.realpath(
                hard_file
            ):
                raise ValueError("gradient-test simple and hard motions must be different files")
            # Every distributed rank must see both tasks.  The normal multi-motion
            # path shards files by rank, which would make the per-task gradients
            # undefined on each local PPO minibatch.
            if gradient_test_mode == "simple":
                return [simple_file]
            if gradient_test_mode == "hard":
                return [hard_file]
            return [simple_file, hard_file]

        motion_path = os.fspath(self.cfg.motion_path)
        motion_file = os.fspath(self.cfg.motion_file)
        motion_manifest_file = os.fspath(getattr(self.cfg, "motion_manifest_file", ""))
        if motion_path and motion_file:
            raise ValueError(
                "Provide either motion_path for multi-motion input or motion_file for a "
                "single motion, but not both."
            )

        if motion_path:
            if not os.path.exists(motion_path):
                raise FileNotFoundError(f"Invalid motion path: {motion_path}")
            if not os.path.isdir(motion_path):
                raise ValueError(
                    f"motion_path must point to a directory containing .npz files: {motion_path}"
                )
            if motion_manifest_file:
                if not os.path.isfile(motion_manifest_file):
                    raise FileNotFoundError(
                        f"Motion manifest file not found: {motion_manifest_file}"
                    )
                resolved_motion_files = []
                with open(motion_manifest_file, encoding="utf-8") as manifest:
                    for line in manifest:
                        entry = line.strip()
                        if not entry:
                            continue
                        if not os.path.isabs(entry):
                            entry = os.path.join(motion_path, entry)
                        resolved_motion_files.append(entry)
            else:
                rank, world_size = self._runtime_rank_context()
                if world_size > 1:
                    manifest_file = self._automatic_global_manifest_file(motion_path)
                    if rank == 0:
                        resolved_motion_files = self._build_and_publish_global_manifest(
                            motion_path,
                            manifest_file,
                            world_size=world_size,
                        )
                    else:
                        resolved_motion_files = self._wait_for_global_manifest(manifest_file)
                else:
                    resolved_motion_files = self._scan_filter_sort_motion_files(motion_path)
        elif motion_file:
            if not os.path.exists(motion_file):
                raise FileNotFoundError(f"Invalid motion file: {motion_file}")
            if not os.path.isfile(motion_file):
                raise ValueError(f"motion_file must point to a .npz file: {motion_file}")
            resolved_motion_files = [motion_file]
        else:
            resolved_motion_files = []

        # Explicit manifests retain their historical rank-private semantics.
        # Automatically discovered global manifests have already been filtered.
        if motion_manifest_file or not motion_path:
            before_filter = len(resolved_motion_files)
            _multimotion_bootstrap_log(
                "stage=filter start "
                f"motions={before_filter} source={'private_manifest' if motion_manifest_file else 'single_file'}"
            )
            resolved_motion_files = filter_excluded_motion_files(
                resolved_motion_files,
                motion_path=motion_path,
                excluded_motion_files=self.cfg.excluded_motion_files,
                motion_exclude_files=getattr(self.cfg, "motion_exclude_files", ()),
                motion_exclude_file=self.cfg.motion_exclude_file,
            )
            _multimotion_bootstrap_log(
                "stage=filter done "
                f"motions_before={before_filter} motions_after={len(resolved_motion_files)} "
                f"removed={before_filter - len(resolved_motion_files)}"
            )

        rank, world_size = self._runtime_rank_context()
        if not motion_manifest_file and world_size > 1 and len(resolved_motion_files) > 1:
            global_count = len(resolved_motion_files)
            resolved_motion_files = resolved_motion_files[rank::world_size]
            _multimotion_bootstrap_log(
                "stage=shard done "
                f"global_motions={global_count} local_motions={len(resolved_motion_files)}"
            )

        if len(resolved_motion_files) == 0:
            raise ValueError(
                "No motion files found. Provide either:\n"
                "  - motion_path: path to a directory containing .npz files\n"
                "  - motion_file: path to a single .npz file"
            )
        return resolved_motion_files

    @staticmethod
    def _runtime_rank_context() -> tuple[int, int]:
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
        return rank, max(world_size, 1)

    def _scan_filter_sort_motion_files(self, motion_path: str) -> list[str]:
        resolved_motion_files = self._scan_motion_path(motion_path)
        before_filter = len(resolved_motion_files)
        _multimotion_bootstrap_log(
            "stage=filter start "
            f"motions={before_filter} "
            f"direct_exclusions={len(self.cfg.excluded_motion_files)} "
            f"exclude_files={len(getattr(self.cfg, 'motion_exclude_files', ()))} "
            f"exclude_file={self.cfg.motion_exclude_file or '(none)'}"
        )
        resolved_motion_files = filter_excluded_motion_files(
            resolved_motion_files,
            motion_path=motion_path,
            excluded_motion_files=self.cfg.excluded_motion_files,
            motion_exclude_files=getattr(self.cfg, "motion_exclude_files", ()),
            motion_exclude_file=self.cfg.motion_exclude_file,
        )
        _multimotion_bootstrap_log(
            "stage=filter done "
            f"motions_before={before_filter} motions_after={len(resolved_motion_files)} "
            f"removed={before_filter - len(resolved_motion_files)}"
        )
        _multimotion_bootstrap_log(f"stage=sort start motions={len(resolved_motion_files)}")
        resolved_motion_files.sort()
        _multimotion_bootstrap_log(f"stage=sort done motions={len(resolved_motion_files)}")
        return resolved_motion_files

    def _scan_motion_path(self, motion_path: str) -> list[str]:
        backend = str(getattr(self.cfg, "motion_scan_backend", "auto")).lower()
        if backend not in {"auto", "fd", "python"}:
            raise ValueError("motion_scan_backend must be one of: auto, fd, python")

        if backend in {"auto", "fd"}:
            fd_executable = str(getattr(self.cfg, "motion_scan_fd_executable", "fd"))
            fd_path = shutil.which(fd_executable)
            if fd_path:
                try:
                    return self._scan_motion_path_with_fd(motion_path, fd_path)
                except (OSError, subprocess.SubprocessError) as exc:
                    if backend == "fd":
                        raise RuntimeError(
                            f"fd motion scan failed for path {motion_path}: {exc}"
                        ) from exc
                    _multimotion_bootstrap_log(
                        f"stage=scan fd_failed fallback=python path={motion_path} error={exc}"
                    )
            elif backend == "fd":
                raise FileNotFoundError(
                    f"motion_scan_backend='fd' requested, but executable not found: {fd_executable}"
                )

        return self._scan_motion_path_with_python(motion_path)

    def _scan_motion_path_with_fd(self, motion_path: str, fd_path: str) -> list[str]:
        worker_count = int(getattr(self.cfg, "motion_scan_workers", 0))
        if worker_count < 0:
            raise ValueError("motion_scan_workers must be non-negative")
        command = [
            fd_path,
            "--hidden",
            "--no-ignore",
            "--type",
            "f",
            "--color",
            "never",
        ]
        if worker_count > 0:
            command.extend(["--threads", str(worker_count)])
        command.extend([r"(?i)\.npz$", motion_path])

        start = time.perf_counter()
        log_interval = max(float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0)), 0.0)
        last_log_time = start
        _multimotion_bootstrap_log(
            "stage=scan start "
            f"backend=fd path={motion_path} executable={fd_path} workers={worker_count}"
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        resolved_motion_files: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            motion_file = line.strip()
            if motion_file:
                resolved_motion_files.append(motion_file)
            now = time.perf_counter()
            if log_interval > 0.0 and now - last_log_time >= log_interval:
                _multimotion_bootstrap_log(
                    "stage=scan progress "
                    f"backend=fd motions={len(resolved_motion_files)} "
                    f"elapsed={now - start:.3f}s"
                )
                last_log_time = now

        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output="\n".join(resolved_motion_files),
                stderr=stderr,
            )
        if stderr.strip():
            _multimotion_bootstrap_log(f"stage=scan stderr backend=fd message={stderr.strip()}")
        _multimotion_bootstrap_log(
            "stage=scan done "
            f"backend=fd motions={len(resolved_motion_files)} "
            f"elapsed={time.perf_counter() - start:.3f}s"
        )
        return resolved_motion_files

    def _scan_motion_path_with_python(self, motion_path: str) -> list[str]:
        start = time.perf_counter()
        log_interval = max(float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0)), 0.0)
        last_log_time = start
        resolved_motion_files: list[str] = []
        scanned_dirs = 0
        scanned_files = 0
        _multimotion_bootstrap_log(f"stage=scan start backend=python path={motion_path}")
        for root, _, files in os.walk(motion_path):
            scanned_dirs += 1
            scanned_files += len(files)
            for filename in files:
                if filename.lower().endswith(".npz"):
                    resolved_motion_files.append(os.path.join(root, filename))
            now = time.perf_counter()
            if log_interval > 0.0 and now - last_log_time >= log_interval:
                _multimotion_bootstrap_log(
                    "stage=scan progress "
                    f"backend=python dirs={scanned_dirs} files={scanned_files} "
                    f"motions={len(resolved_motion_files)} "
                    f"elapsed={now - start:.3f}s root={root}"
                )
                last_log_time = now
        _multimotion_bootstrap_log(
            "stage=scan done "
            f"backend=python dirs={scanned_dirs} files={scanned_files} "
            f"motions={len(resolved_motion_files)} "
            f"elapsed={time.perf_counter() - start:.3f}s"
        )
        return resolved_motion_files

    def _automatic_global_manifest_file(self, motion_path: str) -> str:
        manifest_dir = os.environ.get(
            "SP_TRACKING_MULTIMOTION_MANIFEST_DIR",
            os.environ.get("MJLAB_BOOTSTRAP_DEBUG_DIR", ""),
        )
        job_identity = os.environ.get("TORCHELASTIC_RUN_ID", "")
        if not job_identity or job_identity.lower() == "none":
            job_identity = "|".join(
                (
                    os.environ.get("MASTER_ADDR", ""),
                    os.environ.get("MASTER_PORT", ""),
                    os.environ.get("LOCAL_WORLD_SIZE", ""),
                )
            )
        if not job_identity.strip("|"):
            job_identity = f"parent-{os.getppid()}"
        if not manifest_dir:
            job_digest = hashlib.sha1(job_identity.encode("utf-8")).hexdigest()[:12]
            manifest_dir = os.path.join(
                tempfile.gettempdir(),
                "sp_tracking_multimotion_manifests",
                job_digest,
            )

        source_identity = json.dumps(
            {
                "job": job_identity,
                "motion_path": os.path.abspath(os.path.expanduser(motion_path)),
                "excluded_motion_files": list(self.cfg.excluded_motion_files),
                "motion_exclude_files": [
                    os.path.abspath(os.path.expanduser(path))
                    for path in getattr(self.cfg, "motion_exclude_files", ())
                ],
                "motion_exclude_file": os.path.abspath(
                    os.path.expanduser(self.cfg.motion_exclude_file)
                )
                if self.cfg.motion_exclude_file
                else "",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        source_digest = hashlib.sha1(source_identity.encode("utf-8")).hexdigest()[:16]
        return os.path.join(
            manifest_dir,
            f"filtered_motion_manifest_{source_digest}.txt",
        )

    def _build_and_publish_global_manifest(
        self,
        motion_path: str,
        manifest_file: str,
        *,
        world_size: int,
    ) -> list[str]:
        error_file = f"{manifest_file}.error"
        try:
            os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
            for stale_file in (manifest_file, error_file):
                try:
                    os.unlink(stale_file)
                except FileNotFoundError:
                    pass
            _multimotion_bootstrap_log(f"stage=global_manifest build_start file={manifest_file}")
            motion_files = self._scan_filter_sort_motion_files(motion_path)
            if not motion_files:
                raise ValueError("No motion files remain after scanning and filtering")
            if 1 < len(motion_files) < world_size:
                raise ValueError(
                    "Filtered motion count is smaller than world size: "
                    f"motions={len(motion_files)}, world_size={world_size}"
                )
            _multimotion_bootstrap_log(
                "stage=global_manifest publish_start "
                f"file={manifest_file} motions={len(motion_files)}"
            )
            self._write_manifest_atomic(manifest_file, motion_files)
            _multimotion_bootstrap_log(
                "stage=global_manifest publish_done "
                f"file={manifest_file} motions={len(motion_files)}"
            )
            return motion_files
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            try:
                self._write_text_atomic(error_file, error_message + "\n")
            except Exception as write_exc:
                _multimotion_bootstrap_log(
                    "stage=global_manifest error_publish_failed "
                    f"file={error_file} error={type(write_exc).__name__}: {write_exc}"
                )
            _multimotion_bootstrap_log(f"stage=global_manifest failed error={error_message}")
            raise

    def _wait_for_global_manifest(self, manifest_file: str) -> list[str]:
        timeout_s = max(
            float(getattr(self.cfg, "motion_manifest_wait_timeout_s", 600.0)),
            0.0,
        )
        poll_interval_s = max(
            float(getattr(self.cfg, "motion_manifest_poll_interval_s", 0.25)),
            0.01,
        )
        log_interval_s = max(
            float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0)),
            1.0,
        )
        error_file = f"{manifest_file}.error"
        start = time.perf_counter()
        last_log_time = start
        _multimotion_bootstrap_log(
            f"stage=global_manifest wait_start file={manifest_file} timeout={timeout_s:.1f}s"
        )
        while True:
            if os.path.isfile(error_file):
                with open(error_file, encoding="utf-8") as file:
                    error_message = file.read().strip()
                raise RuntimeError(f"rank 0 failed to build multimotion manifest: {error_message}")
            if os.path.isfile(manifest_file):
                _multimotion_bootstrap_log(f"stage=global_manifest read_start file={manifest_file}")
                motion_files = self._read_manifest(manifest_file)
                _multimotion_bootstrap_log(
                    "stage=global_manifest read_done "
                    f"file={manifest_file} motions={len(motion_files)} "
                    f"elapsed={time.perf_counter() - start:.3f}s"
                )
                return motion_files

            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= timeout_s:
                raise TimeoutError(
                    "Timed out waiting for rank 0 multimotion manifest: "
                    f"file={manifest_file}, elapsed={elapsed:.3f}s"
                )
            if now - last_log_time >= log_interval_s:
                _multimotion_bootstrap_log(
                    f"stage=global_manifest waiting file={manifest_file} elapsed={elapsed:.3f}s"
                )
                last_log_time = now
            time.sleep(poll_interval_s)

    @classmethod
    def _write_manifest_atomic(cls, manifest_file: str, motion_files: list[str]) -> None:
        cls._write_text_atomic(
            manifest_file,
            "".join(f"{motion_file}\n" for motion_file in motion_files),
        )

    @staticmethod
    def _write_text_atomic(path: str, contents: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_file = f"{path}.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as file:
            file.write(contents)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_file, path)

    @staticmethod
    def _read_manifest(manifest_file: str) -> list[str]:
        with open(manifest_file, encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor):
            self._clear_shared_joint_observation_cache()
            self.boot_indicator[env_ids] = float(max(int(self.cfg.boot_indicator_max), 0))
            self.feet_standing[env_ids] = False
            for name in (
                "_body_z_termination_buffer",
                "_gravity_dir_termination_buffer",
                "_global_key_body_pos_termination_buffer",
            ):
                buffer = getattr(self, name, None)
                if isinstance(buffer, torch.Tensor):
                    buffer[env_ids] = 0
        return extras

    def maybe_write_adaptive_bin_snapshot(
        self,
        *,
        iteration: int,
        default_snapshot_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        interval = int(self.cfg.adaptive_bin_snapshot_interval_iterations)
        if interval <= 0 or int(iteration) <= 0 or int(iteration) % interval != 0:
            return
        snapshot_dir = os.fspath(self.cfg.adaptive_bin_snapshot_dir)
        if not snapshot_dir:
            if default_snapshot_dir is None:
                return
            snapshot_dir = os.fspath(default_snapshot_dir)

        writer_key = (snapshot_dir,)
        if (
            self._adaptive_bin_snapshot_writer is None
            or writer_key != self._adaptive_bin_snapshot_writer_key
        ):
            from intact_tracking.environment.snapshot import (
                PerRankAdaptiveBinSnapshotWriter,
            )

            self._adaptive_bin_snapshot_writer = PerRankAdaptiveBinSnapshotWriter(
                snapshot_dir=snapshot_dir,
                motion_files=self.motion_files,
                file_lengths=self.motion.file_lengths,
                fps_list=self.motion.fps_list,
                motion_bin_counts=self.motion_bin_counts,
                bin_width_steps=self.bin_width_steps,
                failure_rate_ema_iterations=_resolve_adaptive_ema_iterations(self.cfg),
                prior_visit_count=self.adaptive_prior_visit_count,
                prior_failure_count=self.adaptive_prior_failure_count,
            )
            self._adaptive_bin_snapshot_writer_key = writer_key
        self._adaptive_bin_snapshot_writer.write(
            visit_count=self.bin_visit_count,
            failure_count=self.bin_failure_count,
            iteration=int(iteration),
        )

    def _compute_motion_bin_indices(
        self, time_steps: torch.Tensor, motion_indices: torch.Tensor
    ) -> torch.Tensor:
        raw_bin_indices = torch.div(time_steps, self.bin_width_steps, rounding_mode="floor")
        max_bin_indices = self.motion_bin_counts[motion_indices] - 1
        return torch.minimum(raw_bin_indices, max_bin_indices)

    def _compute_failure_rate(self) -> torch.Tensor:
        failure_rate = self.bin_failure_count / torch.clamp(self.bin_visit_count, min=1e-12)
        return failure_rate.clamp_(0.0, 1.0).masked_fill(~self.bin_valid_mask, 0.0)

    def _init_adaptive_sampling_ema(self) -> None:
        """Allocate one pending-stat buffer and update the sampler once per iteration."""
        self._adaptive_pending_visit_count = torch.zeros_like(self.bin_visit_count)
        self._adaptive_pending_failure_count = torch.zeros_like(self.bin_failure_count)
        self._adaptive_ema_last_iteration: int | None = None

    def _init_adaptive_visit_tracking(self) -> None:
        """Track the currently open bin visit for every environment."""
        self._adaptive_visit_motion_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._adaptive_visit_bin_ids = torch.full_like(self._adaptive_visit_motion_ids, -1)
        self._skip_current_adaptive_visit_update = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def begin_adaptive_sampling_iteration(self, iteration: int) -> None:
        if self.cfg.sampling_mode != "adaptive":
            return
        iteration = int(iteration)
        if self._adaptive_ema_last_iteration is None:
            self._adaptive_ema_last_iteration = iteration
            return
        elapsed_iterations = iteration - self._adaptive_ema_last_iteration
        if elapsed_iterations <= 0:
            return
        decay = _adaptive_ema_decay(self.cfg) ** elapsed_iterations
        self._apply_adaptive_sampling_ema_update(
            self._adaptive_pending_visit_count,
            self._adaptive_pending_failure_count,
            decay=decay,
        )
        self._adaptive_pending_visit_count.zero_()
        self._adaptive_pending_failure_count.zero_()
        self._adaptive_ema_last_iteration = iteration

    def _apply_adaptive_sampling_ema_update(
        self,
        visit_increments: torch.Tensor,
        failure_increments: torch.Tensor,
        *,
        decay: float,
    ) -> None:
        self.bin_visit_count.sub_(self.adaptive_prior_visit_count).mul_(decay)
        self.bin_visit_count.add_(visit_increments).add_(self.adaptive_prior_visit_count)
        self.bin_failure_count.sub_(self.adaptive_prior_failure_count).mul_(decay)
        self.bin_failure_count.add_(failure_increments).add_(self.adaptive_prior_failure_count)
        self.bin_visit_count.masked_fill_(~self.bin_valid_mask, 0.0)
        self.bin_failure_count.masked_fill_(~self.bin_valid_mask, 0.0)

    def _record_completed_adaptive_visits(
        self,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
        failure_mask: torch.Tensor | None,
    ) -> None:
        if motion_ids.numel() == 0:
            return

        current_bin_indices = self._compute_motion_bin_indices(time_steps, motion_ids)
        linear_indices = motion_ids * self.bin_count + current_bin_indices
        current_counts = torch.bincount(
            linear_indices, minlength=self.motion.num_files * self.bin_count
        ).view(self.motion.num_files, self.bin_count)
        # Each row represents one completed visit. Failures are a subset of the
        # same completed visits, so their ratio is a per-visit probability rather
        # than a residence-time failure intensity.
        visit_increments = current_counts.float()
        self._adaptive_pending_visit_count += visit_increments

        if failure_mask is None or not failure_mask.any():
            return

        failed_linear_indices = linear_indices[failure_mask]
        failed_counts = torch.bincount(
            failed_linear_indices, minlength=self.motion.num_files * self.bin_count
        ).view(self.motion.num_files, self.bin_count)
        self._adaptive_pending_failure_count += failed_counts.float()

    def _sync_adaptive_bin_visits(self, env_ids: torch.Tensor) -> None:
        """Open current visits and close successful visits after a bin transition."""
        if self.cfg.sampling_mode != "adaptive" or env_ids.numel() == 0:
            return

        current_motion_ids = self.motion_idx[env_ids]
        current_bin_ids = self._compute_motion_bin_indices(
            self.time_steps[env_ids], current_motion_ids
        )
        visit_motion_ids = self._adaptive_visit_motion_ids[env_ids]
        visit_bin_ids = self._adaptive_visit_bin_ids[env_ids]
        active = visit_motion_ids >= 0
        transitioned = active & (
            (visit_motion_ids != current_motion_ids) | (visit_bin_ids != current_bin_ids)
        )
        if transitioned.any():
            self._record_completed_adaptive_visits(
                visit_motion_ids[transitioned],
                visit_bin_ids[transitioned] * self.bin_width_steps,
                failure_mask=None,
            )

        start_current = ~active | transitioned
        if start_current.any():
            start_env_ids = env_ids[start_current]
            self._adaptive_visit_motion_ids[start_env_ids] = current_motion_ids[start_current]
            self._adaptive_visit_bin_ids[start_env_ids] = current_bin_ids[start_current]

    def _finalize_adaptive_bin_visits(
        self, env_ids: torch.Tensor, failure_mask: torch.Tensor
    ) -> None:
        """Close active visits and record their success/failure outcomes together."""
        if self.cfg.sampling_mode != "adaptive" or env_ids.numel() == 0:
            return
        failure_mask = torch.as_tensor(failure_mask, dtype=torch.bool, device=self.device).reshape(
            -1
        )
        if failure_mask.numel() != env_ids.numel():
            raise ValueError("failure_mask must match env_ids")

        visit_motion_ids = self._adaptive_visit_motion_ids[env_ids]
        active = visit_motion_ids >= 0
        if active.any():
            visit_bin_ids = self._adaptive_visit_bin_ids[env_ids]
            self._record_completed_adaptive_visits(
                visit_motion_ids[active],
                visit_bin_ids[active] * self.bin_width_steps,
                failure_mask[active],
            )
        self._adaptive_visit_motion_ids[env_ids] = -1
        self._adaptive_visit_bin_ids[env_ids] = -1

    def _stage_pre_resample_adaptive_stats(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sampling_mode != "adaptive" or env_ids.numel() == 0:
            return
        if self._adaptive_sampling_phase != "idle":
            return

        active_env_ids = env_ids[self._env.episode_length_buf[env_ids] > 0]
        if active_env_ids.numel() == 0:
            return

        self._sync_adaptive_bin_visits(active_env_ids)
        failure_mask = self._failure_mask(active_env_ids)
        self._finalize_adaptive_bin_visits(active_env_ids, failure_mask)
        self._skip_current_adaptive_visit_update[active_env_ids] = True

    def _accumulate_current_adaptive_sampling_stats(self) -> None:
        active_env_ids = torch.where(~self._skip_current_adaptive_visit_update)[0]
        self._skip_current_adaptive_visit_update.zero_()
        if active_env_ids.numel() == 0:
            return
        self._sync_adaptive_bin_visits(active_env_ids)

    def _clamp_motion_time_steps(
        self, motion_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        max_time_steps = self.motion.file_lengths[motion_ids] - 1
        if time_steps.ndim > max_time_steps.ndim:
            max_time_steps = max_time_steps.reshape(
                max_time_steps.shape + (1,) * (time_steps.ndim - max_time_steps.ndim)
            )
        clamped_time_steps = torch.clamp_min(time_steps, 0)
        return torch.minimum(clamped_time_steps, max_time_steps)

    def _get_frame_indices(
        self, motion_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        clamped_time_steps = self._clamp_motion_time_steps(motion_ids, time_steps)
        frame_starts = self.motion.length_starts[motion_ids]
        if clamped_time_steps.ndim > frame_starts.ndim:
            frame_starts = frame_starts.reshape(
                frame_starts.shape + (1,) * (clamped_time_steps.ndim - frame_starts.ndim)
            )
        return frame_starts + clamped_time_steps

    def _gather_motion_field(
        self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        frame_indices = self._get_frame_indices(motion_ids, time_steps)
        return getattr(self.motion, field_name)[frame_indices]

    def _get_reference_time_steps(self) -> torch.Tensor:
        offset_tensor = torch.as_tensor(
            self._configured_reference_steps(), device=self.device, dtype=torch.long
        )
        return self.time_steps.unsqueeze(1) + offset_tensor.unsqueeze(0)

    def _compute_adaptive_pair_sampling_probabilities(
        self,
        valid_motion_ids: torch.Tensor,
        valid_bin_ids: torch.Tensor,
        num_motions: int,
        *,
        apply_probability_caps: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pure failure-adaptive distribution and raw failure rates."""
        failure_rate = self._compute_failure_rate()
        valid_failure_rate = failure_rate[valid_motion_ids, valid_bin_ids]
        temperature_scaled_failure_rate = _temperature_scale_adaptive_signal(
            valid_failure_rate,
            self.cfg.adaptive_sampling.temperature,
        )

        pair_bin_weights = self.bin_weights[valid_motion_ids, valid_bin_ids]
        failure_scores = temperature_scaled_failure_rate * pair_bin_weights
        failure_score_sum = failure_scores.sum()
        adaptive_probabilities = (
            pair_bin_weights / torch.clamp(pair_bin_weights.sum(), min=1e-12)
            if failure_score_sum <= 0.0
            else failure_scores / failure_score_sum
        )
        if apply_probability_caps:
            adaptive_probabilities = _apply_final_probability_caps(
                adaptive_probabilities,
                valid_motion_ids,
                num_motions=num_motions,
                max_prob_per_bin=self.cfg.adaptive_max_prob_per_bin,
                max_prob_per_motion=self.cfg.adaptive_max_prob_per_motion,
                auto_cap_over_mean=self.cfg.adaptive_probability_max_over_mean,
            )
        return adaptive_probabilities, valid_failure_rate

    def _adaptive_random_probability(self, *, random_probability: float | None = None) -> float:
        if random_probability is None:
            configured = self.cfg.adaptive_sampling.random_probability
            random_probability = (
                self.cfg.adaptive_uniform_ratio if configured is None else configured
            )
        return float(max(0.0, min(1.0, float(random_probability))))

    def _init_adaptive_sampling_metrics(self) -> None:
        """Initialize metrics that have a defined adaptive-sampler interpretation."""
        if self.cfg.sampling_mode != "adaptive":
            return
        self._adaptive_sampling_metric_state: dict[str, torch.Tensor] = {}
        for name in _ADAPTIVE_SAMPLING_DISTRIBUTION_METRICS:
            metric = torch.zeros(self.num_envs, device=self.device)
            self.metrics[name] = metric
            # CommandTerm.reset() clears metrics after logging. Keep the distribution
            # associated with each env's latest non-rewind sample separately so a
            # rewind can preserve it instead of fabricating zero-valued diagnostics.
            self._adaptive_sampling_metric_state[name] = torch.zeros_like(metric)
        self.metrics["sampling_reset_rewind"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_reset_uniform"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_reset_adaptive"] = torch.zeros(self.num_envs, device=self.device)

    def _record_adaptive_sampling_distribution_metrics(
        self,
        env_ids: torch.Tensor,
        sampling_probabilities: torch.Tensor,
        valid_failure_probability: torch.Tensor,
        *,
        uniform_mix_ratio: float,
        adaptive_sampling_probabilities: torch.Tensor | None = None,
        uniform_env_ids: torch.Tensor | None = None,
        adaptive_env_ids: torch.Tensor | None = None,
    ) -> None:
        if not self.cfg.if_log_metrics or env_ids.numel() == 0:
            return
        metric_state = getattr(self, "_adaptive_sampling_metric_state", None)
        if not isinstance(metric_state, dict):
            return

        if adaptive_sampling_probabilities is None:
            adaptive_sampling_probabilities = sampling_probabilities
        num_bins = max(int(sampling_probabilities.numel()), 1)
        entropy_denom = math.log(num_bins) if num_bins > 1 else 1.0
        uniform_baseline = 1.0 / float(num_bins)

        def distribution_statistics(
            probabilities: torch.Tensor,
        ) -> tuple[float | torch.Tensor, torch.Tensor, torch.Tensor]:
            entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
            normalized_entropy = entropy / entropy_denom if num_bins > 1 else 0.0
            top1_probability = probabilities.max()
            effective_num_bins = 1.0 / torch.clamp((probabilities**2).sum(), min=1e-12)
            return normalized_entropy, top1_probability, effective_num_bins

        final_entropy, final_top1, final_effective_bins = distribution_statistics(
            sampling_probabilities
        )
        adaptive_entropy, adaptive_top1, adaptive_effective_bins = distribution_statistics(
            adaptive_sampling_probabilities
        )
        adaptive_sampling_cfg = getattr(self.cfg, "adaptive_sampling", None)
        configured_temperature = getattr(adaptive_sampling_cfg, "temperature", None)
        adaptive_temperature = (
            0.0 if configured_temperature is None else float(configured_temperature)
        )
        adaptive_probability_cap = float(
            getattr(
                self.cfg,
                "adaptive_probability_max_over_mean",
                getattr(self.cfg, "adaptive_failure_rate_max_over_mean", 200.0),
            )
        )
        values: dict[str, float | torch.Tensor] = {
            "sampling_adaptive_temperature": adaptive_temperature,
            "sampling_adaptive_probability_cap_over_mean": adaptive_probability_cap,
            "sampling_adaptive_normalized_entropy": adaptive_entropy,
            "sampling_adaptive_top1_prob": adaptive_top1,
            "sampling_adaptive_top1_over_uniform": (adaptive_top1 / uniform_baseline),
            "sampling_adaptive_effective_num_bins": adaptive_effective_bins,
            "sampling_final_normalized_entropy": final_entropy,
            "sampling_uniform_mix_ratio_pre_cap": float(uniform_mix_ratio),
            "sampling_uniform_branch_probability": float(uniform_mix_ratio),
            "sampling_uniform_baseline_per_bin": uniform_baseline,
            "sampling_final_top1_prob": final_top1,
            "sampling_final_top1_over_uniform": final_top1 / uniform_baseline,
            "sampling_failure_probability_mean": valid_failure_probability.mean(),
            "sampling_failure_probability_max": valid_failure_probability.max(),
            "sampling_final_effective_num_bins": final_effective_bins,
        }
        for name, value in values.items():
            self.metrics[name][env_ids] = value
            metric_state[name][env_ids] = value
        self.metrics["sampling_reset_rewind"][env_ids] = 0.0
        self.metrics["sampling_reset_uniform"][env_ids] = 0.0
        self.metrics["sampling_reset_adaptive"][env_ids] = 0.0
        if uniform_env_ids is not None:
            self.metrics["sampling_reset_uniform"][uniform_env_ids] = 1.0
        if adaptive_env_ids is not None:
            self.metrics["sampling_reset_adaptive"][adaptive_env_ids] = 1.0

    def _restore_rewind_sampling_metrics(self, rewind_env_ids: torch.Tensor) -> None:
        """Keep the originating sampler distribution attached to a rewind episode."""
        if not self.cfg.if_log_metrics or rewind_env_ids.numel() == 0:
            return
        metric_state = getattr(self, "_adaptive_sampling_metric_state", None)
        if not isinstance(metric_state, dict):
            return
        for name in _ADAPTIVE_SAMPLING_DISTRIBUTION_METRICS:
            self.metrics[name][rewind_env_ids] = metric_state[name][rewind_env_ids]
        self.metrics["sampling_reset_rewind"][rewind_env_ids] = 1.0
        self.metrics["sampling_reset_uniform"][rewind_env_ids] = 0.0
        self.metrics["sampling_reset_adaptive"][rewind_env_ids] = 0.0

    @property
    def command(self) -> torch.Tensor:
        cmd = torch.cat([self.motion_joint_pos, self.motion_joint_vel], dim=1)
        return cmd

    @property
    def command_joint_pos(self) -> torch.Tensor:
        return self.motion_joint_pos

    @property
    def command_joint_vel(self) -> torch.Tensor:
        return self.motion_joint_vel

    @property
    def command_current_joint_pos(self) -> torch.Tensor:
        return self.current_motion_joint_pos

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.gather_reference("joint_pos", (0,))[:, 0]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.gather_reference("joint_vel", (0,))[:, 0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.gather_reference("body_pos_w", (0,))[:, 0]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.gather_reference("body_quat_w", (0,))[:, 0]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.gather_reference("body_lin_vel_w", (0,))[:, 0]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.gather_reference("body_ang_vel_w", (0,))[:, 0]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.gather_root_reference("body_pos_w", (0,))[:, 0] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.gather_root_reference("body_quat_w", (0,))[:, 0]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Anchor linear velocities with history and future steps.

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        reference_lin_vel = self.gather_root_reference(
            "body_lin_vel_w", self._configured_reference_steps()
        )
        return reference_lin_vel.reshape(self.num_envs, -1)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Anchor angular velocities with history and future steps.

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        reference_ang_vel = self.gather_root_reference(
            "body_ang_vel_w", self._configured_reference_steps()
        )
        return reference_ang_vel.reshape(self.num_envs, -1)

    @property
    def anchor_projected_gravity(self) -> torch.Tensor:
        """Anchor projected gravity with history and future steps.

        Converts anchor quaternions to projected gravity vectors using the formula:
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        Shape: (num_envs, num_steps * 3) where num_steps = history_steps + 1 + (future_steps - 1)
        """
        anchor_quat = self.gather_reference("body_quat_w", self._configured_reference_steps())[
            :, :, self.motion_anchor_body_index
        ]

        # Extract quaternion components: (w, x, y, z) format
        qw = anchor_quat[..., 0]  # (num_envs, num_steps)
        qx = anchor_quat[..., 1]  # (num_envs, num_steps)
        qy = anchor_quat[..., 2]  # (num_envs, num_steps)
        qz = anchor_quat[..., 3]  # (num_envs, num_steps)

        # Compute projected gravity for each step
        gravity_x = 2 * (-qz * qx + qw * qy)
        gravity_y = -2 * (qz * qy + qw * qx)
        gravity_z = 1 - 2 * (qw * qw + qz * qz)

        # Stack to (num_envs, num_steps, 3)
        projected_gravity = torch.stack([gravity_x, gravity_y, gravity_z], dim=-1)

        # Reshape to (num_envs, num_steps * 3)
        return projected_gravity.reshape(self.num_envs, -1)

    # Motion reference properties with history and future steps
    @property
    def motion_joint_pos(self) -> torch.Tensor:
        """Joint positions reference with history and future steps.

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        reference_joint_pos = self.gather_reference("joint_pos", self._configured_reference_steps())
        return reference_joint_pos.reshape(self.num_envs, -1)

    @property
    def current_motion_joint_pos(self) -> torch.Tensor:
        """Joint positions reference at current step only."""
        return self.joint_pos

    @property
    def motion_joint_vel(self) -> torch.Tensor:
        """Joint velocities reference with history and future steps.

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        reference_joint_vel = self.gather_reference("joint_vel", self._configured_reference_steps())
        return reference_joint_vel.reshape(self.num_envs, -1)

    @property
    def motion_anchor_pos(self) -> torch.Tensor:
        """Anchor positions reference with history and future steps.

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        reference_anchor_pos = self.gather_reference(
            "body_pos_w", self._configured_reference_steps()
        )[:, :, self.motion_anchor_body_index]
        reference_anchor_pos = reference_anchor_pos + self._env.scene.env_origins[:, None, :]
        return reference_anchor_pos.reshape(self.num_envs, -1)

    @property
    def motion_anchor_quat(self) -> torch.Tensor:
        """Anchor quaternions reference with history and future steps.

        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        reference_anchor_quat = self.gather_reference(
            "body_quat_w", self._configured_reference_steps()
        )[:, :, self.motion_anchor_body_index]
        return reference_anchor_quat.reshape(self.num_envs, -1)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        if not self.cfg.if_log_metrics:
            return

        # Extract current step data from multi-step properties
        # anchor_lin_vel_w and anchor_ang_vel_w contain [history_steps, current, future_steps]
        # Current step is at index: history_steps
        # Calculate total number of steps: history_steps + 1 (current) + (future_steps - 1)
        num_steps_total = self.cfg.history_steps + 1 + max(0, self.cfg.future_steps - 1)
        current_step_idx = self.cfg.history_steps

        # For anchor_lin_vel_w and anchor_ang_vel_w, extract current step
        # Reshape from (num_envs, num_steps * 3) to (num_envs, num_steps, 3) and extract current step
        if num_steps_total > 1:
            anchor_lin_vel_current = self.anchor_lin_vel_w.reshape(
                self.num_envs, num_steps_total, 3
            )[:, current_step_idx, :]
            anchor_ang_vel_current = self.anchor_ang_vel_w.reshape(
                self.num_envs, num_steps_total, 3
            )[:, current_step_idx, :]
        else:
            # No history/future, use directly (shape is already (num_envs, 3))
            anchor_lin_vel_current = self.anchor_lin_vel_w
            anchor_ang_vel_current = self.anchor_ang_vel_w

        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            anchor_lin_vel_current - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            anchor_ang_vel_current - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        if self._uses_qpos_only_actor_fk():
            root = self._qpos_actor_root_velocity_state((0,))
            body = self.gather_reference_body_state_b((0,))
            robot_lin_b = fk_quat_apply_inverse(
                root.root_quat_w[:, 0, None],
                self.robot_body_lin_vel_w - root.root_lin_vel_w[:, 0, None],
            )
            robot_ang_b = fk_quat_apply_inverse(
                root.root_quat_w[:, 0, None],
                self.robot_body_ang_vel_w - root.root_ang_vel_w[:, 0, None],
            )
            self.metrics["error_body_lin_vel"] = torch.norm(
                body.lin_vel_b[:, 0] - robot_lin_b, dim=-1
            ).mean(dim=-1)
            self.metrics["error_body_ang_vel"] = torch.norm(
                body.ang_vel_b[:, 0] - robot_ang_b, dim=-1
            ).mean(dim=-1)
        else:
            self.metrics["error_body_lin_vel"] = torch.norm(
                self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
            ).mean(dim=-1)
            self.metrics["error_body_ang_vel"] = torch.norm(
                self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
            ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        uniform_probability = self._adaptive_random_probability()
        strategy = str(self.cfg.adaptive_sampling.strategy)
        adaptive_probabilities, valid_failure_rate = (
            self._compute_adaptive_pair_sampling_probabilities(
                self.valid_motion_ids,
                self.valid_bin_ids,
                self.motion.num_files,
                apply_probability_caps=strategy == "branch",
            )
        )
        uniform_probabilities = torch.full_like(
            adaptive_probabilities,
            1.0 / float(max(adaptive_probabilities.numel(), 1)),
        )
        final_probabilities = (
            1.0 - uniform_probability
        ) * adaptive_probabilities + uniform_probability * uniform_probabilities

        if strategy == "branch":
            sampled_pair_indices, uniform_mask = _sample_adaptive_uniform_branches(
                adaptive_probabilities,
                len(env_ids),
                uniform_probability,
            )
            uniform_env_ids = env_ids[uniform_mask]
            adaptive_env_ids = env_ids[~uniform_mask]
        elif strategy == "mixture":
            # Retain the old single-distribution behavior for configurations that
            # have not opted into independently attributable sampling branches.
            final_probabilities = _apply_final_probability_caps(
                final_probabilities,
                self.valid_motion_ids,
                num_motions=self.motion.num_files,
                max_prob_per_bin=self.cfg.adaptive_max_prob_per_bin,
                max_prob_per_motion=self.cfg.adaptive_max_prob_per_motion,
                auto_cap_over_mean=self.cfg.adaptive_probability_max_over_mean,
            )
            sampled_pair_indices = torch.multinomial(
                final_probabilities, len(env_ids), replacement=True
            )
            uniform_env_ids = None
            adaptive_env_ids = None
        else:
            raise ValueError(f"Unsupported adaptive sampling strategy: {strategy!r}")

        sampled_motion_indices = self.valid_motion_ids[sampled_pair_indices]
        sampled_bin_indices = self.valid_bin_ids[sampled_pair_indices]
        self._record_adaptive_sampling_distribution_metrics(
            env_ids,
            final_probabilities,
            valid_failure_rate,
            uniform_mix_ratio=uniform_probability,
            adaptive_sampling_probabilities=adaptive_probabilities,
            uniform_env_ids=uniform_env_ids,
            adaptive_env_ids=adaptive_env_ids,
        )

        self.motion_idx[env_ids] = sampled_motion_indices
        self.motion_length[env_ids] = self.motion.file_lengths[sampled_motion_indices]

        bin_starts = sampled_bin_indices * self.bin_width_steps
        bin_ends = torch.minimum(bin_starts + self.bin_width_steps, self.motion_length[env_ids])
        bin_lengths = torch.clamp(bin_ends - bin_starts, min=1)
        offsets = (
            sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * bin_lengths.float()
        ).long()
        self.time_steps[env_ids] = torch.minimum(
            bin_starts + offsets, self.motion_length[env_ids] - 1
        )
        if self.cfg.adaptive_pre_failure_sample_window_steps > 0:
            pre_failure_env_mask = (
                torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
                if strategy == "mixture"
                else ~uniform_mask
            )
            pre_failure_env_ids = env_ids[pre_failure_env_mask]
            pre_failure_offsets = torch.randint(
                self.cfg.adaptive_pre_failure_sample_window_steps,
                (len(pre_failure_env_ids),),
                device=self.device,
            )
            self.time_steps[pre_failure_env_ids] = (
                self.time_steps[pre_failure_env_ids] - pre_failure_offsets
            ).clamp_min(0)

    def _uniform_sampling(self, env_ids: torch.Tensor):
        lower = max(int(self.cfg.skip_initial_frames), 0)
        positive_steps = [step for step in self._configured_reference_steps() if step > 0]
        future_margin = max(positive_steps, default=0)
        upper = (
            self.motion_length[env_ids]
            - future_margin
            - max(int(self.cfg.sample_tail_margin_steps), 0)
        ).clamp_min(lower)
        span = (upper - lower).clamp_min(0)
        self.time_steps[env_ids] = (
            lower + (sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * span).long()
        )

    def _failure_rewind_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        rewind_cfg = self.cfg.rewind
        if not rewind_cfg.enabled or env_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        failure_mask = self._failure_mask(env_ids)
        if not failure_mask.any():
            return torch.empty(0, dtype=torch.long, device=self.device)
        probability = float(max(0.0, min(1.0, rewind_cfg.failure_probability)))
        if probability <= 0.0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        use_rewind = torch.rand(len(env_ids), device=self.device) < probability
        return env_ids[failure_mask & use_rewind]

    def _failure_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return terminations, including failures coincident with a timeout."""
        termination_manager = getattr(self._env, "termination_manager", None)
        terminated = getattr(termination_manager, "terminated", None)
        if not isinstance(terminated, torch.Tensor):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        return terminated[env_ids].clone()

    def _prepare_reset_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Record adaptive stats, optionally rewind failures, and return resample IDs."""
        self._invalidate_reference_cache()
        self._stage_pre_resample_adaptive_stats(env_ids)
        rewind_env_ids = self._failure_rewind_env_ids(env_ids)
        if rewind_env_ids.numel() == 0:
            return env_ids
        rewind_cfg = self.cfg.rewind
        minimum = max(int(rewind_cfg.min_steps), 0)
        maximum = max(int(rewind_cfg.max_steps), minimum)
        if maximum > 0:
            rewind = torch.randint(
                minimum,
                maximum + 1,
                (rewind_env_ids.numel(),),
                device=self.device,
            )
            first_valid_frame = max(int(getattr(self.cfg, "skip_initial_frames", 0)), 0)
            self.time_steps[rewind_env_ids] = (self.time_steps[rewind_env_ids] - rewind).clamp_min(
                first_valid_frame
            )
        else:
            bin_ids = self._compute_motion_bin_indices(
                self.time_steps[rewind_env_ids], self.motion_idx[rewind_env_ids]
            )
            self.time_steps[rewind_env_ids] = bin_ids * self.bin_width_steps
        self._restore_rewind_sampling_metrics(rewind_env_ids)
        keep_adaptive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep_adaptive[rewind_env_ids] = False
        return env_ids[keep_adaptive[env_ids]]

    def _reset_robot_to_reference(self, env_ids: torch.Tensor) -> None:
        """Write a sampled/re-wound reference state to the simulator."""
        if self._uses_qpos_only_actor_fk():
            root = self._qpos_actor_root_state((0,), env_ids=env_ids)
            helper = self._reference_fk_helper
            if helper is None:
                raise RuntimeError("qpos-only reference FK helper is not initialized")
            qpos = self._gather_qpos_reference((0,), env_ids=env_ids)[:, 0]
            if self._qpos_body_pose_tile_cache_enabled():
                pose_tile = self._refill_qpos_body_pose_tile_cache(env_ids)
                phase = self._qpos_body_pose_cache_tick % self._qpos_body_pose_cache_tile_steps
                current = phase - self._qpos_body_pose_cache_min_offset
                body_pos_b = pose_tile[0][:, current]
            else:
                body_pos_b, _ = helper.body_pose(qpos[:, 7:])
            root_pos = (root.root_pos_w[:, 0] + self._env.scene.env_origins[env_ids]).clone()
            root_ori = root.root_quat_w[:, 0].clone()
            root_lin_vel = root.root_lin_vel_w[:, 0].clone()
            root_ang_vel = root.root_ang_vel_w[:, 0].clone()
            body_pos_w = root_pos[:, None] + fk_quat_apply(root_ori[:, None], body_pos_b)
            joint_pos = root.joint_pos[:, 0].clone()
            joint_vel = root.joint_vel[:, 0].clone()
        else:
            body_pos_w = self.body_pos_w[env_ids]
            root_pos = body_pos_w[:, 0].clone()
            root_ori = self.body_quat_w[env_ids, 0].clone()
            root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
            root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
            # Preserve the legacy full-mode RNG stream: joint reset noise has always
            # been sampled for every environment before selecting ``env_ids``.
            joint_pos = self.joint_pos.clone()
            joint_vel = self.joint_vel.clone()
        init_noise = self.cfg.init_noise
        if init_noise:
            pos_std = float(init_noise.get("root_pos", 0.0))
            ori_std = float(init_noise.get("root_ori", 0.0))
            rand_samples = torch.cat(
                (
                    torch.randn((len(env_ids), 3), device=self.device).clamp(-1, 1) * pos_std,
                    torch.zeros((len(env_ids), 3), device=self.device),
                ),
                dim=-1,
            )
            rand_samples[:, 2].clamp_min_(0.0)
        else:
            range_list = [
                self.cfg.pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_pos = apply_reset_ground_clearance(
            root_pos,
            body_pos_w,
            self._env.scene.env_origins[env_ids],
            rand_samples[:, 0:3],
            orientations_delta,
            root_lift_height=self.cfg.reset_root_lift_height,
            min_body_z=self.cfg.reset_min_body_z,
        )
        root_ori = quat_mul(orientations_delta, root_ori)
        if init_noise:
            root_ori = self._quaternion_noise(root_ori, ori_std)
        if init_noise:
            lin_std = float(init_noise.get("root_lin_vel", 0.0))
            ang_std = float(init_noise.get("root_ang_vel", 0.0))
            rand_samples = torch.cat(
                (
                    torch.randn((len(env_ids), 3), device=self.device).clamp(-1, 1) * lin_std,
                    torch.randn((len(env_ids), 3), device=self.device).clamp(-1, 1) * ang_std,
                ),
                dim=-1,
            )
        else:
            range_list = [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]
        if init_noise:
            joint_pos += torch.randn_like(joint_pos).clamp(-1, 1) * float(
                init_noise.get("joint_pos", 0.0)
            )
            joint_vel += torch.randn_like(joint_vel).clamp(-1, 1) * float(
                init_noise.get("joint_vel", 0.0)
            )
        else:
            joint_pos += sample_uniform(
                lower=self.cfg.joint_position_range[0],
                upper=self.cfg.joint_position_range[1],
                size=joint_pos.shape,
                device=joint_pos.device,
            )
        # MJLab stores invariant limits with a singleton world dimension.  Do
        # not index that dimension with vector-environment ids; broadcast it
        # to the reset batch instead.  A per-world tensor is still supported
        # for compatibility with randomized limit providers.
        all_soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits
        if all_soft_joint_pos_limits.shape[0] == 1:
            soft_joint_pos_limits = all_soft_joint_pos_limits.expand(len(env_ids), -1, -1)
        else:
            soft_joint_pos_limits = all_soft_joint_pos_limits[env_ids]
        if not self._uses_qpos_only_actor_fk():
            joint_pos = joint_pos[env_ids]
            joint_vel = joint_vel[env_ids]
        joint_pos = torch.clip(
            joint_pos,
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        joint_vel = clamp_reset_joint_velocity(joint_vel, self.cfg.reset_joint_vel_limit)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos,
                    root_ori,
                    root_lin_vel,
                    root_ang_vel,
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )
        self.robot.clear_state(env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._set_action_boot_target(env_ids, joint_pos)
        self._reset_sp_tracking_state(env_ids, root_pos)

    def _synchronized_full_groups(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return group IDs only when every member participates in this reset."""

        group_size = int(getattr(self.cfg, "synchronized_group_size", 1))
        if group_size <= 1:
            return env_ids
        if self.num_envs % group_size:
            raise RuntimeError("synchronized_group_size must divide num_envs")
        selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        selected[env_ids] = True
        selected = selected.view(-1, group_size)
        return selected.all(dim=1).nonzero(as_tuple=False).flatten()

    def _broadcast_synchronized_motion(self, group_ids: torch.Tensor) -> None:
        group_size = int(getattr(self.cfg, "synchronized_group_size", 1))
        if group_size <= 1 or group_ids.numel() == 0:
            return
        leaders = group_ids * group_size
        members = leaders[:, None] + torch.arange(group_size, device=self.device)[None]
        for name in ("motion_idx", "motion_length", "time_steps"):
            value = getattr(self, name)
            value[members] = value[leaders, None]
        gradient_labels = getattr(self, "gradient_test_motion_label", None)
        if isinstance(gradient_labels, torch.Tensor):
            gradient_labels[members] = gradient_labels[leaders, None]

    def _resample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        sample_env_ids = self._prepare_reset_sampling(env_ids)
        group_size = int(getattr(self.cfg, "synchronized_group_size", 1))
        full_group_ids = self._synchronized_full_groups(sample_env_ids)
        if group_size > 1:
            sample_env_ids = full_group_ids * group_size
        if sample_env_ids.numel() > 0:
            gradient_test_mode = self.cfg.gradient_test_mode
            if gradient_test_mode is not None:
                motion_indices, semantic_labels = gradient_test_motion_assignment(
                    gradient_test_mode, sample_env_ids, self.num_envs
                )
                self.gradient_test_motion_label[sample_env_ids] = semantic_labels
            else:
                motion_indices = torch.randint(
                    0,
                    self.motion.num_files,
                    (len(sample_env_ids),),
                    device=self.device,
                )
            if self.cfg.sampling_mode == "start":
                self.motion_idx[sample_env_ids] = motion_indices
                self.motion_length[sample_env_ids] = self.motion.file_lengths[motion_indices]
                self.time_steps[sample_env_ids] = 0
            elif self.cfg.sampling_mode == "uniform":
                self.motion_idx[sample_env_ids] = motion_indices
                self.motion_length[sample_env_ids] = self.motion.file_lengths[motion_indices]
                self._uniform_sampling(sample_env_ids)
            else:
                assert self.cfg.sampling_mode == "adaptive"
                self._adaptive_sampling(sample_env_ids)
            self._broadcast_synchronized_motion(full_group_ids)
        self._set_motion_origin_offset(env_ids)
        self._invalidate_reference_cache()
        self._reset_robot_to_reference(env_ids)

    def _set_action_boot_target(self, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> None:
        action_manager = getattr(self._env, "action_manager", None)
        get_term = getattr(action_manager, "get_term", None)
        if not callable(get_term):
            return
        try:
            action_term = get_term("joint_pos")
        except KeyError:
            return
        set_boot_target = getattr(action_term, "set_boot_target", None)
        if callable(set_boot_target):
            target_ids = getattr(action_term, "target_ids", None)
            if isinstance(target_ids, torch.Tensor):
                joint_pos = joint_pos.index_select(1, target_ids.to(device=joint_pos.device))
            set_boot_target(env_ids, joint_pos)

    def prepare_for_collection_forward(self, dt: float) -> bool:
        """Optionally move state-writing work before the environment forward.

        This execution-order optimization is explicitly opt-in. Command metrics
        require the legacy path so they can observe the freshly forwarded robot
        state against the pre-resample reference.
        """
        if not getattr(self.cfg, "collection_preforward_enabled", False):
            return False
        if getattr(self.cfg, "if_log_metrics", True):
            return False
        if getattr(self, "_collection_preforward_pending", False):
            raise RuntimeError("Motion command already has a prepared collection step")
        self._update_metrics()
        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._begin_command_update()
        self._collection_preforward_pending = True
        return True

    def compute(self, dt: float) -> None:
        """Finish a prepared update, or retain CommandTerm's legacy ordering."""
        if not getattr(self, "_collection_preforward_pending", False):
            super().compute(dt)
            return
        self._collection_preforward_pending = False
        self._finish_command_update()

    def _begin_command_update(self) -> bool:
        """Advance/resample reference state without reading derived robot state."""
        motion_resample_boundary = getattr(self, "motion_resample_boundary", None)
        if isinstance(motion_resample_boundary, torch.Tensor):
            motion_resample_boundary.zero_()
        if self.cfg.sampling_mode == "adaptive":
            self._adaptive_sampling_phase = "updating"
            self._accumulate_current_adaptive_sampling_stats()

        boot_indicator = getattr(self, "boot_indicator", None)
        if isinstance(boot_indicator, torch.Tensor):
            boot_indicator.sub_(1).clamp_min_(0.0)
        self.time_steps += 1
        self._tick_qpos_body_pose_tile_cache()
        self._invalidate_reference_cache()
        env_ids = (
            torch.where(self.time_steps >= self.motion_length)[0]
            if getattr(self.cfg, "resample_on_motion_end", True)
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        if env_ids.numel() > 0:
            if isinstance(motion_resample_boundary, torch.Tensor):
                motion_resample_boundary[env_ids] = True
            if self.cfg.sampling_mode == "adaptive":
                self._finalize_adaptive_bin_visits(
                    env_ids, torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
                )
            self._resample_command(env_ids)
        return bool(env_ids.numel() > 0)

    def _finish_command_update(self) -> None:
        """Read forwarded state and refresh targets derived from it."""
        # Feet-contact targets request pose before velocity.  Build the fused state
        # first so qpos-only mode does not run a one-frame FK and then a seven-frame
        # FK for the same current target.
        body_state = (
            self.gather_reference_body_state_b((0,)) if self._uses_qpos_only_actor_fk() else None
        )
        self._update_feet_standing()
        self._update_reward_root_target()

        anchor_pos_w = self.anchor_pos_w
        delta_pos_w = torch.cat((self.robot_anchor_pos_w[:, :2], anchor_pos_w[:, 2:]), dim=-1)[
            :, None
        ]
        delta_ori_w = yaw_quat(
            quat_mul(
                self.robot_anchor_quat_w,
                quat_inv(self.anchor_quat_w),
            )
        )[:, None].expand(-1, len(self.cfg.body_names), -1)

        if self._uses_qpos_only_actor_fk():
            assert body_state is not None
            body_pos_b = body_state.pos_b[:, 0]
            body_quat_b = body_state.quat_b[:, 0]
            reference_offsets_w = fk_quat_apply(self.anchor_quat_w[:, None], body_pos_b)
            reference_quat_w = fk_quat_mul(self.anchor_quat_w[:, None], body_quat_b)
            self.body_quat_relative_w = quat_mul(delta_ori_w, reference_quat_w)
            self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, reference_offsets_w)
        else:
            self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
            self.body_pos_relative_w = delta_pos_w + quat_apply(
                delta_ori_w, self.body_pos_w - anchor_pos_w[:, None]
            )
        if self.cfg.sampling_mode == "adaptive":
            self._adaptive_sampling_phase = "idle"

    def _update_command(self):
        resampled = self._begin_command_update()
        if resampled:
            # Legacy/non-training callers do not install the collection pre-forward
            # hook, so retain their correctness with an explicit refresh.
            self._env.sim.forward()
        self._finish_command_update()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw ghost robot or frames based on visualization mode."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self.cfg.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self._ghost_color
            if (
                self.extra_reference_motion is not None
                and self._extra_reference_ghost_model is None
            ):
                self._extra_reference_ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._extra_reference_ghost_model.geom_rgba[:] = self._extra_reference_ghost_color

            entity: Entity = self._env.scene[self.cfg.entity_name]
            indexing = entity.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            for batch in env_indices:
                qpos = np.zeros(self._env.sim.mj_model.nq)
                qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
                qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
                qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

                visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")
                if self.extra_reference_motion is not None:
                    assert self._extra_reference_ghost_model is not None
                    extra_time_step = torch.clamp(
                        self.time_steps[batch],
                        min=0,
                        max=self.extra_reference_motion.time_step_total - 1,
                    )
                    extra_qpos = np.zeros(self._env.sim.mj_model.nq)
                    extra_body_pos_w = (
                        self.extra_reference_motion.body_pos_w[extra_time_step]
                        + self._env.scene.env_origins[batch]
                    )
                    extra_qpos[free_joint_q_adr[0:3]] = extra_body_pos_w[0].cpu().numpy()
                    extra_qpos[free_joint_q_adr[3:7]] = (
                        self.extra_reference_motion.body_quat_w[extra_time_step, 0].cpu().numpy()
                    )
                    extra_qpos[joint_q_adr] = (
                        self.extra_reference_motion.joint_pos[extra_time_step].cpu().numpy()
                    )
                    visualizer.add_ghost_mesh(
                        extra_qpos,
                        model=self._extra_reference_ghost_model,
                        label=f"extra_reference_ghost_{batch}",
                    )

        elif self.cfg.viz.mode == "frames":
            for batch in env_indices:
                desired_body_pos = self.body_pos_w[batch].cpu().numpy()
                desired_body_quat = self.body_quat_w[batch]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
                current_body_quat = self.robot_body_quat_w[batch]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.cfg.body_names):
                    visualizer.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_{batch}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    visualizer.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_{batch}",
                    )

                desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
                desired_anchor_quat = self.anchor_quat_w[batch]
                desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=desired_anchor_pos,
                    rotation_matrix=desired_rotation_matrix,
                    scale=0.1,
                    label=f"desired_anchor_{batch}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )

                current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
                current_anchor_quat = self.robot_anchor_quat_w[batch]
                current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=current_anchor_pos,
                    rotation_matrix=current_rotation_matrix,
                    scale=0.15,
                    label=f"current_anchor_{batch}",
                )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    entity_name: str
    motion_path: str = ""
    motion_file: str = ""
    # Manifest entries are loaded exactly as listed and are not sharded again.
    motion_manifest_file: str = ""
    # Automatic multi-rank discovery uses a rank-0 global manifest.  These
    # controls do not change the rank-private semantics of motion_manifest_file.
    motion_manifest_wait_timeout_s: float = 600.0
    motion_manifest_poll_interval_s: float = 0.25
    motion_scan_backend: Literal["auto", "fd", "python"] = "auto"
    motion_scan_workers: int = 0
    motion_scan_fd_executable: str = "fd"
    motion_scan_log_interval_s: float = 10.0
    excluded_motion_files: tuple[str, ...] = ()
    motion_exclude_files: tuple[str, ...] = ()
    motion_exclude_file: str = ""
    extra_reference_motion_file: str = ""
    # Optional two-motion diagnostic mode.  These fields are inert for every
    # existing task.  When enabled, each rank loads the same explicit files and
    # environment-to-motion assignment remains fixed across resets.
    gradient_test_mode: Literal["simple", "hard", "mixed"] | None = None
    gradient_test_simple_motion_file: str = ""
    gradient_test_hard_motion_file: str = ""
    motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"
    fk_from_joint_pos: bool = False
    # Kept separate from body FK so tasks can ablate either preprocessing path.
    recompute_joint_vel_from_joint_pos: bool = False
    # Opt-in exact compact qpos for online v2-9 routing/training.
    load_compact_qpos: bool = False
    # ``qpos_only_actor_fk`` keeps only clean [root xyz, root quat, joint q] on
    # the accelerator and reconstructs task references with the Actor's FK path.
    reference_storage_mode: Literal["full", "qpos_only_actor_fk"] = "full"
    actor_reference_fps: float = 50.0
    # Pose-only, phase-aligned rolling tiles amortize qpos FK without restoring
    # the six corpus-sized derived tensors. Zero keeps the direct-FK fallback.
    qpos_body_pose_cache_tile_steps: int = 0
    qpos_body_pose_cache_range: tuple[int, int] = (-8, 20)
    # Reference-frame and task-state features are opt-in so existing tracking
    # tasks retain their legacy behavior.
    motion_origin_recenter: bool = False
    sliding_root_xy_reward: bool = False
    boot_indicator_max: int = 0
    # The reference task masks failure terminations during the first few
    # control steps after reset.  Kept configurable so non-SP users retain the
    # legacy zero-warmup behavior.
    termination_warmup_steps: int = 0
    feet_standing_body_names: tuple[str, ...] = ()
    feet_standing: dict[str, float] = field(default_factory=dict)
    resample_on_motion_end: bool = True
    # Forward-predictor collection only: each contiguous group contains one
    # copy of every fixed dynamics class and shares motion identity/phase.
    synchronized_group_size: int = 1
    anchor_body_name: str
    body_names: tuple[str, ...]
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    reset_root_lift_height: float = 0.0
    reset_min_body_z: float | None = None
    reset_joint_vel_limit: float | None = None
    init_noise: dict[str, float] = field(default_factory=dict)
    skip_initial_frames: int = 0
    sample_tail_margin_steps: int = 0
    student_motion_randomization: dict = field(default_factory=dict)

    # Ref Motion: Future/History steps configuration for N-step lookahead
    future_steps: int = 5  # 1
    history_steps: int = 5  # 0
    reference_cache_enabled: bool = True
    reference_cache_steps: dict[str, tuple[int, ...]] | None = None

    adaptive_uniform_ratio: float = 0.1
    adaptive_sampling: AdaptiveSamplingCfg = field(default_factory=AdaptiveSamplingCfg)
    rewind: RewindCfg = field(default_factory=RewindCfg)
    adaptive_bin_width_s: float = 1.0
    adaptive_bin_width_steps: int | None = None
    # Completed-visit and failed-visit priors keep unseen bins neutral in the
    # failure-driven branch: 0 / 1 = 0. Uniform mixing provides coverage.
    adaptive_prior_visit_count: float = 1.0
    adaptive_prior_failure_count: float = 0.0
    # Effective per-iteration EMA horizon. None retains cumulative statistics.
    adaptive_failure_rate_ema_iterations: int | None = None
    # Deprecated alias for old launch overrides.
    adaptive_failure_rate_window_iterations: int | None = None
    # ``auto`` bin/motion probability caps are this multiple of their respective
    # mean probabilities. Failure rates themselves are never clipped.
    adaptive_probability_max_over_mean: float = 200.0
    adaptive_sequence_length_agnostic: bool = False
    adaptive_max_prob_per_bin: float | Literal["auto"] | None = "auto"
    adaptive_max_prob_per_motion: float | Literal["auto"] | None = "auto"
    adaptive_pre_failure_sample_window_steps: int = 200
    adaptive_bin_snapshot_interval_iterations: int = 0
    adaptive_bin_snapshot_dir: str = ""
    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

    # for downstream task training
    if_log_metrics: bool = True
    collection_preforward_enabled: bool = False

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)


# Keep the public interface aligned with the single-motion module.
MotionCommand = MultiMotionCommand
MotionCommandCfg = MultiMotionCommandCfg
