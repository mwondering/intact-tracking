from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from mjlab.managers import CommandTerm
from mjlab.utils.lab_api.math import sample_uniform

from .motion_fk import MotionFKHelper
from .multi_commands import (
    _ISAACLAB_TO_MUJOCO_BODY_REINDEX,
    _ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    DEFAULT_MOTION_FPS,
    REFERENCE_STORAGE_FULL,
    MotionLoader,
    MultiMotionCommand,
    MultiMotionCommandCfg,
    _adaptive_ema_decay,
    _apply_final_probability_caps,
    _resolve_adaptive_ema_iterations,
    _resolve_adaptive_prior_counts,
    _select_or_fk_body_fields,
    _select_or_recompute_joint_vel,
    _temperature_scale_adaptive_signal,
    extract_motion_fps,
    filter_excluded_motion_files,
)


def _build_motion_group_probabilities(
    motion_files: list[str], groups: list[dict], device
) -> torch.Tensor | None:
    """Build HEFT-style per-group motion-uniform sampling probabilities."""
    if not groups:
        return None
    probabilities = torch.zeros(len(motion_files), dtype=torch.float32, device=device)
    assignments = torch.full((len(motion_files),), -1, dtype=torch.long)
    normalized_paths = [
        f"/{os.path.normpath(path).replace(os.sep, '/').strip('/')}/" for path in motion_files
    ]
    for group_id, group in enumerate(groups):
        fragment = f"/{str(group['path_fragment']).strip('/')}/"
        matched = torch.tensor([fragment in path for path in normalized_paths], dtype=torch.bool)
        if bool((assignments[matched] >= 0).any()):
            raise ValueError(f"motion sampling groups overlap at {fragment!r}")
        assignments[matched] = group_id
    # A custom/flat dataset should remain usable without silently dropping files.
    if bool((assignments < 0).any()):
        return None
    weights = torch.tensor([float(group["weight"]) for group in groups], device=device)
    if bool((weights <= 0).any()):
        raise ValueError("motion sampling group weights must be positive")
    weights /= weights.sum()
    for group_id, weight in enumerate(weights):
        mask = assignments.to(device) == group_id
        if not bool(mask.any()):
            return None
        probabilities[mask] = weight / mask.sum()
    return probabilities / probabilities.sum()


def _bootstrap_log_line(message: str) -> str:
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    pid = os.getpid()
    return (
        f"[BOOT][{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"rank={rank} local_rank={local_rank} pid={pid}: large_motion: {message}"
    )


def _bootstrap_should_print_stdout() -> bool:
    stdout_flag = os.environ.get("SP_TRACKING_BOOTSTRAP_STDOUT", "1").lower()
    if stdout_flag in {"0", "false", "no", "off"}:
        return False
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _bootstrap_debug(message: str, *, stdout: bool = False) -> None:
    line = _bootstrap_log_line(message)
    if stdout and _bootstrap_should_print_stdout():
        print(line, flush=True)

    debug_dir = os.environ.get("MJLAB_BOOTSTRAP_DEBUG_DIR", "")
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        pid = os.getpid()
        log_file = os.path.join(debug_dir, f"rank_{rank}_local_{local_rank}_pid_{pid}.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def _read_large_dataset_motion_metadata_file(
    motion_file: str,
) -> tuple[int, float, bool, bool]:
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Invalid motion file path: {motion_file}")
    with np.load(motion_file) as data:
        file_length = int(data["joint_pos"].shape[0])
        fps_value, is_non_scalar_fps, is_empty_fps = extract_motion_fps(data)
    return file_length, fps_value, is_non_scalar_fps, is_empty_fps


def _read_large_dataset_motion_metadata_job(
    job: tuple[int, str],
) -> tuple[int, int, float, bool, bool]:
    index, motion_file = job
    file_length, fps_value, is_non_scalar_fps, is_empty_fps = (
        _read_large_dataset_motion_metadata_file(motion_file)
    )
    return index, file_length, fps_value, is_non_scalar_fps, is_empty_fps


@dataclass(frozen=True)
class SubsetRefreshResult:
    replaced_slot_ids: torch.Tensor
    old_motion_ids: torch.Tensor
    new_motion_ids: torch.Tensor

    @property
    def num_replaced(self) -> int:
        return int(self.new_motion_ids.numel())


class ActiveMotionSubset:
    """Bookkeeping for the unique per-rank active motion subset."""

    def __init__(
        self,
        *,
        total_motion_count: int,
        subset_size: int,
        min_resident_iterations: int,
        device: str | torch.device,
    ) -> None:
        if total_motion_count <= 0:
            raise ValueError("total_motion_count must be positive")
        if subset_size <= 0:
            raise ValueError("subset_size must be positive")
        self.total_motion_count = int(total_motion_count)
        self.subset_size = min(int(subset_size), self.total_motion_count)
        self.min_resident_iterations = max(int(min_resident_iterations), 0)
        self.device = torch.device(device)

        self.active_motion_ids = torch.empty(self.subset_size, dtype=torch.long, device=self.device)
        self.active_mask = torch.zeros(
            self.total_motion_count, dtype=torch.bool, device=self.device
        )
        self.pending_mask = torch.zeros_like(self.active_mask)
        self.motion_to_slot = torch.full(
            (self.total_motion_count,), -1, dtype=torch.long, device=self.device
        )
        self.slot_loaded_iteration = torch.zeros(
            self.subset_size, dtype=torch.long, device=self.device
        )
        self.slot_ref_count = torch.zeros(self.subset_size, dtype=torch.long, device=self.device)
        self._initialized = False

    def initialize(self, motion_ids: torch.Tensor, *, iteration: int) -> None:
        motion_ids = self._normalize_motion_ids(motion_ids)
        if motion_ids.numel() != self.subset_size:
            raise ValueError(
                f"Expected {self.subset_size} initial motion ids, got {motion_ids.numel()}"
            )
        if torch.unique(motion_ids).numel() != motion_ids.numel():
            raise ValueError("Initial active subset must contain unique motion ids")

        self.active_motion_ids.copy_(motion_ids)
        self.active_mask.zero_()
        self.active_mask[motion_ids] = True
        self.pending_mask.zero_()
        self.motion_to_slot.fill_(-1)
        self.motion_to_slot[motion_ids] = torch.arange(
            self.subset_size, dtype=torch.long, device=self.device
        )
        self.slot_loaded_iteration.fill_(int(iteration))
        self.slot_ref_count.zero_()
        self._initialized = True

    def mark_pending(self, motion_ids: torch.Tensor) -> None:
        motion_ids = self._normalize_motion_ids(motion_ids)
        self.pending_mask[motion_ids] = True

    def clear_pending(self, motion_ids: torch.Tensor) -> None:
        motion_ids = self._normalize_motion_ids(motion_ids)
        self.pending_mask[motion_ids] = False

    def available_motion_ids(self) -> torch.Tensor:
        unavailable = self.active_mask | self.pending_mask
        return torch.where(~unavailable)[0]

    def set_slot_ref_counts_from_motion_ids(self, motion_ids: torch.Tensor) -> None:
        self.slot_ref_count.zero_()
        if motion_ids.numel() == 0:
            return
        motion_ids = self._normalize_motion_ids(motion_ids)
        slot_ids = self.motion_to_slot[motion_ids]
        slot_ids = slot_ids[slot_ids >= 0]
        if slot_ids.numel() == 0:
            return
        counts = torch.bincount(slot_ids, minlength=self.subset_size)
        self.slot_ref_count.copy_(counts.to(dtype=torch.long, device=self.device))

    def eligible_slot_ids(self, *, iteration: int) -> torch.Tensor:
        if not self._initialized:
            return torch.empty(0, dtype=torch.long, device=self.device)
        resident_iterations = int(iteration) - self.slot_loaded_iteration
        eligible = (resident_iterations >= self.min_resident_iterations) & (
            self.slot_ref_count == 0
        )
        return torch.where(eligible)[0]

    def refresh(
        self,
        replacement_motion_ids: torch.Tensor,
        *,
        iteration: int,
        max_replacements: int,
        generator: torch.Generator | None = None,
    ) -> SubsetRefreshResult:
        if not self._initialized:
            raise RuntimeError("ActiveMotionSubset.initialize() must be called first")
        if max_replacements <= 0:
            return self._empty_refresh_result()

        replacement_motion_ids = self._filter_replacement_ids(replacement_motion_ids)
        if replacement_motion_ids.numel() == 0:
            return self._empty_refresh_result()

        eligible_slots = self.eligible_slot_ids(iteration=iteration)
        if eligible_slots.numel() == 0:
            return self._empty_refresh_result()

        num_replacements = min(
            int(max_replacements),
            int(replacement_motion_ids.numel()),
            int(eligible_slots.numel()),
        )
        slot_order = torch.randperm(eligible_slots.numel(), generator=generator, device=self.device)
        selected_slots = eligible_slots[slot_order[:num_replacements]]
        selected_replacements = replacement_motion_ids[:num_replacements]
        old_motion_ids = self.active_motion_ids[selected_slots].clone()

        self.active_mask[old_motion_ids] = False
        self.motion_to_slot[old_motion_ids] = -1

        self.active_motion_ids[selected_slots] = selected_replacements
        self.active_mask[selected_replacements] = True
        self.pending_mask[selected_replacements] = False
        self.motion_to_slot[selected_replacements] = selected_slots
        self.slot_loaded_iteration[selected_slots] = int(iteration)
        self.slot_ref_count[selected_slots] = 0

        return SubsetRefreshResult(
            replaced_slot_ids=selected_slots,
            old_motion_ids=old_motion_ids,
            new_motion_ids=selected_replacements,
        )

    def _filter_replacement_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = self._normalize_motion_ids(motion_ids)
        if motion_ids.numel() == 0:
            return motion_ids
        unique_ids = torch.unique(motion_ids, sorted=False)
        available = ~(self.active_mask[unique_ids] | self.pending_mask[unique_ids])
        return unique_ids[available]

    def _normalize_motion_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if motion_ids.ndim != 1:
            motion_ids = motion_ids.reshape(-1)
        if motion_ids.numel() == 0:
            return motion_ids
        if motion_ids.min() < 0 or motion_ids.max() >= self.total_motion_count:
            raise IndexError("Motion id is outside the full dataset range")
        return motion_ids

    def _empty_refresh_result(self) -> SubsetRefreshResult:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        return SubsetRefreshResult(empty, empty, empty)


@dataclass
class LargeDatasetMotionBuffer:
    global_motion_ids: torch.Tensor
    file_lengths: torch.Tensor
    length_starts: torch.Tensor
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor

    @property
    def num_files(self) -> int:
        return int(self.global_motion_ids.numel())


class LargeDatasetMotionSlotBuffer:
    """Per-slot GPU cache that can replace a few motions without rebuilding all slots."""

    _FIELD_NAMES = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(
        self,
        *,
        global_motion_ids: torch.Tensor,
        chunks: dict[str, list[torch.Tensor]],
        file_lengths: torch.Tensor,
        fps: float,
    ) -> None:
        self.global_motion_ids = global_motion_ids
        self.file_lengths = file_lengths
        self.fps = fps
        self._bucket_capacities: list[int] = []
        self._bucket_id_by_capacity: dict[int, int] = {}
        self._bucket_fields: dict[str, list[torch.Tensor]] = {
            field_name: [] for field_name in self._FIELD_NAMES
        }
        self._bucket_free_local_ids: list[list[int]] = []
        self._field_tail_shapes: dict[str, torch.Size] = {}
        self._field_dtypes: dict[str, torch.dtype] = {}
        self._field_devices: dict[str, torch.device] = {}
        self.slot_bucket_ids = torch.empty_like(self.file_lengths)
        self.slot_bucket_local_ids = torch.empty_like(self.file_lengths)
        self._refresh_length_starts()
        self._build_bucket_storage(chunks)

    @property
    def num_files(self) -> int:
        return int(self.global_motion_ids.numel())

    @property
    def length_starts(self) -> torch.Tensor:
        return self._length_starts

    def gather(
        self,
        field_name: str,
        slot_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        return self.gather_many((field_name,), slot_ids, time_steps)[field_name]

    def gather_many(
        self,
        field_names: tuple[str, ...],
        slot_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Gather several fields while sharing slot/bucket index preparation."""
        flat_slot_ids, flat_time_steps, output_shape = self._flatten_indices(slot_ids, time_steps)
        outputs = {
            field_name: torch.empty(
                (*flat_time_steps.shape, *self._field_tail_shapes[field_name]),
                dtype=self._field_dtypes[field_name],
                device=self._field_devices[field_name],
            )
            for field_name in field_names
        }
        if flat_time_steps.numel() == 0:
            return {
                field_name: output.reshape(*output_shape, *self._field_tail_shapes[field_name])
                for field_name, output in outputs.items()
            }

        bucket_ids = self.slot_bucket_ids[flat_slot_ids]
        bucket_local_ids = self.slot_bucket_local_ids[flat_slot_ids]
        for bucket_id in range(len(self._bucket_capacities)):
            output_indices = torch.where(bucket_ids == bucket_id)[0]
            if output_indices.numel() == 0:
                continue
            local_ids = bucket_local_ids[output_indices]
            selected_time_steps = flat_time_steps[output_indices]
            for field_name in field_names:
                outputs[field_name][output_indices] = self._bucket_fields[field_name][bucket_id][
                    local_ids, selected_time_steps
                ]
        return {
            field_name: output.reshape(*output_shape, *self._field_tail_shapes[field_name])
            for field_name, output in outputs.items()
        }

    def replace_slots(
        self,
        slot_ids: torch.Tensor,
        new_motion_ids: torch.Tensor,
        store: "LargeDatasetMotionStore",
    ) -> None:
        if slot_ids.numel() == 0:
            return
        loaded = store.load_motion_chunks(new_motion_ids)
        for offset, slot in enumerate(slot_ids.tolist()):
            self.global_motion_ids[slot] = loaded["global_motion_ids"][offset]
            self.file_lengths[slot] = loaded["file_lengths"][offset]
            self._replace_slot_storage(
                slot,
                int(loaded["file_lengths"][offset].item()),
                {field_name: loaded[field_name][offset] for field_name in self._FIELD_NAMES},
            )
        self._refresh_length_starts()

    def _flatten_indices(
        self, slot_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        if time_steps.ndim == 1:
            return slot_ids.reshape(-1), time_steps.reshape(-1), time_steps.shape
        expanded_slots = slot_ids.unsqueeze(-1).expand_as(time_steps)
        return expanded_slots.reshape(-1), time_steps.reshape(-1), time_steps.shape

    def _refresh_length_starts(self) -> None:
        self._length_starts = torch.empty_like(self.file_lengths)
        if self.file_lengths.numel() == 0:
            return
        self._length_starts[0] = 0
        if self.file_lengths.numel() > 1:
            self._length_starts[1:] = self.file_lengths[:-1].cumsum(dim=0)

    def _build_bucket_storage(self, chunks: dict[str, list[torch.Tensor]]) -> None:
        if self.num_files == 0:
            raise ValueError("Slot buffer requires at least one motion")
        lengths = [int(length) for length in self.file_lengths.detach().cpu().tolist()]
        capacities = [self._bucket_capacity_for_length(length) for length in lengths]
        for field_name in self._FIELD_NAMES:
            prototype = chunks[field_name][0]
            self._field_tail_shapes[field_name] = prototype.shape[1:]
            self._field_dtypes[field_name] = prototype.dtype
            self._field_devices[field_name] = prototype.device

        for capacity in sorted(set(capacities)):
            self._add_bucket(capacity)

        bucket_slots: dict[int, list[int]] = {
            bucket_id: [] for bucket_id in range(len(self._bucket_capacities))
        }
        for slot, capacity in enumerate(capacities):
            bucket_id = self._bucket_id_by_capacity[capacity]
            local_id = len(bucket_slots[bucket_id])
            bucket_slots[bucket_id].append(slot)
            self.slot_bucket_ids[slot] = bucket_id
            self.slot_bucket_local_ids[slot] = local_id

        for bucket_id, slot_list in bucket_slots.items():
            self._allocate_bucket_storage(bucket_id, len(slot_list))

        for slot, length in enumerate(lengths):
            self._copy_slot_fields(
                int(self.slot_bucket_ids[slot].item()),
                int(self.slot_bucket_local_ids[slot].item()),
                length,
                {field_name: chunks[field_name][slot] for field_name in self._FIELD_NAMES},
            )

        padded_frames = sum(
            len(bucket_slots[bucket_id]) * capacity
            for bucket_id, capacity in enumerate(self._bucket_capacities)
        )
        active_frames = max(sum(lengths), 1)
        bucket_slot_counts = [len(bucket_slots[i]) for i in range(len(self._bucket_capacities))]
        bucket_summary = list(zip(self._bucket_capacities, bucket_slot_counts, strict=True))
        _bootstrap_debug(
            "slot bucket storage built "
            f"buckets={bucket_summary} "
            f"padding_overhead={padded_frames / active_frames:.3f}x"
        )

    @staticmethod
    def _bucket_capacity_for_length(length: int) -> int:
        return 1 << (max(int(length), 1) - 1).bit_length()

    def _add_bucket(self, capacity: int) -> int:
        bucket_id = len(self._bucket_capacities)
        self._bucket_capacities.append(int(capacity))
        self._bucket_id_by_capacity[int(capacity)] = bucket_id
        self._bucket_free_local_ids.append([])
        for field_name in self._FIELD_NAMES:
            if field_name in self._field_tail_shapes:
                empty = torch.empty(
                    (0, int(capacity), *self._field_tail_shapes[field_name]),
                    dtype=self._field_dtypes[field_name],
                    device=self._field_devices[field_name],
                )
                self._bucket_fields[field_name].append(empty)
        return bucket_id

    def _bucket_id_for_length(self, length: int) -> int:
        capacity = self._bucket_capacity_for_length(length)
        bucket_id = self._bucket_id_by_capacity.get(capacity)
        if bucket_id is not None:
            return bucket_id
        return self._add_bucket(capacity)

    def _allocate_bucket_storage(self, bucket_id: int, size: int) -> None:
        capacity = self._bucket_capacities[bucket_id]
        for field_name in self._FIELD_NAMES:
            self._bucket_fields[field_name][bucket_id] = torch.empty(
                (size, capacity, *self._field_tail_shapes[field_name]),
                dtype=self._field_dtypes[field_name],
                device=self._field_devices[field_name],
            )

    def _grow_bucket(self, bucket_id: int) -> int:
        old_size = int(self._bucket_fields[self._FIELD_NAMES[0]][bucket_id].shape[0])
        grow_by = max(1, min(64, old_size // 8 if old_size > 0 else 1))
        new_size = old_size + grow_by
        capacity = self._bucket_capacities[bucket_id]
        for field_name in self._FIELD_NAMES:
            old_bucket = self._bucket_fields[field_name][bucket_id]
            new_bucket = torch.empty(
                (new_size, capacity, *self._field_tail_shapes[field_name]),
                dtype=self._field_dtypes[field_name],
                device=self._field_devices[field_name],
            )
            if old_size > 0:
                new_bucket[:old_size].copy_(old_bucket)
            self._bucket_fields[field_name][bucket_id] = new_bucket
        self._bucket_free_local_ids[bucket_id].extend(range(old_size + 1, new_size))
        return old_size

    def _allocate_bucket_local_id(self, bucket_id: int) -> int:
        free_ids = self._bucket_free_local_ids[bucket_id]
        if free_ids:
            return free_ids.pop()
        return self._grow_bucket(bucket_id)

    def _replace_slot_storage(
        self, slot: int, length: int, fields: dict[str, torch.Tensor]
    ) -> None:
        old_bucket_id = int(self.slot_bucket_ids[slot].item())
        old_local_id = int(self.slot_bucket_local_ids[slot].item())
        new_bucket_id = self._bucket_id_for_length(length)
        if new_bucket_id == old_bucket_id:
            new_local_id = old_local_id
        else:
            self._bucket_free_local_ids[old_bucket_id].append(old_local_id)
            new_local_id = self._allocate_bucket_local_id(new_bucket_id)
            self.slot_bucket_ids[slot] = new_bucket_id
            self.slot_bucket_local_ids[slot] = new_local_id
        self._copy_slot_fields(new_bucket_id, new_local_id, length, fields)

    def _copy_slot_fields(
        self,
        bucket_id: int,
        local_id: int,
        length: int,
        fields: dict[str, torch.Tensor],
    ) -> None:
        for field_name in self._FIELD_NAMES:
            target = self._bucket_fields[field_name][bucket_id][local_id]
            target[:length].copy_(fields[field_name])

    def _as_flat_field(self, field_name: str) -> torch.Tensor:
        pieces = []
        for slot in range(self.num_files):
            length = int(self.file_lengths[slot].item())
            bucket_id = int(self.slot_bucket_ids[slot].item())
            local_id = int(self.slot_bucket_local_ids[slot].item())
            pieces.append(self._bucket_fields[field_name][bucket_id][local_id, :length])
        return torch.cat(pieces, dim=0)

    def __getattr__(self, name: str) -> torch.Tensor:
        if name in self._FIELD_NAMES:
            return self._as_flat_field(name)
        raise AttributeError(name)


class LargeDatasetMotionStore:
    """CPU/disk-side motion store that only stages requested motions on device."""

    _FIELD_NAMES = LargeDatasetMotionSlotBuffer._FIELD_NAMES
    _DEFAULT_FPS = DEFAULT_MOTION_FPS

    def __init__(
        self,
        motion_files: list[str],
        body_indexes: torch.Tensor,
        motion_type: Literal["isaaclab", "mujoco"] = "isaaclab",
        device: str | torch.device = "cpu",
        metadata_cache_file: str = "",
        metadata_cache_wait_timeout_s: float = 7200.0,
        metadata_cache_poll_interval_s: float = 0.25,
        metadata_read_workers: int = 0,
        metadata_read_backend: Literal["thread", "process", "serial"] = "thread",
        metadata_read_chunksize: int = 64,
        fk_from_joint_pos: bool = False,
        recompute_joint_vel_from_joint_pos: bool = False,
        fk_helper: MotionFKHelper | None = None,
    ) -> None:
        if len(motion_files) == 0:
            raise ValueError("motion_files cannot be empty")
        start = time.perf_counter()
        _bootstrap_debug(
            f"LargeDatasetMotionStore init start num_motion_files={len(motion_files)} device={device}",
            stdout=True,
        )
        self.motion_files = list(motion_files)
        self.num_files = len(self.motion_files)
        self.device = torch.device(device)
        self.motion_type = motion_type
        self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long).cpu()
        self.fk_from_joint_pos = bool(fk_from_joint_pos)
        self.recompute_joint_vel_from_joint_pos = bool(recompute_joint_vel_from_joint_pos)
        self.fk_helper = fk_helper
        self.metadata_read_workers = max(int(metadata_read_workers), 0)
        self.metadata_read_backend = str(metadata_read_backend).lower()
        if self.metadata_read_backend not in {"thread", "process", "serial"}:
            raise ValueError("metadata_read_backend must be one of: thread, process, serial")
        self.metadata_read_chunksize = max(int(metadata_read_chunksize), 1)
        self._joint_reindex: list[int] | None = None
        self._body_reindex: list[int] | None = None
        if motion_type == "isaaclab":
            self._joint_reindex = _ISAACLAB_TO_MUJOCO_JOINT_REINDEX
            self._body_reindex = _ISAACLAB_TO_MUJOCO_BODY_REINDEX
        elif motion_type != "mujoco":
            raise ValueError(f"Unsupported motion_type: {motion_type}")

        metadata_cache_file = os.fspath(metadata_cache_file)
        cached_metadata = self._try_load_metadata_cache(metadata_cache_file)
        rank, world_size = self._runtime_rank_context()
        if cached_metadata is None and metadata_cache_file and world_size > 1 and rank != 0:
            cached_metadata = self._wait_for_metadata_cache(
                metadata_cache_file,
                timeout_s=float(metadata_cache_wait_timeout_s),
                poll_interval_s=float(metadata_cache_poll_interval_s),
            )
        if cached_metadata is None:
            file_lengths, fps_values, non_scalar_fps_count, empty_fps_count = (
                self._read_motion_metadata_from_files()
            )
            self._write_metadata_cache(
                metadata_cache_file,
                file_lengths=file_lengths,
                fps_values=fps_values,
                non_scalar_fps_count=non_scalar_fps_count,
                empty_fps_count=empty_fps_count,
            )
        else:
            file_lengths, fps_values, non_scalar_fps_count, empty_fps_count = cached_metadata
        self.file_lengths = torch.tensor(file_lengths, dtype=torch.long, device=self.device)
        self.fps_list = fps_values
        self.fps = fps_values[0]
        self.non_scalar_fps_count = non_scalar_fps_count
        self.empty_fps_count = empty_fps_count
        _bootstrap_debug(
            "LargeDatasetMotionStore init done "
            f"num_files={self.num_files} total_frames={int(sum(file_lengths))} "
            f"non_scalar_fps_count={non_scalar_fps_count} "
            f"empty_fps_count={empty_fps_count} "
            f"elapsed={time.perf_counter() - start:.3f}s",
            stdout=True,
        )

    def _read_motion_metadata_from_files(self) -> tuple[list[int], list[float], int, int]:
        worker_count = self.metadata_read_workers
        if self.metadata_read_backend == "serial" or worker_count <= 1:
            return self._read_motion_metadata_from_files_serial()
        if self.metadata_read_backend == "process":
            return self._read_motion_metadata_from_files_process(worker_count)
        if self.metadata_read_backend == "thread":
            return self._read_motion_metadata_from_files_parallel(worker_count)
        raise ValueError(f"Unsupported metadata_read_backend: {self.metadata_read_backend}")

    def _read_motion_metadata_from_files_serial(
        self,
    ) -> tuple[list[int], list[float], int, int]:
        start = time.perf_counter()
        file_lengths: list[int] = []
        fps_values: list[float] = []
        non_scalar_fps_count = 0
        empty_fps_count = 0
        total_files = len(self.motion_files)
        _bootstrap_debug(f"metadata read start count={total_files} backend=serial", stdout=True)
        for index, motion_file in enumerate(self.motion_files):
            file_length, fps_value, is_non_scalar_fps, is_empty_fps = (
                self._read_one_motion_metadata(motion_file)
            )
            file_lengths.append(file_length)
            fps_values.append(fps_value)
            if is_non_scalar_fps:
                non_scalar_fps_count += 1
            if is_empty_fps:
                empty_fps_count += 1
            completed_count = index + 1
            if completed_count % 5000 == 0 or completed_count == total_files:
                _bootstrap_debug(
                    f"metadata progress {completed_count}/{total_files} "
                    f"elapsed={time.perf_counter() - start:.3f}s file={motion_file}",
                    stdout=True,
                )
        return file_lengths, fps_values, non_scalar_fps_count, empty_fps_count

    def _read_motion_metadata_from_files_parallel(
        self, worker_count: int
    ) -> tuple[list[int], list[float], int, int]:
        start = time.perf_counter()
        total_files = len(self.motion_files)
        worker_count = min(max(int(worker_count), 1), total_files)
        file_lengths = [0] * total_files
        fps_values = [0.0] * total_files
        non_scalar_fps_count = 0
        empty_fps_count = 0
        completed_count = 0
        _bootstrap_debug(
            f"metadata read start count={total_files} backend=parallel workers={worker_count}",
            stdout=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending: dict[concurrent.futures.Future, tuple[int, str]] = {}
            next_index = 0
            max_pending = max(worker_count * 4, worker_count)

            def submit_until_full() -> None:
                nonlocal next_index
                while next_index < total_files and len(pending) < max_pending:
                    motion_file = self.motion_files[next_index]
                    future = executor.submit(self._read_one_motion_metadata, motion_file)
                    pending[future] = (next_index, motion_file)
                    next_index += 1

            submit_until_full()
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    index, motion_file = pending.pop(future)
                    file_length, fps_value, is_non_scalar_fps, is_empty_fps = future.result()
                    file_lengths[index] = file_length
                    fps_values[index] = fps_value
                    if is_non_scalar_fps:
                        non_scalar_fps_count += 1
                    if is_empty_fps:
                        empty_fps_count += 1
                    completed_count += 1
                    if completed_count % 5000 == 0 or completed_count == total_files:
                        _bootstrap_debug(
                            f"metadata progress {completed_count}/{total_files} "
                            f"elapsed={time.perf_counter() - start:.3f}s file={motion_file}",
                            stdout=True,
                        )
                submit_until_full()
        return file_lengths, fps_values, non_scalar_fps_count, empty_fps_count

    def _read_motion_metadata_from_files_process(
        self, worker_count: int
    ) -> tuple[list[int], list[float], int, int]:
        start = time.perf_counter()
        total_files = len(self.motion_files)
        worker_count = min(max(int(worker_count), 1), total_files)
        chunksize = self.metadata_read_chunksize
        file_lengths = [0] * total_files
        fps_values = [0.0] * total_files
        non_scalar_fps_count = 0
        empty_fps_count = 0
        completed_count = 0
        _bootstrap_debug(
            "metadata read start "
            f"count={total_files} backend=process workers={worker_count} chunksize={chunksize}",
            stdout=True,
        )
        with mp.Pool(processes=worker_count) as pool:
            for (
                index,
                file_length,
                fps_value,
                is_non_scalar_fps,
                is_empty_fps,
            ) in pool.imap_unordered(
                _read_large_dataset_motion_metadata_job,
                enumerate(self.motion_files),
                chunksize=chunksize,
            ):
                file_lengths[index] = file_length
                fps_values[index] = fps_value
                if is_non_scalar_fps:
                    non_scalar_fps_count += 1
                if is_empty_fps:
                    empty_fps_count += 1
                completed_count += 1
                if completed_count % 5000 == 0 or completed_count == total_files:
                    _bootstrap_debug(
                        f"metadata progress {completed_count}/{total_files} "
                        f"elapsed={time.perf_counter() - start:.3f}s "
                        f"file={self.motion_files[index]}",
                        stdout=True,
                    )
        return file_lengths, fps_values, non_scalar_fps_count, empty_fps_count

    @staticmethod
    def _read_one_motion_metadata(motion_file: str) -> tuple[int, float, bool, bool]:
        return _read_large_dataset_motion_metadata_file(motion_file)

    def _try_load_metadata_cache(
        self, metadata_cache_file: str
    ) -> tuple[list[int], list[float], int, int] | None:
        if not metadata_cache_file or not os.path.exists(metadata_cache_file):
            return None
        try:
            with open(metadata_cache_file, encoding="utf-8") as f:
                data = json.load(f)
            if int(data["version"]) != 1:
                return None
            if int(data["num_files"]) != self.num_files:
                return None
            cached_hash = str(data["motion_files_hash"])
            if cached_hash != self._motion_files_hash():
                return None
            file_lengths = [int(value) for value in data["file_lengths"]]
            fps_values = [float(value) for value in data["fps_values"]]
            if len(file_lengths) != self.num_files or len(fps_values) != self.num_files:
                return None
            non_scalar_fps_count = int(data["non_scalar_fps_count"])
            empty_fps_count = int(data["empty_fps_count"])
            _bootstrap_debug(
                f"LargeDatasetMotionStore metadata cache hit file={metadata_cache_file}",
                stdout=True,
            )
            return file_lengths, fps_values, non_scalar_fps_count, empty_fps_count
        except Exception as exc:
            _bootstrap_debug(
                f"LargeDatasetMotionStore metadata cache ignored file={metadata_cache_file} error={exc}",
                stdout=True,
            )
            return None

    def _wait_for_metadata_cache(
        self,
        metadata_cache_file: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> tuple[list[int], list[float], int, int] | None:
        start = time.perf_counter()
        timeout_s = max(float(timeout_s), 0.0)
        poll_interval_s = max(float(poll_interval_s), 0.01)
        _bootstrap_debug(
            f"LargeDatasetMotionStore waiting for metadata cache file={metadata_cache_file}",
            stdout=True,
        )
        while time.perf_counter() - start <= timeout_s:
            cached_metadata = self._try_load_metadata_cache(metadata_cache_file)
            if cached_metadata is not None:
                return cached_metadata
            time.sleep(poll_interval_s)
        _bootstrap_debug(
            f"LargeDatasetMotionStore metadata cache wait timed out file={metadata_cache_file}",
            stdout=True,
        )
        return None

    def _write_metadata_cache(
        self,
        metadata_cache_file: str,
        *,
        file_lengths: list[int],
        fps_values: list[float],
        non_scalar_fps_count: int,
        empty_fps_count: int,
    ) -> None:
        if not metadata_cache_file:
            return
        try:
            cache_dir = os.path.dirname(os.path.abspath(metadata_cache_file))
            os.makedirs(cache_dir, exist_ok=True)
            tmp_file = f"{metadata_cache_file}.tmp.{os.getpid()}.json"
            payload = {
                "version": 1,
                "num_files": self.num_files,
                "motion_files_hash": self._motion_files_hash(),
                "file_lengths": [int(value) for value in file_lengths],
                "fps_values": [float(value) for value in fps_values],
                "non_scalar_fps_count": int(non_scalar_fps_count),
                "empty_fps_count": int(empty_fps_count),
            }
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, metadata_cache_file)
            _bootstrap_debug(
                f"LargeDatasetMotionStore metadata cache wrote file={metadata_cache_file}",
                stdout=True,
            )
        except Exception as exc:
            _bootstrap_debug(
                f"LargeDatasetMotionStore metadata cache write skipped file={metadata_cache_file} error={exc}",
                stdout=True,
            )

    def _motion_files_hash(self) -> str:
        digest = hashlib.sha1()
        for motion_file in self.motion_files:
            digest.update(os.path.abspath(motion_file).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

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

    @staticmethod
    def _extract_fps_value(fps_data: np.ndarray) -> tuple[float, bool, bool]:
        fps_array = np.asarray(fps_data, dtype=np.float32)
        if fps_array.size == 0:
            return DEFAULT_MOTION_FPS, False, True
        return float(fps_array.reshape(-1)[0]), fps_array.size > 1, False

    @staticmethod
    def _validate_motion_field(motion_file: str, field_name: str, value: np.ndarray) -> None:
        if np.isfinite(value).all():
            return
        bad_indices = np.argwhere(~np.isfinite(value))
        first_index = tuple(int(i) for i in bad_indices[0].tolist())
        bad_count = int(bad_indices.shape[0])
        raise ValueError(
            "Non-finite motion data "
            f"field={field_name} file={motion_file} "
            f"first_index={first_index} count={bad_count}"
        )

    def load_motion_ids(self, motion_ids: torch.Tensor) -> LargeDatasetMotionBuffer:
        loaded = self.load_motion_chunks(motion_ids)
        length_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),
                loaded["file_lengths"][:-1].cumsum(dim=0),
            ]
        )
        return LargeDatasetMotionBuffer(
            global_motion_ids=loaded["global_motion_ids"],
            file_lengths=loaded["file_lengths"],
            length_starts=length_starts,
            fps=self.fps,
            joint_pos=torch.cat(loaded["joint_pos"], dim=0),
            joint_vel=torch.cat(loaded["joint_vel"], dim=0),
            body_pos_w=torch.cat(loaded["body_pos_w"], dim=0),
            body_quat_w=torch.cat(loaded["body_quat_w"], dim=0),
            body_lin_vel_w=torch.cat(loaded["body_lin_vel_w"], dim=0),
            body_ang_vel_w=torch.cat(loaded["body_ang_vel_w"], dim=0),
        )

    def load_slot_buffer(self, motion_ids: torch.Tensor) -> LargeDatasetMotionSlotBuffer:
        loaded = self.load_motion_chunks(motion_ids)
        chunks = {field_name: loaded[field_name] for field_name in self._FIELD_NAMES}
        return LargeDatasetMotionSlotBuffer(
            global_motion_ids=loaded["global_motion_ids"],
            chunks=chunks,
            file_lengths=loaded["file_lengths"],
            fps=self.fps,
        )

    def load_motion_chunks(self, motion_ids: torch.Tensor) -> dict[str, object]:
        start = time.perf_counter()
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if motion_ids.ndim != 1:
            motion_ids = motion_ids.reshape(-1)
        if motion_ids.numel() == 0:
            raise ValueError("motion_ids cannot be empty")
        if motion_ids.min() < 0 or motion_ids.max() >= self.num_files:
            raise IndexError("Motion id is outside the full dataset range")
        should_log = motion_ids.numel() >= 100
        try:
            progress_interval = max(
                int(os.environ.get("SP_TRACKING_MOTION_LOAD_LOG_INTERVAL", "100")),
                1,
            )
        except ValueError:
            progress_interval = 100
        motion_id_list = motion_ids.detach().cpu().tolist()
        requested_file_lengths = self.file_lengths[motion_ids].detach().cpu().tolist()
        requested_total_frames = int(sum(requested_file_lengths))
        if should_log:
            first_motion_id = int(motion_id_list[0])
            _bootstrap_debug(
                f"load_motion_chunks start count={int(motion_ids.numel())} "
                f"total_frames={requested_total_frames} "
                f"progress_interval={progress_interval} "
                f"first_ids={motion_ids[:5].detach().cpu().tolist()} "
                f"first_file={self.motion_files[first_motion_id]}",
                stdout=True,
            )

        loaded: dict[str, object] = {
            "global_motion_ids": motion_ids.clone(),
            "file_lengths": self.file_lengths[motion_ids].clone(),
        }
        for field_name in self._FIELD_NAMES:
            loaded[field_name] = []

        loaded_frames = 0
        for offset, motion_id in enumerate(motion_id_list):
            if should_log and offset % progress_interval == 0:
                _bootstrap_debug(
                    f"load_motion_chunks loading {offset + 1}/{len(motion_id_list)} "
                    f"elapsed={time.perf_counter() - start:.3f}s "
                    f"file={self.motion_files[motion_id]}",
                    stdout=True,
                )
            fields = self._load_one_motion(motion_id)
            for field_name in self._FIELD_NAMES:
                loaded[field_name].append(fields[field_name])
            loaded_frames += int(requested_file_lengths[offset])
            completed_count = offset + 1
            if should_log and (
                completed_count % progress_interval == 0 or completed_count == len(motion_id_list)
            ):
                elapsed = time.perf_counter() - start
                files_per_s = completed_count / max(elapsed, 1.0e-9)
                _bootstrap_debug(
                    f"load_motion_chunks progress {completed_count}/{len(motion_id_list)} "
                    f"frames={loaded_frames}/{requested_total_frames} "
                    f"elapsed={elapsed:.3f}s files_per_s={files_per_s:.1f} "
                    f"file={self.motion_files[motion_id]}",
                    stdout=True,
                )
        if should_log:
            allocated = (
                torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            )
            reserved = torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
            _bootstrap_debug(
                f"load_motion_chunks done count={int(motion_ids.numel())} "
                f"total_frames={requested_total_frames} "
                f"elapsed={time.perf_counter() - start:.3f}s "
                f"cuda_allocated={allocated} cuda_reserved={reserved}"
                f" device={self.device}",
                stdout=True,
            )
        return loaded

    def _load_one_motion(self, motion_id: int) -> dict[str, torch.Tensor]:
        motion_file = self.motion_files[motion_id]
        with np.load(motion_file) as data:
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        raw_fields = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
        }
        for field_name, value in raw_fields.items():
            self._validate_motion_field(motion_file, field_name, value)

        if self._joint_reindex is not None:
            joint_pos = joint_pos[:, self._joint_reindex]
            joint_vel = joint_vel[:, self._joint_reindex]
        if self._body_reindex is not None:
            body_pos_w = body_pos_w[:, self._body_reindex, :]
            body_quat_w = body_quat_w[:, self._body_reindex, :]
            body_lin_vel_w = body_lin_vel_w[:, self._body_reindex, :]
            body_ang_vel_w = body_ang_vel_w[:, self._body_reindex, :]

        joint_pos_t = torch.as_tensor(joint_pos, dtype=torch.float32, device=self.device)
        joint_vel_t = torch.as_tensor(joint_vel, dtype=torch.float32, device=self.device)
        joint_vel_t = _select_or_recompute_joint_vel(
            joint_pos=joint_pos_t,
            joint_vel=joint_vel_t,
            fps=self.fps_list[motion_id],
            recompute_joint_vel_from_joint_pos=self.recompute_joint_vel_from_joint_pos,
        )
        body_pos_w_t = torch.as_tensor(body_pos_w, dtype=torch.float32, device=self.device)
        body_quat_w_t = torch.as_tensor(body_quat_w, dtype=torch.float32, device=self.device)
        body_lin_vel_w_t = torch.as_tensor(body_lin_vel_w, dtype=torch.float32, device=self.device)
        body_ang_vel_w_t = torch.as_tensor(body_ang_vel_w, dtype=torch.float32, device=self.device)
        body_pos_w_t, body_quat_w_t, body_lin_vel_w_t, body_ang_vel_w_t = _select_or_fk_body_fields(
            joint_pos=joint_pos_t,
            body_pos_w=body_pos_w_t,
            body_quat_w=body_quat_w_t,
            body_lin_vel_w=body_lin_vel_w_t,
            body_ang_vel_w=body_ang_vel_w_t,
            body_indexes=self._body_indexes,
            fps=self.fps_list[motion_id],
            fk_from_joint_pos=self.fk_from_joint_pos,
            fk_helper=self.fk_helper,
        )

        return {
            "joint_pos": joint_pos_t,
            "joint_vel": joint_vel_t,
            "body_pos_w": body_pos_w_t,
            "body_quat_w": body_quat_w_t,
            "body_lin_vel_w": body_lin_vel_w_t,
            "body_ang_vel_w": body_ang_vel_w_t,
        }


class GlobalAdaptiveBinPool:
    """Global full-dataset adaptive statistics with deferred distributed sync."""

    def __init__(
        self,
        file_lengths: torch.Tensor,
        *,
        bin_width_steps: int,
        prior_visit_count: float,
        prior_failure_count: float,
        device: str | torch.device,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.file_lengths = torch.as_tensor(file_lengths, dtype=torch.long, device=self.device)
        self.num_files = int(self.file_lengths.numel())
        self.bin_width_steps = max(int(bin_width_steps), 1)
        self.bin_count = int(self.file_lengths.max().item() // self.bin_width_steps) + 1
        resolved_rank, resolved_world_size = self._resolve_distributed_context()
        self.rank = resolved_rank if rank is None else int(rank)
        self.world_size = resolved_world_size if world_size is None else max(int(world_size), 1)
        self.is_sharded = self.world_size > 1

        self.motion_bin_counts = torch.clamp(
            torch.div(
                self.file_lengths + self.bin_width_steps - 1,
                self.bin_width_steps,
                rounding_mode="floor",
            ),
            min=1,
        )
        self.num_valid_motion_bins = max(int(self.motion_bin_counts.sum().item()), 1)
        self.mean_bin_length = torch.clamp(
            self.file_lengths.float().sum() / float(self.num_valid_motion_bins),
            min=1.0,
        )
        if self.is_sharded:
            self.bin_valid_mask = None
            self.valid_motion_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self.valid_bin_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self.bin_lengths = None
            self.base_bin_weights = None
        else:
            self.bin_valid_mask = self._bin_valid_mask_for(
                torch.arange(self.num_files, dtype=torch.long, device=self.device)
            )
            self.valid_motion_ids, self.valid_bin_ids = torch.where(self.bin_valid_mask)
            self.bin_lengths = self._bin_lengths_for_rows(
                torch.arange(self.num_files, dtype=torch.long, device=self.device)
            )
            self.base_bin_weights = self.bin_lengths.float() / self.mean_bin_length
            self.base_bin_weights.masked_fill_(~self.bin_valid_mask, 0.0)

        self.prior_visit_count = max(float(prior_visit_count), 0.0)
        self.prior_failure_count = max(float(prior_failure_count), 0.0)
        if self.prior_failure_count > self.prior_visit_count:
            raise ValueError("prior_failure_count cannot exceed prior_visit_count")
        self.failure_rate_ema_iterations: int | None = None
        self.owned_motion_ids = torch.arange(
            self.rank, self.num_files, self.world_size, dtype=torch.long, device=self.device
        )
        owned_valid_mask = self._bin_valid_mask_for(self.owned_motion_ids)
        self.bin_visit_count = torch.full(
            (int(self.owned_motion_ids.numel()), self.bin_count),
            self.prior_visit_count,
            dtype=torch.float,
            device=self.device,
        )
        self.bin_failure_count = torch.full_like(self.bin_visit_count, self.prior_failure_count)
        self.bin_visit_count.masked_fill_(~owned_valid_mask, 0.0)
        self.bin_failure_count.masked_fill_(~owned_valid_mask, 0.0)
        self.pending_visit_delta = torch.zeros_like(self.bin_visit_count)
        self.pending_failure_delta = torch.zeros_like(self.bin_failure_count)
        self.last_visit_delta = torch.zeros_like(self.bin_visit_count)
        self.last_failure_delta = torch.zeros_like(self.bin_failure_count)
        self.active_motion_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.active_motion_to_slot = torch.full(
            (self.num_files,), -1, dtype=torch.long, device=self.device
        )
        self.active_visit_count = torch.empty(
            (0, self.bin_count), dtype=torch.float, device=self.device
        )
        self.active_failure_count = torch.empty_like(self.active_visit_count)
        self.active_bin_valid_mask = torch.empty(
            (0, self.bin_count), dtype=torch.bool, device=self.device
        )
        self.active_bin_lengths = torch.empty(
            (0, self.bin_count), dtype=torch.long, device=self.device
        )
        self._pending_sparse_keys: list[torch.Tensor] = []
        self._pending_sparse_visit_values: list[torch.Tensor] = []
        self._pending_sparse_failure_keys: list[torch.Tensor] = []
        self._pending_sparse_failure_values: list[torch.Tensor] = []
        self._last_timing_stats = self._empty_timing_stats()

    @staticmethod
    def _empty_timing_stats() -> dict[str, float]:
        return {
            "global_bin_update_time": 0.0,
            "global_bin_update_pack_time": 0.0,
            "global_bin_update_gather_time": 0.0,
            "global_bin_update_apply_time": 0.0,
            "global_bin_update_visit_key_count": 0.0,
            "global_bin_update_failure_key_count": 0.0,
            "adaptive_bin_pool_reset_time": 0.0,
            "adaptive_bin_pool_reset_applied": 0.0,
        }

    def get_timing_stats(self) -> dict[str, float]:
        return dict(self._last_timing_stats)

    @staticmethod
    def _resolve_distributed_context() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
        return rank, max(world_size, 1)

    def _bin_valid_mask_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
        bin_indices = torch.arange(self.bin_count, device=self.device)
        return bin_indices.unsqueeze(0) < self.motion_bin_counts[motion_ids].unsqueeze(1)

    def _bin_lengths_for_rows(self, motion_ids: torch.Tensor) -> torch.Tensor:
        bin_indices = torch.arange(self.bin_count, device=self.device)
        bin_starts = bin_indices.unsqueeze(0) * self.bin_width_steps
        remaining_lengths = (self.file_lengths[motion_ids].unsqueeze(1) - bin_starts).clamp(min=0)
        bin_lengths = torch.minimum(
            remaining_lengths,
            torch.full_like(remaining_lengths, self.bin_width_steps),
        )
        return bin_lengths.masked_fill(~self._bin_valid_mask_for(motion_ids), 0)

    def _bin_lengths_for_pairs(
        self, motion_ids: torch.Tensor, bin_ids: torch.Tensor
    ) -> torch.Tensor:
        remaining_lengths = (
            self.file_lengths[motion_ids] - bin_ids.to(dtype=torch.long) * self.bin_width_steps
        ).clamp(min=0)
        return torch.minimum(
            remaining_lengths,
            torch.full_like(remaining_lengths, self.bin_width_steps),
        )

    def compute_motion_bin_indices(
        self, time_steps: torch.Tensor, motion_ids: torch.Tensor
    ) -> torch.Tensor:
        raw_bin_indices = torch.div(time_steps, self.bin_width_steps, rounding_mode="floor")
        max_bin_indices = self.motion_bin_counts[motion_ids] - 1
        return torch.minimum(raw_bin_indices, max_bin_indices)

    def compute_failure_rate(self) -> torch.Tensor:
        if self.is_sharded:
            failure_rate = self.active_failure_count / torch.clamp(
                self.active_visit_count, min=1e-12
            )
            return failure_rate.clamp_(0.0, 1.0).masked_fill(~self.active_bin_valid_mask, 0.0)
        failure_rate = self.bin_failure_count / torch.clamp(self.bin_visit_count, min=1e-12)
        return failure_rate.clamp_(0.0, 1.0).masked_fill(~self.bin_valid_mask, 0.0)

    def apply_ema_decay(self, decay: float) -> None:
        """Decay observed statistics while keeping initialization priors persistent."""
        decay = float(max(0.0, min(1.0, decay)))
        owned_valid_mask = self._bin_valid_mask_for(self.owned_motion_ids)
        self.bin_visit_count.sub_(self.prior_visit_count).mul_(decay).add_(self.prior_visit_count)
        self.bin_failure_count.sub_(self.prior_failure_count).mul_(decay).add_(
            self.prior_failure_count
        )
        self.bin_visit_count.masked_fill_(~owned_valid_mask, 0.0)
        self.bin_failure_count.masked_fill_(~owned_valid_mask, 0.0)

        if self.is_sharded and self.active_motion_ids.numel() > 0:
            self.active_visit_count.sub_(self.prior_visit_count).mul_(decay).add_(
                self.prior_visit_count
            )
            self.active_failure_count.sub_(self.prior_failure_count).mul_(decay).add_(
                self.prior_failure_count
            )
            self.active_visit_count.masked_fill_(~self.active_bin_valid_mask, 0.0)
            self.active_failure_count.masked_fill_(~self.active_bin_valid_mask, 0.0)

    def set_active_motion_ids(self, active_motion_ids: torch.Tensor) -> None:
        active_motion_ids = torch.as_tensor(active_motion_ids, dtype=torch.long, device=self.device)
        if active_motion_ids.ndim != 1:
            active_motion_ids = active_motion_ids.reshape(-1)
        self.active_motion_ids = active_motion_ids.clone()
        self.active_motion_to_slot.fill_(-1)
        if active_motion_ids.numel() > 0:
            self.active_motion_to_slot[active_motion_ids] = torch.arange(
                active_motion_ids.numel(), dtype=torch.long, device=self.device
            )
        self.active_bin_valid_mask = self._bin_valid_mask_for(active_motion_ids)
        self.active_bin_lengths = self._bin_lengths_for_rows(active_motion_ids)
        visit_rows, failure_rows = self._fetch_motion_stat_rows(active_motion_ids)
        self.active_visit_count = visit_rows
        self.active_failure_count = failure_rows

    def replace_active_motion_ids(
        self, slot_ids: torch.Tensor, new_motion_ids: torch.Tensor
    ) -> None:
        slot_ids = torch.as_tensor(slot_ids, dtype=torch.long, device=self.device)
        new_motion_ids = torch.as_tensor(new_motion_ids, dtype=torch.long, device=self.device)
        if slot_ids.numel() == 0:
            if self.is_sharded and dist.is_available() and dist.is_initialized():
                self._fetch_motion_stat_rows(new_motion_ids)
            return
        old_motion_ids = self.active_motion_ids[slot_ids]
        self.active_motion_to_slot[old_motion_ids] = -1
        self.active_motion_ids[slot_ids] = new_motion_ids
        self.active_motion_to_slot[new_motion_ids] = slot_ids
        self.active_bin_valid_mask[slot_ids] = self._bin_valid_mask_for(new_motion_ids)
        self.active_bin_lengths[slot_ids] = self._bin_lengths_for_rows(new_motion_ids)
        visit_rows, failure_rows = self._fetch_motion_stat_rows(new_motion_ids)
        self.active_visit_count[slot_ids] = visit_rows
        self.active_failure_count[slot_ids] = failure_rows

    def _fetch_motion_stat_rows(
        self, motion_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        visit_rows = torch.full(
            (motion_ids.numel(), self.bin_count),
            self.prior_visit_count,
            dtype=torch.float,
            device=self.device,
        )
        failure_rows = torch.full_like(visit_rows, self.prior_failure_count)
        valid_mask = self._bin_valid_mask_for(motion_ids)
        visit_rows.masked_fill_(~valid_mask, 0.0)
        failure_rows.masked_fill_(~valid_mask, 0.0)

        local_mask = motion_ids % self.world_size == self.rank
        if local_mask.any():
            local_rows = motion_ids[local_mask] // self.world_size
            visit_rows[local_mask] = self.bin_visit_count[local_rows]
            failure_rows[local_mask] = self.bin_failure_count[local_rows]

        if self.is_sharded and dist.is_available() and dist.is_initialized():
            visit_rows, failure_rows = self._fetch_motion_stat_rows_distributed(
                motion_ids, visit_rows, failure_rows
            )
        return visit_rows, failure_rows

    def _fetch_motion_stat_rows_distributed(
        self,
        motion_ids: torch.Tensor,
        visit_rows: torch.Tensor,
        failure_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_ids = motion_ids.detach().cpu().tolist()
        all_requests: list[list[int]] = [None for _ in range(self.world_size)]  # type: ignore[list-item]
        dist.all_gather_object(all_requests, request_ids)

        responses: dict[int, tuple[list[int], torch.Tensor, torch.Tensor]] = {}
        for requester_rank, requested_ids in enumerate(all_requests):
            owned_ids = [
                motion_id for motion_id in requested_ids if motion_id % self.world_size == self.rank
            ]
            if not owned_ids:
                responses[requester_rank] = (
                    [],
                    torch.empty((0, self.bin_count), dtype=torch.float),
                    torch.empty((0, self.bin_count), dtype=torch.float),
                )
                continue
            owned_tensor = torch.tensor(owned_ids, dtype=torch.long, device=self.device)
            local_rows = owned_tensor // self.world_size
            responses[requester_rank] = (
                owned_ids,
                self.bin_visit_count[local_rows].detach().cpu(),
                self.bin_failure_count[local_rows].detach().cpu(),
            )

        all_responses: list[dict[int, tuple[list[int], torch.Tensor, torch.Tensor]]] = [
            None for _ in range(self.world_size)
        ]  # type: ignore[list-item]
        dist.all_gather_object(all_responses, responses)

        slot_by_motion = {int(motion_id): index for index, motion_id in enumerate(request_ids)}
        for response_by_rank in all_responses:
            motion_list, visit_cpu, failure_cpu = response_by_rank.get(
                self.rank,
                (
                    [],
                    torch.empty((0, self.bin_count), dtype=torch.float),
                    torch.empty((0, self.bin_count), dtype=torch.float),
                ),
            )
            for offset, motion_id in enumerate(motion_list):
                slot = slot_by_motion[int(motion_id)]
                visit_rows[slot] = visit_cpu[offset].to(self.device)
                failure_rows[slot] = failure_cpu[offset].to(self.device)
        return visit_rows, failure_rows

    def record_completed_visits(
        self,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
        failure_mask: torch.Tensor | None,
    ) -> None:
        if motion_ids.numel() == 0:
            return
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        time_steps = torch.as_tensor(time_steps, dtype=torch.long, device=self.device)
        current_bin_indices = self.compute_motion_bin_indices(time_steps, motion_ids)
        linear_indices = motion_ids * self.bin_count + current_bin_indices
        visit_values = torch.ones(linear_indices.shape, dtype=torch.float, device=self.device)
        if self.is_sharded:
            self._pending_sparse_keys.append(linear_indices.detach())
            self._pending_sparse_visit_values.append(visit_values.detach())
        else:
            self.pending_visit_delta.view(-1).index_add_(
                0, linear_indices, visit_values.to(dtype=self.pending_visit_delta.dtype)
            )

        if failure_mask is None:
            return
        failure_mask = torch.as_tensor(failure_mask, dtype=torch.bool, device=self.device)
        failed_linear_indices = linear_indices[failure_mask]
        failure_values = torch.ones(
            failed_linear_indices.shape,
            dtype=self.pending_failure_delta.dtype,
            device=self.device,
        )
        if self.is_sharded:
            self._pending_sparse_failure_keys.append(failed_linear_indices.detach())
            self._pending_sparse_failure_values.append(failure_values.detach())
        else:
            self.pending_failure_delta.view(-1).index_add_(0, failed_linear_indices, failure_values)

    def synchronize(self) -> float:
        start = time.perf_counter()
        self._last_timing_stats = self._empty_timing_stats()
        if self.is_sharded:
            pack_start = time.perf_counter()
            local_update = self._consume_local_sparse_update()
            pack_time = time.perf_counter() - pack_start
            gather_start = time.perf_counter()
            gathered_updates = self._gather_sparse_updates(local_update)
            gather_time = time.perf_counter() - gather_start
            apply_start = time.perf_counter()
            visit_delta, failure_delta = self._apply_sparse_updates(gathered_updates)
            apply_time = time.perf_counter() - apply_start
            self.last_visit_delta.copy_(visit_delta)
            self.last_failure_delta.copy_(failure_delta)
            elapsed = time.perf_counter() - start
            self._last_timing_stats.update(
                {
                    "global_bin_update_time": float(elapsed),
                    "global_bin_update_pack_time": float(pack_time),
                    "global_bin_update_gather_time": float(gather_time),
                    "global_bin_update_apply_time": float(apply_time),
                    "global_bin_update_visit_key_count": float(
                        sum(update["visit_keys"].numel() for update in gathered_updates)
                    ),
                    "global_bin_update_failure_key_count": float(
                        sum(update["failure_keys"].numel() for update in gathered_updates)
                    ),
                }
            )
            return elapsed

        pack_start = time.perf_counter()
        visit_delta = self.pending_visit_delta.clone()
        failure_delta = self.pending_failure_delta.clone()
        pack_time = time.perf_counter() - pack_start
        gather_start = time.perf_counter()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(visit_delta, op=dist.ReduceOp.SUM)
            dist.all_reduce(failure_delta, op=dist.ReduceOp.SUM)
        gather_time = time.perf_counter() - gather_start
        apply_start = time.perf_counter()
        self.bin_visit_count += visit_delta
        self.bin_failure_count += failure_delta
        self.last_visit_delta.copy_(visit_delta)
        self.last_failure_delta.copy_(failure_delta)
        self.pending_visit_delta.zero_()
        self.pending_failure_delta.zero_()
        apply_time = time.perf_counter() - apply_start
        elapsed = time.perf_counter() - start
        self._last_timing_stats.update(
            {
                "global_bin_update_time": float(elapsed),
                "global_bin_update_pack_time": float(pack_time),
                "global_bin_update_gather_time": float(gather_time),
                "global_bin_update_apply_time": float(apply_time),
            }
        )
        return elapsed

    def reset_counts_if_due(
        self,
        *,
        iteration: int,
        interval_iterations: int,
    ) -> float:
        self._last_timing_stats["adaptive_bin_pool_reset_time"] = 0.0
        self._last_timing_stats["adaptive_bin_pool_reset_applied"] = 0.0
        interval_iterations = int(interval_iterations)
        if interval_iterations <= 0 or int(iteration) <= 0:
            return 0.0
        if int(iteration) % interval_iterations != 0:
            return 0.0

        start = time.perf_counter()
        self._reset_count_rows()
        if self.active_motion_ids.numel() > 0:
            self.set_active_motion_ids(self.active_motion_ids)
        elapsed = time.perf_counter() - start
        self._last_timing_stats["adaptive_bin_pool_reset_time"] = float(elapsed)
        self._last_timing_stats["adaptive_bin_pool_reset_applied"] = 1.0
        return elapsed

    def _reset_count_rows(self) -> None:
        valid_mask = self._bin_valid_mask_for(self.owned_motion_ids)
        self.bin_visit_count.fill_(self.prior_visit_count)
        self.bin_failure_count.fill_(self.prior_failure_count)
        self.bin_visit_count.masked_fill_(~valid_mask, 0.0)
        self.bin_failure_count.masked_fill_(~valid_mask, 0.0)
        self.pending_visit_delta.zero_()
        self.pending_failure_delta.zero_()
        self.last_visit_delta.zero_()
        self.last_failure_delta.zero_()
        self._pending_sparse_keys.clear()
        self._pending_sparse_visit_values.clear()
        self._pending_sparse_failure_keys.clear()
        self._pending_sparse_failure_values.clear()

    def _consume_local_sparse_update(self) -> dict[str, torch.Tensor]:
        if self._pending_sparse_keys:
            visit_keys = torch.cat(self._pending_sparse_keys)
            visit_values = torch.cat(self._pending_sparse_visit_values)
            unique_visit_keys, inverse = torch.unique(visit_keys, return_inverse=True)
            unique_visit_values = torch.zeros(
                unique_visit_keys.shape, dtype=torch.float, device=self.device
            )
            unique_visit_values.scatter_add_(0, inverse, visit_values)
        else:
            unique_visit_keys = torch.empty(0, dtype=torch.long, device=self.device)
            unique_visit_values = torch.empty(0, dtype=torch.float, device=self.device)

        if self._pending_sparse_failure_keys:
            failure_keys = torch.cat(self._pending_sparse_failure_keys)
            failure_values = torch.cat(self._pending_sparse_failure_values)
            unique_failure_keys, inverse = torch.unique(failure_keys, return_inverse=True)
            unique_failure_values = torch.zeros(
                unique_failure_keys.shape, dtype=torch.float, device=self.device
            )
            unique_failure_values.scatter_add_(0, inverse, failure_values)
        else:
            unique_failure_keys = torch.empty(0, dtype=torch.long, device=self.device)
            unique_failure_values = torch.empty(0, dtype=torch.float, device=self.device)

        self._pending_sparse_keys.clear()
        self._pending_sparse_visit_values.clear()
        self._pending_sparse_failure_keys.clear()
        self._pending_sparse_failure_values.clear()
        return {
            "visit_keys": unique_visit_keys.detach().cpu(),
            "visit_values": unique_visit_values.detach().cpu(),
            "failure_keys": unique_failure_keys.detach().cpu(),
            "failure_values": unique_failure_values.detach().cpu(),
        }

    def _gather_sparse_updates(
        self, local_update: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        if dist.is_available() and dist.is_initialized():
            gathered_updates: list[dict[str, torch.Tensor]] = [None for _ in range(self.world_size)]  # type: ignore[list-item]
            dist.all_gather_object(gathered_updates, local_update)
            return gathered_updates
        return [local_update]

    def _apply_sparse_updates(
        self, gathered_updates: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visit_delta = torch.zeros_like(self.bin_visit_count)
        failure_delta = torch.zeros_like(self.bin_failure_count)
        for update in gathered_updates:
            self._apply_sparse_update_tensor(
                update["visit_keys"].to(self.device),
                update["visit_values"].to(self.device),
                visit_delta,
                self.bin_visit_count,
                self.active_visit_count,
            )
            self._apply_sparse_update_tensor(
                update["failure_keys"].to(self.device),
                update["failure_values"].to(self.device),
                failure_delta,
                self.bin_failure_count,
                self.active_failure_count,
            )
        return visit_delta, failure_delta

    def _apply_sparse_update_tensor(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        shard_delta: torch.Tensor,
        shard_target: torch.Tensor,
        active_target: torch.Tensor,
    ) -> None:
        if keys.numel() == 0:
            return
        motion_ids = keys // self.bin_count
        bin_ids = keys % self.bin_count

        owner_mask = motion_ids % self.world_size == self.rank
        if owner_mask.any():
            local_rows = motion_ids[owner_mask] // self.world_size
            flat_indices = local_rows * self.bin_count + bin_ids[owner_mask]
            shard_delta.view(-1).index_add_(0, flat_indices, values[owner_mask])
            shard_target.view(-1).index_add_(0, flat_indices, values[owner_mask])

        if self.active_motion_ids.numel() == 0:
            return
        active_slots = self.active_motion_to_slot[motion_ids]
        active_mask = active_slots >= 0
        if active_mask.any():
            flat_active_indices = active_slots[active_mask] * self.bin_count + bin_ids[active_mask]
            active_target.view(-1).index_add_(
                0, flat_active_indices, values[active_mask].to(active_target.dtype)
            )

    def compute_active_pair_sampling_probabilities(
        self,
        active_motion_ids: torch.Tensor,
        *,
        adaptive_uniform_ratio: float,
        adaptive_probability_max_over_mean: float,
        adaptive_sequence_length_agnostic: bool,
        adaptive_max_prob_per_bin: float | Literal["auto"] | None = "auto",
        adaptive_max_prob_per_motion: float | Literal["auto"] | None = "auto",
        adaptive_temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active_motion_ids = torch.as_tensor(active_motion_ids, dtype=torch.long, device=self.device)
        if self.is_sharded:
            if not torch.equal(active_motion_ids, self.active_motion_ids):
                self.set_active_motion_ids(active_motion_ids)
            active_row_ids, valid_bin_ids = torch.where(self.active_bin_valid_mask)
            valid_motion_ids = self.active_motion_ids[active_row_ids]
            if valid_motion_ids.numel() == 0:
                raise RuntimeError("Active subset contains no valid motion bins")
            probabilities, valid_failure_rate = self._compute_active_pair_probabilities_from_cache(
                active_row_ids,
                valid_motion_ids,
                valid_bin_ids,
                num_active_motions=int(active_motion_ids.numel()),
                adaptive_uniform_ratio=adaptive_uniform_ratio,
                adaptive_temperature=adaptive_temperature,
                adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
                adaptive_max_prob_per_bin=adaptive_max_prob_per_bin,
                adaptive_max_prob_per_motion=adaptive_max_prob_per_motion,
                auto_cap_over_mean=adaptive_probability_max_over_mean,
            )
            return valid_motion_ids, valid_bin_ids, probabilities, valid_failure_rate

        active_bin_mask = self.bin_valid_mask[active_motion_ids]
        active_row_ids, valid_bin_ids = torch.where(active_bin_mask)
        valid_motion_ids = active_motion_ids[active_row_ids]
        if valid_motion_ids.numel() == 0:
            raise RuntimeError("Active subset contains no valid motion bins")

        probabilities, valid_failure_rate = self._compute_pair_probabilities(
            valid_motion_ids,
            valid_bin_ids,
            num_motions=int(active_motion_ids.numel()),
            adaptive_uniform_ratio=adaptive_uniform_ratio,
            adaptive_temperature=adaptive_temperature,
            adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
            adaptive_max_prob_per_bin=adaptive_max_prob_per_bin,
            adaptive_max_prob_per_motion=adaptive_max_prob_per_motion,
            auto_cap_over_mean=adaptive_probability_max_over_mean,
        )
        return valid_motion_ids, valid_bin_ids, probabilities, valid_failure_rate

    def compute_motion_sampling_probabilities(
        self,
        candidate_motion_ids: torch.Tensor,
        *,
        adaptive_uniform_ratio: float,
        adaptive_probability_max_over_mean: float,
        adaptive_sequence_length_agnostic: bool,
        adaptive_temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_sharded:
            candidate_motion_ids = torch.as_tensor(
                candidate_motion_ids, dtype=torch.long, device=self.device
            )
            if candidate_motion_ids.ndim != 1:
                candidate_motion_ids = candidate_motion_ids.reshape(-1)
            candidate_valid_mask = self._bin_valid_mask_for(candidate_motion_ids)
            candidate_row_ids, valid_bin_ids = torch.where(candidate_valid_mask)
            valid_motion_ids = candidate_motion_ids[candidate_row_ids]
            if valid_motion_ids.numel() == 0:
                probabilities = torch.full(
                    (candidate_motion_ids.numel(),),
                    1.0 / float(max(candidate_motion_ids.numel(), 1)),
                    dtype=torch.float,
                    device=self.device,
                )
                return candidate_motion_ids, probabilities
            visit_rows, failure_rows = self._fetch_motion_stat_rows(candidate_motion_ids)
            probabilities, _ = self._compute_pair_probabilities_from_rows(
                candidate_row_ids,
                valid_motion_ids,
                valid_bin_ids,
                visit_rows,
                failure_rows,
                num_rows=int(candidate_motion_ids.numel()),
                adaptive_uniform_ratio=adaptive_uniform_ratio,
                adaptive_temperature=adaptive_temperature,
                adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
                adaptive_max_prob_per_bin=None,
                adaptive_max_prob_per_motion="auto",
                auto_cap_over_mean=adaptive_probability_max_over_mean,
            )
            motion_probabilities = torch.zeros(
                candidate_motion_ids.numel(), dtype=probabilities.dtype, device=self.device
            )
            motion_probabilities.scatter_add_(0, candidate_row_ids, probabilities)
            motion_probabilities = motion_probabilities / torch.clamp(
                motion_probabilities.sum(), min=1e-12
            )
            return candidate_motion_ids, motion_probabilities

        motion_ids, _, pair_probabilities, _ = self.compute_active_pair_sampling_probabilities(
            candidate_motion_ids,
            adaptive_uniform_ratio=adaptive_uniform_ratio,
            adaptive_probability_max_over_mean=adaptive_probability_max_over_mean,
            adaptive_temperature=adaptive_temperature,
            adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
            adaptive_max_prob_per_bin=None,
            adaptive_max_prob_per_motion="auto",
        )
        motion_probabilities = torch.zeros(
            self.num_files, dtype=pair_probabilities.dtype, device=self.device
        )
        motion_probabilities.scatter_add_(0, motion_ids, pair_probabilities)
        candidate_probabilities = motion_probabilities[candidate_motion_ids]
        candidate_probabilities = candidate_probabilities / torch.clamp(
            candidate_probabilities.sum(), min=1e-12
        )
        return candidate_motion_ids, candidate_probabilities

    def _compute_active_pair_probabilities_from_cache(
        self,
        active_row_ids: torch.Tensor,
        valid_motion_ids: torch.Tensor,
        valid_bin_ids: torch.Tensor,
        *,
        num_active_motions: int,
        adaptive_uniform_ratio: float,
        adaptive_temperature: float | None,
        adaptive_sequence_length_agnostic: bool,
        adaptive_max_prob_per_bin: float | Literal["auto"] | None,
        adaptive_max_prob_per_motion: float | Literal["auto"] | None,
        auto_cap_over_mean: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compute_pair_probabilities_from_rows(
            active_row_ids,
            valid_motion_ids,
            valid_bin_ids,
            self.active_visit_count,
            self.active_failure_count,
            num_rows=num_active_motions,
            adaptive_uniform_ratio=adaptive_uniform_ratio,
            adaptive_temperature=adaptive_temperature,
            adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
            adaptive_max_prob_per_bin=adaptive_max_prob_per_bin,
            adaptive_max_prob_per_motion=adaptive_max_prob_per_motion,
            auto_cap_over_mean=auto_cap_over_mean,
        )

    def _compute_pair_probabilities_from_rows(
        self,
        row_ids: torch.Tensor,
        valid_motion_ids: torch.Tensor,
        valid_bin_ids: torch.Tensor,
        visit_rows: torch.Tensor,
        failure_rows: torch.Tensor,
        *,
        num_rows: int,
        adaptive_uniform_ratio: float,
        adaptive_temperature: float | None,
        adaptive_sequence_length_agnostic: bool,
        adaptive_max_prob_per_bin: float | Literal["auto"] | None,
        adaptive_max_prob_per_motion: float | Literal["auto"] | None,
        auto_cap_over_mean: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        failure_rate = failure_rows / torch.clamp(visit_rows, min=1e-12)
        valid_failure_rate = failure_rate[row_ids, valid_bin_ids]
        temperature_scaled_failure_rate = _temperature_scale_adaptive_signal(
            valid_failure_rate, adaptive_temperature
        )
        uniform_probabilities = torch.full(
            (len(valid_motion_ids),),
            1.0 / float(max(len(valid_motion_ids), 1)),
            dtype=torch.float,
            device=self.device,
        )
        failure_bin_weights = self._compute_bin_weights_for_pairs(
            valid_motion_ids, valid_bin_ids, adaptive_sequence_length_agnostic
        )
        failure_scores = temperature_scaled_failure_rate * failure_bin_weights
        failure_score_sum = failure_scores.sum()
        failure_based_probabilities = (
            failure_bin_weights / torch.clamp(failure_bin_weights.sum(), min=1e-12)
            if failure_score_sum <= 0.0
            else failure_scores / failure_score_sum
        )
        uniform_ratio = float(max(0.0, min(1.0, adaptive_uniform_ratio)))
        probabilities = (
            1.0 - uniform_ratio
        ) * failure_based_probabilities + uniform_ratio * uniform_probabilities
        probabilities = _apply_final_probability_caps(
            probabilities,
            row_ids,
            num_motions=num_rows,
            max_prob_per_bin=adaptive_max_prob_per_bin,
            max_prob_per_motion=adaptive_max_prob_per_motion,
            auto_cap_over_mean=auto_cap_over_mean,
        )
        return probabilities, valid_failure_rate

    def _compute_pair_probabilities(
        self,
        valid_motion_ids: torch.Tensor,
        valid_bin_ids: torch.Tensor,
        *,
        num_motions: int,
        adaptive_uniform_ratio: float,
        adaptive_temperature: float | None,
        adaptive_sequence_length_agnostic: bool,
        adaptive_max_prob_per_bin: float | Literal["auto"] | None,
        adaptive_max_prob_per_motion: float | Literal["auto"] | None,
        auto_cap_over_mean: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        failure_rate = self.compute_failure_rate()
        valid_failure_rate = failure_rate[valid_motion_ids, valid_bin_ids]
        temperature_scaled_failure_rate = _temperature_scale_adaptive_signal(
            valid_failure_rate, adaptive_temperature
        )
        uniform_probabilities = torch.full(
            (len(valid_motion_ids),),
            1.0 / float(max(len(valid_motion_ids), 1)),
            dtype=torch.float,
            device=self.device,
        )
        bin_weights = self._compute_bin_weights(adaptive_sequence_length_agnostic)
        failure_scores = (
            temperature_scaled_failure_rate * bin_weights[valid_motion_ids, valid_bin_ids]
        )
        failure_score_sum = failure_scores.sum()
        fallback_bin_weights = bin_weights[valid_motion_ids, valid_bin_ids]
        failure_based_probabilities = (
            fallback_bin_weights / torch.clamp(fallback_bin_weights.sum(), min=1e-12)
            if failure_score_sum <= 0.0
            else failure_scores / failure_score_sum
        )
        uniform_ratio = float(max(0.0, min(1.0, adaptive_uniform_ratio)))
        probabilities = (
            1.0 - uniform_ratio
        ) * failure_based_probabilities + uniform_ratio * uniform_probabilities
        probabilities = _apply_final_probability_caps(
            probabilities,
            valid_motion_ids,
            num_motions=num_motions,
            max_prob_per_bin=adaptive_max_prob_per_bin,
            max_prob_per_motion=adaptive_max_prob_per_motion,
            auto_cap_over_mean=auto_cap_over_mean,
        )
        return probabilities, valid_failure_rate

    def _compute_bin_weights(self, sequence_length_agnostic: bool) -> torch.Tensor:
        bin_weights = self.base_bin_weights
        if sequence_length_agnostic:
            bin_weights = bin_weights / self.motion_bin_counts.unsqueeze(1).float()
            bin_weights = bin_weights.masked_fill(~self.bin_valid_mask, 0.0)
        return bin_weights

    def _compute_bin_weights_for_pairs(
        self,
        motion_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        sequence_length_agnostic: bool,
    ) -> torch.Tensor:
        bin_lengths = self._bin_lengths_for_pairs(motion_ids, bin_ids).float()
        bin_weights = bin_lengths / self.mean_bin_length
        if sequence_length_agnostic:
            bin_weights = bin_weights / self.motion_bin_counts[motion_ids].float()
        return bin_weights

    def count_valid_bins(self, motion_ids: torch.Tensor) -> int:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if motion_ids.numel() == 0:
            return 0
        return int(self.motion_bin_counts[motion_ids].sum().item())


class LargeDatasetMultiMotionCommand(MultiMotionCommand):
    cfg: "LargeDatasetMultiMotionCommandCfg"

    def __init__(self, cfg: "LargeDatasetMultiMotionCommandCfg", env):
        _bootstrap_debug("LargeDatasetMultiMotionCommand init start")
        if getattr(cfg, "reference_storage_mode", REFERENCE_STORAGE_FULL) != REFERENCE_STORAGE_FULL:
            raise ValueError(
                "LargeDatasetMultiMotionCommand supports only reference_storage_mode=full"
            )
        CommandTerm.__init__(self, cfg, env)

        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        _bootstrap_debug("before resolve motion files")
        motion_files = self._resolve_all_motion_files()
        _bootstrap_debug(f"after resolve motion files count={len(motion_files)}")
        fk_helper = self._build_fk_helper()
        store_start = time.perf_counter()
        self.motion_store = LargeDatasetMotionStore(
            motion_files,
            self.body_indexes,
            motion_type=self.cfg.motion_type,
            device=self.device,
            metadata_cache_file=self._resolve_motion_metadata_cache_file(),
            metadata_cache_wait_timeout_s=self.cfg.motion_metadata_cache_wait_timeout_s,
            metadata_cache_poll_interval_s=self.cfg.motion_metadata_cache_poll_interval_s,
            metadata_read_workers=self.cfg.motion_metadata_read_workers,
            metadata_read_backend=self.cfg.motion_metadata_read_backend,
            metadata_read_chunksize=self.cfg.motion_metadata_read_chunksize,
            fk_from_joint_pos=self.cfg.fk_from_joint_pos,
            recompute_joint_vel_from_joint_pos=self.cfg.recompute_joint_vel_from_joint_pos,
            fk_helper=fk_helper,
        )
        self._global_motion_sampling_probabilities = _build_motion_group_probabilities(
            self.motion_store.motion_files,
            list(self.cfg.motion_sampling_groups),
            self.device,
        )
        _bootstrap_debug(
            f"after LargeDatasetMotionStore elapsed={time.perf_counter() - store_start:.3f}s"
        )
        subset_size = min(self.cfg.active_subset_size, self.motion_store.num_files)
        _bootstrap_debug(
            f"initial active subset sampling start subset_size={subset_size} "
            f"total_motion_count={self.motion_store.num_files}"
        )
        initial_motion_ids = self._sample_unique_motion_ids(
            torch.arange(self.motion_store.num_files, dtype=torch.long, device=self.device),
            subset_size,
            probabilities=self._global_motion_sampling_probabilities,
        )
        _bootstrap_debug(
            f"initial active subset sampled first_ids={initial_motion_ids[:5].detach().cpu().tolist()}"
        )
        self.active_subset = ActiveMotionSubset(
            total_motion_count=self.motion_store.num_files,
            subset_size=subset_size,
            min_resident_iterations=self.cfg.subset_min_resident_iterations,
            device=self.device,
        )
        self.active_subset.initialize(initial_motion_ids, iteration=0)
        load_start = time.perf_counter()
        _bootstrap_debug("before initial active subset load_slot_buffer")
        self.motion = self.motion_store.load_slot_buffer(self.active_subset.active_motion_ids)
        _bootstrap_debug(
            f"after initial active subset load_slot_buffer elapsed={time.perf_counter() - load_start:.3f}s"
        )

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initialize_sp_tracking_state()
        self._initialize_env_motion_assignments()

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.cfg.adaptive_bin_width_steps is not None:
            self.bin_width_steps = max(int(self.cfg.adaptive_bin_width_steps), 1)
        else:
            self.bin_width_steps = max(
                int(round(float(self.cfg.adaptive_bin_width_s) / env.step_dt)), 1
            )
        bin_pool_start = time.perf_counter()
        _bootstrap_debug("before GlobalAdaptiveBinPool")
        (
            self.adaptive_prior_visit_count,
            self.adaptive_prior_failure_count,
        ) = _resolve_adaptive_prior_counts(self.cfg)
        self.global_bin_pool = GlobalAdaptiveBinPool(
            self.motion_store.file_lengths,
            bin_width_steps=self.bin_width_steps,
            prior_visit_count=self.adaptive_prior_visit_count,
            prior_failure_count=self.adaptive_prior_failure_count,
            device=self.device,
        )
        self.global_bin_pool.failure_rate_ema_iterations = _resolve_adaptive_ema_iterations(
            self.cfg
        )
        _bootstrap_debug(
            f"after GlobalAdaptiveBinPool elapsed={time.perf_counter() - bin_pool_start:.3f}s "
            f"bin_count={self.global_bin_pool.bin_count}"
        )
        self.global_bin_pool.set_active_motion_ids(self.active_subset.active_motion_ids)
        self._bind_global_bin_pool_tensors()
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

        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
        self._extra_reference_ghost_model = None
        self._extra_reference_ghost_color = np.array((1.0, 0.45, 0.1, 0.45), dtype=np.float32)
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
        self._last_global_bin_update_time = 0.0
        self._last_subset_update_time = 0.0
        self._motion_gather_time_accum = 0.0
        self._motion_gather_call_count = 0
        self._adaptive_bin_snapshot_writer = None
        self._adaptive_bin_snapshot_writer_key = None
        self._initialize_reference_cache()
        _bootstrap_debug("LargeDatasetMultiMotionCommand init done")

    def _resolve_all_motion_files(self) -> list[str]:
        motion_path = os.fspath(self.cfg.motion_path)
        motion_file = os.fspath(self.cfg.motion_file)
        if motion_path and motion_file:
            raise ValueError(
                "Provide either motion_path for multi-motion input or motion_file for a "
                "single motion, but not both."
            )

        if motion_path:
            self._validate_motion_path(motion_path)
            manifest_file = self._resolve_motion_manifest_file(motion_path)
            if manifest_file:
                resolved_motion_files = self._resolve_motion_files_with_manifest(
                    motion_path, manifest_file
                )
            else:
                resolved_motion_files = self._scan_motion_path(motion_path)
        elif motion_file:
            if not os.path.exists(motion_file):
                raise FileNotFoundError(f"Invalid motion file: {motion_file}")
            if not os.path.isfile(motion_file):
                raise ValueError(f"motion_file must point to a .npz file: {motion_file}")
            resolved_motion_files = [motion_file]
        else:
            resolved_motion_files = []

        resolved_motion_files = filter_excluded_motion_files(
            resolved_motion_files,
            motion_path=motion_path,
            excluded_motion_files=self.cfg.excluded_motion_files,
            motion_exclude_files=getattr(self.cfg, "motion_exclude_files", ()),
            motion_exclude_file=self.cfg.motion_exclude_file,
        )

        if len(resolved_motion_files) == 0:
            raise ValueError(
                "No motion files found. Provide either:\n"
                "  - motion_path: path to a directory containing .npz files\n"
                "  - motion_file: path to a single motion file"
            )
        return resolved_motion_files

    def _validate_motion_path(self, motion_path: str) -> None:
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"Invalid motion path: {motion_path}")
        if not os.path.isdir(motion_path):
            raise ValueError(
                f"motion_path must point to a directory containing .npz files: {motion_path}"
            )

    def _resolve_motion_manifest_file(self, motion_path: str) -> str:
        configured_manifest = os.fspath(getattr(self.cfg, "motion_manifest_file", ""))
        if configured_manifest:
            return configured_manifest

        _, world_size = self._runtime_rank_context()
        debug_dir = os.environ.get("MJLAB_BOOTSTRAP_DEBUG_DIR", "")
        if world_size <= 1 or not debug_dir:
            return ""

        motion_path_key = hashlib.sha1(os.path.abspath(motion_path).encode("utf-8")).hexdigest()[
            :12
        ]
        return os.path.join(debug_dir, f"motion_manifest_{motion_path_key}.txt")

    def _resolve_motion_metadata_cache_file(self) -> str:
        configured_cache = os.fspath(getattr(self.cfg, "motion_metadata_cache_file", ""))
        if configured_cache:
            return configured_cache
        configured_manifest = os.fspath(getattr(self.cfg, "motion_manifest_file", ""))
        if configured_manifest:
            return f"{configured_manifest}.metadata.json"
        return ""

    def _resolve_motion_files_with_manifest(
        self, motion_path: str, manifest_file: str
    ) -> list[str]:
        rank, world_size = self._runtime_rank_context()
        _bootstrap_debug(
            "resolve motion files with manifest "
            f"path={motion_path} manifest={manifest_file} rank={rank} world_size={world_size}",
            stdout=True,
        )

        if os.path.exists(manifest_file):
            motion_files = self._read_motion_manifest(manifest_file)
            _bootstrap_debug(
                f"read existing motion manifest count={len(motion_files)} file={manifest_file}",
                stdout=True,
            )
            return motion_files

        if world_size <= 1 or rank == 0:
            motion_files = self._scan_motion_path(motion_path)
            self._write_motion_manifest(manifest_file, motion_files)
            _bootstrap_debug(
                f"wrote motion manifest count={len(motion_files)} file={manifest_file}",
                stdout=True,
            )
            return motion_files

        return self._wait_for_motion_manifest(manifest_file)

    def _runtime_rank_context(self) -> tuple[int, int]:
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
        return rank, max(world_size, 1)

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
                    _bootstrap_debug(
                        "fd motion scan failed, falling back to python scanner "
                        f"path={motion_path} error={exc}",
                        stdout=True,
                    )
            elif backend == "fd":
                raise FileNotFoundError(
                    f"motion_scan_backend='fd' requested, but executable not found: {fd_executable}"
                )

        return self._scan_motion_path_with_python(motion_path)

    def _scan_motion_path_with_fd(self, motion_path: str, fd_path: str) -> list[str]:
        start = time.perf_counter()
        worker_count = int(getattr(self.cfg, "motion_scan_workers", 0))
        if worker_count < 0:
            raise ValueError("motion_scan_workers must be non-negative")

        cmd = [
            fd_path,
            "--hidden",
            "--no-ignore",
            "--type",
            "f",
            "--color",
            "never",
        ]
        if worker_count > 0:
            cmd.extend(["--threads", str(worker_count)])
        cmd.extend([r"(?i)\.npz$", motion_path])

        _bootstrap_debug(
            "scan motion path start "
            f"path={motion_path} backend=fd fd={fd_path} workers={worker_count}",
            stdout=True,
        )
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        resolved_motion_files: list[str] = []
        last_log_time = start
        log_interval = float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0))
        assert process.stdout is not None
        for line in process.stdout:
            motion_file = line.strip()
            if motion_file:
                resolved_motion_files.append(motion_file)
            now = time.perf_counter()
            if log_interval > 0.0 and now - last_log_time >= log_interval:
                _bootstrap_debug(
                    "scan motion path progress "
                    f"backend=fd motions={len(resolved_motion_files)} "
                    f"elapsed={now - start:.3f}s",
                    stdout=True,
                )
                last_log_time = now

        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, cmd, output="\n".join(resolved_motion_files), stderr=stderr
            )
        resolved_motion_files.sort()
        if stderr.strip():
            _bootstrap_debug(
                f"fd motion scan stderr path={motion_path} stderr={stderr.strip()}",
                stdout=True,
            )
        _bootstrap_debug(
            "scan motion path done "
            f"backend=fd motions={len(resolved_motion_files)} "
            f"elapsed={time.perf_counter() - start:.3f}s",
            stdout=True,
        )
        return resolved_motion_files

    def _scan_motion_path_with_python(self, motion_path: str) -> list[str]:
        worker_count = self._motion_scan_worker_count()
        if worker_count <= 1:
            return self._scan_motion_path_with_os_walk(motion_path)
        return self._scan_motion_path_with_parallel_os_walk(motion_path, worker_count)

    def _motion_scan_worker_count(self, item_count: int | None = None) -> int:
        configured_workers = int(getattr(self.cfg, "motion_scan_workers", 0))
        if configured_workers < 0:
            raise ValueError("motion_scan_workers must be non-negative")
        if configured_workers > 0:
            return configured_workers

        worker_count = min(os.cpu_count() or 1, 32)
        if item_count is not None:
            worker_count = min(worker_count, max(item_count, 1))
        return max(worker_count, 1)

    def _scan_motion_path_with_parallel_os_walk(
        self, motion_path: str, worker_count: int
    ) -> list[str]:
        root_motion_files: list[str] = []
        child_dirs: list[str] = []
        root_file_count = 0
        with os.scandir(motion_path) as entries:
            for entry in entries:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError as exc:
                    _bootstrap_debug(
                        f"scan motion path skipped entry path={entry.path} error={exc}"
                    )
                    continue

                if is_dir:
                    child_dirs.append(entry.path)
                else:
                    root_file_count += 1
                    if entry.name.lower().endswith(".npz"):
                        root_motion_files.append(entry.path)

        worker_count = self._motion_scan_worker_count(len(child_dirs))
        if worker_count <= 1 or len(child_dirs) < 2:
            return self._scan_motion_path_with_os_walk(motion_path)

        start = time.perf_counter()
        last_log_time = start
        log_interval = float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0))
        resolved_motion_files = list(root_motion_files)
        scanned_dirs = 1
        scanned_files = root_file_count
        completed_roots = 0
        _bootstrap_debug(
            "scan motion path start "
            f"path={motion_path} backend=python_parallel workers={worker_count} "
            f"root_dirs={len(child_dirs)}",
            stdout=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._collect_motion_files_os_walk, child_dir)
                for child_dir in child_dirs
            ]
            for future in concurrent.futures.as_completed(futures):
                motion_files, dir_count, file_count = future.result()
                resolved_motion_files.extend(motion_files)
                scanned_dirs += dir_count
                scanned_files += file_count
                completed_roots += 1

                now = time.perf_counter()
                if log_interval > 0.0 and now - last_log_time >= log_interval:
                    _bootstrap_debug(
                        "scan motion path progress "
                        f"backend=python_parallel dirs={scanned_dirs} files={scanned_files} "
                        f"motions={len(resolved_motion_files)} "
                        f"completed_roots={completed_roots}/{len(child_dirs)} "
                        f"elapsed={now - start:.3f}s",
                        stdout=True,
                    )
                    last_log_time = now

        resolved_motion_files.sort()
        _bootstrap_debug(
            "scan motion path done "
            f"backend=python_parallel dirs={scanned_dirs} files={scanned_files} "
            f"motions={len(resolved_motion_files)} "
            f"elapsed={time.perf_counter() - start:.3f}s",
            stdout=True,
        )
        return resolved_motion_files

    @staticmethod
    def _collect_motion_files_os_walk(motion_path: str) -> tuple[list[str], int, int]:
        resolved_motion_files: list[str] = []
        scanned_dirs = 0
        scanned_files = 0
        for root, _, files in os.walk(motion_path):
            scanned_dirs += 1
            scanned_files += len(files)
            for filename in files:
                if filename.lower().endswith(".npz"):
                    resolved_motion_files.append(os.path.join(root, filename))
        return resolved_motion_files, scanned_dirs, scanned_files

    def _scan_motion_path_with_os_walk(self, motion_path: str) -> list[str]:
        start = time.perf_counter()
        last_log_time = start
        log_interval = float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0))
        resolved_motion_files: list[str] = []
        scanned_dirs = 0
        scanned_files = 0
        _bootstrap_debug(f"scan motion path start path={motion_path} backend=python", stdout=True)
        for root, _, files in os.walk(motion_path):
            scanned_dirs += 1
            scanned_files += len(files)
            for filename in files:
                if filename.lower().endswith(".npz"):
                    resolved_motion_files.append(os.path.join(root, filename))

            now = time.perf_counter()
            if log_interval > 0.0 and now - last_log_time >= log_interval:
                _bootstrap_debug(
                    "scan motion path progress "
                    f"dirs={scanned_dirs} files={scanned_files} "
                    f"motions={len(resolved_motion_files)} elapsed={now - start:.3f}s "
                    f"root={root}",
                    stdout=True,
                )
                last_log_time = now

        resolved_motion_files.sort()
        _bootstrap_debug(
            "scan motion path done "
            f"backend=python dirs={scanned_dirs} files={scanned_files} "
            f"motions={len(resolved_motion_files)} "
            f"elapsed={time.perf_counter() - start:.3f}s",
            stdout=True,
        )
        return resolved_motion_files

    def _write_motion_manifest(self, manifest_file: str, motion_files: list[str]) -> None:
        manifest_dir = os.path.dirname(manifest_file)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        tmp_file = f"{manifest_file}.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for motion_file in motion_files:
                f.write(motion_file + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, manifest_file)

    def _read_motion_manifest(self, manifest_file: str) -> list[str]:
        with open(manifest_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _wait_for_motion_manifest(self, manifest_file: str) -> list[str]:
        timeout_s = float(getattr(self.cfg, "motion_manifest_wait_timeout_s", 600.0))
        poll_interval_s = max(
            float(getattr(self.cfg, "motion_manifest_poll_interval_s", 0.25)), 0.01
        )
        log_interval_s = max(float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0)), 1.0)
        start = time.perf_counter()
        last_log_time = start
        _bootstrap_debug(
            f"waiting for motion manifest file={manifest_file} timeout={timeout_s:.1f}s",
            stdout=True,
        )
        while True:
            if os.path.exists(manifest_file):
                motion_files = self._read_motion_manifest(manifest_file)
                _bootstrap_debug(
                    f"read motion manifest count={len(motion_files)} file={manifest_file}",
                    stdout=True,
                )
                return motion_files

            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for motion manifest: {manifest_file}"
                )
            if now - last_log_time >= log_interval_s:
                _bootstrap_debug(
                    f"still waiting for motion manifest elapsed={elapsed:.3f}s "
                    f"file={manifest_file}",
                    stdout=True,
                )
                last_log_time = now
            time.sleep(poll_interval_s)

    def _bind_global_bin_pool_tensors(self) -> None:
        self.bin_count = self.global_bin_pool.bin_count
        self.motion_bin_counts = self.global_bin_pool.motion_bin_counts
        self.bin_valid_mask = self.global_bin_pool.bin_valid_mask
        self.valid_motion_ids = self.global_bin_pool.valid_motion_ids
        self.valid_bin_ids = self.global_bin_pool.valid_bin_ids
        self.num_valid_motion_bins = self.global_bin_pool.num_valid_motion_bins
        self.bin_lengths = self.global_bin_pool.bin_lengths
        self.bin_weights = (
            None
            if self.global_bin_pool.is_sharded
            else self.global_bin_pool._compute_bin_weights(
                self.cfg.adaptive_sequence_length_agnostic
            )
        )
        self.bin_visit_count = self.global_bin_pool.bin_visit_count
        self.bin_failure_count = self.global_bin_pool.bin_failure_count

    def _init_adaptive_sampling_ema(self) -> None:
        # The global pool already owns the pending distributed increments.
        self._adaptive_pending_visit_count = None
        self._adaptive_pending_failure_count = None
        self._adaptive_ema_last_iteration: int | None = None

    def begin_adaptive_sampling_iteration(self, iteration: int) -> None:
        if self.cfg.sampling_mode == "adaptive":
            iteration = int(iteration)
            if self._adaptive_ema_last_iteration is None:
                elapsed_iterations = 0
            else:
                elapsed_iterations = iteration - self._adaptive_ema_last_iteration
            if self._adaptive_ema_last_iteration is None or elapsed_iterations > 0:
                if elapsed_iterations > 0:
                    decay = _adaptive_ema_decay(self.cfg) ** elapsed_iterations
                    self.global_bin_pool.apply_ema_decay(decay)
                self._last_global_bin_update_time = self.global_bin_pool.synchronize()
                self._adaptive_ema_last_iteration = iteration
            else:
                self._last_global_bin_update_time = 0.0
            self.global_bin_pool.reset_counts_if_due(
                iteration=iteration,
                interval_iterations=self.cfg.adaptive_bin_pool_reset_interval_iterations,
            )
        else:
            self._last_global_bin_update_time = 0.0
        start = time.perf_counter()
        self._refresh_active_subset(iteration)
        self._last_subset_update_time = time.perf_counter() - start

    def get_large_dataset_timing_stats(self, *, reset: bool = False) -> dict[str, float]:
        pool_stats = self.global_bin_pool.get_timing_stats()
        stats = {
            "global_bin_update_time": float(self._last_global_bin_update_time),
            "global_bin_update_pack_time": float(
                pool_stats.get("global_bin_update_pack_time", 0.0)
            ),
            "global_bin_update_gather_time": float(
                pool_stats.get("global_bin_update_gather_time", 0.0)
            ),
            "global_bin_update_apply_time": float(
                pool_stats.get("global_bin_update_apply_time", 0.0)
            ),
            "global_bin_update_visit_key_count": float(
                pool_stats.get("global_bin_update_visit_key_count", 0.0)
            ),
            "global_bin_update_failure_key_count": float(
                pool_stats.get("global_bin_update_failure_key_count", 0.0)
            ),
            "adaptive_bin_pool_reset_time": float(
                pool_stats.get("adaptive_bin_pool_reset_time", 0.0)
            ),
            "adaptive_bin_pool_reset_applied": float(
                pool_stats.get("adaptive_bin_pool_reset_applied", 0.0)
            ),
            "subset_update_time": float(self._last_subset_update_time),
            "motion_gather_time": float(self._motion_gather_time_accum),
            "motion_gather_call_count": float(self._motion_gather_call_count),
        }
        if reset:
            self._motion_gather_time_accum = 0.0
            self._motion_gather_call_count = 0
        return stats

    def maybe_write_adaptive_bin_snapshot(
        self,
        *,
        iteration: int,
        default_snapshot_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        interval = int(getattr(self.cfg, "adaptive_bin_snapshot_interval_iterations", 0))
        if interval <= 0 or int(iteration) <= 0 or int(iteration) % interval != 0:
            return
        snapshot_dir = os.fspath(getattr(self.cfg, "adaptive_bin_snapshot_dir", ""))
        if not snapshot_dir:
            if default_snapshot_dir is None:
                return
            snapshot_dir = os.fspath(default_snapshot_dir)

        num_buckets = int(getattr(self.cfg, "adaptive_bin_snapshot_num_buckets", 2048))
        writer_key = (snapshot_dir, num_buckets)
        if (
            self._adaptive_bin_snapshot_writer is None
            or writer_key != self._adaptive_bin_snapshot_writer_key
        ):
            from intact_tracking.environment.snapshot import (
                AdaptiveBinPoolSnapshotWriter,
            )

            self._adaptive_bin_snapshot_writer = AdaptiveBinPoolSnapshotWriter(
                snapshot_dir=snapshot_dir,
                num_buckets=num_buckets,
                motion_files=self.motion_store.motion_files,
                manifest_file=os.fspath(getattr(self.cfg, "motion_manifest_file", "")),
            )
            self._adaptive_bin_snapshot_writer_key = writer_key
        self._adaptive_bin_snapshot_writer.write(self.global_bin_pool, iteration=int(iteration))

    def _compute_failure_rate(self) -> torch.Tensor:
        return self.global_bin_pool.compute_failure_rate()

    def _compute_motion_bin_indices(
        self, time_steps: torch.Tensor, motion_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.global_bin_pool.compute_motion_bin_indices(time_steps, motion_indices)

    def _record_completed_adaptive_visits(
        self,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
        failure_mask: torch.Tensor | None,
    ) -> None:
        self.global_bin_pool.record_completed_visits(motion_ids, time_steps, failure_mask)

    def _clamp_motion_time_steps(
        self, motion_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        max_time_steps = self.motion_store.file_lengths[motion_ids] - 1
        if time_steps.ndim > 1:
            max_time_steps = max_time_steps.unsqueeze(-1)
        clamped_time_steps = torch.clamp_min(time_steps, 0)
        return torch.minimum(clamped_time_steps, max_time_steps)

    def _gather_motion_field(
        self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        return self._gather_motion_fields((field_name,), motion_ids, time_steps)[field_name]

    def _gather_motion_fields(
        self,
        field_names: tuple[str, ...],
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        slot_ids = self.active_subset.motion_to_slot[motion_ids]
        if torch.any(slot_ids < 0):
            missing_motion_ids = motion_ids[slot_ids < 0]
            raise RuntimeError(
                "Requested motion ids are not resident in the active subset: "
                f"{missing_motion_ids.detach().cpu().tolist()}"
            )
        clamped_time_steps = self._clamp_motion_time_steps(motion_ids, time_steps)
        start = time.perf_counter()
        gathered = self.motion.gather_many(field_names, slot_ids, clamped_time_steps)
        self._motion_gather_time_accum += time.perf_counter() - start
        self._motion_gather_call_count += 1
        return gathered

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        random_probability = self._adaptive_random_probability()
        valid_motion_ids, valid_bin_ids, sampling_probabilities, valid_failure_rate = (
            self.global_bin_pool.compute_active_pair_sampling_probabilities(
                self.active_subset.active_motion_ids,
                adaptive_uniform_ratio=random_probability,
                adaptive_temperature=self.cfg.adaptive_sampling.temperature,
                adaptive_probability_max_over_mean=self.cfg.adaptive_probability_max_over_mean,
                adaptive_sequence_length_agnostic=self.cfg.adaptive_sequence_length_agnostic,
                adaptive_max_prob_per_bin=self.cfg.adaptive_max_prob_per_bin,
                adaptive_max_prob_per_motion=self.cfg.adaptive_max_prob_per_motion,
            )
        )
        sampled_pair_indices = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        sampled_motion_indices = valid_motion_ids[sampled_pair_indices]
        sampled_bin_indices = valid_bin_ids[sampled_pair_indices]
        self._record_adaptive_sampling_distribution_metrics(
            env_ids,
            sampling_probabilities,
            valid_failure_rate,
            uniform_mix_ratio=random_probability,
        )

        self.motion_idx[env_ids] = sampled_motion_indices
        self.motion_length[env_ids] = self.motion_store.file_lengths[sampled_motion_indices]

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
            pre_failure_offsets = torch.randint(
                self.cfg.adaptive_pre_failure_sample_window_steps,
                (len(env_ids),),
                device=self.device,
            )
            self.time_steps[env_ids] = (self.time_steps[env_ids] - pre_failure_offsets).clamp_min(0)

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

    def _initialize_env_motion_assignments(self) -> None:
        if self.num_envs == 0:
            return
        motion_indices = self._sample_active_motion_ids(self.num_envs)
        self.motion_idx.copy_(motion_indices)
        self.motion_length.copy_(self.motion_store.file_lengths[motion_indices])
        self.time_steps.zero_()

    def _resample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        sample_env_ids = self._prepare_reset_sampling(env_ids)
        if sample_env_ids.numel() > 0:
            if self.cfg.sampling_mode == "start":
                motion_indices = self._sample_active_motion_ids(len(sample_env_ids))
                self.motion_idx[sample_env_ids] = motion_indices
                self.motion_length[sample_env_ids] = self.motion_store.file_lengths[motion_indices]
                self.time_steps[sample_env_ids] = 0
            elif self.cfg.sampling_mode == "uniform":
                motion_indices = self._sample_active_motion_ids(len(sample_env_ids))
                self.motion_idx[sample_env_ids] = motion_indices
                self.motion_length[sample_env_ids] = self.motion_store.file_lengths[motion_indices]
                self._uniform_sampling(sample_env_ids)
            else:
                assert self.cfg.sampling_mode == "adaptive"
                self._adaptive_sampling(sample_env_ids)
        self._set_motion_origin_offset(env_ids)
        self._invalidate_reference_cache()
        self._reset_robot_to_reference(env_ids)

    def _refresh_active_subset(self, iteration: int) -> None:
        refresh_count = int(self.cfg.subset_refresh_count)
        if refresh_count <= 0 or self.motion_store.num_files <= self.active_subset.subset_size:
            return
        refresh_result = self.active_subset._empty_refresh_result()
        self.active_subset.set_slot_ref_counts_from_motion_ids(self.motion_idx)
        replacement_ids = self._sample_subset_replacement_ids(refresh_count)
        if replacement_ids.numel() > 0:
            refresh_result = self.active_subset.refresh(
                replacement_ids,
                iteration=iteration,
                max_replacements=refresh_count,
            )
        if refresh_result.num_replaced > 0:
            self.motion.replace_slots(
                refresh_result.replaced_slot_ids,
                refresh_result.new_motion_ids,
                self.motion_store,
            )
            self._invalidate_reference_cache()
        self.global_bin_pool.replace_active_motion_ids(
            refresh_result.replaced_slot_ids,
            refresh_result.new_motion_ids,
        )

    def _sample_subset_replacement_ids(self, count: int) -> torch.Tensor:
        available_ids = self.active_subset.available_motion_ids()
        if available_ids.numel() == 0:
            return available_ids
        count = min(int(count), int(available_ids.numel()))
        adaptive_count = int(round(count * float(self.cfg.subset_adaptive_refresh_ratio)))
        adaptive_count = max(0, min(count, adaptive_count))
        sampled_parts: list[torch.Tensor] = []
        if adaptive_count > 0 and self.cfg.sampling_mode == "adaptive":
            adaptive_candidate_ids = available_ids
            candidate_pool_size = int(
                getattr(self.cfg, "subset_adaptive_candidate_pool_size", 10_000)
            )
            if candidate_pool_size > 0 and adaptive_candidate_ids.numel() > candidate_pool_size:
                candidate_order = torch.randperm(adaptive_candidate_ids.numel(), device=self.device)
                adaptive_candidate_ids = adaptive_candidate_ids[
                    candidate_order[:candidate_pool_size]
                ]
            candidate_ids, candidate_probabilities = (
                self.global_bin_pool.compute_motion_sampling_probabilities(
                    adaptive_candidate_ids,
                    adaptive_uniform_ratio=self._adaptive_random_probability(),
                    adaptive_probability_max_over_mean=(
                        self.cfg.adaptive_probability_max_over_mean
                    ),
                    adaptive_temperature=self.cfg.adaptive_sampling.temperature,
                    adaptive_sequence_length_agnostic=self.cfg.adaptive_sequence_length_agnostic,
                )
            )
            positive_probability = candidate_probabilities > 0.0
            positive_candidate_ids = candidate_ids[positive_probability]
            positive_probabilities = candidate_probabilities[positive_probability]
            adaptive_count = min(adaptive_count, int(positive_candidate_ids.numel()))
            if adaptive_count > 0:
                positive_probabilities = positive_probabilities / torch.clamp(
                    positive_probabilities.sum(), min=1e-12
                )
                sampled_indices = torch.multinomial(
                    positive_probabilities, adaptive_count, replacement=False
                )
                sampled_parts.append(positive_candidate_ids[sampled_indices])

        remaining_count = count - sum(part.numel() for part in sampled_parts)
        if remaining_count > 0:
            excluded = torch.zeros(
                self.motion_store.num_files, dtype=torch.bool, device=self.device
            )
            for part in sampled_parts:
                excluded[part] = True
            random_pool = available_ids[~excluded[available_ids]]
            if random_pool.numel() > 0:
                random_order = torch.randperm(random_pool.numel(), device=self.device)
                sampled_parts.append(random_pool[random_order[:remaining_count]])

        if not sampled_parts:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.cat(sampled_parts)[:count]

    def _sample_active_motion_ids(self, count: int) -> torch.Tensor:
        active_ids = self.active_subset.active_motion_ids
        probabilities = self._global_motion_sampling_probabilities
        if probabilities is None:
            random_indices = torch.randint(active_ids.numel(), (count,), device=self.device)
        else:
            active_probabilities = probabilities[active_ids]
            active_probabilities /= active_probabilities.sum()
            random_indices = torch.multinomial(active_probabilities, count, replacement=True)
        return active_ids[random_indices]

    def _sample_unique_motion_ids(
        self,
        candidate_ids: torch.Tensor,
        count: int,
        probabilities: torch.Tensor | None,
    ) -> torch.Tensor:
        if probabilities is None:
            order = torch.randperm(candidate_ids.numel(), device=self.device)
            return candidate_ids[order[:count]]
        sampled_indices = torch.multinomial(probabilities, count, replacement=False)
        return candidate_ids[sampled_indices]


@dataclass(kw_only=True)
class LargeDatasetMultiMotionCommandCfg(MultiMotionCommandCfg):
    """Opt-in large-dataset motion command configuration."""

    active_subset_size: int = 20_000
    motion_sampling_groups: tuple[dict, ...] = ()
    subset_refresh_count: int = 10
    subset_min_resident_iterations: int = 50
    subset_adaptive_refresh_ratio: float = 0.5
    subset_adaptive_candidate_pool_size: int = 10_000
    adaptive_bin_pool_reset_interval_iterations: int = 0
    adaptive_bin_snapshot_num_buckets: int = 2048
    motion_metadata_cache_file: str = ""
    motion_metadata_cache_wait_timeout_s: float = 7200.0
    motion_metadata_cache_poll_interval_s: float = 0.25
    motion_metadata_read_workers: int = 0
    motion_metadata_read_backend: Literal["thread", "process", "serial"] = "thread"
    motion_metadata_read_chunksize: int = 64
    motion_manifest_wait_timeout_s: float = 600.0
    motion_manifest_poll_interval_s: float = 0.25
    motion_scan_backend: Literal["auto", "fd", "python"] = "auto"
    motion_scan_workers: int = 0
    motion_scan_fd_executable: str = "fd"
    motion_scan_log_interval_s: float = 10.0

    def build(self, env) -> LargeDatasetMultiMotionCommand:
        return LargeDatasetMultiMotionCommand(self, env)


MotionCommand = LargeDatasetMultiMotionCommand
MotionCommandCfg = LargeDatasetMultiMotionCommandCfg
