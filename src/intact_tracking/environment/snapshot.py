from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class CompactBinPoolSnapshot:
    access_sum: np.ndarray
    failure_sum: np.ndarray
    valid_count: np.ndarray
    bucket_start_motion_ids: np.ndarray
    bucket_end_motion_ids: np.ndarray


def build_compact_bin_pool_snapshot(
    pool,
    *,
    num_buckets: int,
) -> CompactBinPoolSnapshot:
    """Compress a full adaptive bin pool shard into global motion-id buckets."""

    num_buckets = max(1, min(int(num_buckets), int(pool.num_files)))
    bin_count = int(pool.bin_count)
    device = pool.device

    with torch.no_grad():
        owned_motion_ids = pool.owned_motion_ids.to(device=device, dtype=torch.long)
        bucket_ids = torch.div(
            owned_motion_ids * num_buckets,
            int(pool.num_files),
            rounding_mode="floor",
        ).clamp_(0, num_buckets - 1)
        bin_ids = torch.arange(bin_count, dtype=torch.long, device=device)
        flat_indices = (bucket_ids.unsqueeze(1) * bin_count + bin_ids).reshape(-1)
        valid_mask = pool._bin_valid_mask_for(owned_motion_ids)

        access_values = torch.clamp_min(pool.bin_visit_count - float(pool.prior_visit_count), 0.0)
        failure_values = torch.clamp_min(
            pool.bin_failure_count - float(pool.prior_failure_count), 0.0
        )
        access_values = access_values.masked_fill(~valid_mask, 0.0).reshape(-1)
        failure_values = failure_values.masked_fill(~valid_mask, 0.0).reshape(-1)
        valid_values = valid_mask.to(dtype=torch.float32).reshape(-1)

        flat_size = num_buckets * bin_count
        access_sum = torch.zeros(flat_size, dtype=torch.float32, device=device)
        failure_sum = torch.zeros_like(access_sum)
        valid_count = torch.zeros_like(access_sum)
        access_sum.index_add_(0, flat_indices, access_values.to(dtype=torch.float32))
        failure_sum.index_add_(0, flat_indices, failure_values.to(dtype=torch.float32))
        valid_count.index_add_(0, flat_indices, valid_values)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(access_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(failure_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)

    bucket_ids_np = np.arange(num_buckets + 1, dtype=np.int64)
    bucket_edges = (bucket_ids_np * int(pool.num_files)) // num_buckets
    return CompactBinPoolSnapshot(
        access_sum=access_sum.reshape(num_buckets, bin_count).detach().cpu().numpy(),
        failure_sum=failure_sum.reshape(num_buckets, bin_count).detach().cpu().numpy(),
        valid_count=valid_count.reshape(num_buckets, bin_count)
        .to(dtype=torch.int32)
        .detach()
        .cpu()
        .numpy(),
        bucket_start_motion_ids=bucket_edges[:-1],
        bucket_end_motion_ids=bucket_edges[1:],
    )


class AdaptiveBinPoolSnapshotWriter:
    def __init__(
        self,
        *,
        snapshot_dir: str | os.PathLike[str],
        num_buckets: int,
        motion_files: list[str],
        manifest_file: str = "",
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.num_buckets = int(num_buckets)
        self.motion_files = list(motion_files)
        self.manifest_file = str(manifest_file)

    def write(self, pool, *, iteration: int) -> None:
        snapshot = build_compact_bin_pool_snapshot(pool, num_buckets=self.num_buckets)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        first_paths, last_paths = self._bucket_paths(
            snapshot.bucket_start_motion_ids, snapshot.bucket_end_motion_ids
        )

        self._write_array_atomic("access_sum.f32", snapshot.access_sum.astype(np.float32))
        self._write_array_atomic("failure_sum.f32", snapshot.failure_sum.astype(np.float32))
        self._write_array_atomic("valid_count.i32", snapshot.valid_count.astype(np.int32))
        metadata = {
            "version": 4,
            "iteration": int(iteration),
            "updated_at_unix": time.time(),
            "num_files": int(pool.num_files),
            "bucket_count": int(snapshot.access_sum.shape[0]),
            "bin_count": int(pool.bin_count),
            "bin_width_steps": int(pool.bin_width_steps),
            "prior_visit_count": float(pool.prior_visit_count),
            "prior_failure_count": float(pool.prior_failure_count),
            "failure_rate_ema_iterations": getattr(pool, "failure_rate_ema_iterations", None),
            "count_semantics": ("exponentially_decayed_completed_bin_visits_excluding_priors"),
            "manifest_file": self.manifest_file,
            "access_file": "access_sum.f32",
            "failure_file": "failure_sum.f32",
            "valid_file": "valid_count.i32",
            "bucket_start_motion_ids": snapshot.bucket_start_motion_ids.astype(int).tolist(),
            "bucket_end_motion_ids": snapshot.bucket_end_motion_ids.astype(int).tolist(),
            "bucket_first_paths": first_paths,
            "bucket_last_paths": last_paths,
        }
        self._write_json_atomic("latest.json", metadata)

    def _bucket_paths(
        self, bucket_starts: np.ndarray, bucket_ends: np.ndarray
    ) -> tuple[list[str], list[str]]:
        first_paths: list[str] = []
        last_paths: list[str] = []
        for start, end in zip(bucket_starts.tolist(), bucket_ends.tolist(), strict=True):
            if start >= end or start >= len(self.motion_files):
                first_paths.append("")
                last_paths.append("")
                continue
            last_index = min(int(end) - 1, len(self.motion_files) - 1)
            first_paths.append(self.motion_files[int(start)])
            last_paths.append(self.motion_files[last_index])
        return first_paths, last_paths

    def _write_array_atomic(self, filename: str, array: np.ndarray) -> None:
        path = self.snapshot_dir / filename
        tmp_path = path.with_name(f".{path.name}.tmp")
        array.tofile(tmp_path)
        os.replace(tmp_path, path)

    def _write_json_atomic(self, filename: str, data: dict) -> None:
        path = self.snapshot_dir / filename
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, path)


def _environment_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _distributed_identity() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return _environment_int("RANK", 0), max(_environment_int("WORLD_SIZE", 1), 1)


class PerRankAdaptiveBinSnapshotWriter:
    """Write exact per-motion adaptive statistics for one distributed rank."""

    def __init__(
        self,
        *,
        snapshot_dir: str | os.PathLike[str],
        motion_files: list[str] | tuple[str, ...],
        file_lengths: torch.Tensor | list[int] | tuple[int, ...],
        fps_list: list[float] | tuple[float, ...],
        motion_bin_counts: torch.Tensor | list[int] | tuple[int, ...],
        bin_width_steps: int,
        failure_rate_ema_iterations: int | None,
        prior_visit_count: float = 0.0,
        prior_failure_count: float = 0.0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        resolved_rank, resolved_world_size = _distributed_identity()
        self.rank = resolved_rank if rank is None else int(rank)
        self.world_size = resolved_world_size if world_size is None else max(int(world_size), 1)
        self.snapshot_dir = Path(snapshot_dir)
        self.rank_dir = self.snapshot_dir / f"rank_{self.rank:03d}"
        self.motion_files = tuple(os.fspath(path) for path in motion_files)
        self.file_lengths = torch.as_tensor(file_lengths, dtype=torch.long).cpu().tolist()
        self.fps_list = [float(fps) for fps in fps_list]
        self.motion_bin_counts = torch.as_tensor(motion_bin_counts, dtype=torch.long).cpu().tolist()
        self.bin_width_steps = max(int(bin_width_steps), 1)
        self.failure_rate_ema_iterations = (
            None if failure_rate_ema_iterations is None else int(failure_rate_ema_iterations)
        )
        self.prior_visit_count = max(float(prior_visit_count), 0.0)
        self.prior_failure_count = max(float(prior_failure_count), 0.0)
        motion_count = len(self.motion_files)
        if not (
            len(self.file_lengths)
            == len(self.fps_list)
            == len(self.motion_bin_counts)
            == motion_count
        ):
            raise ValueError(
                "Per-rank snapshot metadata lengths must match: "
                f"files={motion_count}, lengths={len(self.file_lengths)}, "
                f"fps={len(self.fps_list)}, bins={len(self.motion_bin_counts)}"
            )
        if any(fps <= 0.0 for fps in self.fps_list):
            raise ValueError("Per-rank snapshot FPS values must be positive")

        self.sorted_local_motion_ids = tuple(
            sorted(
                range(motion_count),
                key=lambda motion_id: (
                    self.file_lengths[motion_id] / self.fps_list[motion_id],
                    self.motion_files[motion_id],
                    motion_id,
                ),
            )
        )
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self._write_motion_metadata()
        if self.rank == 0:
            self._write_layout()

    def write(
        self,
        *,
        visit_count: torch.Tensor,
        failure_count: torch.Tensor,
        iteration: int,
    ) -> None:
        motion_count = len(self.motion_files)
        minimum_bin_count = max(self.motion_bin_counts, default=0)
        if visit_count.ndim != 2 or failure_count.shape != visit_count.shape:
            raise ValueError(
                "Per-rank adaptive counts must be matching rank-2 tensors, got "
                f"visit={tuple(visit_count.shape)}, "
                f"failure={tuple(failure_count.shape)}"
            )
        if visit_count.shape[0] != motion_count:
            raise ValueError(
                "Per-rank adaptive count motion dimension does not match metadata: "
                f"counts={visit_count.shape[0]}, metadata={motion_count}"
            )
        if visit_count.shape[1] < minimum_bin_count:
            raise ValueError(
                "Per-rank adaptive count bin dimension is shorter than a valid motion: "
                f"counts={visit_count.shape[1]}, required={minimum_bin_count}"
            )

        order = torch.as_tensor(
            self.sorted_local_motion_ids,
            dtype=torch.long,
            device=visit_count.device,
        )
        with torch.no_grad():
            access = (
                torch.clamp_min(
                    visit_count.index_select(0, order) - self.prior_visit_count,
                    0.0,
                )
                .to(dtype=torch.float32)
                .detach()
                .cpu()
                .numpy()
            )
            failure = (
                torch.clamp_min(
                    failure_count.index_select(0, order) - self.prior_failure_count,
                    0.0,
                )
                .to(dtype=torch.float32)
                .detach()
                .cpu()
                .numpy()
            )

        self._write_array_atomic("access.f32", access)
        self._write_array_atomic("failure.f32", failure)
        metadata = {
            "version": 5,
            "layout": "per_rank_exact_motion",
            "iteration": int(iteration),
            "updated_at_unix": time.time(),
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": _environment_int("LOCAL_RANK", self.rank),
            "hostname": socket.gethostname(),
            "cuda_device": (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else None
            ),
            "motion_count": int(access.shape[0]),
            "bin_count": int(access.shape[1]),
            "bin_width_steps": self.bin_width_steps,
            "failure_rate_ema_iterations": self.failure_rate_ema_iterations,
            "count_semantics": ("exponentially_decayed_completed_bin_visits_excluding_priors"),
            "prior_visit_count": self.prior_visit_count,
            "prior_failure_count": self.prior_failure_count,
            "motion_order": "duration_seconds_ascending",
            "motion_metadata_file": "motions.json",
            "access_file": "access.f32",
            "failure_file": "failure.f32",
        }
        self._write_json_atomic("latest.json", metadata)

    def _write_layout(self) -> None:
        ranks = [
            {
                "rank": rank,
                "snapshot": f"rank_{rank:03d}/latest.json",
            }
            for rank in range(self.world_size)
        ]
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic_at(
            self.snapshot_dir / "layout.json",
            {
                "version": 2,
                "layout": "per_rank_exact_motion",
                "world_size": self.world_size,
                "ranks": ranks,
            },
        )

    def _write_motion_metadata(self) -> None:
        motions = []
        for sorted_index, local_motion_id in enumerate(self.sorted_local_motion_ids):
            length_steps = int(self.file_lengths[local_motion_id])
            fps = float(self.fps_list[local_motion_id])
            motions.append(
                {
                    "sorted_index": sorted_index,
                    "local_motion_id": local_motion_id,
                    "path": self.motion_files[local_motion_id],
                    "length_steps": length_steps,
                    "fps": fps,
                    "duration_seconds": length_steps / fps,
                    "valid_bin_count": int(self.motion_bin_counts[local_motion_id]),
                }
            )
        self._write_json_atomic(
            "motions.json",
            {
                "version": 2,
                "rank": self.rank,
                "motion_order": "duration_seconds_ascending",
                "motions": motions,
            },
        )

    def _write_array_atomic(self, filename: str, array: np.ndarray) -> None:
        path = self.rank_dir / filename
        tmp_path = path.with_name(f".{path.name}.tmp")
        np.asarray(array, dtype=np.float32).tofile(tmp_path)
        os.replace(tmp_path, path)

    def _write_json_atomic(self, filename: str, data: dict) -> None:
        self._write_json_atomic_at(self.rank_dir / filename, data)

    @staticmethod
    def _write_json_atomic_at(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, path)
