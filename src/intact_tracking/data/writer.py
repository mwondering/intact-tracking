"""Streaming, dependency-light NumPy rollout shards."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import (
    SCHEMA_VERSION,
    RolloutDimensions,
    core_field_specs,
    diagnostic_field_specs,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class RolloutShardWriter:
    """Write vectorized transition batches without adding dependencies to SP.

    Each shard is a directory of aligned ``.npy`` columns plus metadata.  The
    format is mmap-readable by the trainer and avoids importing h5py inside the
    configured MJLab rollout environment.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dimensions: RolloutDimensions | None = None,
        shard_size: int = 100_000,
        include_diagnostics: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.dimensions = dimensions or RolloutDimensions()
        self.shard_size = int(shard_size)
        self.include_diagnostics = bool(include_diagnostics)
        self.metadata = dict(metadata or {})
        self._specs = core_field_specs(self.dimensions)
        if self.include_diagnostics:
            self._specs.update(diagnostic_field_specs(self.dimensions))
        self._buffers = {
            name: np.empty((self.shard_size, *spec.shape), dtype=spec.dtype)
            for name, spec in self._specs.items()
        }
        self._position = 0
        self._total = 0
        self._shards: list[dict[str, Any]] = []
        self._world_ids: set[int] = set()
        self._closed = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "manifest.json"
        if manifest.exists():
            raise FileExistsError(f"Refusing to overwrite an existing rollout manifest: {manifest}")

    @property
    def total_transitions(self) -> int:
        return self._total

    def _validate_batch(self, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
        missing = sorted(set(self._specs).difference(batch))
        if missing:
            raise KeyError(f"Rollout batch is missing fields: {missing}")
        unexpected = sorted(set(batch).difference(self._specs))
        if unexpected:
            raise KeyError(f"Rollout batch has unknown fields: {unexpected}")
        arrays: dict[str, np.ndarray] = {}
        batch_size: int | None = None
        for name, spec in self._specs.items():
            value = _as_numpy(batch[name])
            if value.ndim != 1 + len(spec.shape) or tuple(value.shape[1:]) != spec.shape:
                raise ValueError(
                    f"Field {name!r} must be [B,{','.join(map(str, spec.shape))}], "
                    f"got {value.shape}"
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"Field {name!r} has batch {value.shape[0]}, expected {batch_size}"
                )
            if np.issubdtype(spec.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"Field {name!r} contains non-finite values")
            arrays[name] = value.astype(spec.dtype, copy=False)
        if batch_size is None or batch_size < 1:
            raise ValueError("Cannot append an empty rollout batch")
        return arrays

    def append(self, batch: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed rollout writer")
        arrays = self._validate_batch(batch)
        cursor = 0
        batch_size = next(iter(arrays.values())).shape[0]
        while cursor < batch_size:
            count = min(batch_size - cursor, self.shard_size - self._position)
            target = slice(self._position, self._position + count)
            source = slice(cursor, cursor + count)
            for name, array in arrays.items():
                self._buffers[name][target] = array[source]
            self._world_ids.update(int(value) for value in arrays["world_id"][source].reshape(-1))
            self._position += count
            self._total += count
            cursor += count
            if self._position == self.shard_size:
                self._flush_shard()

    def _flush_shard(self) -> None:
        if self._position == 0:
            return
        shard_index = len(self._shards)
        relative = Path(f"shard_{shard_index:05d}")
        shard_dir = self.output_dir / relative
        shard_dir.mkdir(parents=False, exist_ok=False)
        for name, buffer in self._buffers.items():
            np.save(shard_dir / f"{name}.npy", buffer[: self._position], allow_pickle=False)
        shard_worlds = sorted(
            int(value) for value in np.unique(self._buffers["world_id"][: self._position])
        )
        shard_metadata = {
            "schema_version": SCHEMA_VERSION,
            "length": self._position,
            "fields": {
                name: {"shape": list(spec.shape), "dtype": spec.dtype.name}
                for name, spec in self._specs.items()
            },
            "world_ids": shard_worlds,
        }
        (shard_dir / "metadata.json").write_text(
            json.dumps(shard_metadata, indent=2, sort_keys=True) + "\n"
        )
        self._shards.append(
            {"path": str(relative), "length": self._position, "world_ids": shard_worlds}
        )
        self._position = 0

    def close(self) -> Path:
        if self._closed:
            return self.output_dir / "manifest.json"
        self._flush_shard()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dimensions": self.dimensions.to_dict(),
            "include_diagnostics": self.include_diagnostics,
            "total_transitions": self._total,
            "world_ids": sorted(self._world_ids),
            "shards": self._shards,
            "metadata": _json_ready(self.metadata),
        }
        path = self.output_dir / "manifest.json"
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        self._closed = True
        return path

    def __enter__(self) -> "RolloutShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
