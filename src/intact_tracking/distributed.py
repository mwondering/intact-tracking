"""Small torchrun/DDP runtime used by pure-online INTACT training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def _environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass
class DistributedContext:
    """Process identity, device ownership, and small collective helpers."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    initialized_here: bool = False

    @classmethod
    def initialize(
        cls,
        *,
        requested_device: str | None = None,
        requested_backend: str | None = None,
    ) -> DistributedContext:
        environment_world_size = max(_environment_integer("WORLD_SIZE", 1), 1)
        environment_rank = _environment_integer("RANK", 0)
        environment_local_rank = _environment_integer("LOCAL_RANK", environment_rank)
        preinitialized = dist.is_available() and dist.is_initialized()
        distributed = preinitialized or environment_world_size > 1

        if distributed and requested_device is not None:
            raise ValueError(
                "Do not pass --device under torchrun; each process is assigned "
                "cuda:LOCAL_RANK automatically"
            )

        if distributed:
            if torch.cuda.is_available():
                if environment_local_rank >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"LOCAL_RANK={environment_local_rank} but only "
                        f"{torch.cuda.device_count()} CUDA devices are visible"
                    )
                torch.cuda.set_device(environment_local_rank)
                device = torch.device("cuda", environment_local_rank)
                backend = requested_backend or "nccl"
            else:
                device = torch.device("cpu")
                backend = requested_backend or "gloo"
            if backend == "nccl" and device.type != "cuda":
                raise RuntimeError("The NCCL backend requires CUDA")

            initialized_here = False
            if not preinitialized:
                dist.init_process_group(backend=backend, init_method="env://")
                initialized_here = True
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            actual_backend = dist.get_backend()
            if (actual_rank, actual_world_size) != (
                environment_rank,
                environment_world_size,
            ):
                raise RuntimeError(
                    "torchrun environment and process group disagree: "
                    f"env=({environment_rank},{environment_world_size}), "
                    f"group=({actual_rank},{actual_world_size})"
                )
            return cls(
                rank=actual_rank,
                local_rank=environment_local_rank,
                world_size=actual_world_size,
                device=device,
                backend=str(actual_backend),
                initialized_here=initialized_here,
            )

        device = torch.device(
            requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "cuda":
            index = 0 if device.index is None else device.index
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested cuda:{index}, but only {torch.cuda.device_count()} "
                    "CUDA devices are visible"
                )
            torch.cuda.set_device(index)
            device = torch.device("cuda", index)
        return cls(
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
            backend=None,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def all_true(self, condition: bool) -> bool:
        value = torch.tensor(
            1 if condition else 0,
            device=self.device,
            dtype=torch.int64,
        )
        if self.enabled:
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def sum_integers(self, values: dict[str, int]) -> dict[str, int]:
        names = tuple(values)
        packed = torch.tensor(
            [values[name] for name in names],
            device=self.device,
            dtype=torch.int64,
        )
        self.all_reduce_sum(packed)
        return {name: int(value) for name, value in zip(names, packed.tolist(), strict=True)}

    def mean_scalars(self, values: dict[str, float]) -> dict[str, float]:
        names = tuple(values)
        packed = torch.tensor(
            [values[name] for name in names],
            device=self.device,
            dtype=torch.float64,
        )
        self.all_reduce_sum(packed)
        packed.div_(self.world_size)
        return {name: float(value) for name, value in zip(names, packed.tolist(), strict=True)}

    def broadcast_object(self, value: Any, source: int = 0) -> Any:
        objects = [value if self.rank == source else None]
        if self.enabled:
            dist.broadcast_object_list(objects, src=source, device=self.device)
        return objects[0]

    def all_gather_object(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        objects: list[Any] = [None] * self.world_size
        dist.all_gather_object(objects, value)
        return objects

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized_here = False
