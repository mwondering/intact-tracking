"""Compact online replay for nominal five-step Forward Predictor training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intact_tracking.forward_predictor import physical_state_delta

from .schema import RolloutDimensions


def _float_tuple(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().tolist())


class _RunningMoments:
    def __init__(self, width: int, device: torch.device) -> None:
        self.width = int(width)
        self.device = device
        self.count = 0
        self.total = torch.zeros(width, dtype=torch.float64, device=device)
        self.square_total = torch.zeros_like(self.total)

    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().to(device=self.device, dtype=torch.float64).reshape(-1, self.width)
        if flat.numel() == 0:
            return
        if not torch.isfinite(flat).all():
            raise ValueError("Forward Predictor normalization received a non-finite value")
        self.count += flat.size(0)
        self.total.add_(flat.sum(dim=0))
        self.square_total.add_(flat.square().sum(dim=0))

    def packed(self) -> torch.Tensor:
        return torch.cat((self.total.new_tensor([self.count]), self.total, self.square_total))


@dataclass(frozen=True)
class ForwardPredictorNormalizationStats:
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    delta_mean: tuple[float, ...]
    delta_std: tuple[float, ...]
    world_ids: tuple[int, ...]
    epsilon: float = 1.0e-6

    def to_json(self, path: str | Path) -> None:
        import json

        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


class ForwardPredictorNormalization:
    """Running state/action/delta moments that can be frozen after warmup."""

    def __init__(
        self,
        dimensions: RolloutDimensions,
        *,
        epsilon: float = 1.0e-6,
        device: torch.device | str | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.epsilon = float(epsilon)
        self.device = torch.device(device or "cpu")
        self.state = _RunningMoments(dimensions.robot_state, self.device)
        self.action = _RunningMoments(dimensions.action, self.device)
        self.delta = _RunningMoments(70, self.device)
        self.frozen = False

    def update(self, batch: dict[str, torch.Tensor], valid: torch.Tensor) -> None:
        if self.frozen:
            return
        if valid.dtype != torch.bool or valid.shape != (batch["robot_state"].size(0),):
            raise ValueError("Forward Predictor normalization mask must be [num_worlds] bool")
        if not bool(valid.any()):
            return
        current = batch["robot_state"][valid]
        following = batch["next_robot_state"][valid]
        self.state.update(torch.cat((current, following), dim=0))
        self.action.update(batch["action"][valid])
        self.delta.update(physical_state_delta(current, following))

    def freeze(self) -> None:
        self.frozen = True

    @property
    def packed_size(self) -> int:
        return sum(
            1 + 2 * width for width in (self.dimensions.robot_state, self.dimensions.action, 70)
        )

    def packed_statistics(self, device: torch.device | str | None = None) -> torch.Tensor:
        packed = torch.cat((self.state.packed(), self.action.packed(), self.delta.packed()))
        return packed.to(device=device) if device is not None else packed

    def snapshot_from_packed(
        self,
        packed: torch.Tensor,
        world_ids: tuple[int, ...],
    ) -> ForwardPredictorNormalizationStats:
        flat = packed.detach().to(device=self.device, dtype=torch.float64).flatten()
        if flat.numel() != self.packed_size:
            raise ValueError(
                f"Packed Forward Predictor normalization has {flat.numel()} values, "
                f"expected {self.packed_size}"
            )
        offset = 0

        def read(width: int) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal offset
            count_value = float(flat[offset].item())
            count = int(count_value)
            if count < 1 or count_value != count:
                raise ValueError(f"Invalid Forward Predictor normalization count: {count_value}")
            offset += 1
            total = flat[offset : offset + width]
            offset += width
            square_total = flat[offset : offset + width]
            offset += width
            mean = total / count
            variance = (square_total / count - mean.square()).clamp_min(self.epsilon**2)
            return mean.float(), variance.sqrt().float()

        state_mean, state_std = read(self.dimensions.robot_state)
        action_mean, action_std = read(self.dimensions.action)
        delta_mean, delta_std = read(70)
        return ForwardPredictorNormalizationStats(
            state_mean=_float_tuple(state_mean),
            state_std=_float_tuple(state_std),
            action_mean=_float_tuple(action_mean),
            action_std=_float_tuple(action_std),
            delta_mean=_float_tuple(delta_mean),
            delta_std=_float_tuple(delta_std),
            world_ids=world_ids,
            epsilon=self.epsilon,
        )


class ForwardPredictorReplayBuffer:
    """Store strictly causal, reset-free five-step state/action windows."""

    REQUIRED_FIELDS = (
        "action",
        "robot_state",
        "next_robot_state",
        "reset_boundary",
        "world_id",
        "episode_step",
        "motion_id",
        "motion_step",
    )

    def __init__(
        self,
        *,
        num_worlds: int,
        dimensions: RolloutDimensions | None = None,
        horizon: int = 5,
        capacity: int = 16384,
        seed: int = 0,
        world_id_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if num_worlds < 1 or capacity < 1:
            raise ValueError("num_worlds and capacity must be positive")
        if horizon != 5:
            raise ValueError("Forward Predictor replay horizon is fixed to five")
        if world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")
        self.num_worlds = int(num_worlds)
        self.dimensions = dimensions or RolloutDimensions()
        self.horizon = int(horizon)
        self.capacity = int(capacity)
        self.world_id_offset = int(world_id_offset)
        self.device = torch.device(device or "cpu")
        self._world_ids = tuple(range(world_id_offset, world_id_offset + num_worlds))
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._query_offsets = torch.arange(4, -1, -1, device=self.device)
        self._history: dict[str, torch.Tensor] = {}
        self._samples: dict[str, torch.Tensor] = {}
        self._reset_history: torch.Tensor | None = None
        self._history_write = 0
        self._sample_write = 0
        self._size = 0
        self._world_ids_validated = False
        self.total_samples_generated = 0
        self.total_transitions = 0
        self.normalizer = ForwardPredictorNormalization(
            self.dimensions,
            device=self.device,
        )

    def __len__(self) -> int:
        return self._size

    @property
    def world_ids(self) -> tuple[int, ...]:
        return self._world_ids

    @property
    def storage_bytes(self) -> int:
        tensors = [*self._history.values(), *self._samples.values()]
        if self._reset_history is not None:
            tensors.append(self._reset_history)
        return sum(value.numel() * value.element_size() for value in tensors)

    @property
    def estimated_storage_bytes(self) -> int:
        dims = self.dimensions
        history_floats = self.horizon * self.num_worlds * (dims.robot_state + dims.action)
        sample_floats = self.capacity * (
            (self.horizon + 1) * dims.robot_state + self.horizon * dims.action
        )
        history_integers = 3 * self.horizon * self.num_worlds
        sample_integers = 3 * self.capacity
        flags = self.horizon * self.num_worlds
        return (
            4 * (history_floats + sample_floats) + 8 * (history_integers + sample_integers) + flags
        )

    def _allocate(self) -> None:
        if self._history:
            return
        dims = self.dimensions
        history_prefix = (self.horizon, self.num_worlds)
        self._history = {
            "state": torch.empty((*history_prefix, dims.robot_state), device=self.device),
            "action": torch.empty((*history_prefix, dims.action), device=self.device),
            "episode_step": torch.empty(history_prefix, dtype=torch.long, device=self.device),
            "motion_id": torch.empty(history_prefix, dtype=torch.long, device=self.device),
            "motion_step": torch.empty(history_prefix, dtype=torch.long, device=self.device),
        }
        self._reset_history = torch.empty(history_prefix, dtype=torch.bool, device=self.device)
        self._samples = {
            "state": torch.empty(
                (self.capacity, self.horizon + 1, dims.robot_state), device=self.device
            ),
            "action": torch.empty((self.capacity, self.horizon, dims.action), device=self.device),
            "world_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
        }

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        missing = sorted(set(self.REQUIRED_FIELDS).difference(batch))
        if missing:
            raise KeyError(f"Forward Predictor transition batch is missing fields: {missing}")
        dims = self.dimensions
        expected = {
            "action": (self.num_worlds, dims.action),
            "robot_state": (self.num_worlds, dims.robot_state),
            "next_robot_state": (self.num_worlds, dims.robot_state),
            "reset_boundary": (self.num_worlds,),
            "world_id": (self.num_worlds,),
            "episode_step": (self.num_worlds,),
            "motion_id": (self.num_worlds,),
            "motion_step": (self.num_worlds,),
        }
        for name, shape in expected.items():
            value = batch[name]
            if not isinstance(value, torch.Tensor) or value.device != self.device:
                raise ValueError(f"Forward Predictor field {name!r} must be on {self.device}")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Forward Predictor field {name!r} has {tuple(value.shape)}, expected {shape}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"Forward Predictor field {name!r} contains non-finite values")
        if batch["reset_boundary"].dtype != torch.bool:
            raise ValueError("reset_boundary must be boolean")
        if not self._world_ids_validated:
            expected_ids = torch.arange(
                self.world_id_offset,
                self.world_id_offset + self.num_worlds,
                device=self.device,
            )
            if not torch.equal(batch["world_id"], expected_ids):
                raise ValueError("Forward Predictor replay world IDs do not match vector slots")
            self._world_ids_validated = True

    def _append_samples(self, samples: dict[str, torch.Tensor], count: int) -> None:
        retained = count
        if retained > self.capacity:
            selection = torch.randperm(retained, generator=self._generator, device=self.device)[
                : self.capacity
            ]
            samples = {name: value.index_select(0, selection) for name, value in samples.items()}
            retained = self.capacity
        positions = (self._sample_write + torch.arange(retained, device=self.device)).remainder(
            self.capacity
        )
        for name, value in samples.items():
            self._samples[name].index_copy_(0, positions, value)
        self._sample_write = (self._sample_write + retained) % self.capacity
        self._size = min(self.capacity, self._size + retained)
        self.total_samples_generated += count

    def add_step(self, batch: dict[str, torch.Tensor]) -> int:
        self._validate_batch(batch)
        self._allocate()
        assert self._reset_history is not None
        valid_transition = ~batch["reset_boundary"]
        self.normalizer.update(batch, valid_transition)

        position = self._history_write
        self._history["state"][position].copy_(batch["robot_state"])
        self._history["action"][position].copy_(batch["action"])
        self._history["episode_step"][position].copy_(batch["episode_step"])
        self._history["motion_id"][position].copy_(batch["motion_id"])
        self._history["motion_step"][position].copy_(batch["motion_step"])
        self._reset_history[position].copy_(batch["reset_boundary"])

        time_ids = (position - self._query_offsets).remainder(self.horizon)
        episode_steps = self._history["episode_step"][time_ids]
        complete_window = (
            episode_steps.remainder(self.horizon)
            == torch.arange(self.horizon, device=self.device)[:, None]
        ).all(dim=0)
        crosses_reset = self._reset_history[time_ids].any(dim=0)
        valid = (
            (batch["episode_step"] + 1 >= self.horizon)
            & complete_window
            & ~crosses_reset
            & ~batch["reset_boundary"]
        )
        env_ids = valid.nonzero(as_tuple=False).flatten()
        count = int(env_ids.numel())
        if count:
            state_history = self._history["state"][time_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            action_history = self._history["action"][time_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            samples = {
                "state": torch.cat(
                    (state_history, batch["next_robot_state"][env_ids, None]), dim=1
                ),
                "action": action_history,
                "world_id": batch["world_id"][env_ids],
                "motion_id": self._history["motion_id"][time_ids[0], env_ids],
                "motion_step": self._history["motion_step"][time_ids[0], env_ids],
            }
            self._append_samples(samples, count)

        self._history_write = (position + 1) % self.horizon
        self.total_transitions += self.num_worlds
        return count

    @staticmethod
    def _normalization_tensors(
        stats: ForwardPredictorNormalizationStats,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.tensor(getattr(stats, name), dtype=torch.float32, device=device)
            for name in (
                "state_mean",
                "state_std",
                "action_mean",
                "action_std",
                "delta_mean",
                "delta_std",
            )
        )

    def sample_batch(
        self,
        batch_size: int,
        normalization: ForwardPredictorNormalizationStats,
    ) -> dict[str, torch.Tensor]:
        if batch_size < 1 or self._size < batch_size:
            raise RuntimeError(
                f"Forward Predictor replay has {self._size} samples, batch_size={batch_size}"
            )
        indices = torch.randperm(self._size, generator=self._generator, device=self.device)[
            :batch_size
        ]
        selected = {name: value.index_select(0, indices) for name, value in self._samples.items()}
        state_mean, state_std, action_mean, action_std, delta_mean, delta_std = (
            self._normalization_tensors(normalization, self.device)
        )
        return {
            "state": (selected["state"] - state_mean) / state_std,
            "action": (selected["action"] - action_mean) / action_std,
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "world_id": selected["world_id"],
            "motion_id": selected["motion_id"],
            "motion_step": selected["motion_step"],
        }
