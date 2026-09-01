"""Device-resident causal replay for five-step residual-policy training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .schema import RolloutDimensions


def _float_tuple(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().to("cpu").tolist())


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
            raise ValueError("Residual normalization received a non-finite value")
        self.count += flat.size(0)
        self.total.add_(flat.sum(dim=0))
        self.square_total.add_(flat.square().sum(dim=0))

    def packed(self) -> torch.Tensor:
        return torch.cat((self.total.new_tensor([self.count]), self.total, self.square_total))


@dataclass(frozen=True)
class ResidualNormalizationStats:
    proprio_mean: tuple[float, ...]
    proprio_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    world_ids: tuple[int, ...]
    epsilon: float = 1.0e-6

    def to_json(self, path: str | Path) -> None:
        import json

        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


class ResidualOnlineNormalization:
    """Additive statistics that can be globally reduced across DDP ranks."""

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
        self.proprio = _RunningMoments(dimensions.proprio, self.device)
        self.action = _RunningMoments(dimensions.action, self.device)
        self.state = _RunningMoments(dimensions.robot_state, self.device)

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        self.proprio.update(torch.cat((batch["proprio"], batch["next_proprio"]), dim=0))
        self.action.update(batch["action"])
        self.state.update(
            torch.cat(
                (
                    batch["robot_state"],
                    batch["next_robot_state"],
                    batch["reference_state"],
                    batch["next_reference_state"],
                ),
                dim=0,
            )
        )

    @property
    def packed_size(self) -> int:
        return sum(
            1 + 2 * width
            for width in (
                self.dimensions.proprio,
                self.dimensions.action,
                self.dimensions.robot_state,
            )
        )

    def packed_statistics(self, device: torch.device | str | None = None) -> torch.Tensor:
        packed = torch.cat((self.proprio.packed(), self.action.packed(), self.state.packed()))
        return packed.to(device=device) if device is not None else packed

    def snapshot_from_packed(
        self,
        packed: torch.Tensor,
        world_ids: tuple[int, ...],
    ) -> ResidualNormalizationStats:
        flat = packed.detach().to(device=self.device, dtype=torch.float64).flatten()
        if flat.numel() != self.packed_size:
            raise ValueError(
                f"Packed residual normalization has {flat.numel()} values, "
                f"expected {self.packed_size}"
            )
        offset = 0

        def read(width: int) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal offset
            count_value = float(flat[offset].item())
            count = int(count_value)
            if count < 1 or count_value != count:
                raise ValueError(f"Invalid packed residual normalization count: {count_value}")
            offset += 1
            total = flat[offset : offset + width]
            offset += width
            square_total = flat[offset : offset + width]
            offset += width
            mean = total / count
            variance = (square_total / count - mean.square()).clamp_min(self.epsilon**2)
            return mean.float(), variance.sqrt().float()

        proprio_mean, proprio_std = read(self.dimensions.proprio)
        action_mean, action_std = read(self.dimensions.action)
        state_mean, state_std = read(self.dimensions.robot_state)
        return ResidualNormalizationStats(
            proprio_mean=_float_tuple(proprio_mean),
            proprio_std=_float_tuple(proprio_std),
            action_mean=_float_tuple(action_mean),
            action_std=_float_tuple(action_std),
            state_mean=_float_tuple(state_mean),
            state_std=_float_tuple(state_std),
            world_ids=world_ids,
            epsilon=self.epsilon,
        )


class ResidualOnlineReplayBuffer:
    """Causal 5-step windows plus same-world 16-token interaction context."""

    REQUIRED_FIELDS = (
        "proprio",
        "next_proprio",
        "policy_observation",
        "tracker_action",
        "action",
        "robot_state",
        "next_robot_state",
        "reference_state",
        "next_reference_state",
        "reset_boundary",
        "world_id",
        "episode_id",
        "episode_step",
        "collector_step",
    )

    def __init__(
        self,
        *,
        num_worlds: int,
        policy_observation_dim: int,
        dimensions: RolloutDimensions | None = None,
        horizon: int = 5,
        context_chunk_steps: int = 5,
        sample_stride: int = 1,
        context_tokens: int = 16,
        capacity: int = 8192,
        seed: int = 0,
        world_id_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        positive = {
            "num_worlds": num_worlds,
            "policy_observation_dim": policy_observation_dim,
            "horizon": horizon,
            "context_chunk_steps": context_chunk_steps,
            "sample_stride": sample_stride,
            "capacity": capacity,
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"Residual replay arguments must be positive: {invalid}")
        if horizon != 5:
            raise ValueError("Residual replay horizon is fixed to five steps")
        if context_tokens != 16:
            raise ValueError("Residual replay context is fixed to 16 tokens")
        if horizon % context_chunk_steps:
            raise ValueError("horizon must be divisible by context_chunk_steps")
        if world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")

        self.num_worlds = int(num_worlds)
        self.policy_observation_dim = int(policy_observation_dim)
        self.dimensions = dimensions or RolloutDimensions()
        self.horizon = int(horizon)
        self.context_chunk_steps = int(context_chunk_steps)
        self.sample_stride = int(sample_stride)
        self.context_tokens = int(context_tokens)
        self.capacity = int(capacity)
        self.world_id_offset = int(world_id_offset)
        self.device = torch.device(device or "cpu")
        self._query_context_chunks = self.horizon // self.context_chunk_steps
        self._context_capacity = self.context_tokens + self._query_context_chunks
        self.minimum_steps = self.context_tokens * self.context_chunk_steps + self.horizon
        self._history_length = self.horizon + 1
        self._world_ids = tuple(
            range(self.world_id_offset, self.world_id_offset + self.num_worlds)
        )
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._context_chunk_offsets = torch.arange(
            self.context_chunk_steps - 1,
            -1,
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._query_offsets = torch.arange(
            self.horizon - 1, -1, -1, dtype=torch.long, device=self.device
        )
        self._sample_context_offsets = torch.arange(
            self._context_capacity - 1,
            self._query_context_chunks - 1,
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._latest_context_offsets = torch.arange(
            self.context_tokens - 1, -1, -1, dtype=torch.long, device=self.device
        )
        self._history: dict[str, torch.Tensor] = {}
        self._reset_history: torch.Tensor | None = None
        self._context: dict[str, torch.Tensor] = {}
        self._samples: dict[str, torch.Tensor] = {}
        self._context_counts: torch.Tensor | None = None
        self._history_write = 0
        self._sample_write = 0
        self._size = 0
        self._world_ids_validated = False
        self.total_samples_generated = 0
        self.total_transitions = 0
        self.normalizer = ResidualOnlineNormalization(
            self.dimensions, device=self.device
        )

    def __len__(self) -> int:
        return self._size

    @property
    def world_ids(self) -> tuple[int, ...]:
        return self._world_ids

    @property
    def storage_bytes(self) -> int:
        tensors = [*self._history.values(), *self._context.values(), *self._samples.values()]
        if self._reset_history is not None:
            tensors.append(self._reset_history)
        if self._context_counts is not None:
            tensors.append(self._context_counts)
        return sum(value.numel() * value.element_size() for value in tensors)

    @property
    def estimated_storage_bytes(self) -> int:
        dims = self.dimensions
        history_width = (
            dims.proprio
            + self.policy_observation_dim
            + 2 * dims.action
            + 2 * dims.robot_state
        )
        context_width = 2 * dims.proprio + self.context_chunk_steps * dims.action
        sample_width = (
            self.horizon * self.policy_observation_dim
            + (2 * self.horizon + 1) * dims.action
            + (2 * self.horizon + 1) * dims.robot_state
            + self.context_tokens * context_width
        )
        floats = (
            self._history_length * self.num_worlds * history_width
            + self._context_capacity * self.num_worlds * context_width
            + self.capacity * sample_width
        )
        integers = self.num_worlds + 2 * self.capacity
        boundary_flags = self._history_length * self.num_worlds
        return 4 * floats + 8 * integers + boundary_flags

    def _allocate(self) -> None:
        if self._history:
            return
        dims = self.dimensions
        hp = (self._history_length, self.num_worlds)
        cp = (self.num_worlds, self._context_capacity)
        self._history = {
            "proprio": torch.zeros((*hp, dims.proprio), device=self.device),
            "policy_observation": torch.zeros(
                (*hp, self.policy_observation_dim), device=self.device
            ),
            "tracker_action": torch.zeros((*hp, dims.action), device=self.device),
            "action": torch.zeros((*hp, dims.action), device=self.device),
            "robot_state": torch.zeros((*hp, dims.robot_state), device=self.device),
            "next_robot_state": torch.zeros((*hp, dims.robot_state), device=self.device),
            "next_reference_state": torch.zeros((*hp, dims.reference_state), device=self.device),
        }
        self._reset_history = torch.zeros(hp, dtype=torch.bool, device=self.device)
        self._context = {
            "before": torch.zeros((*cp, dims.proprio), device=self.device),
            "actions": torch.zeros(
                (*cp, self.context_chunk_steps, dims.action), device=self.device
            ),
            "after": torch.zeros((*cp, dims.proprio), device=self.device),
        }
        self._context_counts = torch.zeros(
            self.num_worlds, dtype=torch.long, device=self.device
        )
        self._samples = {
            "policy_observation": torch.empty(
                (self.capacity, self.horizon, self.policy_observation_dim), device=self.device
            ),
            "tracker_action": torch.empty(
                (self.capacity, self.horizon, dims.action), device=self.device
            ),
            "action": torch.empty(
                (self.capacity, self.horizon, dims.action), device=self.device
            ),
            "previous_action": torch.empty(
                (self.capacity, dims.action), device=self.device
            ),
            "state": torch.empty(
                (self.capacity, self.horizon + 1, dims.robot_state), device=self.device
            ),
            "reference_state": torch.empty(
                (self.capacity, self.horizon, dims.reference_state), device=self.device
            ),
            "context_before": torch.empty(
                (self.capacity, self.context_tokens, dims.proprio), device=self.device
            ),
            "context_actions": torch.empty(
                (self.capacity, self.context_tokens, self.context_chunk_steps, dims.action),
                device=self.device,
            ),
            "context_after": torch.empty(
                (self.capacity, self.context_tokens, dims.proprio), device=self.device
            ),
            "world_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "episode_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
        }

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        missing = sorted(set(self.REQUIRED_FIELDS).difference(batch))
        if missing:
            raise KeyError(f"Residual transition batch is missing fields: {missing}")
        dims = self.dimensions
        expected = {
            "proprio": (self.num_worlds, dims.proprio),
            "next_proprio": (self.num_worlds, dims.proprio),
            "policy_observation": (self.num_worlds, self.policy_observation_dim),
            "tracker_action": (self.num_worlds, dims.action),
            "action": (self.num_worlds, dims.action),
            "robot_state": (self.num_worlds, dims.robot_state),
            "next_robot_state": (self.num_worlds, dims.robot_state),
            "reference_state": (self.num_worlds, dims.reference_state),
            "next_reference_state": (self.num_worlds, dims.reference_state),
        }
        for name in self.REQUIRED_FIELDS:
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Residual field {name!r} must be a Tensor")
            if value.device != self.device:
                raise ValueError(
                    f"Residual field {name!r} must remain on {self.device}, got {value.device}"
                )
            if value.ndim < 1 or value.size(0) != self.num_worlds:
                raise ValueError(
                    f"Residual field {name!r} must start with {self.num_worlds} worlds"
                )
        for name, shape in expected.items():
            if tuple(batch[name].shape) != shape:
                raise ValueError(
                    f"Residual field {name!r} has {tuple(batch[name].shape)}, "
                    f"expected {shape}"
                )
        if not self._world_ids_validated:
            expected_ids = torch.arange(
                self.world_id_offset,
                self.world_id_offset + self.num_worlds,
                dtype=torch.long,
                device=self.device,
            )
            if not torch.equal(batch["world_id"], expected_ids):
                raise ValueError("Residual replay world IDs do not match fixed vector slots")
            self._world_ids_validated = True

    def _append_context(
        self,
        batch: dict[str, torch.Tensor],
        history_position: int,
        valid: torch.Tensor,
    ) -> None:
        assert self._context_counts is not None
        env_ids = valid.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        time_ids = (history_position - self._context_chunk_offsets).remainder(
            self._history_length
        )
        slots = self._context_counts[env_ids].remainder(self._context_capacity)
        self._context["before"][env_ids, slots] = self._history["proprio"][time_ids[0], env_ids]
        actions = self._history["action"][time_ids[:, None], env_ids[None, :]].permute(1, 0, 2)
        self._context["actions"][env_ids, slots] = actions
        self._context["after"][env_ids, slots] = batch["next_proprio"][env_ids]
        self._context_counts[env_ids] += 1

    def _append_samples(self, samples: dict[str, torch.Tensor], count: int) -> None:
        retained = count
        if retained > self.capacity:
            selection = torch.randperm(
                retained, generator=self._generator, device=self.device
            )[: self.capacity]
            samples = {name: value.index_select(0, selection) for name, value in samples.items()}
            retained = self.capacity
        positions = (
            self._sample_write + torch.arange(retained, device=self.device)
        ).remainder(self.capacity)
        for name, value in samples.items():
            self._samples[name].index_copy_(0, positions, value)
        self._sample_write = (self._sample_write + retained) % self.capacity
        self._size = min(self.capacity, self._size + retained)
        self.total_samples_generated += count

    def _materialize(
        self,
        batch: dict[str, torch.Tensor],
        history_position: int,
        completed_steps: torch.Tensor,
        valid: torch.Tensor,
    ) -> int:
        assert self._context_counts is not None
        env_ids = valid.nonzero(as_tuple=False).flatten()
        count = int(env_ids.numel())
        if count == 0:
            return 0
        time_ids = (history_position - self._query_offsets).remainder(self._history_length)
        query = {
            name: self._history[name][time_ids[:, None], env_ids[None, :]].permute(1, 0, 2)
            for name in (
                "policy_observation",
                "tracker_action",
                "action",
                "robot_state",
                "next_reference_state",
            )
        }
        previous_time = (history_position - self.horizon) % self._history_length
        previous = self._history["action"][previous_time, env_ids]
        previous = torch.where(
            (completed_steps[env_ids] >= self.horizon + 1)[:, None],
            previous,
            torch.zeros_like(previous),
        )
        states = torch.cat(
            (query["robot_state"], batch["next_robot_state"][env_ids, None]), dim=1
        )
        context_slots = (
            self._context_counts[env_ids, None] - 1 - self._sample_context_offsets[None, :]
        ).remainder(self._context_capacity)
        context_envs = env_ids[:, None].expand_as(context_slots)
        samples = {
            "policy_observation": query["policy_observation"],
            "tracker_action": query["tracker_action"],
            "action": query["action"],
            "previous_action": previous,
            "state": states,
            "reference_state": query["next_reference_state"],
            "context_before": self._context["before"][context_envs, context_slots],
            "context_actions": self._context["actions"][context_envs, context_slots],
            "context_after": self._context["after"][context_envs, context_slots],
            "world_id": batch["world_id"][env_ids],
            "episode_id": batch["episode_id"][env_ids],
        }
        self._append_samples(samples, count)
        return count

    def add_step(self, batch: dict[str, torch.Tensor]) -> int:
        self._validate_batch(batch)
        self._allocate()
        self.normalizer.update(batch)
        position = self._history_write
        for name in self._history:
            source = batch[name]
            self._history[name][position].copy_(source)
        assert self._reset_history is not None
        self._reset_history[position].copy_(batch["reset_boundary"])
        completed = batch["episode_step"] + 1
        context_time_ids = (position - self._context_chunk_offsets).remainder(
            self._history_length
        )
        context_crosses_reset = self._reset_history[context_time_ids].any(dim=0)
        context_valid = (
            (completed.remainder(self.context_chunk_steps) == 0)
            & (completed >= self.context_chunk_steps)
            & ~context_crosses_reset
            & ~batch["reset_boundary"]
        )
        self._append_context(batch, position, context_valid)
        assert self._context_counts is not None
        query_time_ids = (position - self._query_offsets).remainder(self._history_length)
        query_crosses_reset = self._reset_history[query_time_ids].any(dim=0)
        sample_valid = (
            (completed.remainder(self.sample_stride) == 0)
            & (completed >= self.horizon)
            & (self._context_counts >= self._context_capacity)
            & ~query_crosses_reset
            & ~batch["reset_boundary"]
        )
        generated = self._materialize(batch, position, completed, sample_valid)
        self._history_write = (position + 1) % self._history_length
        self.total_transitions += self.num_worlds
        return generated

    @staticmethod
    def _normalization_tensors(
        stats: ResidualNormalizationStats,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.tensor(getattr(stats, name), dtype=torch.float32, device=device)
            for name in (
                "proprio_mean",
                "proprio_std",
                "action_mean",
                "action_std",
                "state_mean",
                "state_std",
            )
        )

    def latest_context(
        self,
        normalization: ResidualNormalizationStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized context before the next action and a ready mask."""
        self._allocate()
        assert self._context_counts is not None
        slots = (
            self._context_counts[:, None] - 1 - self._latest_context_offsets[None, :]
        ).remainder(self._context_capacity)
        envs = torch.arange(self.num_worlds, device=self.device)[:, None].expand_as(slots)
        before = self._context["before"][envs, slots]
        actions = self._context["actions"][envs, slots]
        after = self._context["after"][envs, slots]
        prop_mean, prop_std, action_mean, action_std, _, _ = self._normalization_tensors(
            normalization, self.device
        )
        context = torch.cat(
            (
                (before - prop_mean) / prop_std,
                ((actions - action_mean) / action_std).flatten(start_dim=2),
                (after - prop_mean) / prop_std,
            ),
            dim=-1,
        )
        ready = self._context_counts >= self.context_tokens
        context = torch.where(ready[:, None, None], context, torch.zeros_like(context))
        return context, ready

    def sample_batch(
        self,
        batch_size: int,
        normalization: ResidualNormalizationStats,
    ) -> dict[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise RuntimeError(f"Residual replay has {self._size} samples, batch_size={batch_size}")
        indices = torch.randperm(
            self._size, generator=self._generator, device=self.device
        )[:batch_size]
        selected = {name: value.index_select(0, indices) for name, value in self._samples.items()}
        prop_mean, prop_std, action_mean, action_std, state_mean, state_std = (
            self._normalization_tensors(normalization, self.device)
        )
        context = torch.cat(
            (
                (selected["context_before"] - prop_mean) / prop_std,
                ((selected["context_actions"] - action_mean) / action_std).flatten(
                    start_dim=2
                ),
                (selected["context_after"] - prop_mean) / prop_std,
            ),
            dim=-1,
        )
        return {
            "context": context,
            "context_mask": torch.ones(
                batch_size, self.context_tokens, dtype=torch.bool, device=self.device
            ),
            "policy_observation": selected["policy_observation"],
            "tracker_action": selected["tracker_action"],
            "action": (selected["action"] - action_mean) / action_std,
            "previous_action": (selected["previous_action"] - action_mean) / action_std,
            "state": (selected["state"] - state_mean) / state_std,
            "reference_state": (selected["reference_state"] - state_mean) / state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "state_mean": state_mean,
            "state_std": state_std,
            "world_id": selected["world_id"],
            "episode_id": selected["episode_id"],
        }
