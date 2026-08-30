"""Device-resident causal replay for pure online INTACT training."""

from __future__ import annotations

import torch

from .dataset import NormalizationStats
from .schema import RolloutDimensions


def _float_tuple(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().to("cpu").tolist())


class _RunningMoments:
    """Additive sufficient statistics kept on the rollout device."""

    def __init__(self, width: int, device: torch.device) -> None:
        self.width = int(width)
        self.device = device
        self.count = 0
        self.total = torch.zeros(self.width, dtype=torch.float64, device=device)
        self.square_total = torch.zeros_like(self.total)

    def update(self, value: torch.Tensor) -> None:
        if value.device != self.device:
            raise ValueError(f"Online normalization expected {self.device}, got {value.device}")
        flat = value.detach().to(dtype=torch.float64).reshape(-1, self.width)
        if flat.numel() == 0:
            return
        self.count += flat.shape[0]
        self.total.add_(flat.sum(dim=0))
        self.square_total.add_(flat.square().sum(dim=0))

    def mean_std(self, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 1:
            raise RuntimeError("Online normalization has not observed any transitions")
        if not bool(torch.isfinite(self.total).all() and torch.isfinite(self.square_total).all()):
            raise ValueError("Online normalization received a non-finite value")
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean.square()).clamp_min(epsilon**2)
        return mean.to(torch.float32), variance.sqrt().to(torch.float32)

    def packed(self) -> torch.Tensor:
        """Return additive sufficient statistics as ``count, sum, square_sum``."""
        return torch.cat((self.total.new_tensor([self.count]), self.total, self.square_total))


class OnlineNormalization:
    """Running train-stream statistics resident on one rank's rollout device."""

    def __init__(
        self,
        dimensions: RolloutDimensions,
        epsilon: float = 1e-6,
        device: torch.device | str | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.epsilon = float(epsilon)
        self.device = torch.device(device or "cpu")
        self.observation = _RunningMoments(dimensions.observation, self.device)
        self.proprio = _RunningMoments(dimensions.proprio, self.device)
        self.action = _RunningMoments(dimensions.action, self.device)

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        self.observation.update(
            torch.cat((batch["observation"], batch["reference_observation"]), dim=0)
        )
        self.proprio.update(batch["proprio"])
        self.action.update(batch["action"])

    @staticmethod
    def _stats(
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        world_ids: tuple[int, ...],
        epsilon: float,
    ) -> NormalizationStats:
        return NormalizationStats(
            observation_mean=_float_tuple(observation_mean),
            observation_std=_float_tuple(observation_std),
            proprio_mean=_float_tuple(proprio_mean),
            proprio_std=_float_tuple(proprio_std),
            action_mean=_float_tuple(action_mean),
            action_std=_float_tuple(action_std),
            world_ids=world_ids,
            epsilon=epsilon,
        )

    def snapshot(self, world_ids: tuple[int, ...]) -> NormalizationStats:
        observation_mean, observation_std = self.observation.mean_std(self.epsilon)
        proprio_mean, proprio_std = self.proprio.mean_std(self.epsilon)
        action_mean, action_std = self.action.mean_std(self.epsilon)
        return self._stats(
            observation_mean,
            observation_std,
            proprio_mean,
            proprio_std,
            action_mean,
            action_std,
            world_ids,
            self.epsilon,
        )

    @property
    def packed_size(self) -> int:
        return sum(
            1 + 2 * width
            for width in (
                self.dimensions.observation,
                self.dimensions.proprio,
                self.dimensions.action,
            )
        )

    def packed_statistics(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Pack additive moments for an optional distributed all-reduce."""
        packed = torch.cat(
            (
                self.observation.packed(),
                self.proprio.packed(),
                self.action.packed(),
            )
        )
        return packed.to(device=device) if device is not None else packed

    def snapshot_from_packed(
        self,
        packed: torch.Tensor,
        world_ids: tuple[int, ...],
    ) -> NormalizationStats:
        """Build normalization from globally summed sufficient statistics."""
        flat = packed.detach().to(device=self.device, dtype=torch.float64).flatten()
        if flat.numel() != self.packed_size:
            raise ValueError(
                f"Packed normalization has {flat.numel()} values, expected {self.packed_size}"
            )
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("Packed normalization contains non-finite values")

        offset = 0

        def read(width: int) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal offset
            count_value = float(flat[offset].item())
            count = int(count_value)
            if count < 1 or count_value != count:
                raise ValueError(f"Invalid packed normalization count: {count_value}")
            offset += 1
            total = flat[offset : offset + width]
            offset += width
            square_total = flat[offset : offset + width]
            offset += width
            mean = total / count
            variance = (square_total / count - mean.square()).clamp_min(self.epsilon**2)
            return mean.to(torch.float32), variance.sqrt().to(torch.float32)

        observation_mean, observation_std = read(self.dimensions.observation)
        proprio_mean, proprio_std = read(self.dimensions.proprio)
        action_mean, action_std = read(self.dimensions.action)
        return self._stats(
            observation_mean,
            observation_std,
            proprio_mean,
            proprio_std,
            action_mean,
            action_std,
            world_ids,
            self.epsilon,
        )


class OnlineReplayBuffer:
    """Vectorized device-resident causal replay.

    Each vector slot is one immutable physics world. Query transitions must be
    contiguous within an episode, while the 16 context chunks may come from
    earlier episodes of that same world. Raw transition history, context rings,
    materialized replay samples, normalization, and sampled mini-batches all stay
    on the rank's rollout device.
    """

    REQUIRED_FIELDS = (
        "proprio",
        "next_proprio",
        "observation",
        "next_observation",
        "reference_observation",
        "next_reference_observation",
        "action",
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
        dimensions: RolloutDimensions | None = None,
        block_size: int = 5,
        horizon: int = 5,
        context_tokens: int = 16,
        capacity: int = 8192,
        seed: int = 0,
        world_id_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if num_worlds < 1 or block_size < 1 or horizon < 1 or capacity < 1:
            raise ValueError("num_worlds, block_size, horizon and capacity must be positive")
        if context_tokens != 16:
            raise ValueError("Online INTACT context_tokens is fixed at 16")
        if world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")
        self.num_worlds = int(num_worlds)
        self.world_id_offset = int(world_id_offset)
        self.dimensions = dimensions or RolloutDimensions()
        self.block_size = int(block_size)
        self.horizon = int(horizon)
        self.context_tokens = int(context_tokens)
        self.capacity = int(capacity)
        self.device = torch.device(device or "cpu")
        self.query_steps = self.block_size * self.horizon
        self.minimum_steps = self.block_size * (self.context_tokens + self.horizon)
        self._history_length = self.query_steps + self.block_size
        self._context_capacity = self.context_tokens + self.horizon
        self._world_ids = tuple(range(self.world_id_offset, self.world_id_offset + self.num_worlds))
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._block_offsets = torch.arange(
            self.block_size - 1, -1, -1, dtype=torch.long, device=self.device
        )
        self._query_offsets = torch.arange(
            self.query_steps - 1, -1, -1, dtype=torch.long, device=self.device
        )
        self._previous_offsets = torch.arange(
            self.query_steps + self.block_size - 1,
            self.query_steps - 1,
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._context_offsets = torch.arange(
            self._context_capacity - 1,
            self.horizon - 1,
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._history: dict[str, torch.Tensor] = {}
        self._context: dict[str, torch.Tensor] = {}
        self._context_counts: torch.Tensor | None = None
        self._samples: dict[str, torch.Tensor] = {}
        self._history_write = 0
        self._sample_write = 0
        self._size = 0
        self._world_ids_validated = False
        self._cached_normalization: NormalizationStats | None = None
        self._cached_normalization_tensors: tuple[torch.Tensor, ...] | None = None
        self.normalizer = OnlineNormalization(self.dimensions, device=self.device)
        self.total_samples_generated = 0
        self.total_transitions = 0

    def __len__(self) -> int:
        return self._size

    @property
    def world_ids(self) -> tuple[int, ...]:
        return self._world_ids

    @property
    def storage_bytes(self) -> int:
        tensors = [*self._history.values(), *self._context.values(), *self._samples.values()]
        if self._context_counts is not None:
            tensors.append(self._context_counts)
        return sum(value.numel() * value.element_size() for value in tensors)

    @property
    def estimated_storage_bytes(self) -> int:
        dims = self.dimensions
        float_count = (
            self._history_length * self.num_worlds * (dims.proprio + dims.observation + dims.action)
            + self._context_capacity
            * self.num_worlds
            * (2 * dims.proprio + self.block_size * dims.action)
            + self.capacity
            * (
                (self.horizon + 2) * dims.observation
                + 2 * self.horizon * self.block_size * dims.action
                + self.context_tokens * (2 * dims.proprio + self.block_size * dims.action)
            )
        )
        integer_count = self.num_worlds + 2 * self.capacity
        return 4 * float_count + 8 * integer_count

    def normalization(self) -> NormalizationStats:
        return self.normalizer.snapshot(self.world_ids)

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        missing = sorted(set(self.REQUIRED_FIELDS).difference(batch))
        if missing:
            raise KeyError(f"Online transition batch is missing fields: {missing}")
        expected_shapes = {
            "proprio": (self.num_worlds, self.dimensions.proprio),
            "next_proprio": (self.num_worlds, self.dimensions.proprio),
            "observation": (self.num_worlds, self.dimensions.observation),
            "next_observation": (self.num_worlds, self.dimensions.observation),
            "reference_observation": (self.num_worlds, self.dimensions.observation),
            "next_reference_observation": (self.num_worlds, self.dimensions.observation),
            "action": (self.num_worlds, self.dimensions.action),
        }
        for name in self.REQUIRED_FIELDS:
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Online field {name!r} must be a Tensor")
            if value.device != self.device:
                raise ValueError(
                    f"Online field {name!r} must remain on replay device {self.device}, "
                    f"got {value.device}"
                )
            if value.ndim < 1 or value.shape[0] != self.num_worlds:
                raise ValueError(
                    f"Online field {name!r} must start with {self.num_worlds} worlds, "
                    f"got {tuple(value.shape)}"
                )
        for name, shape in expected_shapes.items():
            if tuple(batch[name].shape) != shape:
                raise ValueError(
                    f"Online field {name!r} has {tuple(batch[name].shape)}, {shape} expected"
                )
        if not self._world_ids_validated:
            expected_world_ids = torch.arange(
                self.world_id_offset,
                self.world_id_offset + self.num_worlds,
                dtype=torch.long,
                device=self.device,
            )
            if not torch.equal(batch["world_id"], expected_world_ids):
                raise ValueError(
                    "Online fixed-world contract requires world_id == "
                    "world_id_offset + local_env_id"
                )
            self._world_ids_validated = True

    def _allocate_stream_storage(self) -> None:
        if self._history:
            return
        dims = self.dimensions
        history_prefix = (self._history_length, self.num_worlds)
        context_prefix = (self.num_worlds, self._context_capacity)
        self._history = {
            "proprio": torch.zeros(
                (*history_prefix, dims.proprio), dtype=torch.float32, device=self.device
            ),
            "observation": torch.zeros(
                (*history_prefix, dims.observation), dtype=torch.float32, device=self.device
            ),
            "action": torch.zeros(
                (*history_prefix, dims.action), dtype=torch.float32, device=self.device
            ),
        }
        self._context = {
            "before": torch.empty(
                (*context_prefix, dims.proprio), dtype=torch.float32, device=self.device
            ),
            "actions": torch.empty(
                (*context_prefix, self.block_size, dims.action),
                dtype=torch.float32,
                device=self.device,
            ),
            "after": torch.empty(
                (*context_prefix, dims.proprio), dtype=torch.float32, device=self.device
            ),
        }
        self._context_counts = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)

    def _allocate_sample_storage(self) -> None:
        if self._samples:
            return
        dims = self.dimensions
        self._samples = {
            "observation": torch.empty(
                (self.capacity, self.horizon + 1, dims.observation),
                dtype=torch.float32,
                device=self.device,
            ),
            "goal_observation": torch.empty(
                (self.capacity, dims.observation), dtype=torch.float32, device=self.device
            ),
            "action": torch.empty(
                (self.capacity, self.horizon, self.block_size, dims.action),
                dtype=torch.float32,
                device=self.device,
            ),
            "previous_action": torch.empty(
                (self.capacity, self.horizon, self.block_size, dims.action),
                dtype=torch.float32,
                device=self.device,
            ),
            "context_before": torch.empty(
                (self.capacity, self.context_tokens, dims.proprio),
                dtype=torch.float32,
                device=self.device,
            ),
            "context_actions": torch.empty(
                (self.capacity, self.context_tokens, self.block_size, dims.action),
                dtype=torch.float32,
                device=self.device,
            ),
            "context_after": torch.empty(
                (self.capacity, self.context_tokens, dims.proprio),
                dtype=torch.float32,
                device=self.device,
            ),
            "world_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "episode_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
        }

    def _append_context_chunks(
        self,
        batch: dict[str, torch.Tensor],
        history_position: int,
        valid: torch.Tensor,
    ) -> None:
        if self._context_counts is None:
            raise RuntimeError("Online context storage is not initialized")
        env_ids = valid.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        time_ids = (history_position - self._block_offsets).remainder(self._history_length)
        slots = self._context_counts[env_ids].remainder(self._context_capacity)
        self._context["before"][env_ids, slots] = self._history["proprio"][time_ids[0], env_ids]
        actions = self._history["action"][time_ids[:, None], env_ids[None, :]].permute(1, 0, 2)
        self._context["actions"][env_ids, slots] = actions
        self._context["after"][env_ids, slots] = batch["next_proprio"][env_ids]
        self._context_counts[env_ids] = self._context_counts[env_ids] + 1

    def _materialize_samples(
        self,
        batch: dict[str, torch.Tensor],
        history_position: int,
        completed_steps: torch.Tensor,
        valid: torch.Tensor,
    ) -> int:
        if self._context_counts is None:
            raise RuntimeError("Online context storage is not initialized")
        env_ids = valid.nonzero(as_tuple=False).flatten()
        generated = int(env_ids.numel())
        if generated == 0:
            return 0

        query_time_ids = (history_position - self._query_offsets).remainder(self._history_length)
        query_observation = self._history["observation"][
            query_time_ids[:, None], env_ids[None, :]
        ].permute(1, 0, 2)
        observations = torch.cat(
            (
                query_observation[:, :: self.block_size],
                batch["next_observation"][env_ids, None],
            ),
            dim=1,
        )
        query_actions = self._history["action"][query_time_ids[:, None], env_ids[None, :]].permute(
            1, 0, 2
        )
        query_actions = query_actions.reshape(
            generated, self.horizon, self.block_size, self.dimensions.action
        )

        previous_time_ids = (history_position - self._previous_offsets).remainder(
            self._history_length
        )
        previous_first = self._history["action"][
            previous_time_ids[:, None], env_ids[None, :]
        ].permute(1, 0, 2)
        has_previous = completed_steps[env_ids] >= self.query_steps + self.block_size
        previous_first = torch.where(
            has_previous[:, None, None], previous_first, torch.zeros_like(previous_first)
        )
        previous_actions = torch.cat((previous_first[:, None], query_actions[:, :-1]), dim=1)

        context_slots = (
            self._context_counts[env_ids, None] - 1 - self._context_offsets[None, :]
        ).remainder(self._context_capacity)
        context_env_ids = env_ids[:, None].expand_as(context_slots)
        samples = {
            "observation": observations,
            "goal_observation": batch["next_reference_observation"][env_ids],
            "action": query_actions,
            "previous_action": previous_actions,
            "context_before": self._context["before"][context_env_ids, context_slots],
            "context_actions": self._context["actions"][context_env_ids, context_slots],
            "context_after": self._context["after"][context_env_ids, context_slots],
            "world_id": batch["world_id"][env_ids],
            "episode_id": batch["episode_id"][env_ids],
        }
        self._append_samples(samples, generated)
        return generated

    def _append_samples(self, samples: dict[str, torch.Tensor], generated: int) -> None:
        self._allocate_sample_storage()
        retained = generated
        if retained > self.capacity:
            selection = torch.randperm(retained, generator=self._generator, device=self.device)[
                : self.capacity
            ]
            samples = {name: value.index_select(0, selection) for name, value in samples.items()}
            retained = self.capacity
        positions = (
            self._sample_write + torch.arange(retained, dtype=torch.long, device=self.device)
        ).remainder(self.capacity)
        for name, value in samples.items():
            self._samples[name].index_copy_(0, positions, value)
        self._sample_write = (self._sample_write + retained) % self.capacity
        self._size = min(self.capacity, self._size + retained)
        self.total_samples_generated += generated

    def add_step(self, batch: dict[str, torch.Tensor]) -> int:
        """Append one vector-environment step without leaving the replay device."""
        self._validate_batch(batch)
        self._allocate_stream_storage()
        self.normalizer.update(batch)

        history_position = self._history_write
        self._history["proprio"][history_position].copy_(batch["proprio"])
        self._history["observation"][history_position].copy_(batch["observation"])
        self._history["action"][history_position].copy_(batch["action"])

        completed_steps = batch["episode_step"] + 1
        block_boundary = completed_steps.remainder(self.block_size) == 0
        valid_context = (
            block_boundary & (completed_steps >= self.block_size) & ~batch["reset_boundary"]
        )
        self._append_context_chunks(batch, history_position, valid_context)
        if self._context_counts is None:
            raise RuntimeError("Online context storage is not initialized")
        valid_sample = (
            block_boundary
            & (completed_steps >= self.query_steps)
            & (self._context_counts >= self._context_capacity)
            & ~batch["reset_boundary"]
        )
        generated = self._materialize_samples(
            batch, history_position, completed_steps, valid_sample
        )

        self._history_write = (history_position + 1) % self._history_length
        self.total_transitions += self.num_worlds
        return generated

    def _normalization_tensors(
        self,
        stats: NormalizationStats,
    ) -> tuple[torch.Tensor, ...]:
        if stats is self._cached_normalization and self._cached_normalization_tensors is not None:
            return self._cached_normalization_tensors
        tensors = tuple(
            torch.tensor(getattr(stats, name), dtype=torch.float32, device=self.device)
            for name in (
                "observation_mean",
                "observation_std",
                "proprio_mean",
                "proprio_std",
                "action_mean",
                "action_std",
            )
        )
        self._cached_normalization = stats
        self._cached_normalization_tensors = tensors
        return tensors

    def sample_batch(
        self,
        batch_size: int,
        normalization: NormalizationStats | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise RuntimeError(f"Online replay has {self._size} samples, batch_size={batch_size}")
        indices = torch.randperm(self._size, generator=self._generator, device=self.device)[
            :batch_size
        ]
        selected = {name: value.index_select(0, indices) for name, value in self._samples.items()}
        stats = normalization or self.normalization()
        obs_mean, obs_std, prop_mean, prop_std, action_mean, action_std = (
            self._normalization_tensors(stats)
        )

        observations = (selected["observation"] - obs_mean) / obs_std
        goals = (selected["goal_observation"] - obs_mean) / obs_std
        actions = (selected["action"] - action_mean) / action_std
        previous = (selected["previous_action"] - action_mean) / action_std
        context_before = (selected["context_before"] - prop_mean) / prop_std
        context_actions = (selected["context_actions"] - action_mean) / action_std
        context_after = (selected["context_after"] - prop_mean) / prop_std
        actions = actions.flatten(start_dim=2)
        previous = previous.flatten(start_dim=2)
        context = torch.cat(
            (context_before, context_actions.flatten(start_dim=2), context_after), dim=-1
        )
        mask = torch.ones(batch_size, self.horizon, dtype=torch.bool, device=self.device)
        return {
            "observation": observations,
            "goal_observation": goals,
            "action": actions,
            "previous_action": previous,
            "context": context,
            "context_mask": torch.ones(
                batch_size,
                self.context_tokens,
                dtype=torch.bool,
                device=self.device,
            ),
            "transition_mask": mask.clone(),
            "physical_mask": mask.clone(),
            "goal_mask": mask,
            "world_id": selected["world_id"],
            "episode_id": selected["episode_id"],
        }
