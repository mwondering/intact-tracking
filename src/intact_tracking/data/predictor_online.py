"""Compact online replay for context-conditioned five-step predictor training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intact_tracking.forward_predictor import physical_state_delta
from intact_tracking.forward_predictor_inputs import (
    CONTACT_BINARY_DIM,
    CONTACT_FORCE_DIM,
    FOOT_FEATURE_DIM,
)

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
    foot_mean: tuple[float, ...]
    foot_std: tuple[float, ...]
    contact_force_mean: tuple[float, ...]
    contact_force_std: tuple[float, ...]
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
        self.foot = _RunningMoments(FOOT_FEATURE_DIM, self.device)
        self.contact_force = _RunningMoments(CONTACT_FORCE_DIM, self.device)
        self.delta = _RunningMoments(70, self.device)
        self.frozen = False

    def update(self, batch: dict[str, torch.Tensor], valid: torch.Tensor) -> None:
        if self.frozen:
            return
        if valid.dtype != torch.bool or valid.shape != (batch["robot_state"].size(0),):
            raise ValueError("Forward Predictor normalization mask must be [num_worlds] bool")
        current = batch["robot_state"][valid]
        following = batch["next_robot_state"][valid]
        self.state.update(torch.cat((current, following), dim=0))
        self.action.update(batch["joint_target"][valid])
        self.foot.update(torch.cat((batch["foot"][valid], batch["next_foot"][valid]), dim=0))
        self.contact_force.update(
            torch.cat(
                (batch["contact_force"][valid], batch["next_contact_force"][valid]),
                dim=0,
            )
        )
        self.delta.update(physical_state_delta(current, following))

    def freeze(self) -> None:
        self.frozen = True

    @property
    def packed_size(self) -> int:
        return sum(
            1 + 2 * width
            for width in (
                self.dimensions.robot_state,
                self.dimensions.action,
                FOOT_FEATURE_DIM,
                CONTACT_FORCE_DIM,
                70,
            )
        )

    def packed_statistics(self, device: torch.device | str | None = None) -> torch.Tensor:
        packed = torch.cat(
            (
                self.state.packed(),
                self.action.packed(),
                self.foot.packed(),
                self.contact_force.packed(),
                self.delta.packed(),
            )
        )
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
        foot_mean, foot_std = read(FOOT_FEATURE_DIM)
        contact_force_mean, contact_force_std = read(CONTACT_FORCE_DIM)
        delta_mean, delta_std = read(70)
        return ForwardPredictorNormalizationStats(
            state_mean=_float_tuple(state_mean),
            state_std=_float_tuple(state_std),
            action_mean=_float_tuple(action_mean),
            action_std=_float_tuple(action_std),
            foot_mean=_float_tuple(foot_mean),
            foot_std=_float_tuple(foot_std),
            contact_force_mean=_float_tuple(contact_force_mean),
            contact_force_std=_float_tuple(contact_force_std),
            delta_mean=_float_tuple(delta_mean),
            delta_std=_float_tuple(delta_std),
            world_ids=world_ids,
            epsilon=self.epsilon,
        )


class ForwardPredictorReplayBuffer:
    """Store broad A rollouts plus their five-step nominal B counterfactuals."""

    REQUIRED_FIELDS = (
        "joint_target",
        "robot_state",
        "next_robot_state",
        "nominal_next_robot_state",
        "foot",
        "next_foot",
        "contact_force",
        "next_contact_force",
        "contact_binary",
        "next_contact_binary",
        "reset_boundary",
        "is_nominal",
        "world_id",
        "episode_id",
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
        history_steps: int = 10,
        context_history_steps: int = 100,
        positive_offset_steps: int = 5,
        capacity: int = 16384,
        sampling_mode: str = "motion_balanced",
        seed: int = 0,
        world_id_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if num_worlds < 1 or capacity < 1:
            raise ValueError("num_worlds and capacity must be positive")
        if horizon != 5 or history_steps != 10:
            raise ValueError("Forward Predictor replay uses horizon=5 and history_steps=10")
        if context_history_steps < history_steps:
            raise ValueError("context_history_steps must be at least history_steps")
        if positive_offset_steps != horizon:
            raise ValueError("The local representation positive is fixed to an exact 5-step shift")
        if sampling_mode not in {"uniform", "motion_balanced"}:
            raise ValueError("sampling_mode must be 'uniform' or 'motion_balanced'")
        if world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")

        self.num_worlds = int(num_worlds)
        self.dimensions = dimensions or RolloutDimensions()
        self.horizon = int(horizon)
        self.history_steps = int(history_steps)
        self.context_history_steps = int(context_history_steps)
        self.positive_offset_steps = int(positive_offset_steps)
        self.capacity = int(capacity)
        retained_collector_steps = (
            2 * self.horizon * max(1, (self.capacity + self.num_worlds - 1) // self.num_worlds)
        )
        self.ring_steps = self.context_history_steps + self.horizon + retained_collector_steps
        self.sampling_mode = sampling_mode
        self.world_id_offset = int(world_id_offset)
        self.device = torch.device(device or "cpu")
        self._world_ids = tuple(range(world_id_offset, world_id_offset + num_worlds))
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._target_offsets = torch.arange(self.horizon - 1, -1, -1, device=self.device)
        self._history: dict[str, torch.Tensor] = {}
        self._samples: dict[str, torch.Tensor] = {}
        self._reset_history: torch.Tensor | None = None
        self._history_stamp: torch.Tensor | None = None
        self._collector_step = 0
        self._sample_write = 0
        self._size = 0
        self._active_indices: torch.Tensor | None = None
        self._world_ids_validated = False
        self._local_positive_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self.total_samples_generated = 0
        self.total_transitions = 0
        self.normalizer = ForwardPredictorNormalization(self.dimensions, device=self.device)

    def __len__(self) -> int:
        if not self._samples:
            return 0
        return int(self._active_sample_indices().numel())

    @property
    def world_ids(self) -> tuple[int, ...]:
        return self._world_ids

    @property
    def collector_step(self) -> int:
        return self._collector_step

    @property
    def storage_bytes(self) -> int:
        tensors = [*self._history.values(), *self._samples.values()]
        if self._reset_history is not None:
            tensors.append(self._reset_history)
        if self._history_stamp is not None:
            tensors.append(self._history_stamp)
        return sum(value.numel() * value.element_size() for value in tensors)

    @property
    def estimated_storage_bytes(self) -> int:
        dims = self.dimensions
        history_floats = (
            self.ring_steps
            * self.num_worlds
            * (2 * dims.robot_state + dims.action + FOOT_FEATURE_DIM + CONTACT_FORCE_DIM)
        )
        sample_floats = self.capacity * (
            (2 * self.horizon + 1) * dims.robot_state
            + self.horizon * dims.action
            + (self.horizon + 1) * (FOOT_FEATURE_DIM + CONTACT_FORCE_DIM)
        )
        history_integers = 4 * self.ring_steps * self.num_worlds + self.ring_steps
        sample_integers = 7 * self.capacity
        flags = self.ring_steps * self.num_worlds * (1 + CONTACT_BINARY_DIM) + self.capacity * (
            2 + (self.horizon + 1) * CONTACT_BINARY_DIM
        )
        return (
            4 * (history_floats + sample_floats) + 8 * (history_integers + sample_integers) + flags
        )

    def _allocate(self) -> None:
        if self._history:
            return
        dims = self.dimensions
        history_prefix = (self.ring_steps, self.num_worlds)
        self._history = {
            "state": torch.zeros((*history_prefix, dims.robot_state), device=self.device),
            "action": torch.zeros((*history_prefix, dims.action), device=self.device),
            "nominal_next_state": torch.zeros(
                (*history_prefix, dims.robot_state), device=self.device
            ),
            "foot": torch.zeros((*history_prefix, FOOT_FEATURE_DIM), device=self.device),
            "contact_force": torch.zeros((*history_prefix, CONTACT_FORCE_DIM), device=self.device),
            "contact_binary": torch.zeros(
                (*history_prefix, CONTACT_BINARY_DIM), dtype=torch.bool, device=self.device
            ),
            "episode_id": torch.full(history_prefix, -1, dtype=torch.long, device=self.device),
            "episode_step": torch.full(history_prefix, -1, dtype=torch.long, device=self.device),
            "motion_id": torch.full(history_prefix, -1, dtype=torch.long, device=self.device),
            "motion_step": torch.full(history_prefix, -1, dtype=torch.long, device=self.device),
        }
        self._reset_history = torch.ones(history_prefix, dtype=torch.bool, device=self.device)
        self._history_stamp = torch.full(
            (self.ring_steps,), -1, dtype=torch.long, device=self.device
        )
        self._samples = {
            "state": torch.empty(
                (self.capacity, self.horizon + 1, dims.robot_state), device=self.device
            ),
            "nominal_state": torch.empty(
                (self.capacity, self.horizon, dims.robot_state), device=self.device
            ),
            "action": torch.empty((self.capacity, self.horizon, dims.action), device=self.device),
            "foot": torch.empty(
                (self.capacity, self.horizon + 1, FOOT_FEATURE_DIM), device=self.device
            ),
            "contact_force": torch.empty(
                (self.capacity, self.horizon + 1, CONTACT_FORCE_DIM), device=self.device
            ),
            "contact_binary": torch.empty(
                (self.capacity, self.horizon + 1, CONTACT_BINARY_DIM),
                dtype=torch.bool,
                device=self.device,
            ),
            "is_nominal": torch.empty(self.capacity, dtype=torch.bool, device=self.device),
            "world_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "env_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "episode_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "episode_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "collector_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "context_full": torch.empty(self.capacity, dtype=torch.bool, device=self.device),
        }

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        missing = sorted(set(self.REQUIRED_FIELDS).difference(batch))
        if missing:
            raise KeyError(f"Forward Predictor transition batch is missing fields: {missing}")
        dims = self.dimensions
        expected = {
            "joint_target": (self.num_worlds, dims.action),
            "robot_state": (self.num_worlds, dims.robot_state),
            "next_robot_state": (self.num_worlds, dims.robot_state),
            "nominal_next_robot_state": (self.num_worlds, dims.robot_state),
            "foot": (self.num_worlds, FOOT_FEATURE_DIM),
            "next_foot": (self.num_worlds, FOOT_FEATURE_DIM),
            "contact_force": (self.num_worlds, CONTACT_FORCE_DIM),
            "next_contact_force": (self.num_worlds, CONTACT_FORCE_DIM),
            "contact_binary": (self.num_worlds, CONTACT_BINARY_DIM),
            "next_contact_binary": (self.num_worlds, CONTACT_BINARY_DIM),
            "reset_boundary": (self.num_worlds,),
            "is_nominal": (self.num_worlds,),
            "world_id": (self.num_worlds,),
            "episode_id": (self.num_worlds,),
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
        for name in ("reset_boundary", "is_nominal", "contact_binary", "next_contact_binary"):
            if batch[name].dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        if not self._world_ids_validated:
            expected_ids = torch.arange(
                self.world_id_offset,
                self.world_id_offset + self.num_worlds,
                device=self.device,
            )
            if not torch.equal(batch["world_id"], expected_ids):
                raise ValueError("Forward Predictor replay world IDs do not match vector slots")
            self._world_ids_validated = True

    def _invalidate_caches(self) -> None:
        self._active_indices = None
        self._local_positive_cache = None

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
        self._invalidate_caches()
        self.total_samples_generated += count

    def _active_sample_indices(self) -> torch.Tensor:
        """Return samples whose 100-frame context still exists in the archive."""

        if self._active_indices is not None:
            return self._active_indices
        if self._size == 0:
            self._active_indices = torch.empty(0, dtype=torch.long, device=self.device)
            return self._active_indices
        positions = torch.arange(
            self.capacity if self._size == self.capacity else self._size,
            device=self.device,
        )
        latest_step = self._collector_step - 1
        minimum_sample_step = (
            latest_step - self.ring_steps + self.context_history_steps + self.horizon
        )
        fresh = self._samples["collector_step"].index_select(0, positions) >= minimum_sample_step
        self._active_indices = positions[fresh]
        return self._active_indices

    def add_step(self, batch: dict[str, torch.Tensor]) -> int:
        self._validate_batch(batch)
        self._allocate()
        assert self._reset_history is not None
        assert self._history_stamp is not None
        valid_transition = ~batch["reset_boundary"]
        self.normalizer.update(batch, valid_transition)

        position = self._collector_step % self.ring_steps
        self._history["state"][position].copy_(batch["robot_state"])
        self._history["action"][position].copy_(batch["joint_target"])
        self._history["nominal_next_state"][position].copy_(batch["nominal_next_robot_state"])
        self._history["foot"][position].copy_(batch["foot"])
        self._history["contact_force"][position].copy_(batch["contact_force"])
        self._history["contact_binary"][position].copy_(batch["contact_binary"])
        self._history["episode_id"][position].copy_(batch["episode_id"])
        self._history["episode_step"][position].copy_(batch["episode_step"])
        self._history["motion_id"][position].copy_(batch["motion_id"])
        self._history["motion_step"][position].copy_(batch["motion_step"])
        self._reset_history[position].copy_(batch["reset_boundary"])
        self._history_stamp[position] = self._collector_step

        target_ids = (position - self._target_offsets).remainder(self.ring_steps)
        target_steps = self._history["episode_step"][target_ids]
        target_episode_ids = self._history["episode_id"][target_ids]
        target_stamps = self._history_stamp[target_ids]
        target_start = target_steps[0]
        complete_window = (
            target_steps
            == target_start[None] + torch.arange(self.horizon, device=self.device)[:, None]
        ).all(dim=0)
        complete_window &= (target_episode_ids == batch["episode_id"][None]).all(dim=0)
        complete_window &= (target_stamps == self._collector_step - self._target_offsets).all()
        crosses_reset = self._reset_history[target_ids].any(dim=0)
        valid = (
            (batch["episode_step"] + 1 >= self.horizon)
            & complete_window
            & ~crosses_reset
            & ~batch["reset_boundary"]
        )
        if (self._collector_step + 1) % self.horizon:
            valid.zero_()
        env_ids = valid.nonzero(as_tuple=False).flatten()
        count = int(env_ids.numel())
        if count:

            def history_window(name: str) -> torch.Tensor:
                return self._history[name][target_ids[:, None], env_ids[None, :]].permute(1, 0, 2)

            target_state = history_window("state")
            target_action = history_window("action")
            target_foot = history_window("foot")
            target_contact_force = history_window("contact_force")
            target_contact_binary = history_window("contact_binary")
            target_nominal_state = history_window("nominal_next_state")
            context_start = self._collector_step - (self.horizon - 1) - self.context_history_steps
            context_ids = (
                context_start + torch.arange(self.context_history_steps, device=self.device)
            ).remainder(self.ring_steps)
            expected_context_steps = (
                target_start[env_ids, None]
                - self.context_history_steps
                + torch.arange(self.context_history_steps, device=self.device)[None]
            )
            context_full = (
                (expected_context_steps >= 0)
                & (
                    self._history["episode_step"][context_ids[:, None], env_ids[None, :]].T
                    == expected_context_steps
                )
                & (
                    self._history["episode_id"][context_ids[:, None], env_ids[None, :]].T
                    == target_episode_ids[0, env_ids, None]
                )
            ).all(dim=1)
            samples = {
                "state": torch.cat((target_state, batch["next_robot_state"][env_ids, None]), dim=1),
                "nominal_state": target_nominal_state,
                "action": target_action,
                "foot": torch.cat((target_foot, batch["next_foot"][env_ids, None]), dim=1),
                "contact_force": torch.cat(
                    (target_contact_force, batch["next_contact_force"][env_ids, None]), dim=1
                ),
                "contact_binary": torch.cat(
                    (target_contact_binary, batch["next_contact_binary"][env_ids, None]), dim=1
                ),
                "is_nominal": batch["is_nominal"][env_ids],
                "world_id": batch["world_id"][env_ids],
                "env_id": env_ids,
                "episode_id": target_episode_ids[0, env_ids],
                "episode_step": target_start[env_ids],
                "motion_id": self._history["motion_id"][target_ids[0], env_ids],
                "motion_step": self._history["motion_step"][target_ids[0], env_ids],
                "collector_step": torch.full_like(env_ids, self._collector_step),
                "context_full": context_full,
            }
            self._append_samples(samples, count)

        self._collector_step += 1
        self._invalidate_caches()
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
                "foot_mean",
                "foot_std",
                "contact_force_mean",
                "contact_force_std",
                "delta_mean",
                "delta_std",
            )
        )

    def _local_positive_candidates(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Index exact +/-5-frame same-world windows without changing batch sampling."""

        if self._local_positive_cache is not None:
            return self._local_positive_cache
        active = self._active_sample_indices()
        if active.numel() == 0:
            candidates = torch.empty((0, 2), dtype=torch.long, device=self.device)
            valid = torch.empty_like(candidates, dtype=torch.bool)
            lookup = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
            self._local_positive_cache = (active, candidates, valid, lookup)
            return self._local_positive_cache

        identity = torch.stack(
            (
                self._samples["world_id"].index_select(0, active),
                self._samples["episode_id"].index_select(0, active),
                self._samples["motion_id"].index_select(0, active),
            ),
            dim=1,
        )
        _, group_id = torch.unique(identity, dim=0, sorted=True, return_inverse=True)
        episode_step = self._samples["episode_step"].index_select(0, active)
        order = torch.argsort(episode_step, stable=True)
        order = order.index_select(0, torch.argsort(group_id.index_select(0, order), stable=True))
        anchors = active.index_select(0, order)
        sorted_group = group_id.index_select(0, order)
        sorted_step = episode_step.index_select(0, order)
        row = torch.arange(active.numel(), device=self.device)[:, None]
        candidate_row = row + torch.tensor((-1, 1), device=self.device)[None]
        in_bounds = (candidate_row >= 0) & (candidate_row < active.numel())
        safe_row = candidate_row.clamp(0, active.numel() - 1)
        gap = (sorted_step[safe_row] - sorted_step[:, None]).abs()
        valid = (
            in_bounds
            & (sorted_group[safe_row] == sorted_group[:, None])
            & (gap == self.positive_offset_steps)
        )
        candidates = anchors[safe_row]
        lookup = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        lookup[anchors] = torch.arange(anchors.numel(), device=self.device)
        self._local_positive_cache = (anchors, candidates, valid, lookup)
        return self._local_positive_cache

    def _positive_ready_indices(self) -> torch.Tensor:
        anchors, candidates, valid, _ = self._local_positive_candidates()
        if anchors.numel() == 0:
            return anchors
        valid = valid.clone()
        valid &= self._samples["context_full"].index_select(0, anchors)[:, None]
        valid &= self._samples["context_full"][candidates]
        return anchors[valid.any(dim=1)]

    def can_sample_positive_pairs(self, count: int = 1) -> bool:
        return count > 0 and self._positive_ready_indices().numel() >= count

    def _sampling_weights(self, indices: torch.Tensor) -> torch.Tensor:
        motion_ids = self._samples["motion_id"].index_select(0, indices)
        _, inverse, counts = torch.unique(motion_ids, return_inverse=True, return_counts=True)
        return counts[inverse].float().reciprocal()

    def _sample_indices(self, batch_size: int, *, positive_ready_only: bool) -> torch.Tensor:
        candidates = (
            self._positive_ready_indices() if positive_ready_only else self._active_sample_indices()
        )
        if candidates.numel() < batch_size:
            raise RuntimeError(
                f"Forward Predictor replay has {candidates.numel()} eligible samples, "
                f"batch_size={batch_size}"
            )
        if self.sampling_mode == "uniform":
            ranks = torch.randperm(
                candidates.numel(), generator=self._generator, device=self.device
            )[:batch_size]
        else:
            ranks = torch.multinomial(
                self._sampling_weights(candidates),
                batch_size,
                replacement=False,
                generator=self._generator,
            )
        return candidates.index_select(0, ranks)

    def _choose_positive_indices(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchors, candidates, candidate_valid, lookup = self._local_positive_candidates()
        if anchors.numel() == 0:
            return indices, torch.zeros_like(indices, dtype=torch.bool)
        rows = lookup.index_select(0, indices)
        row_valid = rows >= 0
        safe_rows = rows.clamp_min(0)
        choices = candidates.index_select(0, safe_rows)
        valid = candidate_valid.index_select(0, safe_rows) & row_valid[:, None]
        valid &= self._samples["context_full"].index_select(0, indices)[:, None]
        valid &= self._samples["context_full"][choices]
        scores = torch.rand(valid.shape, generator=self._generator, device=self.device).masked_fill(
            ~valid, -1.0
        )
        column = scores.argmax(dim=1)
        selected = choices.gather(1, column[:, None]).squeeze(1)
        pair_valid = valid.any(dim=1)
        selected = torch.where(pair_valid, selected, indices)
        return selected, pair_valid

    def _materialize_context(self, selected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self._history_stamp is not None
        target_start_collector = selected["collector_step"] - (self.horizon - 1)
        context_absolute_steps = (
            target_start_collector[:, None]
            - self.context_history_steps
            + torch.arange(self.context_history_steps, device=self.device)[None]
        )
        context_slots = context_absolute_steps.remainder(self.ring_steps)
        env_ids = selected["env_id"][:, None].expand_as(context_slots)
        archive_present = self._history_stamp[context_slots] == context_absolute_steps
        history_episode = self._history["episode_id"][context_slots, env_ids]
        history_episode_step = self._history["episode_step"][context_slots, env_ids]
        expected_episode_step = (
            selected["episode_step"][:, None]
            - self.context_history_steps
            + torch.arange(self.context_history_steps, device=self.device)[None]
        )
        history_valid = (
            archive_present
            & (expected_episode_step >= 0)
            & (history_episode == selected["episode_id"][:, None])
            & (history_episode_step == expected_episode_step)
        )
        predictor_slots = context_slots[:, -self.history_steps :]
        predictor_env_ids = env_ids[:, -self.history_steps :]
        return {
            "state": self._history["state"][context_slots, env_ids],
            "action": self._history["action"][context_slots, env_ids],
            "valid": history_valid,
            "foot": self._history["foot"][predictor_slots, predictor_env_ids],
            "contact_force": self._history["contact_force"][predictor_slots, predictor_env_ids],
            "contact_binary": self._history["contact_binary"][predictor_slots, predictor_env_ids],
        }

    @staticmethod
    def _normalize_masked(
        value: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (value - mean) / std
        return torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))

    def sample_batch(
        self,
        batch_size: int,
        normalization: ForwardPredictorNormalizationStats,
        *,
        positive_ready_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        indices = self._sample_indices(batch_size, positive_ready_only=positive_ready_only)
        positive_indices, positive_pair_valid = self._choose_positive_indices(indices)
        selected = {name: value.index_select(0, indices) for name, value in self._samples.items()}
        positive = {
            name: value.index_select(0, positive_indices) for name, value in self._samples.items()
        }
        context = self._materialize_context(selected)
        positive_context = self._materialize_context(positive)
        (
            state_mean,
            state_std,
            action_mean,
            action_std,
            foot_mean,
            foot_std,
            contact_force_mean,
            contact_force_std,
            delta_mean,
            delta_std,
        ) = self._normalization_tensors(normalization, self.device)
        history_state = self._normalize_masked(
            context["state"], state_mean, state_std, context["valid"]
        )
        history_action = self._normalize_masked(
            context["action"], action_mean, action_std, context["valid"]
        )
        positive_history_state = self._normalize_masked(
            positive_context["state"], state_mean, state_std, positive_context["valid"]
        )
        positive_history_action = self._normalize_masked(
            positive_context["action"], action_mean, action_std, positive_context["valid"]
        )
        return {
            "state": (selected["state"] - state_mean) / state_std,
            "nominal_state": (selected["nominal_state"] - state_mean) / state_std,
            "action": (selected["action"] - action_mean) / action_std,
            "history_state": history_state,
            "history_action": history_action,
            "history_valid": context["valid"],
            "positive_current_state": (positive["state"][:, 0] - state_mean) / state_std,
            "positive_history_state": positive_history_state,
            "positive_history_action": positive_history_action,
            "positive_history_valid": positive_context["valid"],
            "positive_pair_valid": positive_pair_valid,
            "foot": (selected["foot"] - foot_mean) / foot_std,
            "history_foot": (context["foot"] - foot_mean) / foot_std,
            "contact_force": (selected["contact_force"] - contact_force_mean) / contact_force_std,
            "contact_binary": selected["contact_binary"],
            "history_contact_force": (context["contact_force"] - contact_force_mean)
            / contact_force_std,
            "history_contact_binary": context["contact_binary"],
            "is_nominal": selected["is_nominal"],
            "world_id": selected["world_id"],
            "motion_id": selected["motion_id"],
            "context_full": context["valid"].all(dim=1),
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "foot_mean": foot_mean,
            "foot_std": foot_std,
            "contact_force_mean": contact_force_mean,
            "contact_force_std": contact_force_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
