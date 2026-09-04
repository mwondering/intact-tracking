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
    """Store five-step targets and reconstruct long causal context from a time archive."""

    REQUIRED_FIELDS = (
        "joint_target",
        "robot_state",
        "next_robot_state",
        "foot",
        "next_foot",
        "contact_force",
        "next_contact_force",
        "contact_binary",
        "next_contact_binary",
        "reset_boundary",
        "world_id",
        "episode_id",
        "episode_step",
        "motion_id",
        "motion_step",
        "motion_group_id",
        "dynamics_id",
    )

    def __init__(
        self,
        *,
        num_worlds: int,
        dimensions: RolloutDimensions | None = None,
        horizon: int = 5,
        history_steps: int = 10,
        context_history_steps: int = 100,
        positive_max_offset_steps: int = 20,
        dynamics_classes: int | None = None,
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
        if positive_max_offset_steps < horizon:
            raise ValueError("positive_max_offset_steps must be at least one replay horizon")
        self.grouped_dynamics = dynamics_classes is not None
        resolved_dynamics_classes = num_worlds if dynamics_classes is None else dynamics_classes
        if resolved_dynamics_classes < 1 or num_worlds % resolved_dynamics_classes:
            raise ValueError("dynamics_classes must be positive and divide num_worlds")
        if sampling_mode not in {"uniform", "motion_balanced"}:
            raise ValueError("sampling_mode must be 'uniform' or 'motion_balanced'")
        if world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")
        self.num_worlds = int(num_worlds)
        self.dimensions = dimensions or RolloutDimensions()
        self.horizon = int(horizon)
        self.history_steps = int(history_steps)
        self.context_history_steps = int(context_history_steps)
        self.positive_max_offset_steps = int(positive_max_offset_steps)
        self.dynamics_classes = int(resolved_dynamics_classes)
        self.motion_group_count = self.num_worlds // self.dynamics_classes
        self.capacity = int(capacity)
        # A full replay at the nominal one sample/world/five steps retention rate
        # needs roughly horizon * capacity / num_worlds collector frames.  Keep
        # twice that span so ordinary reset losses cannot invalidate live samples.
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
        self._history_write = 0
        self._collector_step = 0
        self._sample_write = 0
        self._size = 0
        self._active_indices: torch.Tensor | None = None
        self._world_ids_validated = False
        self._sampling_weights: torch.Tensor | None = None
        self._local_positive_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._temporal_run_cache: dict[tuple[int, bool], tuple[torch.Tensor, torch.Tensor]] = {}
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
            * (dims.robot_state + dims.action + FOOT_FEATURE_DIM + CONTACT_FORCE_DIM)
        )
        sample_floats = self.capacity * (
            (self.horizon + 1) * dims.robot_state
            + self.horizon * dims.action
            + (self.horizon + 1) * FOOT_FEATURE_DIM
            + (self.horizon + 1) * CONTACT_FORCE_DIM
        )
        history_integers = 4 * self.ring_steps * self.num_worlds + self.ring_steps
        sample_integers = 10 * self.capacity
        flags = self.ring_steps * self.num_worlds * (1 + CONTACT_BINARY_DIM) + self.capacity * (
            1 + (self.horizon + 1) * CONTACT_BINARY_DIM
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
            "world_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "env_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "episode_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "episode_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "motion_group_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "cohort_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "collector_step": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "dynamics_id": torch.empty(self.capacity, dtype=torch.long, device=self.device),
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
            "foot": (self.num_worlds, FOOT_FEATURE_DIM),
            "next_foot": (self.num_worlds, FOOT_FEATURE_DIM),
            "contact_force": (self.num_worlds, CONTACT_FORCE_DIM),
            "next_contact_force": (self.num_worlds, CONTACT_FORCE_DIM),
            "contact_binary": (self.num_worlds, CONTACT_BINARY_DIM),
            "next_contact_binary": (self.num_worlds, CONTACT_BINARY_DIM),
            "reset_boundary": (self.num_worlds,),
            "world_id": (self.num_worlds,),
            "episode_id": (self.num_worlds,),
            "episode_step": (self.num_worlds,),
            "motion_id": (self.num_worlds,),
            "motion_step": (self.num_worlds,),
            "motion_group_id": (self.num_worlds,),
            "dynamics_id": (self.num_worlds,),
        }
        for name, shape in expected.items():
            value = batch[name]
            if not isinstance(value, torch.Tensor) or value.device != self.device:
                raise ValueError(f"Forward Predictor field {name!r} must be on {self.device}")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Forward Predictor field {name!r} has {tuple(value.shape)}, expected {shape}"
                )
        if batch["reset_boundary"].dtype != torch.bool:
            raise ValueError("reset_boundary must be boolean")
        if batch["contact_binary"].dtype != torch.bool:
            raise ValueError("contact_binary must be boolean")
        if batch["next_contact_binary"].dtype != torch.bool:
            raise ValueError("next_contact_binary must be boolean")
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
        self._active_indices = None
        self._sampling_weights = None
        self._local_positive_cache = None
        self._temporal_run_cache.clear()
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
            target_state = self._history["state"][target_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            target_action = self._history["action"][target_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            target_foot = self._history["foot"][target_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            target_contact_force = self._history["contact_force"][
                target_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
            target_contact_binary = self._history["contact_binary"][
                target_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
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
            cohort_id = (
                torch.full_like(batch["motion_group_id"][env_ids], self._collector_step) << 32
            ) + batch["motion_group_id"][env_ids]
            samples = {
                "state": torch.cat((target_state, batch["next_robot_state"][env_ids, None]), dim=1),
                "action": target_action,
                "foot": torch.cat((target_foot, batch["next_foot"][env_ids, None]), dim=1),
                "contact_force": torch.cat(
                    (target_contact_force, batch["next_contact_force"][env_ids, None]), dim=1
                ),
                "contact_binary": torch.cat(
                    (target_contact_binary, batch["next_contact_binary"][env_ids, None]), dim=1
                ),
                "world_id": batch["world_id"][env_ids],
                "env_id": env_ids,
                "episode_id": target_episode_ids[0, env_ids],
                "episode_step": target_start[env_ids],
                "motion_id": self._history["motion_id"][target_ids[0], env_ids],
                "motion_step": self._history["motion_step"][target_ids[0], env_ids],
                "motion_group_id": batch["motion_group_id"][env_ids],
                "cohort_id": cohort_id,
                "collector_step": torch.full_like(env_ids, self._collector_step),
                "dynamics_id": batch["dynamics_id"][env_ids],
                "context_full": context_full,
            }
            self._append_samples(samples, count)

        self._history_write = (position + 1) % self.ring_steps
        self._collector_step += 1
        self._active_indices = None
        self._sampling_weights = None
        self._local_positive_cache = None
        self._temporal_run_cache.clear()
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

    def _motion_sampling_weights(self) -> torch.Tensor:
        active = self._active_sample_indices()
        if self._sampling_weights is None or self._sampling_weights.numel() != active.numel():
            motion_ids = self._samples["motion_id"].index_select(0, active)
            _, inverse, counts = torch.unique(
                motion_ids,
                return_inverse=True,
                return_counts=True,
            )
            self._sampling_weights = counts[inverse].float().reciprocal()
        return self._sampling_weights

    def can_sample_dynamics_pairs(
        self,
        batch_size: int,
        *,
        contrastive_block_size: int | None = None,
        contrastive_ready_only: bool = False,
    ) -> bool:
        """Return whether replay can build the requested contrastive batch layout."""

        active_count = len(self)
        if batch_size < 2 or batch_size % 2 or active_count < batch_size:
            return False
        if self.grouped_dynamics:
            block_size = contrastive_block_size or batch_size
            if block_size % self.dynamics_classes or block_size // self.dynamics_classes < 2:
                return False
            views = block_size // self.dynamics_classes
            if (views - 1) * self.horizon > self.positive_max_offset_steps:
                return False
            runs, _ = self._grouped_temporal_runs(
                views,
                contrastive_ready_only=contrastive_ready_only,
            )
            return runs.size(0) >= batch_size // block_size
        anchors, candidates, candidate_valid = self._local_positive_candidates()
        if contrastive_ready_only and anchors.numel():
            anchor_full = self._samples["context_full"].index_select(0, anchors)
            candidate_full = self._samples["context_full"][candidates]
            candidate_valid = candidate_valid & anchor_full[:, None] & candidate_full
        eligible = candidate_valid.any(dim=1)
        return int(eligible.sum()) >= batch_size // 2

    def _local_positive_candidates(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return same-world, same-episode local temporal neighbors for each sample."""

        if self._local_positive_cache is not None:
            return self._local_positive_cache
        active = self._active_sample_indices()
        neighbor_count = self.positive_max_offset_steps // self.horizon
        if active.numel() == 0:
            anchors = active
            candidates = torch.empty((0, 2 * neighbor_count), dtype=torch.long, device=self.device)
            valid = torch.empty_like(candidates, dtype=torch.bool)
            self._local_positive_cache = (anchors, candidates, valid)
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

        negative_offsets = -torch.arange(
            neighbor_count, 0, -1, dtype=torch.long, device=self.device
        )
        positive_offsets = torch.arange(1, neighbor_count + 1, dtype=torch.long, device=self.device)
        offsets = torch.cat((negative_offsets, positive_offsets))
        row = torch.arange(active.numel(), device=self.device)[:, None]
        candidate_row = row + offsets[None]
        in_bounds = (candidate_row >= 0) & (candidate_row < active.numel())
        safe_row = candidate_row.clamp(0, active.numel() - 1)
        candidate_step = sorted_step[safe_row]
        gap = (candidate_step - sorted_step[:, None]).abs()
        candidate_valid = (
            in_bounds
            & (sorted_group[safe_row] == sorted_group[:, None])
            & (gap > 0)
            & (gap <= self.positive_max_offset_steps)
        )
        candidates = anchors[safe_row]
        self._local_positive_cache = (anchors, candidates, candidate_valid)
        return self._local_positive_cache

    def _grouped_temporal_runs(
        self,
        views: int,
        *,
        contrastive_ready_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build non-overlapping runs of nearby complete motion/phase cohorts."""

        key = (views, contrastive_ready_only)
        cached = self._temporal_run_cache.get(key)
        if cached is not None:
            return cached
        empty_runs = torch.empty(
            (0, views, self.dynamics_classes), dtype=torch.long, device=self.device
        )
        empty_motion = torch.empty(0, dtype=torch.long, device=self.device)
        if views < 2 or (views - 1) * self.horizon > self.positive_max_offset_steps:
            result = (empty_runs, empty_motion)
            self._temporal_run_cache[key] = result
            return result

        active = self._active_sample_indices()
        if active.numel() == 0:
            result = (empty_runs, empty_motion)
            self._temporal_run_cache[key] = result
            return result
        cohort_ids = self._samples["cohort_id"].index_select(0, active)
        unique_cohorts, inverse, counts = torch.unique(
            cohort_ids,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        complete = (counts == self.dynamics_classes).nonzero(as_tuple=False).flatten()
        if complete.numel() == 0:
            result = (empty_runs, empty_motion)
            self._temporal_run_cache[key] = result
            return result

        order = torch.argsort(inverse)
        starts = counts.cumsum(0) - counts
        class_offsets = torch.arange(self.dynamics_classes, device=self.device)
        complete_ranks = order[starts.index_select(0, complete)[:, None] + class_offsets[None]]
        cohort_positions = active.index_select(0, complete_ranks.flatten()).view(
            -1, self.dynamics_classes
        )
        cohort_dynamics = self._samples["dynamics_id"][cohort_positions]
        dynamics_order = torch.argsort(cohort_dynamics, dim=1)
        cohort_positions = torch.gather(cohort_positions, 1, dynamics_order)
        cohort_dynamics = torch.gather(cohort_dynamics, 1, dynamics_order)
        expected_dynamics = torch.arange(self.dynamics_classes, device=self.device).expand_as(
            cohort_dynamics
        )
        if not torch.equal(cohort_dynamics, expected_dynamics):
            raise RuntimeError("A motion/phase cohort does not contain each dynamics class once")

        complete_ids = unique_cohorts.index_select(0, complete)
        max_offset_units = self.positive_max_offset_steps // self.horizon
        offset_units = torch.arange(views, device=self.device)
        use_sparse_max_offset = views >= 4 and max_offset_units >= views
        if use_sparse_max_offset:
            offset_units[-1] = max_offset_units
        temporal_offsets = self.horizon * offset_units
        desired_ids = complete_ids[:, None] + (temporal_offsets[None] << 32)
        located = torch.searchsorted(complete_ids, desired_ids)
        present = located < complete_ids.numel()
        safe_located = located.clamp(max=complete_ids.numel() - 1)
        present &= complete_ids[safe_located] == desired_ids

        collector_step = self._samples["collector_step"][cohort_positions[:, 0]]
        # Partition stored cohorts into disjoint runs.  Rotating the partition
        # phase after each newly collected cohort avoids permanently dropping
        # the slots not selected by a sparse offset pattern (default: 0/5/10/20).
        partition_width = (max_offset_units if use_sparse_max_offset else views - 1) + 1
        partition_phase = (self._collector_step // self.horizon) % partition_width
        aligned_start = (collector_step // self.horizon - partition_phase).remainder(
            partition_width
        ) == 0
        run_positions = cohort_positions[safe_located]
        run_valid = present.all(dim=1) & aligned_start

        run_collector = self._samples["collector_step"][run_positions]
        expected_collector = collector_step[:, None, None] + temporal_offsets[None, :, None]
        run_valid &= (run_collector == expected_collector).all(dim=(1, 2))
        run_world = self._samples["world_id"][run_positions]
        run_valid &= (run_world == run_world[:, :1]).all(dim=(1, 2))
        run_episode = self._samples["episode_id"][run_positions]
        run_valid &= (run_episode == run_episode[:, :1]).all(dim=(1, 2))
        run_valid &= (run_episode == run_episode[:, :, :1]).all(dim=(1, 2))
        run_motion = self._samples["motion_id"][run_positions]
        run_valid &= (run_motion == run_motion[:, :1]).all(dim=(1, 2))
        run_valid &= (run_motion == run_motion[:, :, :1]).all(dim=(1, 2))
        run_episode_step = self._samples["episode_step"][run_positions]
        expected_episode_step = run_episode_step[:, :1] + temporal_offsets[None, :, None]
        run_valid &= (run_episode_step == expected_episode_step).all(dim=(1, 2))
        run_valid &= (run_episode_step == run_episode_step[:, :, :1]).all(dim=(1, 2))
        run_motion_step = self._samples["motion_step"][run_positions]
        expected_motion_step = run_motion_step[:, :1] + temporal_offsets[None, :, None]
        run_valid &= (run_motion_step == expected_motion_step).all(dim=(1, 2))
        run_valid &= (run_motion_step == run_motion_step[:, :, :1]).all(dim=(1, 2))
        if contrastive_ready_only:
            full_context = self._samples["context_full"][run_positions]
            has_positive = full_context.sum(dim=1)[:, None, :] > 1
            has_negative = full_context.sum(dim=2)[:, :, None] > 1
            run_valid &= (full_context & has_positive & has_negative).any(dim=(1, 2))

        runs = run_positions[run_valid]
        motions = run_motion[run_valid, 0, 0]
        result = (runs, motions)
        self._temporal_run_cache[key] = result
        return result

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        active = self._active_sample_indices()
        if self.sampling_mode == "uniform":
            ranks = torch.randperm(active.numel(), generator=self._generator, device=self.device)[
                :batch_size
            ]
        else:
            ranks = torch.multinomial(
                self._motion_sampling_weights(),
                batch_size,
                replacement=False,
                generator=self._generator,
            )
        return active.index_select(0, ranks)

    def _sample_paired_dynamics_indices(
        self,
        batch_size: int,
        *,
        contrastive_ready_only: bool,
    ) -> torch.Tensor:
        if batch_size % 2:
            raise ValueError("Dynamics-paired batch_size must be even")
        anchors, candidates, candidate_valid = self._local_positive_candidates()
        if contrastive_ready_only and anchors.numel():
            anchor_full = self._samples["context_full"].index_select(0, anchors)
            candidate_full = self._samples["context_full"][candidates]
            candidate_valid = candidate_valid & anchor_full[:, None] & candidate_full
        eligible = candidate_valid.any(dim=1)
        pair_count = batch_size // 2
        if int(eligible.sum()) < pair_count:
            raise RuntimeError(
                "Forward Predictor replay cannot form enough local same-world history "
                f"pairs for batch_size={batch_size}"
            )
        if self.sampling_mode == "uniform":
            anchor_weights = eligible.float()
        else:
            active = self._active_sample_indices()
            anchor_weights = (
                self._motion_sampling_weights().index_select(0, torch.searchsorted(active, anchors))
                * eligible
            )
        chosen_rows = torch.multinomial(
            anchor_weights,
            pair_count,
            replacement=False,
            generator=self._generator,
        )
        chosen_candidates = candidates.index_select(0, chosen_rows)
        chosen_valid = candidate_valid.index_select(0, chosen_rows)
        random_score = torch.rand(
            chosen_valid.shape,
            generator=self._generator,
            device=self.device,
        ).masked_fill(~chosen_valid, -1.0)
        positive_column = random_score.argmax(dim=1)
        positives = chosen_candidates.gather(1, positive_column[:, None]).squeeze(1)
        selected_anchors = anchors.index_select(0, chosen_rows)
        if bool(torch.eq(selected_anchors, positives).any()):
            raise RuntimeError("Dynamics pair sampler selected the same replay window twice")
        return torch.stack((selected_anchors, positives), dim=1).flatten()

    def _sample_grouped_dynamics_indices(
        self,
        batch_size: int,
        block_size: int,
        *,
        contrastive_ready_only: bool,
    ) -> torch.Tensor:
        """Build each block from nearby views of one synchronized environment family."""

        if batch_size % block_size:
            raise ValueError("batch_size must be divisible by contrastive_block_size")
        if block_size % self.dynamics_classes:
            raise ValueError("contrastive_block_size must be divisible by dynamics_classes")
        views = block_size // self.dynamics_classes
        if views < 2:
            raise ValueError("Each contrastive block needs at least two cohort views")
        if (views - 1) * self.horizon > self.positive_max_offset_steps:
            raise ValueError("contrastive block spans more steps than positive_max_offset_steps")
        runs, run_motion = self._grouped_temporal_runs(
            views,
            contrastive_ready_only=contrastive_ready_only,
        )
        needed_runs = batch_size // block_size
        if runs.size(0) < needed_runs:
            raise RuntimeError(
                "Replay does not yet contain enough complete local temporal cohort runs"
            )
        if self.sampling_mode == "motion_balanced":
            _, motion_inverse, motion_counts = torch.unique(
                run_motion,
                return_inverse=True,
                return_counts=True,
            )
            weights = motion_counts[motion_inverse].float().reciprocal()
        else:
            weights = torch.ones(runs.size(0), device=self.device)
        if contrastive_ready_only:
            full_context = self._samples["context_full"][runs]
            has_positive = full_context.sum(dim=1)[:, None, :] > 1
            has_negative = full_context.sum(dim=2)[:, :, None] > 1
            weights = weights * (full_context & has_positive & has_negative).sum(dim=(1, 2)).float()
        chosen = torch.multinomial(
            weights,
            needed_runs,
            replacement=False,
            generator=self._generator,
        )
        return runs.index_select(0, chosen).reshape(-1)

    def sample_batch(
        self,
        batch_size: int,
        normalization: ForwardPredictorNormalizationStats,
        *,
        paired_dynamics: bool = False,
        contrastive_block_size: int | None = None,
        contrastive_ready_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        active_count = len(self)
        if batch_size < 1 or active_count < batch_size:
            raise RuntimeError(
                f"Forward Predictor replay has {active_count} samples, batch_size={batch_size}"
            )
        if paired_dynamics and self.grouped_dynamics:
            indices = self._sample_grouped_dynamics_indices(
                batch_size,
                contrastive_block_size or batch_size,
                contrastive_ready_only=contrastive_ready_only,
            )
        elif paired_dynamics:
            indices = self._sample_paired_dynamics_indices(
                batch_size,
                contrastive_ready_only=contrastive_ready_only,
            )
        elif contrastive_ready_only:
            raise ValueError("contrastive_ready_only requires paired_dynamics=True")
        else:
            indices = self._sample_indices(batch_size)
        selected = {name: value.index_select(0, indices) for name, value in self._samples.items()}
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
        history_state = self._history["state"][context_slots, env_ids]
        history_action = self._history["action"][context_slots, env_ids]
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
        history_foot = self._history["foot"][predictor_slots, predictor_env_ids]
        history_contact_force = self._history["contact_force"][predictor_slots, predictor_env_ids]
        history_contact_binary = self._history["contact_binary"][predictor_slots, predictor_env_ids]
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
        normalized_history_state = (history_state - state_mean) / state_std
        normalized_history_action = (history_action - action_mean) / action_std
        valid_scale = history_valid.unsqueeze(-1)
        normalized_history_state = torch.where(
            valid_scale, normalized_history_state, torch.zeros_like(normalized_history_state)
        )
        normalized_history_action = torch.where(
            valid_scale, normalized_history_action, torch.zeros_like(normalized_history_action)
        )
        return {
            "state": (selected["state"] - state_mean) / state_std,
            "action": (selected["action"] - action_mean) / action_std,
            "history_state": normalized_history_state,
            "history_action": normalized_history_action,
            "foot": (selected["foot"] - foot_mean) / foot_std,
            "history_foot": (history_foot - foot_mean) / foot_std,
            "contact_force": (selected["contact_force"] - contact_force_mean) / contact_force_std,
            "contact_binary": selected["contact_binary"],
            "history_contact_force": (history_contact_force - contact_force_mean)
            / contact_force_std,
            "history_contact_binary": history_contact_binary,
            "history_valid": history_valid,
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
            "world_id": selected["world_id"],
            "dynamics_id": selected["dynamics_id"],
            "motion_group_id": selected["motion_group_id"],
            "cohort_id": selected["cohort_id"],
            "context_full": history_valid.all(dim=1),
            "episode_id": selected["episode_id"],
            "episode_step": selected["episode_step"],
            "motion_id": selected["motion_id"],
            "motion_step": selected["motion_step"],
        }
