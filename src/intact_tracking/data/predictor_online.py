"""Compact online replay for nominal five-step Forward Predictor training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intact_tracking.forward_predictor import physical_state_delta
from intact_tracking.forward_predictor_inputs import (
    CONTACT_BINARY_DIM,
    CONTACT_FORCE_DIM,
    FOOT_FEATURE_DIM,
    G1FootKinematics,
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
        self.foot_kinematics = G1FootKinematics().to(self.device)
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
        self.action.update(batch["joint_target"][valid])
        self.foot.update(self.foot_kinematics(torch.cat((current, following), dim=0)))
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
    """Store five-step targets with ten strictly causal state-action history frames."""

    REQUIRED_FIELDS = (
        "joint_target",
        "robot_state",
        "next_robot_state",
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
    )

    def __init__(
        self,
        *,
        num_worlds: int,
        dimensions: RolloutDimensions | None = None,
        horizon: int = 5,
        history_steps: int = 10,
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
        if sampling_mode not in {"uniform", "motion_balanced"}:
            raise ValueError("sampling_mode must be 'uniform' or 'motion_balanced'")
        if world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")
        self.num_worlds = int(num_worlds)
        self.dimensions = dimensions or RolloutDimensions()
        self.horizon = int(horizon)
        self.history_steps = int(history_steps)
        self.ring_steps = self.horizon + self.history_steps
        self.capacity = int(capacity)
        self.sampling_mode = sampling_mode
        self.world_id_offset = int(world_id_offset)
        self.device = torch.device(device or "cpu")
        self._world_ids = tuple(range(world_id_offset, world_id_offset + num_worlds))
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._target_offsets = torch.arange(self.horizon - 1, -1, -1, device=self.device)
        self._history_offsets = torch.arange(
            self.ring_steps - 1,
            self.horizon - 1,
            -1,
            device=self.device,
        )
        self._history: dict[str, torch.Tensor] = {}
        self._samples: dict[str, torch.Tensor] = {}
        self._reset_history: torch.Tensor | None = None
        self._history_write = 0
        self._sample_write = 0
        self._size = 0
        self._world_ids_validated = False
        self._sampling_weights: torch.Tensor | None = None
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
        history_floats = (
            self.ring_steps * self.num_worlds * (dims.robot_state + dims.action + CONTACT_FORCE_DIM)
        )
        sample_floats = self.capacity * (
            (self.horizon + 1) * dims.robot_state
            + self.horizon * dims.action
            + self.history_steps * (dims.robot_state + dims.action)
            + (self.horizon + 1 + self.history_steps) * CONTACT_FORCE_DIM
        )
        history_integers = 4 * self.ring_steps * self.num_worlds
        sample_integers = 3 * self.capacity
        flags = self.ring_steps * self.num_worlds * (1 + CONTACT_BINARY_DIM) + self.capacity * (
            self.history_steps + (self.horizon + 1 + self.history_steps) * CONTACT_BINARY_DIM
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
        self._samples = {
            "state": torch.empty(
                (self.capacity, self.horizon + 1, dims.robot_state), device=self.device
            ),
            "action": torch.empty((self.capacity, self.horizon, dims.action), device=self.device),
            "history_state": torch.empty(
                (self.capacity, self.history_steps, dims.robot_state), device=self.device
            ),
            "history_action": torch.empty(
                (self.capacity, self.history_steps, dims.action), device=self.device
            ),
            "contact_force": torch.empty(
                (self.capacity, self.horizon + 1, CONTACT_FORCE_DIM), device=self.device
            ),
            "contact_binary": torch.empty(
                (self.capacity, self.horizon + 1, CONTACT_BINARY_DIM),
                dtype=torch.bool,
                device=self.device,
            ),
            "history_contact_force": torch.empty(
                (self.capacity, self.history_steps, CONTACT_FORCE_DIM), device=self.device
            ),
            "history_contact_binary": torch.empty(
                (self.capacity, self.history_steps, CONTACT_BINARY_DIM),
                dtype=torch.bool,
                device=self.device,
            ),
            "history_valid": torch.empty(
                (self.capacity, self.history_steps), dtype=torch.bool, device=self.device
            ),
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
            "joint_target": (self.num_worlds, dims.action),
            "robot_state": (self.num_worlds, dims.robot_state),
            "next_robot_state": (self.num_worlds, dims.robot_state),
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
        self._sampling_weights = None
        self.total_samples_generated += count

    def add_step(self, batch: dict[str, torch.Tensor]) -> int:
        self._validate_batch(batch)
        self._allocate()
        assert self._reset_history is not None
        valid_transition = ~batch["reset_boundary"]
        self.normalizer.update(batch, valid_transition)

        position = self._history_write
        self._history["state"][position].copy_(batch["robot_state"])
        self._history["action"][position].copy_(batch["joint_target"])
        self._history["contact_force"][position].copy_(batch["contact_force"])
        self._history["contact_binary"][position].copy_(batch["contact_binary"])
        self._history["episode_id"][position].copy_(batch["episode_id"])
        self._history["episode_step"][position].copy_(batch["episode_step"])
        self._history["motion_id"][position].copy_(batch["motion_id"])
        self._history["motion_step"][position].copy_(batch["motion_step"])
        self._reset_history[position].copy_(batch["reset_boundary"])

        target_ids = (position - self._target_offsets).remainder(self.ring_steps)
        history_ids = (position - self._history_offsets).remainder(self.ring_steps)
        target_steps = self._history["episode_step"][target_ids]
        target_episode_ids = self._history["episode_id"][target_ids]
        target_start = target_steps[0]
        complete_window = (
            target_steps
            == target_start[None] + torch.arange(self.horizon, device=self.device)[:, None]
        ).all(dim=0)
        complete_window &= target_start.remainder(self.horizon) == 0
        complete_window &= (target_episode_ids == batch["episode_id"][None]).all(dim=0)
        crosses_reset = self._reset_history[target_ids].any(dim=0)
        valid = (
            (batch["episode_step"] + 1 >= self.horizon)
            & complete_window
            & ~crosses_reset
            & ~batch["reset_boundary"]
        )
        env_ids = valid.nonzero(as_tuple=False).flatten()
        count = int(env_ids.numel())
        if count:
            target_state = self._history["state"][target_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            target_action = self._history["action"][target_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            history_state = self._history["state"][history_ids[:, None], env_ids[None, :]].permute(
                1, 0, 2
            )
            history_action = self._history["action"][
                history_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
            target_contact_force = self._history["contact_force"][
                target_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
            target_contact_binary = self._history["contact_binary"][
                target_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
            history_contact_force = self._history["contact_force"][
                history_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
            history_contact_binary = self._history["contact_binary"][
                history_ids[:, None], env_ids[None, :]
            ].permute(1, 0, 2)
            prior_steps = self._history["episode_step"][
                history_ids[:, None], env_ids[None, :]
            ].permute(1, 0)
            prior_episode_ids = self._history["episode_id"][
                history_ids[:, None], env_ids[None, :]
            ].permute(1, 0)
            expected_prior_steps = (
                target_start[env_ids, None]
                - self.history_steps
                + torch.arange(self.history_steps, device=self.device)[None]
            )
            history_valid = (
                (expected_prior_steps >= 0)
                & (prior_steps == expected_prior_steps)
                & (prior_episode_ids == batch["episode_id"][env_ids, None])
            )
            samples = {
                "state": torch.cat((target_state, batch["next_robot_state"][env_ids, None]), dim=1),
                "action": target_action,
                "history_state": history_state,
                "history_action": history_action,
                "contact_force": torch.cat(
                    (target_contact_force, batch["next_contact_force"][env_ids, None]), dim=1
                ),
                "contact_binary": torch.cat(
                    (target_contact_binary, batch["next_contact_binary"][env_ids, None]), dim=1
                ),
                "history_contact_force": history_contact_force,
                "history_contact_binary": history_contact_binary,
                "history_valid": history_valid,
                "world_id": batch["world_id"][env_ids],
                "motion_id": self._history["motion_id"][target_ids[0], env_ids],
                "motion_step": self._history["motion_step"][target_ids[0], env_ids],
            }
            self._append_samples(samples, count)

        self._history_write = (position + 1) % self.ring_steps
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

    def sample_batch(
        self,
        batch_size: int,
        normalization: ForwardPredictorNormalizationStats,
    ) -> dict[str, torch.Tensor]:
        if batch_size < 1 or self._size < batch_size:
            raise RuntimeError(
                f"Forward Predictor replay has {self._size} samples, batch_size={batch_size}"
            )
        if self.sampling_mode == "uniform":
            indices = torch.randperm(self._size, generator=self._generator, device=self.device)[
                :batch_size
            ]
        else:
            if self._sampling_weights is None or self._sampling_weights.numel() != self._size:
                motion_ids = self._samples["motion_id"][: self._size]
                _, inverse, counts = torch.unique(
                    motion_ids,
                    return_inverse=True,
                    return_counts=True,
                )
                self._sampling_weights = counts[inverse].float().reciprocal()
            indices = torch.multinomial(
                self._sampling_weights,
                batch_size,
                replacement=False,
                generator=self._generator,
            )
        selected = {name: value.index_select(0, indices) for name, value in self._samples.items()}
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
        return {
            "state": (selected["state"] - state_mean) / state_std,
            "action": (selected["action"] - action_mean) / action_std,
            "history_state": (selected["history_state"] - state_mean) / state_std,
            "history_action": (selected["history_action"] - action_mean) / action_std,
            "contact_force": (selected["contact_force"] - contact_force_mean) / contact_force_std,
            "contact_binary": selected["contact_binary"],
            "history_contact_force": (selected["history_contact_force"] - contact_force_mean)
            / contact_force_std,
            "history_contact_binary": selected["history_contact_binary"],
            "history_valid": selected["history_valid"],
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
            "motion_id": selected["motion_id"],
            "motion_step": selected["motion_step"],
        }
