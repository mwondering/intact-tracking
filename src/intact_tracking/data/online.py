"""In-memory causal replay for pure online INTACT training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .dataset import NormalizationStats
from .schema import RolloutDimensions


class _RunningMoments:
    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.count = 0
        self.total = torch.zeros(self.width, dtype=torch.float64)
        self.square_total = torch.zeros(self.width, dtype=torch.float64)

    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1, self.width)
        if flat.numel() == 0:
            return
        if not torch.isfinite(flat).all():
            raise ValueError("Online normalization received a non-finite value")
        self.count += flat.shape[0]
        self.total += flat.sum(dim=0)
        self.square_total += flat.square().sum(dim=0)

    def mean_std(self, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 1:
            raise RuntimeError("Online normalization has not observed any transitions")
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean.square()).clamp_min(epsilon**2)
        return mean.to(torch.float32), variance.sqrt().to(torch.float32)


class OnlineNormalization:
    """Running train-stream statistics applied when a replay batch is sampled."""

    def __init__(self, dimensions: RolloutDimensions, epsilon: float = 1e-6) -> None:
        self.dimensions = dimensions
        self.epsilon = float(epsilon)
        self.observation = _RunningMoments(dimensions.observation)
        self.proprio = _RunningMoments(dimensions.proprio)
        self.action = _RunningMoments(dimensions.action)

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        self.observation.update(
            torch.cat((batch["observation"], batch["reference_observation"]), dim=0)
        )
        self.proprio.update(batch["proprio"])
        self.action.update(batch["action"])

    def snapshot(self, world_ids: tuple[int, ...]) -> NormalizationStats:
        observation_mean, observation_std = self.observation.mean_std(self.epsilon)
        proprio_mean, proprio_std = self.proprio.mean_std(self.epsilon)
        action_mean, action_std = self.action.mean_std(self.epsilon)
        return NormalizationStats(
            observation_mean=tuple(float(value) for value in observation_mean),
            observation_std=tuple(float(value) for value in observation_std),
            proprio_mean=tuple(float(value) for value in proprio_mean),
            proprio_std=tuple(float(value) for value in proprio_std),
            action_mean=tuple(float(value) for value in action_mean),
            action_std=tuple(float(value) for value in action_std),
            world_ids=world_ids,
            epsilon=self.epsilon,
        )


@dataclass(frozen=True)
class _Transition:
    proprio: torch.Tensor
    next_proprio: torch.Tensor
    observation: torch.Tensor
    next_observation: torch.Tensor
    reference_observation: torch.Tensor
    next_reference_observation: torch.Tensor
    action: torch.Tensor
    reset_boundary: bool
    world_id: int
    episode_id: int
    episode_step: int
    collector_step: int


@dataclass(frozen=True)
class _ContextChunk:
    first_collector_step: int
    last_collector_step: int
    before: torch.Tensor
    actions: torch.Tensor
    after: torch.Tensor


@dataclass(frozen=True)
class _TrainingSample:
    observations: torch.Tensor
    goal_observation: torch.Tensor
    actions: torch.Tensor
    previous_actions: torch.Tensor
    context_before: torch.Tensor
    context_actions: torch.Tensor
    context_after: torch.Tensor
    world_id: int
    episode_id: int


class OnlineReplayBuffer:
    """Convert a live vector stream into full-context INTACT training samples.

    Each vector slot is one immutable physics world. Query transitions must be
    contiguous within an episode, while context chunks may come from earlier
    episodes of the same world. This matches the offline causal-window contract
    without materializing rollout shards.
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
    ) -> None:
        if num_worlds < 1 or block_size < 1 or horizon < 1 or capacity < 1:
            raise ValueError("num_worlds, block_size, horizon and capacity must be positive")
        if context_tokens != 16:
            raise ValueError("Online INTACT context_tokens is fixed at 16")
        self.num_worlds = int(num_worlds)
        self.dimensions = dimensions or RolloutDimensions()
        self.block_size = int(block_size)
        self.horizon = int(horizon)
        self.context_tokens = int(context_tokens)
        self.capacity = int(capacity)
        self.query_steps = self.block_size * self.horizon
        self.minimum_steps = self.block_size * (self.context_tokens + self.horizon)
        self._rng = np.random.default_rng(seed)
        self._episodes = [
            deque(maxlen=self.query_steps + self.block_size) for _ in range(self.num_worlds)
        ]
        self._contexts = [
            deque(maxlen=self.context_tokens + self.horizon) for _ in range(self.num_worlds)
        ]
        self._samples: deque[_TrainingSample] = deque(maxlen=self.capacity)
        self.normalizer = OnlineNormalization(self.dimensions)
        self.total_samples_generated = 0
        self.total_transitions = 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def world_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_worlds))

    def normalization(self) -> NormalizationStats:
        return self.normalizer.snapshot(self.world_ids)

    def _cpu_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = sorted(set(self.REQUIRED_FIELDS).difference(batch))
        if missing:
            raise KeyError(f"Online transition batch is missing fields: {missing}")
        result = {}
        for name in self.REQUIRED_FIELDS:
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Online field {name!r} must be a Tensor")
            if value.ndim < 1 or value.shape[0] != self.num_worlds:
                raise ValueError(
                    f"Online field {name!r} must start with {self.num_worlds} worlds, "
                    f"got {tuple(value.shape)}"
                )
            result[name] = value.detach().to("cpu").clone()
        expected_shapes = {
            "proprio": (self.num_worlds, self.dimensions.proprio),
            "next_proprio": (self.num_worlds, self.dimensions.proprio),
            "observation": (self.num_worlds, self.dimensions.observation),
            "next_observation": (self.num_worlds, self.dimensions.observation),
            "reference_observation": (self.num_worlds, self.dimensions.observation),
            "next_reference_observation": (self.num_worlds, self.dimensions.observation),
            "action": (self.num_worlds, self.dimensions.action),
        }
        for name, shape in expected_shapes.items():
            if tuple(result[name].shape) != shape:
                raise ValueError(
                    f"Online field {name!r} has {tuple(result[name].shape)}, {shape} expected"
                )
        return result

    @staticmethod
    def _transition(batch: dict[str, torch.Tensor], env_id: int) -> _Transition:
        return _Transition(
            proprio=batch["proprio"][env_id],
            next_proprio=batch["next_proprio"][env_id],
            observation=batch["observation"][env_id],
            next_observation=batch["next_observation"][env_id],
            reference_observation=batch["reference_observation"][env_id],
            next_reference_observation=batch["next_reference_observation"][env_id],
            action=batch["action"][env_id],
            reset_boundary=bool(batch["reset_boundary"][env_id]),
            world_id=int(batch["world_id"][env_id]),
            episode_id=int(batch["episode_id"][env_id]),
            episode_step=int(batch["episode_step"][env_id]),
            collector_step=int(batch["collector_step"][env_id]),
        )

    def _append_context(self, env_id: int, transitions: list[_Transition]) -> None:
        block = transitions[-self.block_size :]
        if len(block) != self.block_size or any(item.reset_boundary for item in block):
            return
        self._contexts[env_id].append(
            _ContextChunk(
                first_collector_step=block[0].collector_step,
                last_collector_step=block[-1].collector_step,
                before=block[0].proprio,
                actions=torch.stack([item.action for item in block]),
                after=block[-1].next_proprio,
            )
        )

    def _make_sample(self, env_id: int, transitions: list[_Transition]) -> _TrainingSample | None:
        query = transitions[-self.query_steps :]
        if len(query) != self.query_steps or any(item.reset_boundary for item in query):
            return None
        if any(item.episode_id != query[0].episode_id for item in query):
            return None
        expected_steps = list(range(query[0].episode_step, query[0].episode_step + len(query)))
        if [item.episode_step for item in query] != expected_steps:
            return None
        contexts = [
            item
            for item in self._contexts[env_id]
            if item.last_collector_step < query[0].collector_step
        ][-self.context_tokens :]
        if len(contexts) != self.context_tokens:
            return None

        action_blocks = torch.stack(
            [
                torch.stack([item.action for item in query[start : start + self.block_size]])
                for start in range(0, self.query_steps, self.block_size)
            ]
        )
        if query[0].episode_step >= self.block_size:
            previous = transitions[-self.query_steps - self.block_size : -self.query_steps]
            if len(previous) != self.block_size:
                raise RuntimeError("Online replay lost the previous action block")
            previous_first = torch.stack([item.action for item in previous])
        else:
            previous_first = torch.zeros(
                self.block_size, self.dimensions.action, dtype=torch.float32
            )
        previous_actions = torch.cat((previous_first[None], action_blocks[:-1]), dim=0)
        observations = torch.stack(
            [query[start].observation for start in range(0, self.query_steps, self.block_size)]
            + [query[-1].next_observation]
        )
        return _TrainingSample(
            observations=observations,
            goal_observation=query[-1].next_reference_observation,
            actions=action_blocks,
            previous_actions=previous_actions,
            context_before=torch.stack([item.before for item in contexts]),
            context_actions=torch.stack([item.actions for item in contexts]),
            context_after=torch.stack([item.after for item in contexts]),
            world_id=query[0].world_id,
            episode_id=query[0].episode_id,
        )

    def add_step(self, batch: dict[str, torch.Tensor]) -> int:
        """Append one vector-environment step and return new sample count."""
        batch = self._cpu_batch(batch)
        self.normalizer.update(batch)
        generated = 0
        for env_id in range(self.num_worlds):
            transition = self._transition(batch, env_id)
            if transition.world_id != env_id:
                raise ValueError(
                    "Online fixed-world contract requires world_id == env_id; "
                    f"slot {env_id} reported {transition.world_id}"
                )
            episode = self._episodes[env_id]
            episode.append(transition)
            transitions = list(episode)
            completed_steps = transition.episode_step + 1
            if completed_steps % self.block_size == 0:
                self._append_context(env_id, transitions)
                if completed_steps >= self.query_steps:
                    sample = self._make_sample(env_id, transitions)
                    if sample is not None:
                        self._samples.append(sample)
                        self.total_samples_generated += 1
                        generated += 1
            if transition.reset_boundary:
                episode.clear()
        self.total_transitions += self.num_worlds
        return generated

    @staticmethod
    def _normalization_tensors(
        stats: NormalizationStats,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.tensor(getattr(stats, name), dtype=torch.float32)
            for name in (
                "observation_mean",
                "observation_std",
                "proprio_mean",
                "proprio_std",
                "action_mean",
                "action_std",
            )
        )

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self._samples) < batch_size:
            raise RuntimeError(
                f"Online replay has {len(self._samples)} samples, batch_size={batch_size}"
            )
        population = list(self._samples)
        indices = self._rng.choice(len(population), size=batch_size, replace=False)
        selected = [population[int(index)] for index in indices]
        stats = self.normalization()
        obs_mean, obs_std, prop_mean, prop_std, action_mean, action_std = (
            self._normalization_tensors(stats)
        )

        observations = (torch.stack([item.observations for item in selected]) - obs_mean) / obs_std
        goals = (torch.stack([item.goal_observation for item in selected]) - obs_mean) / obs_std
        actions = (torch.stack([item.actions for item in selected]) - action_mean) / action_std
        previous = (
            torch.stack([item.previous_actions for item in selected]) - action_mean
        ) / action_std
        context_before = (
            torch.stack([item.context_before for item in selected]) - prop_mean
        ) / prop_std
        context_actions = (
            torch.stack([item.context_actions for item in selected]) - action_mean
        ) / action_std
        context_after = (
            torch.stack([item.context_after for item in selected]) - prop_mean
        ) / prop_std
        actions = actions.flatten(start_dim=2)
        previous = previous.flatten(start_dim=2)
        context = torch.cat(
            (context_before, context_actions.flatten(start_dim=2), context_after), dim=-1
        )
        mask = torch.ones(batch_size, self.horizon, dtype=torch.bool)
        return {
            "observation": observations,
            "goal_observation": goals,
            "action": actions,
            "previous_action": previous,
            "context": context,
            "context_mask": torch.ones(batch_size, self.context_tokens, dtype=torch.bool),
            "transition_mask": mask.clone(),
            "physical_mask": mask.clone(),
            "goal_mask": mask,
            "world_id": torch.tensor([item.world_id for item in selected], dtype=torch.long),
            "episode_id": torch.tensor([item.episode_id for item in selected], dtype=torch.long),
        }
