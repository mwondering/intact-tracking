"""World-disjoint, causal sampling of INTACT tracking windows."""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import SCHEMA_VERSION, RolloutDimensions


@dataclass(frozen=True)
class NormalizationStats:
    """Train-world-only statistics shared by actual and reference observations."""

    observation_mean: tuple[float, ...]
    observation_std: tuple[float, ...]
    proprio_mean: tuple[float, ...]
    proprio_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    world_ids: tuple[int, ...]
    epsilon: float = 1e-6

    @staticmethod
    def _normalize(
        value: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]
    ) -> np.ndarray:
        return (value - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

    def observation(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.observation_mean, self.observation_std)

    def proprio(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.proprio_mean, self.proprio_std)

    def action(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.action_mean, self.action_std)

    def to_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return output

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text())
        for name in (
            "observation_mean",
            "observation_std",
            "proprio_mean",
            "proprio_std",
            "action_mean",
            "action_std",
            "world_ids",
        ):
            payload[name] = tuple(payload[name])
        return cls(**payload)


@dataclass(frozen=True)
class _TransitionRef:
    shard: int
    row: int


@dataclass(frozen=True)
class _ChunkRef:
    episode_key: tuple[int, int]
    start: int
    first_collector_step: int
    last_collector_step: int


@dataclass(frozen=True)
class _SampleRef:
    query: _ChunkRef
    context: tuple[_ChunkRef, ...]


class _Shard:
    def __init__(self, root: Path, entry: dict[str, Any]) -> None:
        self.root = root / entry["path"]
        self.length = int(entry["length"])
        metadata = json.loads((self.root / "metadata.json").read_text())
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported shard schema {metadata['schema_version']!r} in {self.root}"
            )
        self.columns = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in metadata["fields"]
        }
        for name, column in self.columns.items():
            if column.shape[0] != self.length:
                raise ValueError(
                    f"Column {name} in {self.root} has {column.shape[0]} rows, "
                    f"expected {self.length}"
                )

    def value(self, name: str, row: int) -> np.ndarray:
        return self.columns[name][row]


def split_world_ids(
    world_ids: list[int] | tuple[int, ...],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, tuple[int, ...]]:
    """Split by physics world, never by clips or transitions."""
    unique = np.asarray(sorted(set(int(value) for value in world_ids)), dtype=np.int64)
    if unique.size < 3:
        raise ValueError("At least three world IDs are required for train/val/test splits")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must lie in (0,1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than one")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_count = max(1, int(round(unique.size * train_fraction)))
    validation_count = max(1, int(round(unique.size * validation_fraction)))
    if train_count + validation_count >= unique.size:
        train_count = unique.size - 2
        validation_count = 1
    return {
        "train": tuple(sorted(int(value) for value in unique[:train_count])),
        "validation": tuple(
            sorted(int(value) for value in unique[train_count : train_count + validation_count])
        ),
        "test": tuple(sorted(int(value) for value in unique[train_count + validation_count :])),
    }


def _moments(arrays: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(width, dtype=np.float64)
    square = np.zeros(width, dtype=np.float64)
    for array in arrays:
        flat = np.asarray(array, dtype=np.float64).reshape(-1, width)
        count += flat.shape[0]
        total += flat.sum(axis=0)
        square += np.square(flat).sum(axis=0)
    if count == 0:
        raise ValueError("Cannot compute statistics from zero selected transitions")
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class RolloutWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample sliding causal queries with 16 prior same-world context tokens."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        world_ids: list[int] | tuple[int, ...] | None = None,
        effect_steps: int = 5,
        query_transitions: int = 5,
        context_chunk_steps: int = 5,
        sample_stride: int = 1,
        context_tokens: int = 16,
        require_full_context: bool = True,
        normalization: NormalizationStats | None = None,
    ) -> None:
        super().__init__()
        if (
            effect_steps < 1
            or query_transitions < 1
            or context_chunk_steps < 1
            or sample_stride < 1
            or context_tokens != 16
        ):
            raise ValueError(
                "effect_steps, query_transitions, context_chunk_steps and sample_stride "
                "must be positive; context_tokens is fixed at 16"
            )
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema {self.manifest['schema_version']!r}")
        self.dimensions = RolloutDimensions(**self.manifest["dimensions"])
        self.effect_steps = int(effect_steps)
        self.query_transitions = int(query_transitions)
        self.context_chunk_steps = int(context_chunk_steps)
        self.sample_stride = int(sample_stride)
        self.context_tokens = int(context_tokens)
        self.require_full_context = bool(require_full_context)
        self.normalization = normalization
        available_worlds = set(int(value) for value in self.manifest["world_ids"])
        self.world_ids = (
            available_worlds if world_ids is None else set(int(value) for value in world_ids)
        )
        unknown = self.world_ids.difference(available_worlds)
        if unknown:
            raise ValueError(f"Requested unknown world IDs: {sorted(unknown)}")
        self.shards = [_Shard(self.root, entry) for entry in self.manifest["shards"]]
        self.episodes: dict[tuple[int, int], list[_TransitionRef]] = {}
        self._build_episode_index()
        self.samples = self._build_sample_index()
        if not self.samples:
            raise ValueError(
                "No valid training windows were found; collect longer same-world "
                "rollouts or allow padded context"
            )

    def _value(self, ref: _TransitionRef, name: str) -> np.ndarray:
        return self.shards[ref.shard].value(name, ref.row)

    def _build_episode_index(self) -> None:
        for shard_index, shard in enumerate(self.shards):
            worlds = shard.columns["world_id"]
            episodes = shard.columns["episode_id"]
            for row in range(shard.length):
                world = int(worlds[row])
                if world not in self.world_ids:
                    continue
                key = (world, int(episodes[row]))
                self.episodes.setdefault(key, []).append(_TransitionRef(shard_index, row))
        for key, refs in self.episodes.items():
            refs.sort(key=lambda ref: int(self._value(ref, "episode_step")))
            steps = [int(self._value(ref, "episode_step")) for ref in refs]
            if steps != list(range(len(steps))):
                raise ValueError(
                    f"Episode {key} is not contiguous from step zero: "
                    f"first={steps[:3]}, last={steps[-3:]}"
                )

    def _chunk_valid(self, refs: list[_TransitionRef], start: int, length: int) -> bool:
        if start + length > len(refs):
            return False
        selected = refs[start : start + length]
        return not any(bool(self._value(ref, "reset_boundary")) for ref in selected)

    def _chunk(self, key: tuple[int, int], start: int, length: int) -> _ChunkRef:
        refs = self.episodes[key]
        first = int(self._value(refs[start], "collector_step"))
        last = int(self._value(refs[start + length - 1], "collector_step"))
        return _ChunkRef(key, start, first, last)

    def _build_sample_index(self) -> list[_SampleRef]:
        context_by_world: dict[int, list[_ChunkRef]] = {world: [] for world in self.world_ids}
        queries: list[_ChunkRef] = []
        query_length = self.effect_steps * self.query_transitions
        for key, refs in self.episodes.items():
            for start in range(0, len(refs), self.context_chunk_steps):
                if self._chunk_valid(refs, start, self.context_chunk_steps):
                    context_by_world[key[0]].append(
                        self._chunk(key, start, self.context_chunk_steps)
                    )
            for start in range(0, len(refs), self.sample_stride):
                if self._chunk_valid(refs, start, query_length):
                    queries.append(self._chunk(key, start, query_length))
        for chunks in context_by_world.values():
            chunks.sort(key=lambda chunk: (chunk.last_collector_step, chunk.episode_key[1]))

        samples: list[_SampleRef] = []
        for query in queries:
            chunks = context_by_world[query.episode_key[0]]
            ends = [chunk.last_collector_step for chunk in chunks]
            stop = bisect.bisect_left(ends, query.first_collector_step)
            selected = tuple(chunks[max(0, stop - self.context_tokens) : stop])
            if self.require_full_context and len(selected) != self.context_tokens:
                continue
            samples.append(_SampleRef(query=query, context=selected))
        samples.sort(
            key=lambda sample: (
                sample.query.first_collector_step,
                sample.query.episode_key,
                sample.query.start,
            )
        )
        return samples

    def compute_normalization(self) -> NormalizationStats:
        """Compute statistics from this dataset's selected worlds only."""
        observation_arrays: list[np.ndarray] = []
        proprio_arrays: list[np.ndarray] = []
        action_arrays: list[np.ndarray] = []
        for shard in self.shards:
            worlds = np.asarray(shard.columns["world_id"])
            selection = np.isin(worlds, np.asarray(sorted(self.world_ids), dtype=np.int64))
            if not selection.any():
                continue
            observation_arrays.extend(
                [
                    np.asarray(shard.columns["observation"])[selection],
                    np.asarray(shard.columns["reference_observation"])[selection],
                ]
            )
            proprio_arrays.append(np.asarray(shard.columns["proprio"])[selection])
            action_arrays.append(np.asarray(shard.columns["action"])[selection])
        obs_mean, obs_std = _moments(observation_arrays, self.dimensions.observation)
        prop_mean, prop_std = _moments(proprio_arrays, self.dimensions.proprio)
        action_mean, action_std = _moments(action_arrays, self.dimensions.action)
        epsilon = 1e-6
        obs_std = np.maximum(obs_std, epsilon)
        prop_std = np.maximum(prop_std, epsilon)
        action_std = np.maximum(action_std, epsilon)
        return NormalizationStats(
            observation_mean=tuple(float(value) for value in obs_mean),
            observation_std=tuple(float(value) for value in obs_std),
            proprio_mean=tuple(float(value) for value in prop_mean),
            proprio_std=tuple(float(value) for value in prop_std),
            action_mean=tuple(float(value) for value in action_mean),
            action_std=tuple(float(value) for value in action_std),
            world_ids=tuple(sorted(self.world_ids)),
            epsilon=epsilon,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _transition_chunk(
        self, episode_key: tuple[int, int], start: int, length: int
    ) -> list[_TransitionRef]:
        return self.episodes[episode_key][start : start + length]

    def _action_sequence(self, refs: list[_TransitionRef]) -> np.ndarray:
        value = np.stack([self._value(ref, "action") for ref in refs]).astype(
            np.float32, copy=False
        )
        if self.normalization is not None:
            value = self.normalization.action(value)
        return value.reshape(-1)

    def _proprio(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return self.normalization.proprio(value) if self.normalization else value

    def _observation(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return self.normalization.observation(value) if self.normalization else value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        query = sample.query
        query_refs = self.episodes[query.episode_key]
        observations = []
        actions = []
        previous_actions = []
        goals = []
        forward_actions = []
        raw_zero = np.zeros(self.dimensions.action, dtype=np.float32)
        for teacher_step in range(self.query_transitions):
            start = query.start + teacher_step * self.effect_steps
            block = self._transition_chunk(query.episode_key, start, self.effect_steps)
            observations.append(self._observation(self._value(block[0], "observation")))
            forward_actions.append(self._action_sequence(block))
            action = np.asarray(self._value(block[0], "action"), dtype=np.float32)
            if self.normalization is not None:
                action = self.normalization.action(action)
            actions.append(action)
            previous_start = start - 1
            if previous_start >= 0:
                previous_ref = query_refs[previous_start]
                previous = np.asarray(self._value(previous_ref, "action"), dtype=np.float32)
                if self.normalization is not None:
                    previous = self.normalization.action(previous)
                previous_actions.append(previous)
            else:
                previous = raw_zero
                if self.normalization is not None:
                    previous = self.normalization.action(previous)
                previous_actions.append(previous)
            goals.append(self._observation(self._value(block[-1], "next_reference_observation")))
        final_ref = query_refs[query.start + self.query_transitions * self.effect_steps - 1]
        observations.append(self._observation(self._value(final_ref, "next_observation")))

        token_dim = 2 * self.dimensions.proprio + self.context_chunk_steps * self.dimensions.action
        context = np.zeros((self.context_tokens, token_dim), dtype=np.float32)
        context_mask = np.zeros(self.context_tokens, dtype=np.bool_)
        offset = self.context_tokens - len(sample.context)
        for token_index, chunk in enumerate(sample.context, start=offset):
            refs = self._transition_chunk(chunk.episode_key, chunk.start, self.context_chunk_steps)
            before = self._proprio(self._value(refs[0], "proprio"))
            action = self._action_sequence(refs)
            after = self._proprio(self._value(refs[-1], "next_proprio"))
            context[token_index] = np.concatenate((before, action, after))
            context_mask[token_index] = True

        step_mask = np.ones(self.query_transitions, dtype=np.bool_)
        return {
            "observation": torch.from_numpy(np.stack(observations).astype(np.float32)),
            "goal_observation": torch.from_numpy(np.stack(goals).astype(np.float32)),
            "forward_action": torch.from_numpy(np.stack(forward_actions).astype(np.float32)),
            "action": torch.from_numpy(np.stack(actions).astype(np.float32)),
            "previous_action": torch.from_numpy(np.stack(previous_actions).astype(np.float32)),
            "context": torch.from_numpy(context),
            "context_mask": torch.from_numpy(context_mask),
            "transition_mask": torch.from_numpy(step_mask.copy()),
            "physical_mask": torch.from_numpy(step_mask.copy()),
            "goal_mask": torch.from_numpy(step_mask.copy()),
            "world_id": torch.tensor(query.episode_key[0], dtype=torch.long),
            "episode_id": torch.tensor(query.episode_key[1], dtype=torch.long),
        }
