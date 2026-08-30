from __future__ import annotations

import json

import numpy as np
import torch

from intact_tracking.cli.train import build_parser as build_train_parser
from intact_tracking.cli.train import run as run_training
from intact_tracking.data import (
    RolloutDimensions,
    RolloutShardWriter,
    RolloutWindowDataset,
    split_world_ids,
)
from intact_tracking.rollout.mjlab_adapter import _latest_proprio


def _synthetic_batch(dimensions: RolloutDimensions, collector_step: int, num_worlds: int = 3):
    worlds = np.arange(num_worlds, dtype=np.int64)
    scalar_i = np.full(num_worlds, collector_step, dtype=np.int64)
    scalar_f = np.full(num_worlds, collector_step, dtype=np.float32)

    def feature(width: int, offset: float) -> np.ndarray:
        return np.stack(
            [np.full(width, offset + collector_step + world * 0.1, np.float32) for world in worlds]
        )

    action = feature(dimensions.action, 0.5)
    return {
        "proprio": feature(dimensions.proprio, 1.0),
        "next_proprio": feature(dimensions.proprio, 2.0),
        "observation": feature(dimensions.observation, 3.0),
        "next_observation": feature(dimensions.observation, 4.0),
        "reference_observation": feature(dimensions.observation, 5.0),
        "next_reference_observation": feature(dimensions.observation, 6.0),
        "action": action,
        "reward": scalar_f,
        "terminated": np.zeros(num_worlds, dtype=np.bool_),
        "truncated": np.zeros(num_worlds, dtype=np.bool_),
        "reset_boundary": np.zeros(num_worlds, dtype=np.bool_),
        "world_id": worlds,
        "episode_id": np.zeros(num_worlds, dtype=np.int64),
        "episode_step": scalar_i,
        "collector_step": scalar_i,
        "env_id": worlds,
        "motion_id": worlds,
        "motion_step": scalar_i,
        "applied_action": action,
        "joint_target": action + 1,
        "joint_torque": action + 2,
        "robot_state": feature(dimensions.robot_state, 7.0),
        "next_robot_state": feature(dimensions.robot_state, 8.0),
        "reference_state": feature(dimensions.reference_state, 9.0),
        "next_reference_state": feature(dimensions.reference_state, 10.0),
    }


def _write_dataset(tmp_path):
    dimensions = RolloutDimensions(
        proprio=6, observation=4, action=2, robot_state=5, reference_state=5
    )
    root = tmp_path / "rollout"
    with RolloutShardWriter(
        root, dimensions=dimensions, shard_size=71, include_diagnostics=True
    ) as writer:
        for step in range(150):
            writer.append(_synthetic_batch(dimensions, step))
    return root / "manifest.json", dimensions


def test_writer_and_dataset_preserve_shapes_and_world_boundaries(tmp_path) -> None:
    manifest, dimensions = _write_dataset(tmp_path)
    payload = json.loads(manifest.read_text())
    assert payload["total_transitions"] == 450
    assert len(payload["shards"]) > 1

    dataset = RolloutWindowDataset(
        manifest,
        world_ids=[1],
        block_size=2,
        horizon=3,
        context_tokens=16,
        require_full_context=True,
    )
    sample = dataset[0]
    assert sample["observation"].shape == (4, dimensions.observation)
    assert sample["action"].shape == (3, 2 * dimensions.action)
    assert sample["previous_action"].shape == (3, 2 * dimensions.action)
    assert sample["context"].shape == (
        16,
        2 * dimensions.proprio + 2 * dimensions.action,
    )
    assert sample["context_mask"].all()
    assert sample["world_id"].item() == 1
    assert all(dataset[index]["world_id"].item() == 1 for index in range(len(dataset)))


def test_raw_zero_previous_action_is_normalized_after_padding(tmp_path) -> None:
    manifest, dimensions = _write_dataset(tmp_path)
    raw = RolloutWindowDataset(
        manifest,
        world_ids=[0],
        block_size=2,
        horizon=2,
        require_full_context=False,
    )
    stats = raw.compute_normalization()
    normalized = RolloutWindowDataset(
        manifest,
        world_ids=[0],
        block_size=2,
        horizon=2,
        require_full_context=False,
        normalization=stats,
    )
    sample = normalized[0]
    expected_frame = -torch.tensor(stats.action_mean) / torch.tensor(stats.action_std)
    expected = expected_frame.repeat(2)
    assert torch.allclose(sample["previous_action"][0], expected)
    assert sample["context_mask"].sum() == 0


def test_world_split_is_disjoint_and_complete() -> None:
    split = split_world_ids(list(range(10)), seed=7)
    groups = [set(split[name]) for name in ("train", "validation", "test")]
    assert not groups[0].intersection(groups[1])
    assert not groups[0].intersection(groups[2])
    assert not groups[1].intersection(groups[2])
    assert set.union(*groups) == set(range(10))


def test_term_major_history_extracts_latest_122_values() -> None:
    term_dims = (29, 29, 3, 3, 29, 29)
    columns = []
    expected = []
    for term_index, width in enumerate(term_dims):
        term = torch.arange(50 * width, dtype=torch.float32).reshape(1, 50, width)
        term = term + 10_000 * term_index
        columns.append(term.reshape(1, -1))
        expected.append(term[:, -1])
    history = torch.cat(columns, dim=-1)
    assert torch.equal(_latest_proprio(history), torch.cat(expected, dim=-1))


def test_training_smoke_is_limited_to_one_batch(tmp_path) -> None:
    manifest, _ = _write_dataset(tmp_path)
    output_dir = tmp_path / "train"
    args = build_train_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--block-size",
            "2",
            "--horizon",
            "2",
            "--embed-dim",
            "8",
            "--encoder-hidden-dim",
            "16",
            "--context-depth",
            "1",
            "--context-heads",
            "2",
            "--forward-depth",
            "1",
            "--forward-heads",
            "2",
            "--actor-hidden-dim",
            "16",
            "--actor-depth",
            "1",
            "--sigreg-projections",
            "2",
            "--max-train-batches",
            "1",
            "--max-validation-batches",
            "1",
            "--device",
            "cpu",
        ]
    )
    checkpoint = run_training(args)
    history = json.loads((output_dir / "history.json").read_text())
    assert checkpoint == output_dir / "last.pt"
    assert checkpoint.is_file()
    assert history[0]["train_batches"] == 1
    assert history[0]["validation_batches"] == 1
    assert np.isfinite(history[0]["train"]["loss"])
