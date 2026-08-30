from __future__ import annotations

import torch

from intact_tracking.data import OnlineNormalization, OnlineReplayBuffer, RolloutDimensions

DIMENSIONS = RolloutDimensions(
    proprio=3,
    observation=2,
    action=2,
    robot_state=1,
    reference_state=1,
)


def _step_batch(
    collector_step: int,
    *,
    num_worlds: int = 1,
    episode_id: int = 0,
    episode_step: int | None = None,
    boundary: bool = False,
    world_id_offset: int = 0,
) -> dict[str, torch.Tensor]:
    local_step = collector_step if episode_step is None else episode_step
    worlds = torch.arange(num_worlds, dtype=torch.float32)
    scalar = torch.full((num_worlds,), float(collector_step)) + worlds * 0.1
    proprio = torch.stack((scalar, scalar + 1, scalar + 2), dim=-1)
    observation = torch.stack((scalar, -scalar - 1), dim=-1)
    reference = torch.stack((scalar + 10, -scalar - 11), dim=-1)
    action = torch.stack((scalar * 0.1 + 1, scalar * -0.2 - 1), dim=-1)
    return {
        "proprio": proprio,
        "next_proprio": proprio + 0.5,
        "observation": observation,
        "next_observation": observation + 0.25,
        "reference_observation": reference,
        "next_reference_observation": reference + 0.25,
        "action": action,
        "reset_boundary": torch.full((num_worlds,), boundary, dtype=torch.bool),
        "world_id": torch.arange(
            world_id_offset,
            world_id_offset + num_worlds,
            dtype=torch.long,
        ),
        "episode_id": torch.full((num_worlds,), episode_id, dtype=torch.long),
        "episode_step": torch.full((num_worlds,), local_step, dtype=torch.long),
        "collector_step": torch.full((num_worlds,), collector_step, dtype=torch.long),
    }


def test_online_replay_waits_for_sixteen_causal_context_tokens() -> None:
    replay = OnlineReplayBuffer(
        num_worlds=2,
        dimensions=DIMENSIONS,
        block_size=2,
        horizon=2,
        context_tokens=16,
        capacity=8,
        seed=3,
    )
    assert replay.minimum_steps == 36

    for step in range(replay.minimum_steps - 1):
        assert replay.add_step(_step_batch(step, num_worlds=2)) == 0
    assert len(replay) == 0
    assert replay.add_step(_step_batch(35, num_worlds=2)) == 2

    batch = replay.sample_batch(2)
    assert batch["observation"].shape == (2, 3, 2)
    assert batch["goal_observation"].shape == (2, 2)
    assert batch["action"].shape == (2, 2, 4)
    assert batch["previous_action"].shape == (2, 2, 4)
    assert batch["context"].shape == (2, 16, 10)
    assert batch["context_mask"].all()
    assert batch["transition_mask"].all()
    assert torch.isfinite(batch["context"]).all()
    assert set(batch["world_id"].tolist()) == {0, 1}


def test_online_replay_keeps_context_across_reset_but_not_query() -> None:
    replay = OnlineReplayBuffer(
        num_worlds=1,
        dimensions=DIMENSIONS,
        block_size=2,
        horizon=2,
        context_tokens=16,
        capacity=4,
    )
    for step in range(32):
        replay.add_step(_step_batch(step))
    replay.add_step(_step_batch(32, boundary=True))
    for episode_step, collector_step in enumerate(range(33, 37)):
        replay.add_step(
            _step_batch(
                collector_step,
                episode_id=1,
                episode_step=episode_step,
            )
        )

    assert len(replay) == 1
    batch = replay.sample_batch(1)
    stats = replay.normalization()
    action_mean = torch.tensor(stats.action_mean)
    action_std = torch.tensor(stats.action_std)
    expected_raw_zero = ((torch.zeros(2, 2) - action_mean) / action_std).flatten()
    torch.testing.assert_close(batch["previous_action"][0, 0], expected_raw_zero)
    assert batch["episode_id"].item() == 1
    assert batch["context_mask"].all()


def test_online_replay_capacity_evicts_old_samples() -> None:
    replay = OnlineReplayBuffer(
        num_worlds=1,
        dimensions=DIMENSIONS,
        block_size=2,
        horizon=2,
        context_tokens=16,
        capacity=2,
    )
    for step in range(40):
        replay.add_step(_step_batch(step))
    assert replay.total_samples_generated == 3
    assert len(replay) == 2


def test_online_replay_offsets_world_ids_for_distributed_ranks() -> None:
    replay = OnlineReplayBuffer(
        num_worlds=2,
        dimensions=DIMENSIONS,
        block_size=2,
        horizon=2,
        context_tokens=16,
        capacity=4,
        world_id_offset=6,
    )
    for step in range(replay.minimum_steps):
        replay.add_step(
            _step_batch(
                step,
                num_worlds=2,
                world_id_offset=6,
            )
        )

    batch = replay.sample_batch(2)
    assert replay.world_ids == (6, 7)
    assert set(batch["world_id"].tolist()) == {6, 7}
    assert replay.normalization().world_ids == (6, 7)


def test_online_normalization_merges_additive_rank_statistics() -> None:
    left = OnlineNormalization(DIMENSIONS)
    right = OnlineNormalization(DIMENSIONS)
    centralized = OnlineNormalization(DIMENSIONS)
    left_batch = _step_batch(2, num_worlds=2)
    right_batch = _step_batch(11, num_worlds=1)
    left.update(left_batch)
    right.update(right_batch)
    centralized.update(left_batch)
    centralized.update(right_batch)

    packed = left.packed_statistics() + right.packed_statistics()
    merged = left.snapshot_from_packed(packed, (0, 1, 2))
    expected = centralized.snapshot((0, 1, 2))
    for name in (
        "observation_mean",
        "observation_std",
        "proprio_mean",
        "proprio_std",
        "action_mean",
        "action_std",
    ):
        torch.testing.assert_close(
            torch.tensor(getattr(merged, name)),
            torch.tensor(getattr(expected, name)),
        )
    assert merged.world_ids == (0, 1, 2)
