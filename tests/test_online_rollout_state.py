from __future__ import annotations

from types import SimpleNamespace

import torch

import intact_tracking.rollout.online as online_module
from intact_tracking.rollout.online import FixedDRTrackerRollout


def _snapshot(offset: float) -> dict[str, torch.Tensor]:
    num_envs = 3
    return {
        "proprio": torch.full((num_envs, 122), offset),
        "observation": torch.full((num_envs, 64), offset),
        "reference_observation": torch.full((num_envs, 64), offset + 1),
        "motion_id": torch.arange(num_envs),
        "motion_step": torch.full((num_envs,), int(offset), dtype=torch.long),
    }


def test_initial_episode_phases_are_reproducibly_desynchronized() -> None:
    first = SimpleNamespace(
        episode_length_buf=torch.zeros(64, dtype=torch.long),
        max_episode_length=500,
    )
    second = SimpleNamespace(
        episode_length_buf=torch.zeros(64, dtype=torch.long),
        max_episode_length=500,
    )

    summary = online_module._randomize_initial_episode_phases(first, seed=7)
    online_module._randomize_initial_episode_phases(second, seed=7)

    torch.testing.assert_close(first.episode_length_buf, second.episode_length_buf)
    assert summary["unique"] > 1
    assert 0 <= summary["minimum"] <= summary["maximum"] < 500


def test_online_rollout_resets_only_completed_slots(monkeypatch) -> None:
    terminated = torch.tensor([False, True, False])
    truncated = torch.zeros(3, dtype=torch.bool)

    class FakeEnv:
        device = "cpu"

        def __init__(self) -> None:
            self.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(3, 1)))

        def step(self, action: torch.Tensor):
            assert action.shape == (3, 29)
            return "after", torch.ones(3), terminated, truncated, {}

    monkeypatch.setattr(
        online_module,
        "_snapshot",
        lambda _env, observations: _snapshot(0.0 if observations == "before" else 1.0),
    )
    monkeypatch.setattr(
        online_module,
        "_policy_observations",
        lambda raw, _num_envs: raw,
    )

    rollout = FixedDRTrackerRollout.__new__(FixedDRTrackerRollout)
    rollout.config = SimpleNamespace(num_envs=3)
    rollout.closed = False
    rollout.env = FakeEnv()
    rollout.observations = "before"
    rollout.policy = lambda _observations: torch.zeros(3, 29)
    rollout._clip_actions = None
    rollout._fixed_dr_model_fields = {"body_mass": torch.ones(3, 1)}
    rollout.dr_invariance_checks = 0
    rollout.world_ids = torch.arange(3)
    rollout.episode_ids = torch.zeros(3, dtype=torch.long)
    rollout.episode_steps = torch.full((3,), 4, dtype=torch.long)
    rollout.env_ids = torch.arange(3)
    rollout.collector_step = 4
    rollout.reset_events = 0
    rollout.environments_reset = 0
    rollout.synchronous_resets = 0
    rollout._motion_ids_seen = torch.arange(3)

    batch = rollout.step()

    assert batch["reset_boundary"].tolist() == [False, True, False]
    assert batch["episode_id"].tolist() == [0, 0, 0]
    assert batch["episode_step"].tolist() == [4, 4, 4]
    assert rollout.episode_ids.tolist() == [0, 1, 0]
    assert rollout.episode_steps.tolist() == [5, 0, 5]
    assert rollout.observations == "after"
    assert rollout.reset_events == 1
    assert rollout.environments_reset == 1
    assert rollout.synchronous_resets == 0
    assert rollout.dr_invariance_checks == 1
    assert rollout.motions_seen_count == 3


def test_online_rollout_adds_residual_after_frozen_tracker(monkeypatch) -> None:
    class FakeEnv:
        device = "cpu"

        def __init__(self) -> None:
            self.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(1, 1)))

        def step(self, action: torch.Tensor):
            torch.testing.assert_close(action, torch.full((1, 29), 0.75))
            done = torch.zeros(1, dtype=torch.bool)
            return "after", torch.ones(1), done, done, {}

    monkeypatch.setattr(
        online_module,
        "_snapshot",
        lambda _env, observations: {
            **{name: value[:1] for name, value in _snapshot(0.0).items()},
        },
    )
    monkeypatch.setattr(online_module, "_policy_observations", lambda raw, _num_envs: raw)

    rollout = FixedDRTrackerRollout.__new__(FixedDRTrackerRollout)
    rollout.config = SimpleNamespace(num_envs=1)
    rollout.closed = False
    rollout.env = FakeEnv()
    rollout.observations = "before"
    rollout.policy = lambda _observations: torch.full((1, 29), 0.5)
    rollout._clip_actions = 1.0
    rollout._fixed_dr_model_fields = {"body_mass": torch.ones(1, 1)}
    rollout.dr_invariance_checks = 0
    rollout.world_ids = torch.arange(1)
    rollout.episode_ids = torch.zeros(1, dtype=torch.long)
    rollout.episode_steps = torch.zeros(1, dtype=torch.long)
    rollout.env_ids = torch.arange(1)
    rollout.collector_step = 0
    rollout.reset_events = 0
    rollout.environments_reset = 0
    rollout.synchronous_resets = 0
    rollout._motion_ids_seen = torch.empty(0, dtype=torch.long)

    batch = rollout.step(
        lambda observation, tracker: torch.full_like(tracker, 0.25)
    )
    torch.testing.assert_close(batch["tracker_action"], torch.full((1, 29), 0.5))
    torch.testing.assert_close(batch["residual_action"], torch.full((1, 29), 0.25))
    torch.testing.assert_close(batch["action"], torch.full((1, 29), 0.75))
