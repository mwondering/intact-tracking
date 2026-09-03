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


def test_online_rollout_treats_resampling_and_reset_as_boundaries(monkeypatch) -> None:
    terminated = torch.tensor([False, True, False])
    truncated = torch.zeros(3, dtype=torch.bool)
    motion_resample_boundary = torch.tensor([False, False, True])
    motion_command = SimpleNamespace(motion_resample_boundary=motion_resample_boundary)

    class FakeEnv:
        device = "cpu"

        def __init__(self) -> None:
            self.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(3, 1)))
            self.command_manager = SimpleNamespace(get_term=lambda _name: motion_command)
            simulator_target = torch.zeros(3, 29)
            simulator_target[1:] = 1.6676
            self.scene = {
                "robot": SimpleNamespace(data=SimpleNamespace(joint_pos_target=simulator_target))
            }

        def step(self, action: torch.Tensor):
            assert action.shape == (3, 29)
            return "after", torch.ones(3), terminated, truncated, {}

    monkeypatch.setattr(
        online_module,
        "_snapshot",
        lambda _env, observations: _snapshot(0.0 if observations == "before" else 1.0),
    )
    predictor_snapshots = iter((_snapshot(0.0), _snapshot(1.0)))
    monkeypatch.setattr(
        online_module,
        "_forward_predictor_snapshot",
        lambda _env: next(predictor_snapshots),
    )
    monkeypatch.setattr(
        online_module,
        "_policy_observations",
        lambda raw, _num_envs: raw,
    )

    rollout = FixedDRTrackerRollout.__new__(FixedDRTrackerRollout)
    latent_calls = 0

    class Actor:
        def get_latent(self, _observations: object) -> torch.Tensor:
            nonlocal latent_calls
            latent_calls += 1
            return torch.zeros(3, 64)

    rollout._runtime = SimpleNamespace(actor=Actor())
    rollout.config = SimpleNamespace(num_envs=3)
    rollout.closed = False
    rollout.env = FakeEnv()
    rollout.motion_command = motion_command
    rollout.observations = "before"
    rollout.policy = lambda _observations: torch.zeros(3, 29)
    rollout._clip_actions = None
    rollout.predictor_action_transform = lambda action: torch.zeros_like(action)
    rollout.predictor_action_target_verified = False
    rollout.predictor_action_target_max_abs_error = None
    rollout._fixed_dr_model_fields = {"body_mass": torch.ones(3, 1)}
    rollout.dr_invariance_checks = 0
    rollout.world_ids = torch.arange(3)
    rollout.is_nominal = torch.tensor([True, False, False])
    rollout.episode_ids = torch.zeros(3, dtype=torch.long)
    rollout.episode_steps = torch.full((3,), 4, dtype=torch.long)
    rollout.env_ids = torch.arange(3)
    rollout.collector_step = 4
    rollout.reset_events = 0
    rollout.environments_reset = 0
    rollout.synchronous_resets = 0
    rollout._motion_ids_seen = torch.arange(3)

    batch = rollout.step(predictor_only=True)

    assert latent_calls == 0
    assert "policy_observation" not in batch
    assert "reference_observation" not in batch
    assert "tracking_error" not in batch
    assert batch["reset_boundary"].tolist() == [False, True, True]
    assert batch["episode_id"].tolist() == [0, 0, 0]
    assert batch["episode_step"].tolist() == [4, 4, 4]
    assert rollout.episode_ids.tolist() == [0, 1, 1]
    assert rollout.episode_steps.tolist() == [5, 0, 0]
    assert rollout.observations == "after"
    assert rollout.reset_events == 1
    assert rollout.environments_reset == 1
    assert rollout.synchronous_resets == 0
    assert rollout.dr_invariance_checks == 1
    assert rollout.motions_seen_count == 3
    assert rollout.predictor_action_target_verified
    assert rollout.predictor_action_target_max_abs_error == 0.0


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
    rollout.is_nominal = torch.zeros(1, dtype=torch.bool)
    rollout.episode_ids = torch.zeros(1, dtype=torch.long)
    rollout.episode_steps = torch.zeros(1, dtype=torch.long)
    rollout.env_ids = torch.arange(1)
    rollout.collector_step = 0
    rollout.reset_events = 0
    rollout.environments_reset = 0
    rollout.synchronous_resets = 0
    rollout._motion_ids_seen = torch.empty(0, dtype=torch.long)

    batch = rollout.step(lambda observation, tracker: torch.full_like(tracker, 0.25))
    torch.testing.assert_close(batch["tracker_action"], torch.full((1, 29), 0.5))
    torch.testing.assert_close(batch["residual_action"], torch.full((1, 29), 0.25))
    torch.testing.assert_close(batch["action"], torch.full((1, 29), 0.75))
