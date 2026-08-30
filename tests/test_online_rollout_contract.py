from __future__ import annotations

from types import SimpleNamespace

import torch

from intact_tracking.rollout.online import (
    _capture_randomized_model_fields,
    _disable_startup_reset_callbacks,
    _keep_startup_events,
)


def test_online_rollout_removes_every_non_startup_event() -> None:
    startup = SimpleNamespace(mode="startup")
    config = SimpleNamespace(
        events={
            "mass": startup,
            "reset_noise": SimpleNamespace(mode="reset"),
            "push": SimpleNamespace(mode="interval"),
            "force_lifetime": SimpleNamespace(mode="step"),
        }
    )
    kept, removed = _keep_startup_events(config)
    assert kept == ["mass"]
    assert removed == ["force_lifetime", "push", "reset_noise"]
    assert config.events == {"mass": startup}


def test_online_rollout_disables_startup_class_reset_callbacks() -> None:
    first = SimpleNamespace(func=type("MotorRandomization", (), {})())
    second = SimpleNamespace(func=type("EncoderRandomization", (), {})())
    manager = SimpleNamespace(_mode_class_term_cfgs={"startup": [first, second], "interval": []})
    env = SimpleNamespace(event_manager=manager)

    disabled = _disable_startup_reset_callbacks(env)

    assert disabled == ["EncoderRandomization", "MotorRandomization"]
    assert manager._mode_class_term_cfgs["startup"] == []


def test_online_rollout_snapshots_randomized_model_fields() -> None:
    mass = torch.tensor([[1.0], [2.0]])
    manager = SimpleNamespace(domain_randomization_fields=("body_mass", "missing"))
    env = SimpleNamespace(
        event_manager=manager,
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=mass)),
    )
    snapshots = _capture_randomized_model_fields(env)
    mass.add_(1)

    assert list(snapshots) == ["body_mass"]
    torch.testing.assert_close(snapshots["body_mass"], torch.tensor([[1.0], [2.0]]))
