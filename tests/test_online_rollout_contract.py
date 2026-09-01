from __future__ import annotations

from types import SimpleNamespace

import torch

from intact_tracking.rollout.online import (
    FixedDRRolloutConfig,
    _capture_randomized_model_fields,
    _disable_startup_reset_callbacks,
    _keep_startup_events,
    _restore_nominal_physics,
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


def test_online_rollout_restores_only_selected_worlds_to_nominal_physics() -> None:
    body_mass = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    armature = torch.tensor([[0.2, 0.4], [0.3, 0.5]])
    defaults = {
        "body_mass": torch.tensor([1.0, 1.5]),
        "dof_armature": torch.tensor([0.1, 0.1]),
    }
    model = SimpleNamespace(
        body_mass=body_mass,
        dof_armature=armature,
        clear_cache=lambda: None,
    )
    simulator = SimpleNamespace(
        model=model,
        get_default_field=lambda name: defaults[name],
        forward=lambda: None,
    )
    action_term = SimpleNamespace(joint_offset=torch.full((2, 2), 0.25))
    env = SimpleNamespace(
        num_envs=2,
        sim=simulator,
        event_manager=SimpleNamespace(
            domain_randomization_fields=("body_mass", "dof_armature")
        ),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.ones(2, 2)))},
        action_manager=SimpleNamespace(get_term=lambda _: action_term),
    )

    metrics = _restore_nominal_physics(env, torch.tensor([0], dtype=torch.long))

    torch.testing.assert_close(body_mass[0], defaults["body_mass"])
    torch.testing.assert_close(armature[0], defaults["dof_armature"])
    torch.testing.assert_close(body_mass[1], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(armature[1], torch.tensor([0.3, 0.5]))
    torch.testing.assert_close(env.scene["robot"].data.encoder_bias[0], torch.zeros(2))
    torch.testing.assert_close(action_term.joint_offset[0], torch.zeros(2))
    assert metrics == {
        "model_field_max_abs_error": 0.0,
        "encoder_bias_max_abs_error": 0.0,
        "dr_model_field_max_abs_difference": 3.5,
        "dr_encoder_bias_max_abs_difference": 1.0,
    }


def test_online_rollout_requires_an_integral_nominal_world_count() -> None:
    try:
        FixedDRRolloutConfig(
            checkpoint_file="tracker.pt",
            motion_file="motion.npz",
            num_envs=3,
            nominal_fraction=0.5,
        )
    except ValueError as error:
        assert "must be an integer" in str(error)
    else:
        raise AssertionError("Expected an invalid half split for three vector worlds")
