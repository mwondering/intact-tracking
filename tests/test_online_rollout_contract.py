from __future__ import annotations

from types import SimpleNamespace

import torch

from intact_tracking.rollout.online import (
    FixedDRRolloutConfig,
    _capture_privileged_dynamics_targets,
    _capture_randomized_model_fields,
    _disable_startup_reset_callbacks,
    _keep_startup_events,
    _restore_nominal_physics,
    _tile_fixed_dynamics_prototypes,
)


class _TensorProxy:
    """Minimal non-Tensor proxy matching MJLab's TorchArray interface."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def clone(self) -> torch.Tensor:
        return self.tensor.clone()

    def __setitem__(self, index, value: torch.Tensor) -> None:
        self.tensor[index] = value


def test_online_rollout_removes_every_non_startup_event() -> None:
    startup = SimpleNamespace(mode="startup")
    config = SimpleNamespace(
        events={
            "mass": startup,
            "reset_noise": SimpleNamespace(mode="reset"),
            "periodic": SimpleNamespace(mode="interval"),
            "push_robot": SimpleNamespace(mode="step"),
        }
    )
    kept, removed = _keep_startup_events(config)
    assert kept == ["mass"]
    assert removed == ["periodic", "push_robot", "reset_noise"]
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


def test_online_rollout_extracts_compact_physics_targets_and_ignores_encoder_bias() -> None:
    def body_com_offset():
        pass

    def body_mass():
        pass

    def geom_friction():
        pass

    def encoder_bias():
        pass

    asset_cfg = SimpleNamespace(
        name="robot",
        body_ids=[0],
        geom_ids=[0, 1],
    )
    asset = SimpleNamespace(
        body_names=("torso",),
        geom_names=("left_foot", "right_foot"),
        indexing=SimpleNamespace(
            body_ids=torch.tensor([1]),
            geom_ids=torch.tensor([2, 3]),
        ),
    )
    default_ipos = torch.zeros(3, 3)
    body_ipos = default_ipos.repeat(2, 1, 1)
    body_ipos[0, 1] = torch.tensor([0.1, 0.2, 0.3])
    body_ipos[1, 1] = torch.tensor([-0.1, -0.2, -0.3])
    default_mass = torch.tensor([1.0, 2.0, 3.0])
    expanded_mass = default_mass.repeat(2, 1)
    expanded_mass[:, 1] = torch.tensor([2.2, 1.8])
    default_friction = torch.ones(4, 3)
    expanded_friction = default_friction.repeat(2, 1, 1)
    expanded_friction[0, 2:4, 0] = 0.5
    expanded_friction[1, 2:4, 0] = 1.5
    configs = {
        "base_com": SimpleNamespace(
            func=body_com_offset,
            params={"asset_cfg": asset_cfg, "ranges": {0: (-1, 1), 1: (-1, 1), 2: (-1, 1)}},
        ),
        "base_mass": SimpleNamespace(
            func=body_mass,
            params={"asset_cfg": asset_cfg, "shared_random": False},
        ),
        "foot_friction": SimpleNamespace(
            func=geom_friction,
            params={"asset_cfg": asset_cfg, "shared_random": True},
        ),
        "encoder_bias": SimpleNamespace(func=encoder_bias, params={}),
    }
    manager = SimpleNamespace(
        active_terms={"startup": list(configs)},
        get_term_cfg=configs.__getitem__,
    )
    defaults = {
        "body_ipos": default_ipos,
        "body_mass": default_mass,
        "geom_friction": default_friction,
    }
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        event_manager=manager,
        scene={"robot": asset},
        sim=SimpleNamespace(
            model=SimpleNamespace(
                body_ipos=body_ipos,
                body_mass=expanded_mass,
                geom_friction=expanded_friction,
            ),
            get_default_field=defaults.__getitem__,
        ),
    )

    target = _capture_privileged_dynamics_targets(env)

    assert target.names == (
        "base_com/com_offset/torso/x",
        "base_com/com_offset/torso/y",
        "base_com/com_offset/torso/z",
        "base_mass/relative_mass/torso",
        "foot_friction/friction/shared/0",
    )
    torch.testing.assert_close(
        target.values,
        torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.5], [-0.1, -0.2, -0.3, -0.1, 1.5]]),
    )
    assert target.ignored_startup_events == ("encoder_bias",)


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
        event_manager=SimpleNamespace(domain_randomization_fields=("body_mass", "dof_armature")),
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


def test_grouped_rollout_requires_aligned_fixed_dynamics_families() -> None:
    for kwargs, message in (
        ({"num_envs": 10, "dynamics_classes": 4}, "divide num_envs"),
        (
            {"num_envs": 8, "dynamics_classes": 4, "world_id_offset": 2},
            "align to dynamics_classes",
        ),
        (
            {"num_envs": 8, "dynamics_classes": 4, "nominal_fraction": 0.5},
            "does not support a nominal subset",
        ),
    ):
        try:
            FixedDRRolloutConfig(
                checkpoint_file="tracker.pt",
                motion_file="motion.npz",
                **kwargs,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Expected invalid grouped rollout config: {kwargs}")


def test_online_rollout_tiles_fixed_dynamics_prototypes_without_resampling() -> None:
    body_mass = torch.arange(8, dtype=torch.float32)[:, None]
    joint_damping = (10.0 + torch.arange(8, dtype=torch.float32))[:, None]
    gravity = torch.arange(24, dtype=torch.float32).view(8, 3)
    model_gravity = torch.zeros_like(gravity)
    calls = {"clear_cache": 0, "forward": 0}

    gravity_event = type("perturb_gravity", (), {})()
    gravity_event._gravity = gravity
    event_manager = SimpleNamespace(
        domain_randomization_fields=("body_mass", "joint_damping", "shared_field"),
        active_terms={"startup": ["gravity"]},
        get_term_cfg=lambda _name: SimpleNamespace(func=gravity_event),
    )
    model = SimpleNamespace(
        body_mass=body_mass,
        joint_damping=joint_damping,
        shared_field=torch.tensor([123.0]),
        opt=SimpleNamespace(gravity=model_gravity),
        clear_cache=lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
    )
    env = SimpleNamespace(
        num_envs=8,
        device="cpu",
        event_manager=event_manager,
        sim=SimpleNamespace(
            model=model,
            forward=lambda: calls.__setitem__("forward", calls["forward"] + 1),
        ),
    )
    expected_mass = body_mass[:4].repeat(2, 1)
    expected_damping = joint_damping[:4].repeat(2, 1)
    expected_gravity = gravity[:4].repeat(2, 1)

    metrics = _tile_fixed_dynamics_prototypes(env, dynamics_classes=4)

    torch.testing.assert_close(model.body_mass, expected_mass)
    torch.testing.assert_close(model.joint_damping, expected_damping)
    torch.testing.assert_close(gravity_event._gravity, expected_gravity)
    torch.testing.assert_close(model.opt.gravity, expected_gravity)
    torch.testing.assert_close(model.shared_field, torch.tensor([123.0]))
    assert calls == {"clear_cache": 1, "forward": 1}
    assert metrics == {
        "dynamics_classes": 4,
        "motion_groups_per_rank": 2,
        "tiled_model_fields": ["body_mass", "joint_damping"],
        "empty_model_fields": [],
        "gravity_tiled": True,
        "max_abs_replication_error": 0.0,
    }


def test_online_rollout_tiles_mjlab_tensor_proxy_without_changing_samples() -> None:
    sampled_mass = torch.arange(8, dtype=torch.float32)[:, None]
    body_mass = _TensorProxy(sampled_mass.clone())
    empty_tendon = _TensorProxy(torch.empty((8, 0), dtype=torch.float32))
    calls = {"clear_cache": 0, "forward": 0}
    env = SimpleNamespace(
        num_envs=8,
        device="cpu",
        event_manager=SimpleNamespace(
            domain_randomization_fields=("body_mass", "tendon_length0"),
            active_terms={"startup": []},
        ),
        sim=SimpleNamespace(
            model=SimpleNamespace(
                body_mass=body_mass,
                tendon_length0=empty_tendon,
                clear_cache=lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
            ),
            forward=lambda: calls.__setitem__("forward", calls["forward"] + 1),
        ),
    )

    metrics = _tile_fixed_dynamics_prototypes(env, dynamics_classes=4)

    torch.testing.assert_close(body_mass.tensor, sampled_mass[:4].repeat(2, 1))
    assert empty_tendon.tensor.shape == (8, 0)
    assert calls == {"clear_cache": 1, "forward": 1}
    assert metrics["tiled_model_fields"] == ["body_mass"]
    assert metrics["empty_model_fields"] == ["tendon_length0"]
    assert metrics["gravity_tiled"] is False
    assert metrics["max_abs_replication_error"] == 0.0
