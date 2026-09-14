from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

import intact_tracking.residual_policy as residual_module
from intact_tracking.adaptation_policy import (
    ORACLE_HISTORY,
    ORACLE_KEY_BODY,
    PRIVILEGE,
    ContextAdaptationActor,
    DeployableHistoryEncoder,
    PrivilegedAdaptationActor,
    _base_with_optional_update,
    configure_oracle_proprioception,
)
from intact_tracking.adaptation_sensors import IMU_HISTORY, measured_imu_acceleration
from intact_tracking.cli.adaptation_eval import fixed_starts


def test_full_physics_schema_does_not_read_privileged_state_or_labels(monkeypatch):
    from types import SimpleNamespace

    import intact_tracking.adaptation_policy as module

    environment = object()
    values = torch.randn(3, 495)
    calls = []

    def physics(env):
        assert env is environment
        calls.append(env)
        return values

    def forbidden(*args, **kwargs):
        raise AssertionError("Physics-only input must not read current-state truths")

    monkeypatch.setattr(module, "physics_observation", physics)
    monkeypatch.setattr(module, "_robot_raw_state", forbidden)
    wrapper = SimpleNamespace(privilege_schema="physics", unwrapped=environment)
    obs = TensorDict({}, [3])
    result = module.PrivilegedAdaptationWrapper.attach(wrapper, obs)
    torch.testing.assert_close(result[PRIVILEGE], values, atol=0, rtol=0)
    assert list(result.keys()) == [PRIVILEGE]
    module.PrivilegedAdaptationWrapper.attach(wrapper, TensorDict({}, [3]))
    assert len(calls) == 2  # Read actual parameters again, not stale labels.


def test_direct_physics_probe_scores_actor_code_without_refitting():
    from intact_tracking.cli.context_probe import direct_compact_code_probe

    truth = torch.randn(8, 7)
    features = truth.unsqueeze(0).expand(3, -1, -1).clone()
    train, test = torch.arange(4), torch.arange(4, 8)
    exact = direct_compact_code_probe(features, truth, train, test)
    assert exact["physical_rmse"] == [0.0] * 7
    assert exact["r2"] == [1.0] * 7
    features[..., 0] += 0.1
    biased = direct_compact_code_probe(features, truth, train, test)
    assert biased["physical_rmse"][0] == pytest.approx(0.3, abs=1e-6)
    assert biased["physical_bias"][0] == pytest.approx(0.3, abs=1e-6)
    with pytest.raises(ValueError, match="seven"):
        direct_compact_code_probe(features[..., :6], truth, train, test)


def test_mean_only_physics_configuration_and_actor_keep_code_width_seven():
    from types import SimpleNamespace

    from intact_tracking.adaptation_context_memory import (
        CONTEXT_MEAN,
        physics_mean_only_configuration,
    )

    config = {"class_name": "intact_tracking.adaptation_policy:ContextAdaptationActor",
              "adaptation_latent_dim": 7, "physics_low_rank": True}
    changed = physics_mean_only_configuration(config, 50)
    assert "context_mean_only" not in config
    assert changed["context_mean_only"] and not changed["train_context_encoder"]
    with pytest.raises(ValueError, match="positive"):
        physics_mean_only_configuration(config, 0)
    with pytest.raises(ValueError, match="stateless"):
        physics_mean_only_configuration(changed, 50)
    mean = torch.randn(4, 7)
    actor = SimpleNamespace(context_latent_mean=True, context_mean_only=True,
                            instant_context_latent=lambda obs: torch.ones(4, 7))
    obs = {CONTEXT_MEAN: mean}
    torch.testing.assert_close(ContextAdaptationActor.context_latent(actor, obs), mean, atol=0, rtol=0)
    actor.context_ablation = "shuffle"
    torch.testing.assert_close(ContextAdaptationActor.context_latent(actor, obs), mean.roll(1, 0), atol=0, rtol=0)
    actor.context_ablation = "zero"
    assert ContextAdaptationActor.context_latent(actor, obs).count_nonzero() == 0


def test_learned_physical_code_is_nominal_zero_after_arbitrary_encoder_training():
    from types import SimpleNamespace

    encoder = nn.Sequential(nn.Linear(9, 12), nn.ELU(), nn.Linear(12, 7), nn.Tanh())
    actor = SimpleNamespace(privilege_encoder=encoder, privilege_normalizer=nn.Identity(),
                            latent_low_rank=True)
    nominal = {PRIVILEGE: torch.zeros(5, 9)}
    physical = {PRIVILEGE: torch.randn(5, 9)}
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    for _ in range(4):
        assert PrivilegedAdaptationActor.privileged_latent(actor, nominal).count_nonzero() == 0
        code = PrivilegedAdaptationActor.privileged_latent(actor, physical)
        assert code.abs().max() <= 1
        optimizer.zero_grad()
        (code - 0.2).square().mean().backward()
        optimizer.step()
    assert PrivilegedAdaptationActor.privileged_latent(actor, nominal).count_nonzero() == 0


def test_zero_physics_information_control_keeps_trainable_global_latent():
    from types import SimpleNamespace

    encoder = nn.Sequential(nn.Linear(9, 12), nn.ELU(), nn.Linear(12, 7), nn.Tanh())
    actor = SimpleNamespace(privilege_encoder=encoder, privilege_normalizer=nn.Identity(),
                            actor_physics_input="zero", latent_low_rank=False)
    a, b = {PRIVILEGE: torch.randn(5, 9)}, {PRIVILEGE: torch.full((5, 9), float("nan"))}
    za = PrivilegedAdaptationActor.privileged_latent(actor, a)
    zb = PrivilegedAdaptationActor.privileged_latent(actor, b)
    torch.testing.assert_close(za, zb, atol=0, rtol=0)
    assert PrivilegedAdaptationActor._actor_privilege_input(actor, a).count_nonzero() == 0
    before = za.detach().clone()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    optimizer.zero_grad()
    (za - 0.2).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, PrivilegedAdaptationActor.privileged_latent(actor, a))


def test_constant_physics_control_has_no_information_and_keeps_all_ranks_trainable():
    from types import SimpleNamespace

    from rsl_rl.modules import MLP

    from intact_tracking.adaptation_modulation import PhysicsLowRankResidual

    actor = SimpleNamespace(privilege_encoder=nn.Identity(), privilege_normalizer=nn.Identity(),
                            actor_physics_input="constant", latent_low_rank=False)
    a, b = {PRIVILEGE: torch.randn(5, 7)}, {PRIVILEGE: torch.full((5, 7), float("nan"))}
    za = PrivilegedAdaptationActor.privileged_latent(actor, a)
    zb = PrivilegedAdaptationActor.privileged_latent(actor, b)
    torch.testing.assert_close(za, torch.ones_like(za), atol=0, rtol=0)
    torch.testing.assert_close(za, zb, atol=0, rtol=0)
    model = PhysicsLowRankResidual(MLP(5, 3, (8, 6), "elu"), 7, 2)
    value = torch.cat((torch.randn(5, 5), za), -1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        (model(value) - 1).square().mean().backward()
        optimizer.step()
    for layer in model.layers:
        assert (layer.up.weight.abs().sum(0) > 0).all()
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(".up." in k or ".down." in k for k in changed)
    assert model(value).count_nonzero() > 0


def test_encoder_only_transfer_preserves_controller_and_rejects_wrong_semantics():
    from copy import deepcopy

    from omegaconf import OmegaConf

    from intact_tracking.cli.adaptation_distill import transfer_compact_context_encoder

    student = nn.Module()
    student.tracker = nn.Module()
    student.tracker.history_normalizer = nn.BatchNorm1d(4)
    student.context_encoder = nn.Linear(4, 7)
    student.residual_mlp = nn.Linear(11, 3)
    student.frozen_nominal_prior_mlp = nn.Linear(4, 3)
    cfg = {"class_name": "intact_tracking.adaptation_policy:ContextAdaptationActor",
           "adaptation_latent_dim": 7, "physics_low_rank": True,
           "context_imu_accel": True, "context_right_aligned": True}
    before = deepcopy(student.state_dict())
    state = deepcopy(before)
    for key in state:
        if not key.startswith("tracker.history_normalizer."):
            state[key].add_(2)
    source = {"actor_state_dict": state, "cfg": OmegaConf.create({"agent": {"actor": cfg}}),
              "residual_policy": {"latent_target_contract": {"schema": "compact_physics", "actor_physics_input": "normal"}}}
    audit = transfer_compact_context_encoder(student, source, cfg)
    assert audit["source_optimizer_controller_and_prior_transferred"] is False
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, state[key] if key.startswith("context_encoder.") else before[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="semantics"):
        transfer_compact_context_encoder(student, source, dict(cfg, context_imu_accel=False))
    source["actor_state_dict"]["tracker.history_normalizer.running_mean"].add_(1)
    with pytest.raises(ValueError, match="normalization"):
        transfer_compact_context_encoder(student, source, cfg)
    source["residual_policy"]["latent_target_contract"]["actor_physics_input"] = "constant"
    with pytest.raises(ValueError, match="target semantics"):
        transfer_compact_context_encoder(student, source, cfg)


def test_context_warm_start_validates_semantics_not_trainability():
    from intact_tracking.cli.adaptation_distill import validate_context_initialization

    original = {
        "class_name": "intact_tracking.adaptation_policy:ContextAdaptationActor",
        "context_imu_accel": True,
        "residual_scale": 1.0,
    }
    fine_tune = dict(original, train_reference_encoder=True, train_context_encoder=False)
    validate_context_initialization(fine_tune, original)
    for name, value in (("context_right_aligned", True), ("residual_scale", 0.25)):
        with pytest.raises(ValueError, match="input/action contract"):
            validate_context_initialization(dict(fine_tune, **{name: value}), original)
    with pytest.raises(ValueError, match="deployable context"):
        validate_context_initialization(fine_tune, {"class_name": "PrivilegedAdaptationActor"})
    with_memory = dict(fine_tune, context_latent_mean=True, context_mean_horizon=250)
    validate_context_initialization(with_memory, original)
    with pytest.raises(ValueError, match="memory contract"):
        validate_context_initialization(fine_tune, with_memory)
    with pytest.raises(ValueError, match="memory contract"):
        validate_context_initialization(dict(with_memory, context_mean_horizon=50), with_memory)


def test_identity_feature_adapter_upgrade_preserves_source_and_requires_zero_output():
    from intact_tracking.cli.adaptation_distill import (
        add_identity_feature_adapter_state,
        validate_context_initialization,
    )

    source_cfg = {"class_name": "intact_tracking.adaptation_policy:ContextAdaptationActor"}
    target_cfg = dict(source_cfg, context_feature_adapter=True)
    with pytest.raises(ValueError, match="input/action contract"):
        validate_context_initialization(target_cfg, source_cfg)
    validate_context_initialization(target_cfg, source_cfg, allow_identity_feature_adapter=True)
    student = nn.Module()
    student.feature_adapter = nn.Sequential(nn.Linear(5, 4), nn.ELU(), nn.Linear(4, 3))
    student.base = nn.Linear(3, 2)
    with torch.no_grad():
        student.feature_adapter[-1].weight.zero_()
        student.feature_adapter[-1].bias.zero_()
    source = {key: value.clone() for key, value in student.state_dict().items() if not key.startswith("feature_adapter.")}
    result = add_identity_feature_adapter_state(student, source)
    student.load_state_dict(result, strict=True)
    for key, value in source.items():
        torch.testing.assert_close(result[key], value, atol=0, rtol=0)
    features, context = torch.randn(7, 3), torch.randn(7, 2)
    adapted = features + student.feature_adapter(torch.cat((features, context), -1))
    torch.testing.assert_close(adapted, features, atol=0, rtol=0)
    with pytest.raises(ValueError, match="new feature adapter"):
        add_identity_feature_adapter_state(student, result)
    with torch.no_grad():
        student.feature_adapter[-1].bias.fill_(0.1)
    with pytest.raises(ValueError, match="exactly zero"):
        add_identity_feature_adapter_state(student, source)


def test_context_mean_is_prefix_then_ema_and_resets_only_selected_worlds():
    from intact_tracking.adaptation_context_memory import update_context_mean

    mean, count = torch.zeros(2, 1), torch.zeros(2, 1)
    no_reset = torch.zeros(2, dtype=torch.bool)
    for value, expected in ((1, 1), (3, 2), (5, 3), (9, 5)):
        current = torch.full((2, 1), float(value))
        mean, count = update_context_mean(mean, count, current, no_reset, horizon=3)
        torch.testing.assert_close(mean, torch.full_like(mean, float(expected)))
    mean, count = update_context_mean(
        mean, count, torch.full((2, 1), 11.0), torch.tensor([True, False]), horizon=3
    )
    torch.testing.assert_close(mean, torch.tensor([[11.0], [7.0]]))
    torch.testing.assert_close(count, torch.tensor([[1.0], [3.0]]))
    with pytest.raises(ValueError, match="positive"):
        update_context_mean(mean, count, mean, no_reset, horizon=0)


def test_bias_readout_expansion_preserves_old_outputs_and_action_weights():
    from intact_tracking.cli.adaptation_distill import (
        expand_bias_readout_state,
        validate_context_initialization,
    )

    source = {"class_name": "intact_tracking.adaptation_policy:ContextAdaptationActor", "context_auxiliary_dim": 10}
    target = dict(source, context_auxiliary_dim=39)
    with pytest.raises(ValueError, match="input/action contract"):
        validate_context_initialization(target, source)
    validate_context_initialization(target, source, allow_bias_head_expansion=True)
    state = {
        "context_auxiliary_head.2.weight": torch.randn(10, 64),
        "context_auxiliary_head.2.bias": torch.randn(10),
        "residual_mlp.0.weight": torch.randn(8, 9),
    }
    expected = {
        "context_auxiliary_head.2.weight": torch.empty(39, 64),
        "context_auxiliary_head.2.bias": torch.empty(39),
    }
    expanded = expand_bias_readout_state(state, expected)
    torch.testing.assert_close(expanded["residual_mlp.0.weight"], state["residual_mlp.0.weight"], atol=0, rtol=0)
    for suffix in ("weight", "bias"):
        key = f"context_auxiliary_head.2.{suffix}"
        torch.testing.assert_close(expanded[key][:10], state[key], atol=0, rtol=0)
        assert expanded[key][10:].count_nonzero() == 0
        assert state[key].shape[0] == 10
    with pytest.raises(ValueError, match="ten to39"):
        expand_bias_readout_state(expanded, expected)


def test_context_mean_wrapper_is_causal_snapshot_and_deployable_only():
    from types import SimpleNamespace

    from intact_tracking.adaptation_context_memory import CONTEXT_MEAN, CausalContextMeanWrapper

    env = SimpleNamespace(common_step_counter=0)
    inner = SimpleNamespace(num_envs=2, device="cpu", unwrapped=env)
    wrapper = CausalContextMeanWrapper(inner, latent_dim=1, horizon=3)
    obs = TensorDict({"sensor": torch.tensor([[1.0], [2.0]]), "priv": torch.full((2, 3), torch.nan)}, [2])
    assert wrapper.attach(obs)[CONTEXT_MEAN].count_nonzero() == 0
    encoder = nn.Linear(1, 1)

    def encode(inputs):
        assert set(inputs.keys()) == {"sensor"}
        return inputs["sensor"]

    actor = SimpleNamespace(
        context_encoder=encoder, instant_context_latent=encode,
        deployable_observation_groups=("sensor", CONTEXT_MEAN),
    )
    with pytest.raises(ValueError, match="frozen"):
        wrapper.bind(actor)
    encoder.requires_grad_(False)
    wrapper.bind(actor)
    snapshot = wrapper.attach(obs)[CONTEXT_MEAN]
    torch.testing.assert_close(snapshot, obs["sensor"])
    changed = obs.clone()
    changed["sensor"].fill_(9)
    torch.testing.assert_close(wrapper.attach(changed)[CONTEXT_MEAN], snapshot)
    assert wrapper.count.eq(1).all()
    env.common_step_counter += 1
    result = wrapper.attach(changed, reset=torch.tensor([True, False]))[CONTEXT_MEAN]
    torch.testing.assert_close(result, torch.tensor([[9.0], [5.5]]))
    torch.testing.assert_close(snapshot, torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(wrapper.count, torch.tensor([[1.0], [2.0]]))


def test_causal_orientation_filter_respects_gyro_motion_and_quaternion_sign():
    from intact_tracking.adaptation_filters import causal_orientation_filter

    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gyro = torch.tensor([[0.0, 0.0, 1.0]])
    measured = torch.tensor([[torch.cos(torch.tensor(0.01)), 0.0, 0.0, torch.sin(torch.tensor(0.01))]])
    for weight in (0.1, 0.3, 1.0):
        result = causal_orientation_filter(-measured, gyro, previous, gyro, 0.02, weight)
        torch.testing.assert_close(result, measured)
        torch.testing.assert_close(result.norm(dim=-1), torch.ones(1))
    noisy = torch.tensor([[torch.cos(torch.tensor(0.1)), 0.0, 0.0, torch.sin(torch.tensor(0.1))]])
    result = causal_orientation_filter(noisy, gyro * 0, previous, gyro * 0, 0.02, 0.1)
    assert (result - previous).norm() < (noisy - previous).norm() / 5


def test_causal_action_filter_uses_only_known_previous_actions_and_handles_reset():
    from intact_tracking.adaptation_filters import causal_action_filter

    history = torch.zeros(3, 6100)
    action = torch.full((3, 29), 3.0)
    torch.testing.assert_close(causal_action_filter(action, history, 0.2, 2), action)
    past = history[:, 3200:4650].reshape(3, 50, 29)
    past[:, -1] = 2
    past[:, -2] = 1
    torch.testing.assert_close(causal_action_filter(action, history, 0.2, 2), action)
    torch.testing.assert_close(
        causal_action_filter(action, history, 0.2, 1), torch.full_like(action, 2.8)
    )
    history[:, :3200] = float("nan")
    history[:, 4650:] = float("nan")
    torch.testing.assert_close(causal_action_filter(action, history, 0.2, 2), action)


def test_oracle_channels_are_separate_from_unchanged_deployment_noise():
    from types import SimpleNamespace

    def group():
        return SimpleNamespace(
            enable_corruption=True,
            terms={
                "sensor": SimpleNamespace(
                    noise="unchanged", params={"biased": True, "noise_std": 0.5}
                )
            },
        )

    cfg = SimpleNamespace(observations={"estimator_history": group(), "robot_key_body": group()})
    configure_oracle_proprioception(cfg)
    for name in ("estimator_history", "robot_key_body"):
        original = cfg.observations[name]
        assert original.enable_corruption
        assert original.terms["sensor"].noise == "unchanged"
        assert original.terms["sensor"].params == {"biased": True, "noise_std": 0.5}
    for name in (ORACLE_HISTORY, ORACLE_KEY_BODY):
        privileged = cfg.observations[name]
        assert not privileged.enable_corruption
        assert privileged.terms["sensor"].noise is None
        assert privileged.terms["sensor"].params == {"biased": False, "noise_std": 0.0}


def test_oracle_feature_substitution_does_not_overwrite_student_observations():
    from types import SimpleNamespace

    obs = TensorDict(
        {
            "estimator_history": torch.ones(2, 2),
            "robot_key_body": torch.ones(2, 1),
            ORACLE_HISTORY: torch.full((2, 2), 2.0),
            ORACLE_KEY_BODY: torch.full((2, 1), 3.0),
            "height": torch.zeros(2, 1),
            "contact": torch.zeros(2, 2),
            "reference": torch.zeros(2, 1),
        },
        [2],
    )
    tracker = SimpleNamespace(
        obs_groups=["estimator_history", "robot_key_body"],
        estimator_history_group="estimator_history",
        robot_key_body_group="robot_key_body",
        estimator_target_group="height",
        foot_contact_target_group="contact",
        reference_encoder_target_group="reference",
        policy_normalizer=nn.Identity(),
        _spv5_2_features=lambda inputs, *args: torch.cat(
            (inputs["estimator_history"], inputs["robot_key_body"]), dim=-1
        ),
        mlp=nn.Linear(3, 2),
        distribution=GaussianDistribution(2, init_std=0.3),
    )
    actor = SimpleNamespace(
        tracker=tracker,
        oracle_tracking_features=True,
        oracle_clean_proprio=True,
        train_base_policy=False,
    )
    features, _ = _base_with_optional_update(actor, obs)
    torch.testing.assert_close(features, torch.tensor([[2.0, 2.0, 3.0]]).repeat(2, 1))
    torch.testing.assert_close(obs["estimator_history"], torch.ones(2, 2))
    torch.testing.assert_close(obs["robot_key_body"], torch.ones(2, 1))
    tracker.get_latent = lambda inputs: torch.cat(
        (inputs["estimator_history"], inputs["robot_key_body"]), dim=-1
    )
    actor.oracle_feature_mix = torch.tensor(0.0)
    start, _ = _base_with_optional_update(actor, obs)
    torch.testing.assert_close(start, features, atol=0, rtol=0)
    actor.oracle_feature_mix.fill_(1.0)
    end, _ = _base_with_optional_update(actor, obs)
    torch.testing.assert_close(end, torch.ones(2, 3), atol=0, rtol=0)
    actor.oracle_feature_mix.fill_(0.5)
    middle, _ = _base_with_optional_update(actor, obs)
    torch.testing.assert_close(middle, (start + end) / 2)


def test_oracle_curriculum_advances_only_after_warmup_and_is_bounded(monkeypatch):
    from types import SimpleNamespace

    from intact_tracking.adaptation_ppo import CriticWarmupPPO, OracleFeatureCurriculumPPO

    def update(algorithm):
        algorithm.adaptation_update_count += 1
        return {}

    monkeypatch.setattr(CriticWarmupPPO, "update", update)
    algorithm = object.__new__(OracleFeatureCurriculumPPO)
    algorithm.actor = SimpleNamespace(oracle_feature_mix=torch.tensor(0.0))
    algorithm.adaptation_update_count = 0
    algorithm.critic_warmup_updates = 2
    algorithm.oracle_feature_curriculum_updates = 2
    values = [algorithm.update()["oracle_feature_mix"] for _ in range(6)]
    assert values == [0, 0, 0.5, 1, 1, 1]
    assert algorithm.actor.oracle_feature_mix.item() == 1


@pytest.mark.parametrize(
    "ablation,expected",
    [
        ("normal", [1.0, 2.0, 2.0, 3.0]),
        ("estimated_height_contact", [4.0, 0.5, 0.5, 3.0]),
        ("estimated_reference", [1.0, 2.0, 2.0, 6.0]),
        ("estimated_all", [4.0, 0.5, 0.5, 6.0]),
    ],
)
def test_teacher_feature_diagnostic_replaces_exactly_selected_channels(ablation, expected):
    from types import SimpleNamespace

    obs = TensorDict(
        {"height": torch.ones(2, 1), "contact": torch.full((2, 2), 2.0),
         "reference": torch.full((2, 1), 3.0)}, [2]
    )
    model = SimpleNamespace(
        estimator_target_group="height", foot_contact_target_group="contact",
        reference_encoder_target_group="reference", policy_normalizer=nn.Identity(),
        estimate_height_and_contact=lambda _: (torch.full((2, 1), 4.0), torch.zeros(2, 2)),
        encode_reference=lambda _: torch.full((2, 1), 6.0),
        _spv5_2_features=lambda inputs, h, c, r: torch.cat((h, c, r), -1),
        mlp=nn.Linear(4, 2), distribution=GaussianDistribution(2, init_std=0.3),
    )
    actor = SimpleNamespace(
        tracker=model, oracle_tracking_features=True,
        oracle_feature_ablation=ablation, train_base_policy=False,
    )
    features, _ = _base_with_optional_update(actor, obs)
    torch.testing.assert_close(features, torch.tensor(expected).repeat(2, 1))


@pytest.mark.parametrize("remove", ["height", "contact", "height_contact"])
def test_height_contact_ablation_removes_both_privileged_actor_routes(remove):
    from types import SimpleNamespace

    obs = TensorDict({"height": torch.ones(2, 1), "contact": torch.ones(2, 2),
                      "reference": torch.full((2, 1), 3.0),
                      PRIVILEGE: torch.full((2, 75), 0.25)}, [2])
    if remove in ("height", "height_contact"):
        obs["height"].fill_(torch.nan)
        obs[PRIVILEGE][:, :1] = torch.nan
    if remove in ("contact", "height_contact"):
        obs["contact"].fill_(torch.nan)
        obs[PRIVILEGE][:, 69:71] = torch.nan
    model = SimpleNamespace(
        estimator_target_group="height", foot_contact_target_group="contact",
        reference_encoder_target_group="reference", policy_normalizer=nn.Identity(),
        estimate_height_and_contact=lambda _: (torch.full((2, 1), 4.0), torch.zeros(2, 2)),
        _spv5_2_features=lambda inputs, h, c, r: torch.cat((h, c, r), -1),
        mlp=nn.Linear(4, 2), distribution=GaussianDistribution(2, init_std=0.3),
    )
    actor = SimpleNamespace(
        tracker=model, oracle_tracking_features=True, oracle_state_ablation=remove,
        train_base_policy=False, privilege_schema="state_physics",
        privilege_normalizer=nn.Identity(), privilege_encoder=nn.Identity(),
    )
    features, _ = _base_with_optional_update(actor, obs)
    latent_input = PrivilegedAdaptationActor.privileged_latent(actor, obs)
    assert torch.isfinite(features).all() and torch.isfinite(latent_input).all()
    torch.testing.assert_close(latent_input[:, 1:69], obs[PRIVILEGE][:, 1:69])
    torch.testing.assert_close(latent_input[:, 71:], obs[PRIVILEGE][:, 71:])
    if remove in ("height", "height_contact"):
        assert features[:, :1].eq(4).all() and latent_input[:, :1].eq(4).all()
        assert obs[PRIVILEGE][:, :1].isnan().all()
    if remove in ("contact", "height_contact"):
        assert features[:, 1:3].eq(0.5).all() and latent_input[:, 69:71].eq(0.5).all()
        assert obs[PRIVILEGE][:, 69:71].isnan().all()


def test_partial_evaluation_reset_does_not_advance_surviving_worlds():
    from types import SimpleNamespace

    import pytest

    from intact_tracking.cli.adaptation_eval import reset_finished_worlds

    command = SimpleNamespace(time_steps=torch.tensor([50, 100, 150]))
    data = SimpleNamespace(
        qpos=torch.ones(3, 5), qvel=torch.ones(3, 4), qacc_warmstart=torch.ones(3, 4)
    )

    def partial_reset(ids):
        command.time_steps[ids] = 0
        for value in vars(data).values():
            value[ids] = 0

    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        sim=SimpleNamespace(data=data),
        scene=SimpleNamespace(write_data_to_sim=lambda: None),
        _reset_idx=partial_reset,
    )
    survivors = torch.tensor([False, True, True])
    reset_finished_worlds(env, torch.tensor([0]), survivors)
    torch.testing.assert_close(command.time_steps, torch.tensor([0, 100, 150]))

    def bad_reset(ids):
        partial_reset(ids)
        command.time_steps += 1

    env._reset_idx = bad_reset
    with pytest.raises(RuntimeError, match="surviving-world cursor"):
        reset_finished_worlds(env, torch.tensor([0]), survivors)

    history = SimpleNamespace(
        _pointer=2,
        _num_pushes=torch.tensor([10, 20, 30]),
        _buffer=torch.ones(5, 3, 4),
    )
    env.observation_manager = SimpleNamespace(
        _group_obs_term_history_buffer={"actor": {"sensor": history}}
    )
    env._reset_idx = partial_reset
    reset_finished_worlds(env, torch.tensor([0]), survivors)

    def bad_history_reset(ids):
        partial_reset(ids)
        history._buffer[:, survivors] = 0

    env._reset_idx = bad_history_reset
    with pytest.raises(RuntimeError, match="surviving-world observation history"):
        reset_finished_worlds(env, torch.tensor([0]), survivors)


def test_metric_rewards_use_joint_l2_and_current_body_alignment():
    from types import SimpleNamespace

    from intact_tracking.adaptation_rewards import balanced_body_tracking, joint_l2_tracking

    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = SimpleNamespace(
        joint_pos=torch.tensor([[0.3, 0.4]]),
        robot_joint_pos=torch.zeros(1, 2),
        anchor_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
        robot_anchor_pos_w=torch.tensor([[10.0, 20.0, 1.0]]),
        anchor_quat_w=identity,
        robot_anchor_quat_w=identity,
        body_pos_w=torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
        robot_body_pos_w=torch.tensor([[[10.0, 20.0, 1.0], [11.0, 20.0, 1.0]]]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    for key, value in vars(command).items():
        setattr(command, key, value.repeat(2, *([1] * (value.ndim - 1))))
    torch.testing.assert_close(joint_l2_tracking(env, scale=0.5), torch.ones(2).exp().reciprocal())
    torch.testing.assert_close(balanced_body_tracking(env), torch.ones(2))
    command.robot_body_pos_w[:, 1, 2] += 0.06
    torch.testing.assert_close(balanced_body_tracking(env), torch.ones(2).exp().reciprocal())
    torch.testing.assert_close(joint_l2_tracking(env, scale=0.5, shape="linear"), torch.zeros(2))
    torch.testing.assert_close(
        balanced_body_tracking(env, shape="linear"), torch.zeros(2), atol=2e-6, rtol=0
    )


def test_linear_metric_reward_preserves_error_slope_and_caps_outliers():
    from intact_tracking.adaptation_rewards import error_reward

    errors = torch.tensor([0.0, 0.2, 0.4, 4.0])
    torch.testing.assert_close(
        error_reward(errors, 0.4, "linear"), torch.tensor([1.0, 0.5, 0.0, -4.0])
    )


def test_auxiliary_reward_covers_all_eight_errors_and_qpos_velocity_convention():
    from types import SimpleNamespace

    from intact_tracking.adaptation_rewards import auxiliary_tracking

    zero = torch.zeros(2, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    body_zero = torch.zeros(2, 2, 3)
    body_quat = quat[:, None].repeat(1, 2, 1)
    command = SimpleNamespace(
        cfg=SimpleNamespace(history_steps=0),
        anchor_pos_w=zero,
        robot_anchor_pos_w=zero,
        anchor_quat_w=quat,
        robot_anchor_quat_w=quat,
        anchor_lin_vel_w=zero,
        robot_anchor_lin_vel_w=zero,
        anchor_ang_vel_w=zero,
        robot_anchor_ang_vel_w=zero,
        body_quat_w=body_quat,
        robot_body_quat_w=body_quat,
        joint_vel=torch.zeros(2, 29),
        robot_joint_vel=torch.zeros(2, 29),
        body_lin_vel_w=body_zero,
        robot_body_lin_vel_w=body_zero,
        body_ang_vel_w=body_zero,
        robot_body_ang_vel_w=body_zero,
        _qpos_actor_root_velocity_state=lambda _: SimpleNamespace(
            root_quat_w=quat[:, None], root_lin_vel_w=zero[:, None], root_ang_vel_w=zero[:, None]
        ),
        gather_reference_body_state_b=lambda _: SimpleNamespace(
            lin_vel_b=body_zero[:, None], ang_vel_b=body_zero[:, None]
        ),
    )
    env = SimpleNamespace(num_envs=2, command_manager=SimpleNamespace(get_term=lambda _: command))
    for qpos_only in (False, True):
        command._uses_qpos_only_actor_fk = lambda enabled=qpos_only: enabled
        torch.testing.assert_close(auxiliary_tracking(env), torch.ones(2))
    command.robot_joint_vel[:, 0] = 4.4
    torch.testing.assert_close(auxiliary_tracking(env), (7 + torch.ones(2).neg().exp()) / 8)
    torch.testing.assert_close(auxiliary_tracking(env, shape="linear"), torch.full((2,), 7 / 8))
    torch.testing.assert_close(
        auxiliary_tracking(env, shape="linear", metric_weights=(1, 1, 1, 4, 1, 1, 1, 1)),
        torch.full((2,), 7 / 11),
    )
    with pytest.raises(ValueError, match="Unknown auxiliary"):
        auxiliary_tracking(env, shape="invalid")
    torch.testing.assert_close(
        auxiliary_tracking(env, metric_weights=(1, 1, 1, 4, 1, 1, 1, 1)),
        (7 + 4 * torch.ones(2).neg().exp()) / 11,
    )


def test_physics_probe_keeps_temporal_rows_aligned_with_world_labels():
    from intact_tracking.cli.context_probe import ridge_probe

    generator = torch.Generator().manual_seed(5167)
    targets = torch.randn(64, 3, generator=generator)
    features = targets[None].repeat(4, 1, 1)
    result = ridge_probe(features, targets, torch.arange(48), torch.arange(48, 64), ridge=1e-6)
    assert min(result["heldout_r2"]) > 0.999


def test_context_supervision_uses_body_velocity_not_world_heading():
    from intact_tracking.cli.adaptation_distill import context_auxiliary_targets

    root_velocity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    half = 0.5**0.5
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [half, 0.0, 0.0, half]])
    targets = context_auxiliary_targets(root_velocity, root_quat, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(targets[:, :3], torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1))
    torch.testing.assert_close(targets[:, 3], torch.tensor([-1.0, 1.0]))


def test_reward_target_advance_reads_future_without_mutating_command():
    from types import SimpleNamespace

    from intact_tracking.adaptation_rewards import joint_l2_tracking, reference_value

    command = SimpleNamespace(
        time_steps=torch.tensor([20]),
        joint_pos=torch.zeros(1, 2),
        robot_joint_pos=torch.ones(1, 2),
        gather_reference=lambda field, steps: torch.ones(1, 1, 2) * steps[0],
        gather_root_reference=lambda field, steps: torch.ones(1, 1, 3) * steps[0],
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene=SimpleNamespace(env_origins=torch.tensor([[10.0, 20.0, 0.0]])),
    )
    assert joint_l2_tracking(env).item() < 1.0
    torch.testing.assert_close(joint_l2_tracking(env, reference_offset=1), torch.ones(1))
    torch.testing.assert_close(
        reference_value(env, command, "anchor_pos_w", 1), torch.tensor([[11.0, 21.0, 1.0]])
    )
    torch.testing.assert_close(command.time_steps, torch.tensor([20]))


def test_payload_arm_reward_selects_only_right_arm_without_changing_global_metrics():
    from types import SimpleNamespace

    from intact_tracking.adaptation_rewards import right_arm_angular_tracking

    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["left_wrist_yaw_link", "right_wrist_yaw_link", "pelvis"]),
        _uses_qpos_only_actor_fk=lambda: False,
        body_ang_vel_w=torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        robot_body_ang_vel_w=torch.zeros(1, 3, 3),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    torch.testing.assert_close(right_arm_angular_tracking(env), torch.ones(1))
    command.body_ang_vel_w[:, 1, 0] = 1.0
    torch.testing.assert_close(right_arm_angular_tracking(env), torch.ones(1).neg().exp())
    torch.testing.assert_close(env._adaptation_right_arm_indices, torch.tensor([1]))


def test_right_arm_rotation_reward_ignores_left_arm_and_aligns_global_yaw():
    from types import SimpleNamespace

    from mjlab.utils.lab_api.math import quat_mul

    from intact_tracking.adaptation_rewards import right_arm_rotation_tracking

    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    yaw = torch.tensor([[torch.cos(torch.tensor(0.4)), 0.0, 0.0, torch.sin(torch.tensor(0.4))]])
    roll = torch.tensor([[torch.cos(torch.tensor(0.15)), torch.sin(torch.tensor(0.15)), 0.0, 0.0]])
    actual = yaw[:, None].expand(1, 3, 4).clone()
    actual[:, 0] = quat_mul(yaw, roll)  # left-arm error must be ignored
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["left_wrist_yaw_link", "right_wrist_yaw_link", "pelvis"]),
        anchor_quat_w=identity, robot_anchor_quat_w=yaw,
        body_quat_w=identity[:, None].expand(1, 3, 4), robot_body_quat_w=actual,
        robot_body_ang_vel_w=torch.zeros(1, 3, 3),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    torch.testing.assert_close(right_arm_rotation_tracking(env), torch.ones(1), atol=2e-5, rtol=0)
    actual[:, 1] = quat_mul(yaw, roll)
    torch.testing.assert_close(right_arm_rotation_tracking(env), -torch.ones(1), atol=2e-5, rtol=0)


class FakeTracker(nn.Module):
    def __init__(self, obs, groups, obs_set, output_dim, **kwargs):
        super().__init__()
        self.policy_input_dim = 3
        self.mlp = nn.Linear(3, output_dim)
        self.distribution = GaussianDistribution(output_dim, init_std=0.3)
        self.history_normalizer = nn.Identity()

    def get_latent(self, obs):
        return obs["sensor"]

    def populate_policy_context_cache(self, obs):
        pass


def test_context_mean_actor_preserves_initial_function_without_privileges(tmp_path, monkeypatch):
    from intact_tracking.adaptation_context_memory import CONTEXT_MEAN
    from intact_tracking.context_export import ContextInferenceModule

    monkeypatch.setattr(residual_module, "SPV52HeightContactEstimatorActor", FakeTracker)
    path = tmp_path / "tracker.pt"
    torch.save({"actor_state_dict": FakeTracker(None, None, None, 2).state_dict()}, path)
    obs = TensorDict({
        "sensor": torch.randn(4, 3), "estimator_history": torch.randn(4, 6100),
        CONTEXT_MEAN: torch.randn(4, 4),
    }, [4])
    kwargs = dict(
        tracker_checkpoint=str(path), tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["sensor", "estimator_history"]},
        use_dynamics_latent=False, residual_hidden_dims=(16, 8),
        adaptation_latent_dim=4, train_context_encoder=False,
    )
    original = ContextAdaptationActor(obs, kwargs["tracker_obs_groups"], "actor", 2, **kwargs).eval()
    final = [layer for layer in original.residual_mlp.modules() if isinstance(layer, nn.Linear)][-1]
    nn.init.normal_(final.weight, std=0.1)
    memory = ContextAdaptationActor(
        obs, kwargs["tracker_obs_groups"], "actor", 2, context_latent_mean=True, **kwargs
    ).eval()
    state = dict(original.state_dict())
    key = "residual_mlp.0.weight"
    state[key] = torch.cat((state[key], torch.zeros(16, 4)), -1)
    memory.load_state_dict(state, strict=True)
    torch.testing.assert_close(memory(obs), original(obs))
    assert memory.context_latent(obs).shape == (4, 8)
    with torch.no_grad():
        memory.residual_mlp[0].weight[:, -4:].normal_()
    clean = memory(obs)
    poison = obs.clone()
    poison.set(PRIVILEGE, torch.full((4, 7), torch.nan))
    torch.testing.assert_close(memory(poison), clean)
    no_mean = obs.clone()
    no_mean[CONTEXT_MEAN].zero_()
    assert (memory(no_mean) - clean).abs().max() > 1e-4
    memory.context_ablation = "zero"
    assert memory.context_latent(obs).count_nonzero() == 0
    memory.context_ablation = "shuffle"
    torch.testing.assert_close(memory.context_latent(obs), torch.cat((memory.instant_context_latent(obs), obs[CONTEXT_MEAN]), -1).roll(1, 0))
    memory.context_ablation = "normal"
    memory(obs).square().mean().backward()
    assert all(parameter.grad is None for parameter in memory.context_encoder.parameters())
    assert all(parameter.grad is None for parameter in memory.tracker.parameters())
    assert memory.residual_mlp[0].weight.grad[:, -4:].abs().sum() > 0
    with pytest.raises(ValueError, match="stateful export"):
        ContextInferenceModule(memory)


def test_action_gradients_reach_deployable_reference_encoder_without_truth(tmp_path, monkeypatch):
    class ReferenceTracker(FakeTracker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reference_encoder = nn.Linear(3, 3)
            self.policy_normalizer = nn.Identity()

        def estimate_height_and_contact(self, obs):
            return obs["sensor"][:, :1] * 0, obs["sensor"][:, :2] * 0

        def encode_reference(self, obs):
            return self.reference_encoder(obs["sensor"])

        def _spv5_2_features(self, obs, height, contacts, decoded):
            return decoded

        def get_latent(self, obs):
            raise AssertionError("Action-trained reference path must bypass detached caches")

    monkeypatch.setattr(residual_module, "SPV52HeightContactEstimatorActor", ReferenceTracker)
    path = tmp_path / "tracker.pt"
    tracker = ReferenceTracker(None, None, None, 2)
    torch.save({"actor_state_dict": tracker.state_dict()}, path)
    obs = TensorDict(
        {"sensor": torch.randn(4, 3), "estimator_history": torch.randn(4, 6100)}, [4]
    )
    actor = ContextAdaptationActor(
        obs, {"actor": ["sensor"]}, "actor", 2,
        tracker_checkpoint=str(path), tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["sensor"]}, use_dynamics_latent=False,
        residual_hidden_dims=(16, 8), adaptation_latent_dim=4,
        train_reference_encoder=True,
    )
    output = actor(obs)
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in actor.tracker.reference_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in actor.tracker.mlp.parameters())
    poisoned = obs.clone()
    for key in (
        PRIVILEGE, "reference_encoder_target", "estimator_target", "foot_contact_target",
        "teacher_features",
    ):
        poisoned.set(key, torch.full((4, 7), float("nan")))
    torch.testing.assert_close(output, actor(poisoned))


def test_fixed_evaluation_covers_all_motions_and_is_independent_of_global_rng():
    lengths = torch.tensor([25, 500, 1100, 2500])
    ids, starts = fixed_starts(lengths, repeats=8, horizon=500, seed=10001)
    torch.manual_seed(73)
    torch.randn(1249)
    repeated_ids, repeated_starts = fixed_starts(lengths, 8, 500, 10001)
    torch.testing.assert_close(ids, repeated_ids)
    torch.testing.assert_close(starts, repeated_starts)
    assert torch.equal(torch.bincount(ids), torch.full((4,), 8))
    assert (starts >= 0).all() and (starts < lengths[ids]).all()
    assert ((lengths[ids] < 521) | (lengths[ids] - starts >= 521)).all()
    assert not torch.equal(starts, fixed_starts(lengths, 8, 500, 10002)[1])


@pytest.mark.parametrize("imu_accel", [False, True])
def test_context_history_preserves_time_and_sensor_order_without_cross_world_mixing(imu_accel):
    encoder = DeployableHistoryEncoder(latent_dim=4, imu_accel=imu_accel)
    width = encoder.frame_dim
    frames = torch.arange(2 * 50 * width, dtype=torch.float32).reshape(2, 50, width)
    terms = frames.split(encoder.term_dims, dim=-1)
    flat = torch.cat([term.flatten(1) for term in terms], dim=-1)
    torch.testing.assert_close(encoder.frames(flat), frames)
    with torch.no_grad():
        batched = encoder(flat / 10000)
        separate = torch.cat([encoder(row[None] / 10000) for row in flat], dim=0)
    torch.testing.assert_close(batched, separate)
    assert batched.shape == (2, 4)


@pytest.mark.parametrize("history_steps", [49, 50, 51, 52])
def test_right_aligned_context_uses_latest_frame_and_preserves_weight_shapes(history_steps):
    torch.manual_seed(42)
    legacy = DeployableHistoryEncoder(latent_dim=4, history_steps=history_steps)
    aligned = DeployableHistoryEncoder(
        latent_dim=4, history_steps=history_steps, right_aligned=True
    )
    aligned.load_state_dict(legacy.state_dict(), strict=True)
    frames = torch.randn(2, history_steps, 122, requires_grad=True)
    flat = torch.cat([term.flatten(1) for term in frames.split(legacy.TERM_DIMS, -1)], -1)
    aligned(flat).sum().backward()
    assert frames.grad[:, -1].abs().sum() > 0
    if aligned.history_left_trim:
        assert frames.grad[:, : aligned.history_left_trim].abs().sum() == 0
    frames.grad.zero_()
    legacy(flat).sum().backward()
    if (history_steps - 21) % 4:
        assert frames.grad[:, -1].abs().sum() == 0


@pytest.mark.parametrize("learned_gain", [False, True])
@pytest.mark.parametrize("key_body", [False, True])
@pytest.mark.parametrize("right_aligned", [False, True])
@pytest.mark.parametrize("feature_adapter", [False, True])
@pytest.mark.parametrize("imu_accel", [False, True])
@pytest.mark.parametrize("root_orientation", [False, True])
def test_student_action_path_works_without_privileged_keys(
    tmp_path, monkeypatch, feature_adapter, imu_accel, root_orientation, right_aligned, key_body, learned_gain
):
    monkeypatch.setattr(residual_module, "SPV52HeightContactEstimatorActor", FakeTracker)
    path = tmp_path / "tracker.pt"
    tracker = FakeTracker(None, None, None, 2)
    torch.save({"actor_state_dict": tracker.state_dict()}, path)
    obs = TensorDict({"sensor": torch.randn(4, 3), "estimator_history": torch.randn(4, 6100)}, [4])
    if imu_accel:
        obs.set(IMU_HISTORY, torch.randn(4, 150))
    if root_orientation:
        obs.set("robot_root_quat", torch.randn(4, 4))
    if key_body:
        obs.set("robot_key_body", torch.randn(4, 195))
    actor = ContextAdaptationActor(
        obs,
        {"actor": ["sensor"]},
        "actor",
        2,
        tracker_checkpoint=str(path),
        tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["sensor"]},
        use_dynamics_latent=False,
        residual_hidden_dims=(16, 8),
        adaptation_latent_dim=4,
        context_auxiliary_dim=10 if key_body else 4,
        context_feature_adapter=feature_adapter,
        context_imu_accel=imu_accel,
        context_root_orientation=root_orientation,
        context_right_aligned=right_aligned,
        context_key_body=key_body,
        context_learned_residual_gain=learned_gain,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(actor(obs), actor.tracker.mlp(obs["sensor"]))
    final = [m for m in actor.residual_mlp.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.normal_(final.weight, std=0.1)
    with torch.no_grad():
        clean = actor(obs)
        poisoned = obs.clone()
        for key in (
            PRIVILEGE,
            ORACLE_HISTORY,
            ORACLE_KEY_BODY,
            "priv",
            "estimator_target",
            "foot_contact_target",
            "reference_encoder_target",
            "teacher_features",
            "teacher_action",
            "adaptation_teacher_action_target",
            "adaptation_context_auxiliary_target",
        ):
            poisoned.set(key, torch.full((4, 7), float("nan")))
        torch.testing.assert_close(clean, actor(poisoned))
        assert torch.isfinite(clean).all()
    loss = actor.context_auxiliary_head(actor.context_latent(obs)).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in actor.context_encoder.parameters())
    if feature_adapter:
        actor.zero_grad(set_to_none=True)
        actor(obs).square().mean().backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in actor.feature_adapter.parameters()
        )
        assert all(parameter.grad is None for parameter in actor.tracker.mlp.parameters())


def test_accelerometer_reads_existing_raw_sensor_not_root_state():
    from types import SimpleNamespace

    values = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    model = SimpleNamespace(
        nsensor=2,
        sensor=lambda i: SimpleNamespace(name=("robot/gyro", "robot/imu_lin_acc")[i]),
        sensor_dim=(3, 3),
        sensor_adr=(0, 4),
    )
    env = SimpleNamespace(sim=SimpleNamespace(mj_model=model, data=SimpleNamespace(sensordata=values)))
    torch.testing.assert_close(measured_imu_acceleration(env), values[:, 4:7])
    assert env._adaptation_imu_address == 4


def test_teacher_action_targets_do_not_replace_student_cache():
    from types import SimpleNamespace

    from intact_tracking.adaptation_ppo import teacher_action_targets

    class Teacher:
        def populate_tracker_cache(self, obs):
            obs.set("cache", obs["truth"] * 2)

        def __call__(self, obs):
            return obs["cache"] + 1

    obs = TensorDict(
        {"truth": torch.ones(3, 2, requires_grad=True), "sensor": torch.full((3, 2), 4.0)},
        [3],
    )
    student = SimpleNamespace(
        populate_tracker_cache=lambda inputs: inputs.set("cache", inputs["sensor"] * 3)
    )
    target = teacher_action_targets(Teacher(), student, obs)
    torch.testing.assert_close(target, torch.full((3, 2), 3.0))
    torch.testing.assert_close(obs["cache"], torch.full((3, 2), 12.0))
    assert not target.requires_grad


def test_critic_warmup_preserves_actor_then_restores_trainability(monkeypatch):
    from intact_tracking.adaptation_ppo import CriticWarmupPPO

    actor, critic = nn.Linear(3, 2), nn.Linear(3, 1)
    actor.bias.requires_grad_(False)
    algorithm = object.__new__(CriticWarmupPPO)
    algorithm.actor, algorithm.critic = actor, critic
    algorithm.optimizer = torch.optim.Adam(
        [{"params": actor.parameters()}, {"params": critic.parameters()}], lr=0.01
    )
    algorithm.critic_warmup_updates = 1
    algorithm.adaptation_update_count = 0
    algorithm.schedule = "adaptive"

    def parent_update(self):
        value = torch.ones(4, 3)
        self.optimizer.zero_grad(set_to_none=True)
        (self.actor(value).square().mean() + self.critic(value).square().mean()).backward()
        self.optimizer.step()
        return {}

    monkeypatch.setattr(residual_module.ResidualPPO, "update", parent_update)
    old_actor, old_critic = actor.weight.detach().clone(), critic.weight.detach().clone()
    assert algorithm.update()["critic_warmup_active"] == 1
    torch.testing.assert_close(actor.weight, old_actor, rtol=0, atol=0)
    assert not torch.equal(critic.weight, old_critic)
    assert actor.weight.requires_grad and not actor.bias.requires_grad
    assert algorithm.schedule == "adaptive"
    assert algorithm.update()["critic_warmup_active"] == 0
    assert not torch.equal(actor.weight, old_actor)


def test_nominal_initialization_copies_trunk_and_pads_only_context_columns():
    from types import SimpleNamespace

    from intact_tracking.cli.adaptation_distill import initialize_student_from_nominal

    trunk = nn.Linear(3, 2)
    trunk.policy_input_dim = 3
    nominal = nn.Sequential(nn.Linear(3, 5), nn.ELU(), nn.Linear(5, 2))
    residual = nn.Sequential(nn.Linear(7, 5), nn.ELU(), nn.Linear(5, 2))
    state = {f"tracker.{key}": value.clone() for key, value in trunk.state_dict().items()}
    state.update({f"residual_mlp.{key}": value.clone() for key, value in nominal.state_dict().items()})
    student = SimpleNamespace(
        tracker=trunk, residual_mlp=residual, residual_scale=1.0,
        context_residual_log_gain=nn.Parameter(torch.zeros(2)),
    )
    initialize_student_from_nominal(student, state, 0.25)
    torch.testing.assert_close(residual[0].weight[:, :3], nominal[0].weight)
    assert residual[0].weight[:, 3:].count_nonzero() == 0
    torch.testing.assert_close(residual[2].weight, nominal[2].weight)
    torch.testing.assert_close(residual[2].bias, nominal[2].bias)
    inputs = torch.randn(8, 3) * 100
    context = torch.randn(8, 4)
    torch.testing.assert_close(
        residual(torch.cat((inputs, context), -1)).tanh() * student.context_residual_log_gain.exp(),
        nominal(inputs).tanh() * 0.25,
    )
    assert list(state["residual_mlp.0.weight"].shape) == [5, 3]


def test_physics_supervision_normalizes_labels_without_confusing_payload_with_torso():
    from intact_tracking.cli.adaptation_distill import normalized_context_physics_targets

    values = torch.tensor([[2, 0, 0, 0, 0, 1.15, 1.0], [3, 1, 0.075, -0.075, 0, 2, 1.2]])
    target = normalized_context_physics_targets(values)
    torch.testing.assert_close(target[0], torch.zeros(6))
    torch.testing.assert_close(target[1], torch.tensor([1, 1, -1, 0, 1, 1], dtype=torch.float32))


def test_training_start_curriculum_preserves_sensors_and_physical_events():
    from types import SimpleNamespace

    from intact_tracking.adaptation_curriculum import configure_training_starts

    command = SimpleNamespace(pose_range={"x": (-0.05, 0.05)}, velocity_range={"x": (-0.5, 0.5)}, joint_position_range=(-0.1, 0.1))
    observations, events = {"sensor": object()}, {"physical_dr": object()}
    cfg = SimpleNamespace(commands={"motion": command}, observations=observations, events=events)
    original = configure_training_starts(cfg, "original")
    assert original["before"] == original["after"]
    changed = configure_training_starts(cfg, "reference")
    assert command.pose_range == command.velocity_range == command.init_noise == {}
    assert command.joint_position_range == (0.0, 0.0)
    assert cfg.observations is observations and cfg.events is events
    assert changed["before"]["velocity_range"] == {"x": (-0.5, 0.5)}


def test_pd_bias_proxy_reads_only_observations_and_masks_clipped_torque():
    from intact_tracking.adaptation_identification import sensor_pd_bias_proxy

    values = [torch.zeros(2, 50, dim) for dim in (29, 29, 3, 3, 29, 29)]
    values[0].fill_(0.02)
    values[4].fill_(0.02)
    values[5].fill_(-0.5)
    values[5][:, 0] = 99
    history = torch.cat([value.flatten(1) for value in values], -1)
    contract = {"kp": torch.full((29,), 100.0), "kd": torch.full((29,), 2.0), "action_scale": torch.full((29,), 0.25), "action_to_joint": torch.arange(29), "force_limit": torch.full((29,), 100.0)}
    estimate, raw, count = sensor_pd_bias_proxy(history, contract)
    with pytest.raises(ValueError, match="Encoder bias cancels"):
        sensor_pd_bias_proxy(history, dict(contract, encoder_bias_subtracted_from_target=True))
    torch.testing.assert_close(estimate, torch.full((2, 29), 0.01))
    torch.testing.assert_close(raw, estimate)
    assert (count == 49).all()
    # With the real controller's target-bias subtraction the same algebra
    # returns zero even though the simulated measurement bias is .01 rad.
    values[0].fill_(0.03)  # true q=.02 plus bias=.01
    values[5].fill_(-2.5)  # kp*(.25*.02-.01-.02), velocity=0
    compensated_history = torch.cat([value.flatten(1) for value in values], -1)
    canceled, _, _ = sensor_pd_bias_proxy(compensated_history, contract)
    torch.testing.assert_close(canceled, torch.zeros_like(canceled), atol=1e-8, rtol=0)


def test_alternating_teacher_step_updates_only_student_and_detaches_labels(monkeypatch):
    from types import SimpleNamespace

    from intact_tracking.adaptation_ppo import TEACHER_ACTION, TeacherRegularizedPPO

    class Actor(nn.Linear):
        def forward(self, obs):
            return super().forward(obs["sensor"])

    actor, critic = Actor(3, 2), nn.Linear(3, 1)
    labels = torch.randn(2, 4, 2, requires_grad=True)
    observations = TensorDict(
        {"sensor": torch.randn(2, 4, 3), TEACHER_ACTION: labels}, [2, 4]
    )
    algorithm = object.__new__(TeacherRegularizedPPO)
    algorithm.actor, algorithm.critic = actor, critic
    algorithm.storage = SimpleNamespace(observations=observations)
    algorithm.optimizer = torch.optim.Adam(
        [{"params": actor.parameters()}, {"params": critic.parameters()}], lr=0.001
    )
    algorithm.device = "cpu"
    algorithm.teacher_bc_steps = 2
    algorithm.teacher_bc_weight = 0.1
    algorithm.teacher_bc_batch_size = 8
    algorithm.max_grad_norm = 1.0
    algorithm.is_multi_gpu = False
    monkeypatch.setattr(residual_module.ResidualPPO, "update", lambda self: {"ppo": 1.0})
    old_actor, old_critic = actor.weight.detach().clone(), critic.weight.detach().clone()
    result = algorithm.update()
    assert result["teacher_action_mse"] > 0
    assert not torch.equal(actor.weight, old_actor)
    torch.testing.assert_close(critic.weight, old_critic, rtol=0, atol=0)
    assert labels.grad is None


def test_auxiliary_ppo_isolates_labels_preserves_controller_and_honors_warmup(monkeypatch):
    from types import SimpleNamespace

    from intact_tracking.adaptation_ppo import CONTEXT_TARGET, ContextAuxiliaryPPO, CriticWarmupPPO

    class Actor(nn.Module):
        deployable_observation_groups = ("sensor",)

        def __init__(self):
            super().__init__()
            self.context_encoder = nn.Linear(3, 4)
            self.context_auxiliary_head = nn.Linear(4, 10)
            self.residual_mlp = nn.Linear(4, 2)

        def instant_context_latent(self, obs):
            assert set(obs.keys()) == {"sensor"}
            return self.context_encoder(obs["sensor"])

    actor, critic = Actor(), nn.Linear(3, 1)
    labels = torch.randn(2, 4, 10, requires_grad=True)
    observations = TensorDict({"sensor": torch.randn(2, 4, 3), CONTEXT_TARGET: labels}, [2, 4])
    algorithm = object.__new__(ContextAuxiliaryPPO)
    algorithm.actor, algorithm.critic = actor, critic
    algorithm.storage = SimpleNamespace(observations=observations)
    algorithm.device, algorithm.is_multi_gpu = "cpu", False
    algorithm.context_aux_steps, algorithm.context_aux_batch_size = 2, 8
    algorithm.context_aux_weight, algorithm.context_physics_weight = 0.05, 0.01
    algorithm.max_grad_norm = 1.0
    algorithm.optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.001)
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    critic_before = critic.weight.detach().clone()
    monkeypatch.setattr(CriticWarmupPPO, "update", lambda self: {"critic_warmup_active": 1.0})
    assert algorithm.update()["critic_warmup_active"] == 1.0
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in before.items())
    monkeypatch.setattr(CriticWarmupPPO, "update", lambda self: {"critic_warmup_active": 0.0})
    result = algorithm.update()
    assert result["context_supervision_dynamic_mse"] > 0
    assert result["context_supervision_physics_mse"] > 0
    assert not torch.equal(actor.context_encoder.weight, before["context_encoder.weight"])
    assert not torch.equal(actor.context_auxiliary_head.weight, before["context_auxiliary_head.weight"])
    torch.testing.assert_close(actor.residual_mlp.weight, before["residual_mlp.weight"], atol=0, rtol=0)
    torch.testing.assert_close(critic.weight, critic_before, atol=0, rtol=0)
    assert labels.grad is None


def test_privileged_policy_has_trainable_bottleneck_and_frozen_deployable_base(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(residual_module, "SPV52HeightContactEstimatorActor", FakeTracker)
    tracker = FakeTracker(None, None, None, 2)
    path = tmp_path / "tracker.pt"
    torch.save({"actor_state_dict": tracker.state_dict()}, path)
    obs = TensorDict({"sensor": torch.randn(8, 3), PRIVILEGE: torch.randn(8, 7)}, [8])
    actor = PrivilegedAdaptationActor(
        obs,
        {"actor": ["sensor"]},
        "actor",
        2,
        tracker_checkpoint=str(path),
        tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["sensor"]},
        use_dynamics_latent=False,
        residual_hidden_dims=(16, 8),
        privilege_hidden_dims=(12, 8),
        adaptation_latent_dim=4,
        residual_scale=1.0,
    )
    expected = tracker.mlp(obs["sensor"])
    torch.testing.assert_close(actor(obs), expected)
    assert not any(p.requires_grad for p in actor.tracker.parameters())
    # Once the zero-initialized action layer learns, RL gradients must reach the
    # privileged encoder. Merely adding observations to storage is insufficient.
    final = [m for m in actor.residual_mlp.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.normal_(final.weight, std=0.1)
    actor(obs).square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.privilege_encoder.parameters()) > 0
    assert all(p.grad is None for p in actor.tracker.parameters())
    assert actor.policy_metrics(obs)["privilege_shuffle_action_delta_rms"] > 0
