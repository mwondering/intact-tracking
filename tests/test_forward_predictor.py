from __future__ import annotations

from types import SimpleNamespace

import mujoco
import pytest
import torch

from intact_tracking.cli.forward_predictor_train import (
    _CORE_PROBE_METRICS,
    _collect_counterfactual_block,
    _slice_predictor_batch,
    _validate_arguments,
    build_parser,
)
from intact_tracking.data import (
    ForwardPredictorReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.environment.assets.robots.g1_tracking_bfm import (
    get_g1_tracking_bfm_mesh_arm_spec,
)
from intact_tracking.forward_predictor import (
    ForwardDynamicsTransformer,
    ForwardPredictorConfig,
    apply_physical_state_delta,
    physical_state_delta,
)
from intact_tracking.forward_predictor_inputs import (
    G1_XML_JOINT_NAMES,
    G1FootKinematics,
    JointPositionTargetTransform,
    g1_foot_features_from_link_state,
)
from intact_tracking.forward_predictor_objective import (
    ForwardPredictorLossConfig,
    ForwardPredictorObjective,
    _counterfactual_representation_loss,
)
from intact_tracking.rollout.online import _verify_predictor_action_target


def _identity_state(*prefix: int) -> torch.Tensor:
    state = torch.zeros((*prefix, 71))
    state[..., 3] = 1.0
    return state


def _small_config() -> ForwardPredictorConfig:
    return ForwardPredictorConfig(
        context_history_steps=10,
        transformer_dim=32,
        transformer_depth=2,
        transformer_heads=4,
    )


def _normalization(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.zeros(71, device=device),
        "state_std": torch.ones(71, device=device),
        "foot_mean": torch.zeros(8, device=device),
        "foot_std": torch.ones(8, device=device),
        "contact_force_mean": torch.zeros(6, device=device),
        "contact_force_std": torch.ones(6, device=device),
        "delta_mean": torch.zeros(70, device=device),
        "delta_std": torch.ones(70, device=device),
    }


def _history(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "history_state": torch.zeros(batch_size, 10, 71),
        "history_action": torch.zeros(batch_size, 10, 29),
        "history_foot": torch.zeros(batch_size, 10, 8),
        "history_contact_force": torch.zeros(batch_size, 10, 6),
        "history_contact_binary": torch.zeros(batch_size, 10, 2, dtype=torch.bool),
        "history_valid": torch.zeros(batch_size, 10, dtype=torch.bool),
    }


def _current_privileged(
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch_size, 8),
        torch.zeros(batch_size, 6),
        torch.zeros(batch_size, 2, dtype=torch.bool),
    )


def _privileged_trajectory(
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch_size, 6, 8),
        torch.zeros(batch_size, 6, 6),
        torch.zeros(batch_size, 6, 2, dtype=torch.bool),
    )


def _objective_context_fields(
    state: torch.Tensor,
    history: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size = state.size(0)
    return {
        "nominal_state": state[:, 1:].clone(),
        "positive_current_state": state[:, 0].clone(),
        "positive_history_state": history["history_state"].clone(),
        "positive_history_action": history["history_action"].clone(),
        "positive_history_valid": history["history_valid"].clone(),
        "positive_pair_valid": torch.zeros(batch_size, dtype=torch.bool),
        "is_nominal": torch.arange(batch_size).remainder(2) == 0,
        "world_id": torch.arange(batch_size),
    }


def _rollout_normalization() -> dict[str, torch.Tensor]:
    normalization = _normalization()
    return {
        name: normalization[name]
        for name in (
            "state_mean",
            "state_std",
            "delta_mean",
            "delta_std",
        )
    }


def _rollout_target(initial: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    states = []
    current = initial
    for _ in range(5):
        current = apply_physical_state_delta(current, delta)
        states.append(current)
    return torch.stack(states, dim=1)


def test_default_forward_predictor_has_a_separate_dynamics_context_encoder() -> None:
    model = ForwardDynamicsTransformer()

    assert sum(parameter.numel() for parameter in model.parameters()) == 19_500_310
    assert (
        model.config.architecture_version
        == "nominal_counterfactual_dynamics_context_transformer_v12"
    )
    assert model.config.history_steps == 10
    assert model.config.context_history_steps == 100
    assert model.config.sequence_length == 11
    assert model.config.transformer_dim == 512
    assert model.config.transformer_depth == 6
    assert model.config.transformer_heads == 8
    assert model.config.context_dim == 128
    assert model.config.context_depth == 2
    assert model.config.dynamics_latent_dim == 64
    assert not hasattr(model.config, "privileged_dim")
    assert hasattr(model, "context_encoder")
    assert model.context_encoder.interaction_projection[0].in_features == 2 * 71 + 29
    assert not hasattr(model, "theta_encoder")
    assert not hasattr(model, "privileged_head")
    assert not hasattr(model, "foot_kinematics")
    assert model.causal_mask.shape == (11, 11)
    assert torch.equal(model.causal_mask, torch.triu(torch.ones(11, 11, dtype=torch.bool), 1))


def test_masked_history_values_do_not_affect_transformer_prediction() -> None:
    torch.manual_seed(5)
    model = ForwardDynamicsTransformer(_small_config()).eval()
    state = torch.randn(2, 71)
    action = torch.randn(2, 29)
    history = _history(2)
    changed = {name: value.clone() for name, value in history.items()}
    changed["history_state"].normal_()
    changed["history_action"].normal_()
    changed["history_foot"].normal_()
    changed["history_contact_force"].normal_()
    changed["history_contact_binary"].logical_not_()
    foot, contact_force, contact_binary = _current_privileged(2)

    expected = model(
        state,
        action,
        foot,
        contact_force,
        contact_binary,
        **history,
    )
    actual = model(
        state,
        action,
        foot,
        contact_force,
        contact_binary,
        **changed,
    )

    assert all(
        torch.equal(actual_item, expected_item)
        for actual_item, expected_item in zip(actual, expected, strict=True)
    )


def test_dynamics_latent_uses_only_completed_proprioceptive_interactions() -> None:
    torch.manual_seed(6)
    model = ForwardDynamicsTransformer(_small_config()).eval()
    state = torch.randn(2, 71)
    foot, contact_force, contact_binary = _current_privileged(2)
    history = _history(2)
    history["history_valid"].fill_(True)
    history["history_state"].normal_()
    history["history_action"].normal_()
    history["history_foot"].normal_()
    history["history_contact_force"].normal_()

    first = model(
        state,
        torch.zeros(2, 29),
        foot,
        contact_force,
        contact_binary,
        **history,
        return_context=True,
    )
    second = model(
        state,
        torch.ones(2, 29),
        foot,
        contact_force,
        contact_binary,
        **history,
        return_context=True,
    )

    assert len(first) == 5
    torch.testing.assert_close(first[4], second[4])
    assert not torch.equal(first[0], second[0])

    changed_history = {name: value.clone() for name, value in history.items()}
    changed_history["history_foot"].normal_()
    changed_history["history_contact_force"].normal_()
    changed_history["history_contact_binary"].logical_not_()
    changed_foot = torch.randn_like(foot)
    changed_force = torch.randn_like(contact_force)
    changed_binary = ~contact_binary
    privileged_changed = model(
        state,
        torch.zeros(2, 29),
        changed_foot,
        changed_force,
        changed_binary,
        **changed_history,
        return_context=True,
    )

    torch.testing.assert_close(first[4], privileged_changed[4])
    assert not torch.equal(first[0], privileged_changed[0])


def test_representation_matches_continuous_counterfactual_response_geometry() -> None:
    latent = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]],
        requires_grad=True,
    )
    positive_latent = latent.detach().clone()
    response = torch.zeros(4, 5, 70)
    response[2, :, 0] = 1.0
    response[3, :, 0] = 4.0

    loss, metrics, partner = _counterfactual_representation_loss(
        latent,
        positive_latent,
        response,
        torch.ones(4, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool),
        torch.arange(4),
        response_distance_scale=1.0,
    )
    loss.backward()

    assert torch.equal(partner, torch.tensor([2, 3, 0, 1]))
    assert torch.isfinite(loss)
    assert latent.grad is not None
    assert latent.grad.abs().sum() > 0.0
    torch.testing.assert_close(metrics["latent_positive_cosine"], torch.tensor(1.0))
    assert metrics["latent_response_correlation"] > 0.9


def test_incomplete_long_context_trains_predictor_but_not_representation_pairs() -> None:
    torch.manual_seed(9)
    config = ForwardPredictorConfig(
        context_history_steps=100,
        transformer_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        context_dim=32,
        context_depth=1,
        context_heads=4,
        dynamics_latent_dim=16,
    )
    model = ForwardDynamicsTransformer(config)
    initial = _identity_state(2)
    delta = torch.randn(2, 70) * 0.01
    foot, contact_force, contact_binary = _privileged_trajectory(2)
    history = _history(2)
    history["history_state"] = torch.randn(2, 100, 71)
    history["history_action"] = torch.randn(2, 100, 29)
    history["history_valid"] = torch.zeros(2, 100, dtype=torch.bool)
    history["history_valid"][:, -8:] = True
    batch = {
        "state": torch.cat((initial[:, None], _rollout_target(initial, delta)), dim=1),
        "action": torch.randn(2, 5, 29),
        "foot": foot,
        "contact_force": contact_force,
        "contact_binary": contact_binary,
        **history,
        **_normalization(),
    }
    batch.update(_objective_context_fields(batch["state"], history))
    objective = ForwardPredictorObjective(model)

    output = objective(batch)
    output["loss"].backward()

    assert output["loss"].item() > 0.0
    assert output["representation_loss"].item() == 0.0
    assert model.delta_head.weight.grad is not None
    assert model.delta_head.weight.grad.abs().sum().item() > 0.0


def test_g1_foot_features_are_differentiable_in_q_and_qdot() -> None:
    state = _identity_state(2)
    state[:, 2] = 0.8
    state.requires_grad_()

    foot = G1FootKinematics()(state)
    foot.sum().backward()

    assert foot.shape == (2, 8)
    assert torch.isfinite(foot).all()
    assert state.grad is not None
    assert state.grad[:, 13:42].abs().sum() > 0.0
    assert state.grad[:, 42:71].abs().sum() > 0.0


def test_foot_features_are_read_from_simulator_link_state_without_fk() -> None:
    position = torch.tensor([[[0.0, 0.1, 0.50], [0.0, -0.1, 0.60]]])
    quaternion = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2])
    linear_velocity = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    angular_velocity = torch.zeros(1, 2, 3)

    foot = g1_foot_features_from_link_state(
        position,
        quaternion,
        linear_velocity,
        angular_velocity,
    )

    expected = torch.tensor([[0.463, 1.0, 2.0, 3.0, 0.563, 4.0, 5.0, 6.0]])
    torch.testing.assert_close(foot, expected)


def test_policy_action_is_converted_to_physical_pd_target_outside_model() -> None:
    target_names = tuple(reversed(G1_XML_JOINT_NAMES))
    target_ids = torch.arange(28, -1, -1)
    scale = torch.linspace(0.1, 0.3, 29)
    offset = torch.linspace(-0.2, 0.2, 29)
    encoder_bias = torch.stack((torch.linspace(0.0, 0.028, 29),) * 2)
    robot = SimpleNamespace(
        joint_names=G1_XML_JOINT_NAMES,
        data=SimpleNamespace(encoder_bias=encoder_bias),
    )
    env = SimpleNamespace(scene={"robot": robot})
    term = SimpleNamespace(
        action_dim=29,
        raw_action=torch.zeros(2, 29),
        _scale=scale,
        _offset=offset,
        target_ids=target_ids,
        target_names=target_names,
        cfg=SimpleNamespace(clip=None, raw_action_clip=None, boot_delay_steps=0),
    )
    policy_action = torch.linspace(-1.0, 1.0, 29).repeat(2, 1).requires_grad_()

    transform = JointPositionTargetTransform.from_mjlab(env, term)
    target = transform(policy_action)
    expected_target_order = policy_action * scale + offset - encoder_bias[:, target_ids]
    expected = expected_target_order.index_select(
        -1,
        torch.tensor([target_names.index(name) for name in G1_XML_JOINT_NAMES]),
    )
    target.sum().backward()

    assert torch.allclose(target, expected)
    assert policy_action.grad is not None
    assert torch.all(policy_action.grad > 0.0)
    assert transform.contract["predictor_action"] == "physical_pd_joint_target_rad"

    term.max_delay = 2
    with pytest.raises(ValueError, match="stateful delay"):
        JointPositionTargetTransform.from_mjlab(env, term)


def test_action_target_verification_ignores_discontinuous_slots() -> None:
    expected = torch.tensor(
        (
            (0.1, -0.2, 0.3),
            (1.69402, -0.7, 0.4),
        )
    )
    simulator = expected.clone()
    simulator[1].zero_()

    maximum_error = _verify_predictor_action_target(
        expected,
        simulator,
        torch.tensor((False, True)),
    )

    assert maximum_error == 0.0
    assert (
        _verify_predictor_action_target(
            expected,
            torch.zeros_like(expected),
            torch.ones(2, dtype=torch.bool),
        )
        is None
    )
    simulator[0, 0] += 0.1
    with pytest.raises(RuntimeError, match="non-boundary environments"):
        _verify_predictor_action_target(
            expected,
            simulator,
            torch.tensor((False, True)),
        )


def test_checkpoint_mesh_arm_asset_variant_compiles_locally() -> None:
    model = get_g1_tracking_bfm_mesh_arm_spec().compile()

    for side in ("left", "right"):
        wrist_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{side}_wrist_yaw_collision",
        )
        assert wrist_id >= 0
        assert model.geom_type[wrist_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        for suffix in ("palm", "knuckle", "finger_root"):
            assert (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{side}_hand_{suffix}_collision",
                )
                == -1
            )


def test_full_state_delta_round_trip_preserves_pose_and_velocity() -> None:
    torch.manual_seed(3)
    current = _identity_state(4)
    delta = torch.randn(4, 70) * 0.05
    following = apply_physical_state_delta(current, delta)
    recovered = physical_state_delta(current, following)
    reconstructed = apply_physical_state_delta(current, recovered)

    assert torch.allclose(recovered, delta, atol=1.0e-5)
    assert torch.allclose(reconstructed, following, atol=1.0e-5)


def test_forward_predictor_recursively_propagates_gradients_through_five_steps() -> None:
    torch.manual_seed(7)
    model = ForwardDynamicsTransformer(_small_config())
    initial = _identity_state(3)
    actions = torch.randn(3, 5, 29)

    foot, contact_force, contact_binary = _current_privileged(3)
    prediction, deltas, predicted_foot, predicted_force, contact_logits = model.rollout(
        initial,
        actions,
        foot,
        contact_force,
        contact_binary,
        **_history(3),
        **_rollout_normalization(),
    )
    loss = (
        prediction.square().mean()
        + deltas.square().mean()
        + predicted_foot.square().mean()
        + predicted_force.square().mean()
        + contact_logits.square().mean()
    )
    loss.backward()

    assert prediction.shape == (3, 5, 71)
    assert deltas.shape == (3, 5, 70)
    assert predicted_foot.shape == (3, 5, 8)
    assert predicted_force.shape == (3, 5, 6)
    assert contact_logits.shape == (3, 5, 2)
    assert torch.isfinite(prediction).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert (
        sum(
            float(parameter.grad.square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        > 0.0
    )


def test_teacher_forced_windows_are_vectorized_and_match_sequential_evaluation() -> None:
    torch.manual_seed(11)
    model = ForwardDynamicsTransformer(_small_config()).eval()
    batch_size = 2
    states = torch.randn(batch_size, 6, 71)
    states[..., 3:7] = torch.nn.functional.normalize(states[..., 3:7], dim=-1)
    actions = torch.randn(batch_size, 5, 29)
    feet = torch.randn(batch_size, 6, 8)
    forces = torch.randn(batch_size, 6, 6)
    binaries = torch.rand(batch_size, 6, 2) > 0.5
    history = {
        "history_state": torch.randn(batch_size, 10, 71),
        "history_action": torch.randn(batch_size, 10, 29),
        "history_foot": torch.randn(batch_size, 10, 8),
        "history_contact_force": torch.randn(batch_size, 10, 6),
        "history_contact_binary": torch.rand(batch_size, 10, 2) > 0.5,
        "history_valid": torch.rand(batch_size, 10) > 0.25,
    }
    normalization = _rollout_normalization()
    calls = 0

    def count_forward(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: tuple[torch.Tensor, ...],
    ) -> None:
        nonlocal calls
        calls += 1

    handle = model.register_forward_hook(count_forward)
    vectorized = model.teacher_forced(
        states,
        actions,
        feet,
        forces,
        binaries,
        **history,
        **normalization,
    )
    handle.remove()

    sequential_outputs: list[list[torch.Tensor]] = [[], [], [], [], []]
    rolling = tuple(history.values())
    dynamics_latent = model.encode_context(
        history["history_state"],
        history["history_action"],
        states[:, 0],
        history["history_valid"],
    )
    for index in range(5):
        outputs = model(
            states[:, index],
            actions[:, index],
            feet[:, index],
            forces[:, index],
            binaries[:, index],
            *rolling,
            dynamics_latent=dynamics_latent,
        )
        next_state = model._apply_normalized_delta(
            states[:, index],
            outputs[0],
            **normalization,
        )
        for destination, value in zip(
            sequential_outputs,
            (next_state, *outputs),
            strict=True,
        ):
            destination.append(value)
        rolling = model._roll_history(
            *rolling,
            states[:, index],
            actions[:, index],
            feet[:, index],
            forces[:, index],
            binaries[:, index],
        )
    sequential = tuple(torch.stack(values, dim=1) for values in sequential_outputs)

    assert calls == 1
    for actual, expected in zip(vectorized, sequential, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_rollout_feeds_each_predicted_state_into_the_next_transition() -> None:
    class DoublingRootPositionPredictor(ForwardDynamicsTransformer):
        def forward(
            self,
            normalized_state: torch.Tensor,
            normalized_action: torch.Tensor,
            normalized_foot: torch.Tensor,
            normalized_contact_force: torch.Tensor,
            contact_binary: torch.Tensor,
            history_state: torch.Tensor,
            history_action: torch.Tensor,
            history_foot: torch.Tensor,
            history_contact_force: torch.Tensor,
            history_contact_binary: torch.Tensor,
            history_valid: torch.Tensor,
            *,
            dynamics_latent: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del (
                normalized_action,
                contact_binary,
                history_state,
                history_action,
                history_foot,
                history_contact_force,
                history_contact_binary,
                history_valid,
                dynamics_latent,
            )
            delta = normalized_state.new_zeros((*normalized_state.shape[:-1], 70))
            delta[..., 0] = normalized_state[..., 0]
            logits = normalized_contact_force.new_zeros((*normalized_contact_force.shape[:-1], 2))
            return delta, normalized_foot + 1.0, normalized_contact_force, logits

    model = DoublingRootPositionPredictor(_small_config())
    initial = _identity_state(1)
    initial[:, 0] = 1.0
    foot, contact_force, contact_binary = _current_privileged(1)
    prediction, _, predicted_foot, _, _ = model.rollout(
        initial,
        torch.zeros(1, 5, 29),
        foot,
        contact_force,
        contact_binary,
        **_history(1),
        **_rollout_normalization(),
    )

    assert torch.equal(
        prediction[0, :, 0],
        torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0]),
    )
    assert torch.equal(
        predicted_foot[0, :, 0],
        torch.arange(1.0, 6.0),
    )


def test_five_step_objective_is_zero_for_exact_recursive_trajectory() -> None:
    model = ForwardDynamicsTransformer(_small_config())
    for parameter in model.parameters():
        parameter.data.zero_()
    initial = _identity_state(2)
    physical_delta = torch.zeros(2, 70)
    physical_delta[:, 0] = 0.01
    physical_delta[:, 12:41] = 0.002
    target = _rollout_target(initial, physical_delta)
    foot, contact_force, contact_binary = _privileged_trajectory(2)
    history = _history(2)
    batch = {
        "state": torch.cat((initial[:, None], target), dim=1),
        "action": torch.randn(2, 5, 29),
        "foot": foot,
        "contact_force": contact_force,
        "contact_binary": contact_binary,
        **history,
        **_normalization(),
    }
    batch.update(_objective_context_fields(batch["state"], history))
    batch["delta_mean"] = physical_delta[0]
    objective = ForwardPredictorObjective(
        model,
        ForwardPredictorLossConfig(contact_binary_weight=0.0),
    )

    output = objective(batch)
    fast_output = objective(batch, compute_metrics=False, validate_batch=False)

    assert output["loss"].item() < 1.0e-10
    assert "privileged_dynamics_loss" not in output
    assert set(fast_output) == {
        "loss",
        "prediction_loss",
        "representation_loss",
    }
    torch.testing.assert_close(fast_output["loss"], output["loss"])
    assert output["nominal_five_step_nmse"].item() < 1.0e-8
    assert output["dr_five_step_nmse"].item() < 1.0e-8
    assert output["representation_loss"].item() == 0.0


def test_teacher_only_stage_updates_every_transformer_parameter() -> None:
    torch.manual_seed(17)
    model = ForwardDynamicsTransformer(_small_config())
    initial = _identity_state(3)
    delta = torch.randn(3, 70) * 0.01
    target = _rollout_target(initial, delta)
    foot, contact_force, contact_binary = _privileged_trajectory(3)
    history = _history(3)
    batch = {
        "state": torch.cat((initial[:, None], target), dim=1),
        "action": torch.randn(3, 5, 29),
        "foot": foot,
        "contact_force": contact_force,
        "contact_binary": contact_binary,
        **history,
        **_normalization(),
    }
    batch.update(_objective_context_fields(batch["state"], history))
    objective = ForwardPredictorObjective(model)

    output = objective(batch, recursive_weight=0.0)
    output["loss"].backward()

    assert torch.equal(output["loss"], output["prediction_loss"])
    assert output["representation_loss"].item() == 0.0
    assert all(parameter.grad is not None for parameter in model.parameters())


def _transition_batch(step: int, reset_second_world: bool = False) -> dict[str, torch.Tensor]:
    state = _identity_state(2)
    following = _identity_state(2)
    state[:, 0] = torch.tensor([float(step), 100.0 + step])
    following[:, 0] = state[:, 0] + 1.0
    reset = torch.tensor([False, reset_second_world and step == 2])
    return {
        "joint_target": torch.full((2, 29), float(step)),
        "robot_state": state,
        "next_robot_state": following,
        "nominal_next_robot_state": following.clone(),
        "foot": torch.full((2, 8), float(step)),
        "next_foot": torch.full((2, 8), float(step + 1)),
        "contact_force": torch.full((2, 6), float(step)),
        "next_contact_force": torch.full((2, 6), float(step + 1)),
        "contact_binary": torch.full((2, 2), step % 2 == 0, dtype=torch.bool),
        "next_contact_binary": torch.full((2, 2), step % 2 != 0, dtype=torch.bool),
        "reset_boundary": reset,
        "is_nominal": torch.tensor([True, False]),
        "world_id": torch.tensor([10, 11]),
        "episode_id": torch.zeros(2, dtype=torch.long),
        "episode_step": torch.full((2,), step, dtype=torch.long),
        "motion_id": torch.tensor([20, 21]),
        "motion_step": torch.full((2,), 30 + step, dtype=torch.long),
    }


def test_predictor_replay_materializes_only_reset_free_five_step_windows() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=2,
        dimensions=RolloutDimensions(),
        capacity=8,
        world_id_offset=10,
        context_history_steps=10,
    )
    generated = 0
    for step in range(5):
        generated += replay.add_step(_transition_batch(step, reset_second_world=True))

    assert generated == 1
    assert len(replay) == 1
    assert replay.storage_bytes == replay.estimated_storage_bytes
    packed = replay.normalizer.packed_statistics()
    stats = replay.normalizer.snapshot_from_packed(packed, replay.world_ids)
    replay.normalizer.freeze()
    sampled = replay.sample_batch(1, stats)
    physical_states = sampled["state"] * sampled["state_std"] + sampled["state_mean"]
    physical_actions = sampled["action"] * sampled["action_std"] + sampled["action_mean"]
    physical_foot = sampled["foot"] * sampled["foot_std"] + sampled["foot_mean"]

    assert sampled["world_id"].item() == 10
    assert torch.equal(physical_states[0, :, 0], torch.arange(6, dtype=torch.float32))
    assert torch.equal(physical_actions[0, :, 0], torch.arange(5, dtype=torch.float32))
    assert torch.equal(physical_foot[0, :, 0], torch.arange(6, dtype=torch.float32))
    assert not sampled["history_valid"].any()
    assert replay.normalizer.frozen


def _single_transition(
    *,
    state_value: float,
    episode_id: int,
    episode_step: int,
    reset_boundary: bool = False,
) -> dict[str, torch.Tensor]:
    state = _identity_state(1)
    following = _identity_state(1)
    state[:, 0] = state_value
    following[:, 0] = state_value + 1.0
    return {
        "joint_target": torch.full((1, 29), state_value),
        "robot_state": state,
        "next_robot_state": following,
        "nominal_next_robot_state": following.clone(),
        "foot": torch.full((1, 8), state_value),
        "next_foot": torch.full((1, 8), state_value + 1.0),
        "contact_force": torch.full((1, 6), state_value),
        "next_contact_force": torch.full((1, 6), state_value + 1.0),
        "contact_binary": torch.full((1, 2), episode_step % 2 == 0, dtype=torch.bool),
        "next_contact_binary": torch.full((1, 2), episode_step % 2 != 0, dtype=torch.bool),
        "reset_boundary": torch.tensor([reset_boundary]),
        "is_nominal": torch.tensor([False]),
        "world_id": torch.tensor([0]),
        "episode_id": torch.tensor([episode_id]),
        "episode_step": torch.tensor([episode_step]),
        "motion_id": torch.tensor([100 + episode_id]),
        "motion_step": torch.tensor([episode_step]),
    }


def test_predictor_replay_supplies_ten_causal_history_frames() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=1,
        capacity=4,
        sampling_mode="uniform",
        context_history_steps=10,
    )
    generated = 0
    for step in range(15):
        generated += replay.add_step(
            _single_transition(state_value=float(step), episode_id=0, episode_step=step)
        )

    assert generated == 3
    stats = replay.normalizer.snapshot_from_packed(
        replay.normalizer.packed_statistics(), replay.world_ids
    )
    sampled = replay.sample_batch(3, stats)
    target_state = sampled["state"] * sampled["state_std"] + sampled["state_mean"]
    history_state = sampled["history_state"] * sampled["state_std"] + sampled["state_mean"]
    order = target_state[:, 0, 0].argsort()
    target_state = target_state[order]
    history_state = history_state[order]
    history_valid = sampled["history_valid"][order]

    assert not history_valid[0].any()
    assert not history_valid[1, :5].any()
    assert history_valid[1, 5:].all()
    assert history_valid[2].all()
    assert torch.allclose(history_state[2, :, 0], torch.arange(10, dtype=torch.float32))
    assert torch.allclose(target_state[2, :, 0], torch.arange(10, 16, dtype=torch.float32))


def test_replay_uses_exact_five_frame_full_context_positive_views() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=1,
        capacity=8,
        context_history_steps=10,
        sampling_mode="uniform",
        seed=17,
    )
    for step in range(15):
        replay.add_step(
            _single_transition(state_value=float(step), episode_id=0, episode_step=step)
        )
    assert not replay.can_sample_positive_pairs(1)

    for step in range(15, 20):
        replay.add_step(
            _single_transition(state_value=float(step), episode_id=0, episode_step=step)
        )
    stats = replay.normalizer.snapshot_from_packed(
        replay.normalizer.packed_statistics(), replay.world_ids
    )
    assert replay.can_sample_positive_pairs(2)
    sampled = replay.sample_batch(
        2,
        stats,
        positive_ready_only=True,
    )
    assert sampled["history_valid"].all()
    assert sampled["positive_history_valid"].all()
    assert sampled["positive_pair_valid"].all()
    current = sampled["state"][:, 0] * sampled["state_std"] + sampled["state_mean"]
    positive = sampled["positive_current_state"] * sampled["state_std"] + sampled["state_mean"]
    torch.testing.assert_close((current[:, 0] - positive[:, 0]).abs(), torch.full((2,), 5.0))


def test_reset_state_can_start_target_with_masked_old_history() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=1,
        capacity=4,
        sampling_mode="uniform",
        context_history_steps=10,
    )
    for step in range(3):
        replay.add_step(
            _single_transition(
                state_value=float(step),
                episode_id=0,
                episode_step=step,
                reset_boundary=step == 2,
            )
        )
    generated = 0
    for step in range(7):
        generated += replay.add_step(
            _single_transition(
                state_value=100.0 + step,
                episode_id=1,
                episode_step=step,
            )
        )

    assert generated == 1
    stats = replay.normalizer.snapshot_from_packed(
        replay.normalizer.packed_statistics(), replay.world_ids
    )
    sampled = replay.sample_batch(1, stats)
    target_state = sampled["state"] * sampled["state_std"] + sampled["state_mean"]

    assert torch.equal(target_state[0, :, 0], torch.arange(102, 108, dtype=torch.float32))
    assert sampled["history_valid"].sum().item() == 2
    assert not sampled["context_full"].item()
    assert not sampled["positive_pair_valid"].item()


def test_motion_balanced_sampling_uses_inverse_motion_frequency() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=1,
        capacity=8,
        context_history_steps=10,
    )
    for step in range(10):
        replay.add_step(
            _single_transition(state_value=float(step), episode_id=0, episode_step=step)
        )
    replay.add_step(
        _single_transition(
            state_value=10.0,
            episode_id=0,
            episode_step=10,
            reset_boundary=True,
        )
    )
    for step in range(9):
        replay.add_step(
            _single_transition(
                state_value=100.0 + step,
                episode_id=1,
                episode_step=step,
            )
        )
    indices = replay._active_sample_indices()
    motion_ids = replay._samples["motion_id"].index_select(0, indices)
    weights = replay._sampling_weights(indices)
    assert torch.equal(weights[motion_ids == 100], torch.full((2,), 0.5))
    assert torch.equal(weights[motion_ids == 101], torch.ones(1))


def test_forward_predictor_cli_defaults_to_nominal_counterfactual_training() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint-file",
            "tracker.pt",
            "--motion-file",
            "motion.npz",
            "--output-dir",
            "run",
        ]
    )
    _validate_arguments(args)

    assert args.gradient_steps_per_update == 4
    assert args.batch_size == 4096
    assert args.micro_batch_size == 512
    assert args.amp_dtype == "bfloat16"
    assert args.replay_capacity == 262_144
    assert args.replay_sampling == "motion_balanced"
    assert args.history_steps == 10
    assert args.context_history_steps == 100
    assert args.positive_offset_steps == 5
    assert args.transformer_dim == 512
    assert args.transformer_depth == 6
    assert args.transformer_heads == 8
    assert args.context_dim == 128
    assert args.context_depth == 2
    assert args.context_heads == 4
    assert args.dynamics_latent_dim == 64
    assert args.dropout == 0.0
    assert args.nominal_fraction == 0.5
    assert not hasattr(args, "gradient_clip")
    assert args.rollout_steps_per_update == 5
    assert args.recursive_weight == 0.5
    assert args.foot_weight == 1.0
    assert args.contact_force_weight == 1.0
    assert args.contact_binary_weight == 1.0
    assert not hasattr(args, "privileged_dynamics_weight")
    assert args.representation_weight == 0.01
    assert args.response_distance_scale == 1.0
    assert not hasattr(args, "dynamics_classes")
    assert not hasattr(args, "contrastive_temperature")
    assert _CORE_PROBE_METRICS == (
        "one_step_nmse",
        "nominal_five_step_nmse",
        "dr_five_step_nmse",
        "latent_positive_cosine",
        "latent_response_correlation",
        "latent_shuffle_dr_error_ratio",
        "dr_counterfactual_rms",
        "nominal_counterfactual_rms",
    )


def test_counterfactual_collection_replays_exact_a_targets_and_aligns_b_states() -> None:
    num_worlds = 2
    batches = []
    for step in range(5):
        batches.append(
            {
                "robot_state": torch.full((num_worlds, 71), float(step)),
                "joint_target": torch.full((num_worlds, 29), 10.0 + step),
                "motion_id": torch.tensor([3, 7]),
                "motion_step": torch.tensor([20, 40]) + step,
            }
        )

    class ARollout:
        motion_files = ("motion-a.npz", "motion-b.npz")

        def __init__(self) -> None:
            self.index = 0

        def step(self, *, predictor_only: bool) -> dict[str, torch.Tensor]:
            assert predictor_only
            batch = batches[self.index]
            self.index += 1
            return batch

    class BRollout:
        def __init__(self) -> None:
            self.arguments: tuple[object, ...] | None = None

        def rollout_joint_targets(self, state, joint_targets, **metadata):
            self.arguments = (state.clone(), joint_targets.clone(), metadata)
            nominal = torch.stack(
                [torch.full((num_worlds, 71), 100.0 + step) for step in range(5)],
                dim=1,
            )
            return nominal, {"restore_max_abs_error": 0.0}

    class Replay:
        def __init__(self) -> None:
            self.batches: list[dict[str, torch.Tensor]] = []

        def add_step(self, batch: dict[str, torch.Tensor]) -> None:
            self.batches.append(batch)

    a_rollout = ARollout()
    b_rollout = BRollout()
    replay = Replay()
    diagnostics = _collect_counterfactual_block(a_rollout, b_rollout, replay)

    assert b_rollout.arguments is not None
    b_start, b_targets, metadata = b_rollout.arguments
    torch.testing.assert_close(b_start, batches[0]["robot_state"])
    torch.testing.assert_close(
        b_targets,
        torch.stack([batch["joint_target"] for batch in batches], dim=1),
    )
    assert metadata["motion_files"] == a_rollout.motion_files
    assert diagnostics == {"restore_max_abs_error": 0.0}
    assert len(replay.batches) == 5
    for step, batch in enumerate(replay.batches):
        torch.testing.assert_close(
            batch["nominal_next_robot_state"],
            torch.full((num_worlds, 71), 100.0 + step),
        )


def test_micro_batch_slice_preserves_global_normalization_tensors() -> None:
    batch = {
        "state": torch.arange(24).reshape(4, 6),
        "nominal_state": torch.arange(20).reshape(4, 5),
        "foot": torch.arange(32).reshape(4, 8),
        "contact_force": torch.arange(24).reshape(4, 6),
        "contact_binary": torch.ones(4, 6, 2, dtype=torch.bool),
        "history_foot": torch.ones(4, 10, 8),
        "history_contact_force": torch.ones(4, 10, 6),
        "history_contact_binary": torch.ones(4, 10, 2, dtype=torch.bool),
        "history_valid": torch.ones(4, 10, dtype=torch.bool),
        "positive_current_state": torch.ones(4, 6),
        "positive_history_state": torch.ones(4, 10, 6),
        "positive_history_action": torch.ones(4, 10, 3),
        "positive_history_valid": torch.ones(4, 10, dtype=torch.bool),
        "positive_pair_valid": torch.ones(4, dtype=torch.bool),
        "is_nominal": torch.tensor([True, True, False, False]),
        "motion_id": torch.arange(4),
        "state_mean": torch.arange(6),
        "delta_std": torch.arange(5),
    }

    micro_batch = _slice_predictor_batch(batch, 1, 3)

    assert torch.equal(micro_batch["state"], batch["state"][1:3])
    assert torch.equal(micro_batch["foot"], batch["foot"][1:3])
    assert torch.equal(micro_batch["contact_force"], batch["contact_force"][1:3])
    assert torch.equal(micro_batch["contact_binary"], batch["contact_binary"][1:3])
    assert torch.equal(micro_batch["history_valid"], batch["history_valid"][1:3])
    assert torch.equal(micro_batch["nominal_state"], batch["nominal_state"][1:3])
    assert torch.equal(micro_batch["positive_current_state"], batch["positive_current_state"][1:3])
    assert torch.equal(micro_batch["is_nominal"], batch["is_nominal"][1:3])
    assert torch.equal(micro_batch["motion_id"], batch["motion_id"][1:3])
    assert micro_batch["state_mean"] is batch["state_mean"]
    assert micro_batch["delta_std"] is batch["delta_std"]
