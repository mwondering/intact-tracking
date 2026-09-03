from __future__ import annotations

from types import SimpleNamespace

import mujoco
import pytest
import torch

from intact_tracking.cli.forward_predictor_train import (
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
)
from intact_tracking.forward_predictor_objective import (
    ForwardPredictorLossConfig,
    ForwardPredictorObjective,
)


def _identity_state(*prefix: int) -> torch.Tensor:
    state = torch.zeros((*prefix, 71))
    state[..., 3] = 1.0
    return state


def _small_config() -> ForwardPredictorConfig:
    return ForwardPredictorConfig(
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
        "history_contact_force": torch.zeros(batch_size, 10, 6),
        "history_contact_binary": torch.zeros(batch_size, 10, 2, dtype=torch.bool),
        "history_valid": torch.zeros(batch_size, 10, dtype=torch.bool),
    }


def _current_contacts(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(batch_size, 6), torch.zeros(batch_size, 2, dtype=torch.bool)


def _contact_trajectory(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch_size, 6, 6),
        torch.zeros(batch_size, 6, 2, dtype=torch.bool),
    )


def _forward_normalization() -> dict[str, torch.Tensor]:
    normalization = _normalization()
    return {
        name: normalization[name] for name in ("state_mean", "state_std", "foot_mean", "foot_std")
    }


def _rollout_normalization() -> dict[str, torch.Tensor]:
    normalization = _normalization()
    return {
        name: normalization[name]
        for name in (
            "state_mean",
            "state_std",
            "foot_mean",
            "foot_std",
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


def test_default_forward_predictor_is_a_twenty_million_parameter_transformer() -> None:
    model = ForwardDynamicsTransformer()

    assert sum(parameter.numel() for parameter in model.parameters()) == 19_022_414
    assert model.config.history_steps == 10
    assert model.config.sequence_length == 11
    assert model.config.transformer_dim == 512
    assert model.config.transformer_depth == 6
    assert model.config.transformer_heads == 8
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
    changed["history_contact_force"].normal_()
    changed["history_contact_binary"].logical_not_()
    contact_force, contact_binary = _current_contacts(2)

    expected = model(
        state,
        action,
        contact_force,
        contact_binary,
        **history,
        **_forward_normalization(),
    )
    actual = model(
        state,
        action,
        contact_force,
        contact_binary,
        **changed,
        **_forward_normalization(),
    )

    assert all(
        torch.equal(actual_item, expected_item)
        for actual_item, expected_item in zip(actual, expected, strict=True)
    )


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

    contact_force, contact_binary = _current_contacts(3)
    prediction, deltas, predicted_force, contact_logits = model.rollout(
        initial,
        actions,
        contact_force,
        contact_binary,
        **_history(3),
        **_rollout_normalization(),
    )
    loss = (
        prediction.square().mean()
        + deltas.square().mean()
        + predicted_force.square().mean()
        + contact_logits.square().mean()
    )
    loss.backward()

    assert prediction.shape == (3, 5, 71)
    assert deltas.shape == (3, 5, 70)
    assert predicted_force.shape == (3, 5, 6)
    assert contact_logits.shape == (3, 5, 2)
    assert torch.isfinite(prediction).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(float(parameter.grad.square().sum()) for parameter in model.parameters()) > 0.0


def test_rollout_feeds_each_predicted_state_into_the_next_transition() -> None:
    class DoublingRootPositionPredictor(ForwardDynamicsTransformer):
        def forward(
            self,
            normalized_state: torch.Tensor,
            normalized_action: torch.Tensor,
            normalized_contact_force: torch.Tensor,
            contact_binary: torch.Tensor,
            history_state: torch.Tensor,
            history_action: torch.Tensor,
            history_contact_force: torch.Tensor,
            history_contact_binary: torch.Tensor,
            history_valid: torch.Tensor,
            state_mean: torch.Tensor,
            state_std: torch.Tensor,
            foot_mean: torch.Tensor,
            foot_std: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del (
                normalized_action,
                contact_binary,
                history_state,
                history_action,
                history_contact_force,
                history_contact_binary,
                history_valid,
                state_mean,
                state_std,
                foot_mean,
                foot_std,
            )
            delta = normalized_state.new_zeros((*normalized_state.shape[:-1], 70))
            delta[..., 0] = normalized_state[..., 0]
            logits = normalized_contact_force.new_zeros((*normalized_contact_force.shape[:-1], 2))
            return delta, normalized_contact_force, logits

    model = DoublingRootPositionPredictor(_small_config())
    initial = _identity_state(1)
    initial[:, 0] = 1.0
    contact_force, contact_binary = _current_contacts(1)
    prediction, _, _, _ = model.rollout(
        initial,
        torch.zeros(1, 5, 29),
        contact_force,
        contact_binary,
        **_history(1),
        **_rollout_normalization(),
    )

    assert torch.equal(
        prediction[0, :, 0],
        torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0]),
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
    contact_force, contact_binary = _contact_trajectory(2)
    batch = {
        "state": torch.cat((initial[:, None], target), dim=1),
        "action": torch.randn(2, 5, 29),
        "contact_force": contact_force,
        "contact_binary": contact_binary,
        **_history(2),
        **_normalization(),
    }
    batch["delta_mean"] = physical_delta[0]
    objective = ForwardPredictorObjective(
        model,
        ForwardPredictorLossConfig(contact_binary_weight=0.0),
    )

    output = objective(batch)

    assert output["loss"].item() < 1.0e-10
    assert output["recursive_weight"].item() == 0.5
    assert output["rollout_nmse"].item() < 1.0e-8
    for step in range(1, 6):
        assert output[f"horizon_{step}_loss"].item() < 1.0e-10


def test_teacher_only_stage_updates_every_transformer_parameter() -> None:
    torch.manual_seed(17)
    model = ForwardDynamicsTransformer(_small_config())
    initial = _identity_state(3)
    delta = torch.randn(3, 70) * 0.01
    target = _rollout_target(initial, delta)
    contact_force, contact_binary = _contact_trajectory(3)
    batch = {
        "state": torch.cat((initial[:, None], target), dim=1),
        "action": torch.randn(3, 5, 29),
        "contact_force": contact_force,
        "contact_binary": contact_binary,
        **_history(3),
        **_normalization(),
    }
    objective = ForwardPredictorObjective(model)

    output = objective(batch, recursive_weight=0.0)
    output["loss"].backward()

    assert torch.equal(output["loss"], output["teacher_loss"])
    assert output["recursive_weight"].item() == 0.0
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
        "contact_force": torch.full((2, 6), float(step)),
        "next_contact_force": torch.full((2, 6), float(step + 1)),
        "contact_binary": torch.full((2, 2), step % 2 == 0, dtype=torch.bool),
        "next_contact_binary": torch.full((2, 2), step % 2 != 0, dtype=torch.bool),
        "reset_boundary": reset,
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
    )
    generated = 0
    for step in range(5):
        generated += replay.add_step(_transition_batch(step, reset_second_world=True))

    assert generated == 1
    assert len(replay) == 1
    packed = replay.normalizer.packed_statistics()
    stats = replay.normalizer.snapshot_from_packed(packed, replay.world_ids)
    replay.normalizer.freeze()
    sampled = replay.sample_batch(1, stats)
    physical_states = sampled["state"] * sampled["state_std"] + sampled["state_mean"]
    physical_actions = sampled["action"] * sampled["action_std"] + sampled["action_mean"]

    assert sampled["world_id"].item() == 10
    assert torch.equal(physical_states[0, :, 0], torch.arange(6, dtype=torch.float32))
    assert torch.equal(physical_actions[0, :, 0], torch.arange(5, dtype=torch.float32))
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
        "contact_force": torch.full((1, 6), state_value),
        "next_contact_force": torch.full((1, 6), state_value + 1.0),
        "contact_binary": torch.full((1, 2), episode_step % 2 == 0, dtype=torch.bool),
        "next_contact_binary": torch.full((1, 2), episode_step % 2 != 0, dtype=torch.bool),
        "reset_boundary": torch.tensor([reset_boundary]),
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


def test_reset_state_can_start_target_with_masked_old_history() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=1,
        capacity=4,
        sampling_mode="uniform",
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
    for step in range(5):
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

    assert torch.equal(target_state[0, :, 0], torch.arange(100, 106, dtype=torch.float32))
    assert not sampled["history_valid"].any()


def test_motion_balanced_sampling_uses_inverse_motion_frequency() -> None:
    replay = ForwardPredictorReplayBuffer(num_worlds=1, capacity=8)
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
    for step in range(5):
        replay.add_step(
            _single_transition(
                state_value=100.0 + step,
                episode_id=1,
                episode_step=step,
            )
        )
    stats = replay.normalizer.snapshot_from_packed(
        replay.normalizer.packed_statistics(), replay.world_ids
    )

    replay.sample_batch(1, stats)

    assert replay._sampling_weights is not None
    motion_ids = replay._samples["motion_id"][: len(replay)]
    weights = replay._sampling_weights
    assert torch.equal(weights[motion_ids == 100], torch.full((2,), 0.5))
    assert torch.equal(weights[motion_ids == 101], torch.ones(1))


def test_forward_predictor_cli_defaults_to_recursive_nominal_training() -> None:
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
    assert args.micro_batch_size == 256
    assert args.replay_capacity == 262_144
    assert args.replay_sampling == "motion_balanced"
    assert args.history_steps == 10
    assert args.transformer_dim == 512
    assert args.transformer_depth == 6
    assert args.transformer_heads == 8
    assert args.dropout == 0.0
    assert not hasattr(args, "gradient_clip")
    assert args.rollout_steps_per_update == 5
    assert args.recursive_weight == 0.5
    assert args.contact_force_weight == 1.0
    assert args.contact_binary_weight == 1.0


def test_micro_batch_slice_preserves_global_normalization_tensors() -> None:
    batch = {
        "state": torch.arange(24).reshape(4, 6),
        "contact_force": torch.arange(24).reshape(4, 6),
        "contact_binary": torch.ones(4, 6, 2, dtype=torch.bool),
        "history_contact_force": torch.ones(4, 10, 6),
        "history_contact_binary": torch.ones(4, 10, 2, dtype=torch.bool),
        "history_valid": torch.ones(4, 10, dtype=torch.bool),
        "motion_id": torch.arange(4),
        "state_mean": torch.arange(6),
        "delta_std": torch.arange(5),
    }

    micro_batch = _slice_predictor_batch(batch, 1, 3)

    assert torch.equal(micro_batch["state"], batch["state"][1:3])
    assert torch.equal(micro_batch["contact_force"], batch["contact_force"][1:3])
    assert torch.equal(micro_batch["contact_binary"], batch["contact_binary"][1:3])
    assert torch.equal(micro_batch["history_valid"], batch["history_valid"][1:3])
    assert torch.equal(micro_batch["motion_id"], batch["motion_id"][1:3])
    assert micro_batch["state_mean"] is batch["state_mean"]
    assert micro_batch["delta_std"] is batch["delta_std"]
