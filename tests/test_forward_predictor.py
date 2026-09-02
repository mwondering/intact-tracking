from __future__ import annotations

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
from intact_tracking.forward_predictor import (
    ForwardDynamicsTransformer,
    ForwardPredictorConfig,
    apply_physical_state_delta,
    physical_state_delta,
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
        "delta_mean": torch.zeros(70, device=device),
        "delta_std": torch.ones(70, device=device),
    }


def _history(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "history_state": torch.zeros(batch_size, 10, 71),
        "history_action": torch.zeros(batch_size, 10, 29),
        "history_valid": torch.zeros(batch_size, 10, dtype=torch.bool),
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

    assert sum(parameter.numel() for parameter in model.parameters()) == 19_010_118
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

    expected = model(state, action, **history)
    actual = model(state, action, **changed)

    assert torch.equal(actual, expected)


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
    normalization = _normalization()

    prediction, deltas = model.rollout(initial, actions, **_history(3), **normalization)
    loss = prediction.square().mean() + deltas.square().mean()
    loss.backward()

    assert prediction.shape == (3, 5, 71)
    assert deltas.shape == (3, 5, 70)
    assert torch.isfinite(prediction).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(float(parameter.grad.square().sum()) for parameter in model.parameters()) > 0.0


def test_rollout_feeds_each_predicted_state_into_the_next_transition() -> None:
    class DoublingRootPositionPredictor(ForwardDynamicsTransformer):
        def forward(
            self,
            normalized_state: torch.Tensor,
            normalized_action: torch.Tensor,
            history_state: torch.Tensor,
            history_action: torch.Tensor,
            history_valid: torch.Tensor,
        ) -> torch.Tensor:
            del normalized_action, history_state, history_action, history_valid
            delta = normalized_state.new_zeros((*normalized_state.shape[:-1], 70))
            delta[..., 0] = normalized_state[..., 0]
            return delta

    model = DoublingRootPositionPredictor(_small_config())
    initial = _identity_state(1)
    initial[:, 0] = 1.0
    prediction, _ = model.rollout(
        initial,
        torch.zeros(1, 5, 29),
        **_history(1),
        **_normalization(),
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
    batch = {
        "state": torch.cat((initial[:, None], target), dim=1),
        "action": torch.randn(2, 5, 29),
        **_history(2),
        **_normalization(),
    }
    batch["delta_mean"] = physical_delta[0]
    objective = ForwardPredictorObjective(model, ForwardPredictorLossConfig())

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
    batch = {
        "state": torch.cat((initial[:, None], target), dim=1),
        "action": torch.randn(3, 5, 29),
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
        "action": torch.full((2, 29), float(step)),
        "robot_state": state,
        "next_robot_state": following,
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
        "action": torch.full((1, 29), state_value),
        "robot_state": state,
        "next_robot_state": following,
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


def test_micro_batch_slice_preserves_global_normalization_tensors() -> None:
    batch = {
        "state": torch.arange(24).reshape(4, 6),
        "history_valid": torch.ones(4, 10, dtype=torch.bool),
        "motion_id": torch.arange(4),
        "state_mean": torch.arange(6),
        "delta_std": torch.arange(5),
    }

    micro_batch = _slice_predictor_batch(batch, 1, 3)

    assert torch.equal(micro_batch["state"], batch["state"][1:3])
    assert torch.equal(micro_batch["history_valid"], batch["history_valid"][1:3])
    assert torch.equal(micro_batch["motion_id"], batch["motion_id"][1:3])
    assert micro_batch["state_mean"] is batch["state_mean"]
    assert micro_batch["delta_std"] is batch["delta_std"]
