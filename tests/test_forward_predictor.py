from __future__ import annotations

import torch

from intact_tracking.cli.forward_predictor_train import (
    _recursive_loss_weight,
    _validate_arguments,
    build_parser,
)
from intact_tracking.data import (
    ForwardPredictorReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.forward_predictor import (
    ForwardDynamicsMLP,
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
    return ForwardPredictorConfig(hidden_dim=32, residual_blocks=2)


def _normalization(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.zeros(71, device=device),
        "state_std": torch.ones(71, device=device),
        "delta_mean": torch.zeros(70, device=device),
        "delta_std": torch.ones(70, device=device),
    }


def _history(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "history_state": torch.zeros(batch_size, 5, 71),
        "history_action": torch.zeros(batch_size, 5, 29),
        "history_valid": torch.zeros(batch_size, 5, dtype=torch.bool),
    }


def _rollout_target(initial: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    states = []
    current = initial
    for _ in range(5):
        current = apply_physical_state_delta(current, delta)
        states.append(current)
    return torch.stack(states, dim=1)


def test_default_forward_predictor_has_about_twenty_million_parameters() -> None:
    model = ForwardDynamicsMLP()

    assert sum(parameter.numel() for parameter in model.parameters()) == 20_141_070
    assert model.config.history_steps == 5
    assert model.config.hidden_dim == 1100
    assert model.config.residual_blocks == 8
    assert not hasattr(model, "history_encoder")


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
    model = ForwardDynamicsMLP(_small_config())
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
    class DoublingRootPositionPredictor(ForwardDynamicsMLP):
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
    model = ForwardDynamicsMLP(_small_config())
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
    assert output["rollout_nmse"].item() < 1.0e-8
    for step in range(1, 6):
        assert output[f"horizon_{step}_loss"].item() < 1.0e-10


def test_teacher_only_stage_updates_every_mlp_parameter() -> None:
    torch.manual_seed(17)
    model = ForwardDynamicsMLP(_small_config())
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


def test_predictor_replay_supplies_five_flat_history_frames() -> None:
    replay = ForwardPredictorReplayBuffer(
        num_worlds=1,
        capacity=4,
        sampling_mode="uniform",
    )
    generated = 0
    for step in range(10):
        generated += replay.add_step(
            _single_transition(state_value=float(step), episode_id=0, episode_step=step)
        )

    assert generated == 2
    stats = replay.normalizer.snapshot_from_packed(
        replay.normalizer.packed_statistics(), replay.world_ids
    )
    sampled = replay.sample_batch(2, stats)
    target_state = sampled["state"] * sampled["state_std"] + sampled["state_mean"]
    history_state = sampled["history_state"] * sampled["state_std"] + sampled["state_mean"]
    order = target_state[:, 0, 0].argsort()
    target_state = target_state[order]
    history_state = history_state[order]
    history_valid = sampled["history_valid"][order]

    assert not history_valid[0].any()
    assert history_valid[1].all()
    assert torch.equal(history_state[1, :, 0], torch.arange(5, dtype=torch.float32))
    assert torch.equal(target_state[1, :, 0], torch.arange(5, 11, dtype=torch.float32))


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
    assert args.batch_size == 2048
    assert args.replay_capacity == 262_144
    assert args.replay_sampling == "motion_balanced"
    assert args.history_steps == 5
    assert args.hidden_dim == 1100
    assert args.residual_blocks == 8
    assert args.rollout_steps_per_update == 5
    assert args.recursive_warmup_optimizer_steps == 5_000
    assert args.recursive_ramp_optimizer_steps == 15_000
    assert args.recursive_max_weight == 0.5


def test_recursive_loss_weight_warms_up_then_ramps_to_half() -> None:
    arguments = {"warmup_steps": 5_000, "ramp_steps": 15_000, "maximum": 0.5}

    assert _recursive_loss_weight(5_000, **arguments) == 0.0
    assert _recursive_loss_weight(12_500, **arguments) == 0.25
    assert _recursive_loss_weight(20_000, **arguments) == 0.5
    assert _recursive_loss_weight(100_000, **arguments) == 0.5
