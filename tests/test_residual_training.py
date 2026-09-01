from __future__ import annotations

from types import SimpleNamespace

import torch

from intact_tracking.cli.residual_train import (
    _loss_weight_payload,
    _validate_arguments,
    _wandb_payload,
    build_parser,
)
from intact_tracking.data import (
    ResidualOnlineReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.environment.mdp.spv5 import SPV5_REFERENCE_INPUT_STEPS
from intact_tracking.environment.policy.spv5_models import (
    SPV5_REFERENCE_POLICY_LENGTH,
    SPV5_REFERENCE_POLICY_START,
    SPV5_REFERENCE_SUPPORT_STEPS,
)
from intact_tracking.residual_control import ResidualTrunkController
from intact_tracking.residual_model import ResidualTrackingConfig, ResidualTrackingModel
from intact_tracking.residual_objective import (
    ResidualLossConfig,
    ResidualTrainingObjective,
    _pose_effect_losses,
    _pose_losses,
    _reconstruct_pose,
)
from intact_tracking.rollout.nominal import _make_nominal_dynamics_cfg
from intact_tracking.wandb_logger import WandbLogger


def _config() -> ResidualTrackingConfig:
    return ResidualTrackingConfig(
        policy_observation_dim=8,
        proprio_dim=6,
        action_dim=3,
        state_dim=71,
        context_dim=16,
        context_depth=1,
        context_heads=4,
        hidden_dim=32,
        forward_depth=1,
        backward_depth=1,
        policy_depth=1,
        residual_scale=0.2,
    )


def _model_batch(config: ResidualTrackingConfig, batch_size: int = 4):
    return {
        "context": torch.randn(batch_size, config.context_tokens, config.context_token_dim),
        "context_mask": torch.ones(batch_size, config.context_tokens, dtype=torch.bool),
        "state": torch.randn(batch_size, config.horizon + 1, config.state_dim),
        "reference_state": torch.randn(batch_size, config.horizon, config.state_dim),
        "action": torch.randn(batch_size, config.horizon, config.action_dim),
        "previous_action": torch.randn(batch_size, config.action_dim),
        "tracker_action": torch.randn(batch_size, config.horizon, config.action_dim),
        "policy_observation": torch.randn(batch_size, config.policy_observation_dim),
        "policy_world": torch.randn(batch_size, config.context_dim),
        "action_mean": torch.zeros(config.action_dim),
        "action_std": torch.ones(config.action_dim),
        "state_mean": torch.zeros(config.state_dim),
        "state_std": torch.ones(config.state_dim),
    }


def _gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def test_tracking_loss_only_updates_residual_policy() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(
        model,
        ResidualLossConfig(
            forward_weight=0.0,
            backward_weight=0.0,
            tracking_weight=1.0,
            residual_l2_weight=0.0,
            residual_smooth_weight=0.0,
        ),
    )
    output = objective(_model_batch(config), phase="policy")
    output["loss"].backward()

    assert _gradient_norm(model.residual_policy) > 0.0
    assert _gradient_norm(model.context_encoder) == 0.0
    assert _gradient_norm(model.forward_predictor) == 0.0
    assert _gradient_norm(model.backward_predictor) == 0.0


def test_forward_and_backward_losses_update_context_and_predictors() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(
        model,
        ResidualLossConfig(
            forward_weight=1.0,
            backward_weight=1.0,
            tracking_weight=0.0,
            residual_l2_weight=0.0,
            residual_smooth_weight=0.0,
        ),
    )
    output = objective(_model_batch(config), phase="model")
    output["loss"].backward()

    assert _gradient_norm(model.context_encoder) > 0.0
    assert _gradient_norm(model.forward_predictor) > 0.0
    assert _gradient_norm(model.backward_predictor) > 0.0
    assert _gradient_norm(model.residual_policy) == 0.0


def test_nominal_pair_loss_updates_context_and_forward() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(
        model,
        ResidualLossConfig(
            forward_weight=0.0,
            backward_weight=0.0,
            nominal_pair_weight=1.0,
            nominal_effect_weight=1.0,
            tracking_weight=0.0,
        ),
    )
    batch = _model_batch(config)
    batch["nominal_state"] = batch["state"][:2, 1:].clone()
    batch["nominal_state"][..., 0] += 0.5

    output = objective(batch, phase="model")
    output["loss"].backward()

    assert output["nominal_pair_count"].item() == 2.0
    assert output["nominal_pair_loss"].item() > 0.0
    assert _gradient_norm(model.context_encoder) > 0.0
    assert _gradient_norm(model.forward_predictor) > 0.0
    assert _gradient_norm(model.backward_predictor) == 0.0
    assert _gradient_norm(model.residual_policy) == 0.0


def test_forward_predictor_outputs_five_pose_deltas() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    batch = _model_batch(config)
    world = model.encode_context(batch["context"], batch["context_mask"])

    prediction = model.predict_future(
        world,
        batch["state"][:, 0],
        batch["previous_action"],
        batch["action"],
    )

    assert prediction.shape == (4, config.horizon, config.pose_delta_dim)
    assert config.pose_delta_dim == 35


def test_residual_policy_outputs_one_zero_initialized_five_action_trunk() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    trunk = model.residual_action_trunk(
        torch.randn(4, config.context_dim),
        torch.randn(4, config.policy_observation_dim),
    )

    assert trunk.shape == (4, config.horizon, config.action_dim)
    torch.testing.assert_close(trunk, torch.zeros_like(trunk))


def test_policy_phase_reports_exact_recomputed_executed_trunk() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(model)
    batch = _model_batch(config)
    batch["action"] = batch["tracker_action"].clone()

    output = objective(batch, phase="policy")

    torch.testing.assert_close(output["candidate_action_recompute_abs_max"], torch.zeros(()))
    torch.testing.assert_close(output["residual_saturation_fraction"], torch.zeros(()))
    for index in range(config.horizon):
        torch.testing.assert_close(output[f"residual_slot_{index}_rms"], torch.zeros(()))

    with torch.no_grad():
        model.residual_policy.net[-1].bias.fill_(10.0)
    saturated = objective(batch, phase="policy")
    torch.testing.assert_close(saturated["residual_saturation_fraction"], torch.ones(()))
    for index in range(config.horizon):
        torch.testing.assert_close(
            saturated[f"residual_slot_{index}_rms"],
            torch.tensor(config.residual_scale),
        )


def test_tracker_latent_has_current_plus_four_future_reference_frames() -> None:
    stop = SPV5_REFERENCE_POLICY_START + SPV5_REFERENCE_POLICY_LENGTH
    latent_offsets = SPV5_REFERENCE_SUPPORT_STEPS[SPV5_REFERENCE_POLICY_START:stop]

    assert latent_offsets == (0, 1, 2, 3, 4)
    assert 5 not in latent_offsets
    assert max(SPV5_REFERENCE_INPUT_STEPS) == 7


def test_trunk_controller_consumes_one_slot_and_restarts_only_reset_world() -> None:
    config = _config()
    model = ResidualTrackingModel(config).eval()
    target = torch.arange(1, config.horizon + 1, dtype=torch.float32) * 0.01
    target = target[:, None].expand(config.horizon, config.action_dim)
    with torch.no_grad():
        model.residual_policy.net[-1].bias.copy_(
            torch.atanh((target / config.residual_scale).flatten())
        )
    context = torch.randn(2, config.context_tokens, config.context_token_dim)
    controller = ResidualTrunkController(
        model,
        num_worlds=2,
        context_provider=lambda env_ids: (
            context.index_select(0, env_ids),
            torch.ones(env_ids.numel(), dtype=torch.bool),
        ),
        device="cpu",
    )
    observation = torch.randn(2, config.policy_observation_dim)
    tracker = torch.zeros(2, config.action_dim)

    with torch.inference_mode():
        first = controller(observation, tracker)
        controller.invalidate(torch.tensor([True, False]))
        second = controller(observation, tracker)

    torch.testing.assert_close(first, target[0].expand_as(first))
    torch.testing.assert_close(second[0], target[0])
    torch.testing.assert_close(second[1], target[1])
    assert controller.last_step.tolist() == [0, 1]
    assert controller.trunks_generated == 3


def test_zero_pose_delta_reconstructs_current_pose_and_ignores_velocity() -> None:
    config = _config()
    batch_size = 2
    current = torch.randn(batch_size, config.state_dim)
    current[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target = current[:, None].expand(-1, config.horizon, -1).clone()
    target[..., 7:13] = torch.randn_like(target[..., 7:13]) * 100.0
    target[..., 42:71] = torch.randn_like(target[..., 42:71]) * 100.0
    delta = torch.zeros(batch_size, config.horizon, config.pose_delta_dim)
    state_mean = torch.zeros(config.state_dim)
    state_std = torch.ones(config.state_dim)

    reconstructed = _reconstruct_pose(delta, current, state_mean, state_std)
    loss, components = _pose_losses(
        delta,
        current,
        target,
        state_mean,
        state_std,
        ResidualLossConfig(),
    )

    torch.testing.assert_close(reconstructed["root_position"], target[..., :3])
    torch.testing.assert_close(reconstructed["root_orientation"], target[..., 3:7])
    torch.testing.assert_close(reconstructed["joint_position"], target[..., 13:42])
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert set(components) == {"root_position", "root_orientation", "joint_position"}


def test_root_rotation_delta_is_composed_with_current_orientation() -> None:
    config = _config()
    current = torch.zeros(1, config.state_dim)
    current[:, 3] = 1.0
    delta = torch.zeros(1, config.horizon, config.pose_delta_dim)
    delta[:, :, 5] = torch.pi / 2.0

    reconstructed = _reconstruct_pose(
        delta,
        current,
        torch.zeros(config.state_dim),
        torch.ones(config.state_dim),
    )

    expected = torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5])
    torch.testing.assert_close(
        reconstructed["root_orientation"],
        expected.expand(1, config.horizon, 4),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_pose_effect_loss_is_zero_for_exact_dr_minus_nominal_prediction() -> None:
    config = _config()
    current = torch.zeros(1, config.state_dim)
    current[:, 3] = 1.0
    nominal_target = current[:, None].expand(-1, config.horizon, -1).clone()
    dr_target = nominal_target.clone()
    dr_target[..., 0] += 0.2
    dr_target[..., 13] -= 0.1
    nominal_delta = torch.zeros(1, config.horizon, config.pose_delta_dim)
    dr_delta = nominal_delta.clone()
    dr_delta[..., 0] = 0.2
    dr_delta[..., 6] = -0.1

    loss, zero_effect, _ = _pose_effect_losses(
        dr_delta,
        nominal_delta,
        current,
        dr_target,
        nominal_target,
        torch.zeros(config.state_dim),
        torch.ones(config.state_dim),
        ResidualLossConfig(),
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert zero_effect.item() > 0.0


def test_residual_replay_builds_strictly_causal_five_step_window() -> None:
    dimensions = RolloutDimensions(
        proprio=6,
        observation=4,
        action=3,
        robot_state=71,
        reference_state=71,
    )
    replay = ResidualOnlineReplayBuffer(
        num_worlds=2,
        policy_observation_dim=8,
        context_latent_dim=16,
        dimensions=dimensions,
        capacity=32,
        device="cpu",
    )
    for step in range(85):
        scalar = torch.full((2, 1), float(step))
        state = scalar.expand(2, 71).clone()
        batch = {
            "proprio": scalar.expand(2, 6).clone(),
            "next_proprio": (scalar + 1).expand(2, 6).clone(),
            "policy_observation": scalar.expand(2, 8).clone(),
            "tracker_action": scalar.expand(2, 3).clone(),
            "action": scalar.expand(2, 3).clone(),
            "robot_state": state,
            "next_robot_state": state + 1,
            "reference_state": state + 10,
            "next_reference_state": state + 11,
            "reset_boundary": torch.zeros(2, dtype=torch.bool),
            "world_id": torch.arange(2),
            "episode_id": torch.zeros(2, dtype=torch.long),
            "episode_step": torch.full((2,), step, dtype=torch.long),
            "collector_step": torch.full((2,), step, dtype=torch.long),
            "residual_trunk_step": torch.full((2,), step % 5, dtype=torch.long),
            "residual_world": scalar.expand(2, 16).clone(),
        }
        replay.add_step(batch)

    assert len(replay) == 2
    torch.testing.assert_close(replay._samples["policy_observation"][:2, 0], torch.full((2,), 80.0))
    torch.testing.assert_close(replay._samples["state"][0, :, 0], torch.arange(80.0, 86.0))
    assert replay._samples["previous_action"][0, 0].item() == 79.0
    # The sample context ends at step 79 and therefore never overlaps query 80:85.
    assert replay._samples["context_before"][0, 0, 0].item() == 0.0
    assert replay._samples["context_before"][0, -1, 0].item() == 75.0

    packed = replay.normalizer.packed_statistics()
    stats = replay.normalizer.snapshot_from_packed(packed, replay.world_ids)
    sampled = replay.sample_batch(2, stats)
    assert sampled["state"].shape == (2, 6, 71)
    assert sampled["policy_observation"].shape == (2, 8)
    assert sampled["policy_world"].shape == (2, 16)
    assert sampled["context"].shape == (2, 16, 2 * 6 + 5 * 3)
    assert torch.isfinite(sampled["context"]).all()
    assert replay.storage_bytes == replay.estimated_storage_bytes


def test_reset_state_can_start_query_but_boundary_cannot_enter_it() -> None:
    dimensions = RolloutDimensions(
        proprio=6,
        observation=4,
        action=3,
        robot_state=71,
        reference_state=71,
    )
    replay = ResidualOnlineReplayBuffer(
        num_worlds=1,
        policy_observation_dim=8,
        context_latent_dim=16,
        dimensions=dimensions,
        capacity=8,
        device="cpu",
    )

    def add(
        value: float,
        *,
        episode_id: int,
        episode_step: int,
        next_value: float | None = None,
        reset_boundary: bool = False,
    ) -> None:
        current = torch.full((1, 1), value)
        following = value + 1.0 if next_value is None else next_value
        state = current.expand(1, 71).clone()
        replay.add_step(
            {
                "proprio": current.expand(1, 6).clone(),
                "next_proprio": torch.full((1, 6), following),
                "policy_observation": current.expand(1, 8).clone(),
                "tracker_action": current.expand(1, 3).clone(),
                "action": current.expand(1, 3).clone(),
                "robot_state": state,
                "next_robot_state": torch.full((1, 71), following),
                "reference_state": state + 10.0,
                "next_reference_state": torch.full((1, 71), following + 10.0),
                "reset_boundary": torch.tensor([reset_boundary]),
                "world_id": torch.zeros(1, dtype=torch.long),
                "episode_id": torch.tensor([episode_id]),
                "episode_step": torch.tensor([episode_step]),
                "collector_step": torch.tensor([episode_step]),
                "residual_trunk_step": torch.tensor([episode_step % 5]),
                "residual_world": current.expand(1, 16).clone(),
            }
        )

    for step in range(85):
        add(float(step), episode_id=0, episode_step=step)
    add(85.0, episode_id=0, episode_step=85, next_value=200.0, reset_boundary=True)
    for step in range(5):
        add(200.0 + step, episode_id=1, episode_step=step)

    assert len(replay) == 2
    torch.testing.assert_close(replay._samples["state"][1, :, 0], torch.arange(200.0, 206.0))
    torch.testing.assert_close(replay._samples["action"][1, :, 0], torch.arange(200.0, 205.0))
    assert replay._samples["previous_action"][1, 0].item() == 0.0
    assert replay._samples["episode_id"][1].item() == 1
    assert replay._context_counts is not None
    assert replay._context_counts.item() == 18


def test_residual_cli_enables_wandb_and_five_step_collection_by_default() -> None:
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
    assert args.rollout_steps_per_update == 5
    assert args.forward_weight == 2.0
    assert args.backward_weight == 2.0
    assert args.nominal_pair_batch_size == 64
    assert args.nominal_pair_weight == 1.0
    assert args.nominal_effect_weight == 1.0
    assert args.residual_l2_weight == 0.2
    assert args.root_position_weight == 5.0
    assert args.root_orientation_weight == 2.0
    assert args.joint_position_weight == 1.0
    assert not hasattr(args, "joint_velocity_weight")
    assert args.randomize_initial_episode_phase
    assert args.wandb
    assert args.wandb_project == "intact-residual-tracking"


def test_loss_weight_payload_lists_objective_and_shared_state_terms() -> None:
    payload = _loss_weight_payload(ResidualLossConfig())
    assert payload == {
        "objective_terms": {
            "forward_loss": 2.0,
            "backward_loss": 2.0,
            "nominal_pair_loss": 1.0,
            "nominal_effect_within_pair": 1.0,
            "tracking_loss": 1.0,
            "residual_l2": 0.2,
            "residual_smooth": 1.0e-3,
        },
        "pose_terms_shared_by_forward_and_tracking": {
            "root_position": 5.0,
            "root_orientation": 2.0,
            "joint_position": 1.0,
        },
    }


def test_nominal_dynamics_cfg_removes_dr_and_task_managers() -> None:
    env_cfg = SimpleNamespace(
        events={"dr": object()},
        commands={"motion": object()},
        observations={"policy": object()},
        rewards={"tracking": object()},
        terminations={"timeout": object()},
        curriculum={"difficulty": object()},
        metrics={"cache": object()},
        recorders={"trace": object()},
        auto_reset=True,
    )

    removed = _make_nominal_dynamics_cfg(env_cfg)

    assert removed["events"] == ["dr"]
    assert not env_cfg.events
    assert not env_cfg.commands
    assert not env_cfg.observations
    assert not env_cfg.rewards
    assert not env_cfg.terminations
    assert not env_cfg.curriculum
    assert not env_cfg.metrics
    assert not env_cfg.recorders
    assert not env_cfg.auto_reset


def test_wandb_logger_is_rank_zero_only(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeRun:
        id = "run-id"
        url = "https://wandb.invalid/run-id"

        def define_metric(self, *args, **kwargs):
            calls.append(("define", args, kwargs))

        def log(self, payload, step):
            calls.append(("log", payload, step))

        def finish(self, exit_code):
            calls.append(("finish", exit_code))

    fake_wandb = SimpleNamespace(init=lambda **kwargs: calls.append(("init", kwargs)) or FakeRun())
    monkeypatch.setattr(
        "intact_tracking.wandb_logger.importlib.import_module", lambda _name: fake_wandb
    )
    logger = WandbLogger(
        enabled=True,
        is_main=True,
        project="project",
        output_dir=tmp_path,
        config={"value": 1},
    )
    logger.log({"train/loss": 1.0}, step=3)
    logger.finish()
    assert [call[0] for call in calls].count("init") == 1
    assert ("log", {"train/loss": 1.0}, 3) in calls
    assert ("finish", 0) in calls


def test_wandb_payload_exposes_training_tracking_and_replay_curves() -> None:
    error_name = "error_joint_pos"
    payload = _wandb_payload(
        {
            "update": 2,
            "optimizer_steps": 4,
            "learning_rate_model": 3.0e-4,
            "learning_rate_policy": 1.0e-4,
            "gradient_norm": 1.0,
            "model_gradient_norm": 0.75,
            "policy_gradient_norm": 0.25,
            "replay_size": 32,
            "samples_generated": 40,
            "new_samples_generated": 8,
            "replay_storage_bytes": 1024,
            "transitions": 128,
            "environments_reset": 12,
            "new_environments_reset": 4,
            "new_reset_events": 2,
            "reset_fraction": 0.03125,
            "train": {"loss": 3.0, "residual_rms": 0.02},
            "tracking": {error_name: 0.8},
            "tracking_baseline": {error_name: 1.0},
            "tracking_comparison": {
                f"{error_name}_ratio_to_tracker": 0.8,
                f"{error_name}_relative_improvement": 0.2,
            },
        }
    )

    assert payload["train/loss"] == 3.0
    assert payload[f"tracking/rollout_{error_name}"] == 0.8
    assert payload[f"tracking/{error_name}_relative_improvement"] == 0.2
    assert payload["optimization/policy_gradient_norm"] == 0.25
    assert payload["replay/new_samples_generated"] == 8
    assert payload["rollout/environments_reset_delta"] == 4
    assert payload["rollout/reset_fraction"] == 0.03125
