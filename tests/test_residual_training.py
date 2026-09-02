from __future__ import annotations

from types import SimpleNamespace

import pytest
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
from intact_tracking.residual_model import ResidualTrackingConfig, ResidualTrackingModel
from intact_tracking.residual_objective import (
    ResidualLossConfig,
    ResidualTrainingObjective,
    _pose_effect_losses,
    _pose_losses,
    _reconstruct_pose,
)
from intact_tracking.rollout.nominal import (
    NominalPairRollout,
    _make_nominal_dynamics_cfg,
    _repeat_error_diagnostics,
)
from intact_tracking.wandb_logger import WandbLogger


def _config() -> ResidualTrackingConfig:
    return ResidualTrackingConfig(
        proprio_dim=6,
        action_dim=3,
        state_dim=71,
        context_dim=16,
        context_depth=1,
        context_heads=4,
        hidden_dim=32,
        forward_depth=1,
    )


def _model_batch(config: ResidualTrackingConfig, batch_size: int = 4):
    return {
        "context": torch.randn(batch_size, config.context_tokens, config.context_token_dim),
        "context_mask": torch.ones(batch_size, config.context_tokens, dtype=torch.bool),
        "state": torch.randn(batch_size, config.horizon + 1, config.state_dim),
        "action": torch.randn(batch_size, config.horizon, config.action_dim),
        "previous_action": torch.randn(batch_size, config.action_dim),
        "is_nominal": torch.arange(batch_size).remainder(2).eq(0),
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


def test_forward_loss_only_updates_context_and_forward_predictor() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(
        model,
        ResidualLossConfig(forward_weight=1.0),
    )
    output = objective(_model_batch(config), phase="model")
    output["loss"].backward()

    assert _gradient_norm(model.context_encoder) > 0.0
    assert _gradient_norm(model.forward_predictor) > 0.0
    assert not hasattr(model, "backward_predictor")
    assert not hasattr(model, "residual_policy")


def test_forward_objective_rejects_removed_policy_phase() -> None:
    config = _config()
    objective = ResidualTrainingObjective(ResidualTrackingModel(config))

    with pytest.raises(ValueError, match="Forward-only"):
        objective(_model_batch(config), phase="policy")


def test_nominal_pair_loss_updates_context_and_forward() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(
        model,
        ResidualLossConfig(
            forward_weight=0.0,
            nominal_pair_weight=1.0,
            nominal_effect_weight=1.0,
        ),
    )
    batch = _model_batch(config)
    batch["nominal_state"] = batch["state"][:2, 1:].clone()
    batch["nominal_state"][..., 0] += 0.5
    batch["nominal_context"] = torch.randn(
        2, config.context_tokens, config.context_token_dim
    )
    batch["nominal_context_mask"] = torch.ones(
        2, config.context_tokens, dtype=torch.bool
    )

    output = objective(batch, phase="model")
    output["loss"].backward()

    assert output["nominal_pair_count"].item() == 2.0
    assert output["nominal_pair_loss"].item() > 0.0
    assert output["nominal_source_pair_count"].item() == 1.0
    assert output["dr_source_pair_count"].item() == 1.0
    assert _gradient_norm(model.context_encoder) > 0.0
    assert _gradient_norm(model.forward_predictor) > 0.0


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


def test_forward_objective_reports_domain_and_horizon_metrics() -> None:
    config = _config()
    model = ResidualTrackingModel(config)
    objective = ResidualTrainingObjective(model)
    output = objective(_model_batch(config))

    assert "forward_source_nominal_loss" in output
    assert "forward_source_nominal_nmse" in output
    assert "forward_source_dr_loss" in output
    assert "forward_source_dr_nmse" in output
    assert "forward_zero_context_ratio" in output
    assert "forward_nominal_zero_context_ratio" in output
    assert "forward_dr_zero_context_ratio" in output
    for step in range(1, config.horizon + 1):
        assert f"forward_horizon_{step}_loss" in output
        assert f"forward_horizon_{step}_nmse" in output


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
            "action": scalar.expand(2, 3).clone(),
            "robot_state": state,
            "next_robot_state": state + 1,
            "reset_boundary": torch.zeros(2, dtype=torch.bool),
            "world_id": torch.arange(2),
            "is_nominal": torch.tensor([True, False]),
            "episode_id": torch.zeros(2, dtype=torch.long),
            "motion_id": torch.tensor([7, 9], dtype=torch.long),
            "motion_step": torch.full((2,), step + 100, dtype=torch.long),
            "episode_step": torch.full((2,), step, dtype=torch.long),
            "collector_step": torch.full((2,), step, dtype=torch.long),
            "residual_trunk_step": torch.full((2,), step % 5, dtype=torch.long),
        }
        replay.add_step(batch)

    assert len(replay) == 2
    torch.testing.assert_close(replay._samples["state"][0, :, 0], torch.arange(80.0, 86.0))
    assert replay._samples["previous_action"][0, 0].item() == 79.0
    # The sample context ends at step 79 and therefore never overlaps query 80:85.
    assert replay._samples["context_before"][0, 0, 0].item() == 0.0
    assert replay._samples["context_before"][0, -1, 0].item() == 75.0

    packed = replay.normalizer.packed_statistics()
    stats = replay.normalizer.snapshot_from_packed(packed, replay.world_ids)
    sampled = replay.sample_batch(2, stats, include_nominal_context=True)
    assert sampled["state"].shape == (2, 6, 71)
    assert sampled["context"].shape == (2, 16, 2 * 6 + 5 * 3)
    assert set(sampled["motion_id"].tolist()) == {7, 9}
    assert sampled["is_nominal"].sum().item() == 1
    assert sampled["nominal_context"].shape == sampled["context"].shape
    assert sampled["nominal_context_mask"].all()
    assert sampled["nominal_context_world_id"].eq(0).all()
    assert torch.equal(sampled["motion_step"], torch.full((2,), 180, dtype=torch.long))
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
                "action": current.expand(1, 3).clone(),
                "robot_state": state,
                "next_robot_state": torch.full((1, 71), following),
                "reset_boundary": torch.tensor([reset_boundary]),
                "world_id": torch.zeros(1, dtype=torch.long),
                "is_nominal": torch.ones(1, dtype=torch.bool),
                "episode_id": torch.tensor([episode_id]),
                "motion_id": torch.tensor([episode_id + 10]),
                "motion_step": torch.tensor([episode_step + 100]),
                "episode_step": torch.tensor([episode_step]),
                "collector_step": torch.tensor([episode_step]),
                "residual_trunk_step": torch.tensor([episode_step % 5]),
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
    assert replay._samples["motion_id"][1].item() == 11
    assert replay._samples["motion_step"][1].item() == 100
    assert replay._context_counts is not None
    assert replay._context_counts.item() == 18


def test_forward_cli_enables_wandb_and_five_step_collection_by_default() -> None:
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
    assert not hasattr(args, "backward_weight")
    assert not hasattr(args, "policy_learning_rate")
    assert not hasattr(args, "tracking_weight")
    assert args.nominal_pair_batch_size == 64
    assert args.nominal_pair_weight == 1.0
    assert args.nominal_effect_weight == 1.0
    assert args.nominal_consistency_weight == 1.0
    assert args.nominal_rollout_fraction == 0.5
    assert args.root_position_weight == 5.0
    assert args.root_orientation_weight == 2.0
    assert args.joint_position_weight == 1.0
    assert not hasattr(args, "joint_velocity_weight")
    assert args.randomize_initial_episode_phase
    assert args.wandb
    assert args.wandb_project == "intact-forward-world-model"


def test_nominal_only_forward_run_disables_counterfactual_pair_by_default() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint-file",
            "tracker.pt",
            "--motion-file",
            "motion.npz",
            "--output-dir",
            "run",
            "--nominal-rollout-fraction",
            "1.0",
        ]
    )
    _validate_arguments(args)

    assert args.nominal_rollout_fraction == 1.0
    assert args.nominal_pair_batch_size == 0


def test_loss_weight_payload_lists_objective_and_shared_state_terms() -> None:
    payload = _loss_weight_payload(ResidualLossConfig())
    assert payload == {
        "objective_terms": {
            "forward_loss": 2.0,
            "nominal_pair_loss": 1.0,
            "nominal_effect_within_pair": 1.0,
            "nominal_consistency_within_pair": 1.0,
        },
        "forward_pose_terms": {
            "root_position": 5.0,
            "root_orientation": 2.0,
            "joint_position": 1.0,
        },
    }


def test_nominal_dynamics_cfg_removes_dr_and_task_managers() -> None:
    action_cfg = SimpleNamespace(
        max_delay=2,
        alpha=(0.8, 1.0),
        torque_limit_scale_range=(0.7, 1.0),
        boot_delay_steps=3,
    )
    env_cfg = SimpleNamespace(
        actions={"joint_pos": action_cfg},
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
    assert action_cfg.max_delay == 0
    assert action_cfg.alpha == (1.0, 1.0)
    assert action_cfg.torque_limit_scale_range == (1.0, 1.0)
    assert action_cfg.boot_delay_steps == 0
    assert removed["action_randomization_overrides"]["joint_pos"]["max_delay"] == {
        "from": 2,
        "to": 0,
    }


def test_nominal_repeat_diagnostics_excludes_root_velocity_from_pose() -> None:
    target = torch.zeros(2, 5, 71)
    repeated = target.clone()
    repeated[0, 2, 10] = 4.0
    repeated[1, 4, 13] = 0.25

    diagnostics = _repeat_error_diagnostics(
        target,
        repeated,
        motion_ids=torch.tensor([0, 1]),
        motion_steps=torch.tensor([20, 30]),
        motion_files=("first.npz", "second.npz"),
    )

    assert diagnostics["pose"]["max"] == 0.25
    assert diagnostics["full_state"]["max"] == 4.0
    assert diagnostics["worst_pose"] == {
        "pair_index": 1,
        "horizon": 5,
        "state_index": 13,
        "state_component": "joint_position",
        "abs_error": 0.25,
        "motion_id": 1,
        "motion_path": "second.npz",
        "motion_step": 30,
    }
    assert diagnostics["worst_full_state"]["state_component"] == "root_angular_velocity"
    assert diagnostics["worst_full_state"]["motion_path"] == "first.npz"


def test_nominal_repeat_outlier_warns_without_aborting(tmp_path, capsys) -> None:
    rollout = object.__new__(NominalPairRollout)
    rollout.config = SimpleNamespace(
        num_envs=2,
        horizon=5,
        restore_atol=1.0e-5,
        failure_log_file=str(tmp_path / "repeat_warnings.jsonl"),
    )
    rollout.device = torch.device("cpu")
    rollout.action_dim = 29
    rollout.closed = False
    rollout._validated = False
    rollout._last_repeat_error = 0.0
    rollout._last_repeat_pose_error = 0.0
    rollout._last_repeat_full_state_p99_error = 0.0
    rollout._last_repeat_pose_p99_error = 0.0
    rollout._last_repeat_warning = 0.0
    rollout.env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _name: SimpleNamespace())
    )

    target = torch.zeros(2, 5, 71)
    repeated = target.clone()
    repeated[1, 4, 13] = 0.25
    outputs = iter((target, repeated))
    rollout._restore = lambda _state, _previous_action: 9.0e-7
    rollout._step_actions = lambda _actions: next(outputs)

    returned, metrics = rollout.rollout(
        torch.zeros(2, 71),
        torch.zeros(2, 29),
        torch.zeros(2, 5, 29),
        motion_ids=torch.tensor([0, 1]),
        motion_steps=torch.tensor([20, 30]),
        motion_files=("first.npz", "second.npz"),
    )

    assert returned is target
    assert rollout._validated
    assert metrics["nominal_restore_repeat_warning"] == 1.0
    assert metrics["nominal_restore_repeat_pose_max_abs_error"] == 0.25
    warning = (tmp_path / "repeat_warnings.jsonl").read_text(encoding="utf-8")
    assert '"event": "nominal_repeat_validation_warning"' in warning
    assert '"motion_path": "second.npz"' in warning
    assert "nominal_repeat_validation_warning" in capsys.readouterr().out


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


def test_wandb_payload_exposes_forward_and_replay_curves() -> None:
    payload = _wandb_payload(
        {
            "update": 2,
            "optimizer_steps": 4,
            "learning_rate_model": 3.0e-4,
            "gradient_norm": 1.0,
            "model_gradient_norm": 0.75,
            "replay_size": 32,
            "samples_generated": 40,
            "new_samples_generated": 8,
            "replay_storage_bytes": 1024,
            "transitions": 128,
            "environments_reset": 12,
            "new_environments_reset": 4,
            "new_reset_events": 2,
            "reset_fraction": 0.03125,
            "train": {"loss": 3.0, "forward_source_nominal_nmse": 0.2},
        }
    )

    assert payload["train/loss"] == 3.0
    assert payload["train/forward_source_nominal_nmse"] == 0.2
    assert payload["replay/new_samples_generated"] == 8
    assert payload["rollout/environments_reset_delta"] == 4
    assert payload["rollout/reset_fraction"] == 0.03125
