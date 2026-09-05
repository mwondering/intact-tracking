from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from intact_tracking.cli.model_gradient_residual_train import _real_tracking_error_metrics
from intact_tracking.forward_predictor import ForwardDynamicsTransformer, ForwardPredictorConfig
from intact_tracking.forward_predictor_inputs import JointPositionTargetTransform
from intact_tracking.model_gradient_residual import (
    FrozenForwardPredictorCheckpoint,
    ModelGradientLossConfig,
    ModelGradientResidualPolicy,
    PredictorCausalHistory,
    load_frozen_forward_predictor_checkpoint,
    model_gradient_loss,
)


def _small_config() -> ForwardPredictorConfig:
    return ForwardPredictorConfig(
        context_history_steps=10,
        transformer_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        context_dim=16,
        context_depth=1,
        context_heads=4,
        dynamics_latent_dim=8,
    )


def _normalization() -> dict[str, tuple[float, ...]]:
    return {
        "state_mean": (0.0,) * 71,
        "state_std": (1.0,) * 71,
        "action_mean": (0.0,) * 29,
        "action_std": (1.0,) * 29,
        "foot_mean": (0.0,) * 8,
        "foot_std": (1.0,) * 8,
        "contact_force_mean": (0.0,) * 6,
        "contact_force_std": (1.0,) * 6,
        "delta_mean": (0.0,) * 70,
        "delta_std": (1.0,) * 70,
    }


def _checkpoint(model: ForwardDynamicsTransformer) -> FrozenForwardPredictorCheckpoint:
    return FrozenForwardPredictorCheckpoint(
        model=model,
        config=model.config,
        state_mean=torch.zeros(71),
        state_std=torch.ones(71),
        action_mean=torch.zeros(29),
        action_std=torch.ones(29),
        foot_mean=torch.zeros(8),
        foot_std=torch.ones(8),
        contact_force_mean=torch.zeros(6),
        contact_force_std=torch.ones(6),
        delta_mean=torch.zeros(70),
        delta_std=torch.ones(70),
        path="unused",
        sha256="unused",
        tracker_sha256=None,
    )


def _identity_action_transform(num_worlds: int = 1) -> JointPositionTargetTransform:
    return JointPositionTargetTransform(
        scale=torch.ones(num_worlds, 29),
        offset=torch.zeros(num_worlds, 29),
        encoder_bias=torch.zeros(num_worlds, 29),
        target_reindex=torch.arange(29),
    )


def test_full_forward_checkpoint_loader_freezes_every_model_parameter(tmp_path: Path) -> None:
    config = _small_config()
    model = ForwardDynamicsTransformer(config)
    path = tmp_path / "forward.pt"
    torch.save(
        {
            "architecture_version": config.architecture_version,
            "model_config": asdict(config),
            "model": model.state_dict(),
            "normalization": _normalization(),
            "tracker": {"checkpoint_sha256": "tracker-hash"},
        },
        path,
    )

    loaded = load_frozen_forward_predictor_checkpoint(
        path,
        device="cpu",
        expected_tracker_sha256="tracker-hash",
    )

    assert loaded.config == config
    assert loaded.tracker_sha256 == "tracker-hash"
    assert not loaded.model.training
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    torch.testing.assert_close(loaded.contact_force_std, torch.ones(6))


def test_forward_checkpoint_loader_rejects_non_v12_architecture(tmp_path: Path) -> None:
    config = _small_config()
    payload = asdict(config)
    payload["architecture_version"] = "obsolete_forward_predictor"
    path = tmp_path / "old-forward.pt"
    torch.save(
        {
            "architecture_version": "obsolete_forward_predictor",
            "model_config": payload,
            "model": ForwardDynamicsTransformer(config).state_dict(),
            "normalization": _normalization(),
        },
        path,
    )

    with pytest.raises(ValueError, match="complete v12"):
        load_frozen_forward_predictor_checkpoint(path, device="cpu")


def test_predictor_history_masks_reset_world_and_keeps_other_world() -> None:
    model = ForwardDynamicsTransformer(_small_config()).eval().requires_grad_(False)
    checkpoint = _checkpoint(model)
    history = PredictorCausalHistory(
        checkpoint,
        num_envs=2,
        device="cpu",
        use_bfloat16=False,
    )
    state = torch.zeros(2, 71)
    foot = torch.zeros(2, 8)
    force = torch.zeros(2, 6)
    binary = torch.zeros(2, 2)
    for step in range(3):
        history.append(
            state + step,
            torch.zeros(2, 29),
            foot,
            force,
            binary,
            torch.tensor([step == 2, False]),
        )

    snapshot = history.snapshot(state, foot, force, binary)

    assert snapshot["history_valid"][0].sum() == 0
    assert snapshot["history_valid"][1].sum() == 3
    assert snapshot["history_foot"].shape == (2, 10, 8)
    assert snapshot["latent"].shape == (2, 8)


def test_five_step_model_loss_updates_only_residual_policy() -> None:
    torch.manual_seed(4)
    model = ForwardDynamicsTransformer(_small_config()).eval().requires_grad_(False)
    checkpoint = _checkpoint(model)
    policy = ModelGradientResidualPolicy(
        tracker_feature_dim=3,
        dynamics_latent_dim=8,
        hidden_dims=(16, 8),
    )
    batch_size = 3
    initial_state = torch.zeros(batch_size, 71)
    initial_state[:, 3] = 1.0
    reference = initial_state[:, None].expand(-1, 5, -1).clone()
    reference[..., 13:42] = 0.5
    predictor_inputs = {
        "state": initial_state,
        "foot": torch.zeros(batch_size, 8),
        "contact_force": torch.zeros(batch_size, 6),
        "contact_binary": torch.zeros(batch_size, 2),
        "history_state": torch.zeros(batch_size, 10, 71),
        "history_action": torch.zeros(batch_size, 10, 29),
        "history_foot": torch.zeros(batch_size, 10, 8),
        "history_contact_force": torch.zeros(batch_size, 10, 6),
        "history_contact_binary": torch.zeros(batch_size, 10, 2),
        "history_valid": torch.ones(batch_size, 10, dtype=torch.bool),
    }
    predictor_inputs["history_state"][..., 3] = 1.0
    output = model_gradient_loss(
        policy=policy,
        predictor_inputs=predictor_inputs,
        tracker_features=torch.randn(batch_size, 5, 3),
        latent_sequence=torch.randn(batch_size, 5, 8),
        tracker_actions=torch.zeros(batch_size, 5, 29),
        reference_states=reference,
        valid=torch.ones(batch_size, 5, dtype=torch.bool),
        env_ids=torch.arange(batch_size),
        action_transform=_identity_action_transform(batch_size),
        checkpoint=checkpoint,
        loss_config=ModelGradientLossConfig(),
        action_clip=1.0,
    )

    output["loss"].backward()

    gradients = [parameter.grad for parameter in policy.parameters()]
    assert any(gradient is not None and bool((gradient != 0).any()) for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.isfinite(output["loss"])


def test_action_transform_subsample_uses_the_matching_world_rows() -> None:
    transform = JointPositionTargetTransform(
        scale=torch.stack((torch.ones(29), torch.full((29,), 2.0), torch.full((29,), 3.0))),
        offset=torch.zeros(3, 29),
        encoder_bias=torch.zeros(3, 29),
        target_reindex=torch.arange(29),
    )
    action = torch.ones(2, 29)

    target = transform(action, env_ids=torch.tensor([2, 0]))

    torch.testing.assert_close(target[0], torch.full((29,), 3.0))
    torch.testing.assert_close(target[1], torch.ones(29))


def test_real_tracking_errors_are_logged_separately_in_physical_units() -> None:
    errors = torch.zeros(2, 5, 8)
    for index in range(8):
        errors[..., index] = float(index + 1)

    metrics = _real_tracking_error_metrics(errors)

    assert metrics == {
        "real_root_position_error_m": 1.0,
        "real_root_orientation_error_rad": 2.0,
        "real_root_linear_velocity_error_mps": 3.0,
        "real_root_angular_velocity_error_radps": 4.0,
        "real_joint_position_error_l2_rad": 7.0,
        "real_joint_velocity_error_l2_radps": 8.0,
    }
