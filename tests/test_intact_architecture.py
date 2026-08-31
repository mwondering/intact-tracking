from __future__ import annotations

import pytest
import torch

from intact_tracking.model import SIGReg, TrackingINTACT, TrackingINTACTConfig
from intact_tracking.objective import construct_intents, intact_objective


def _config() -> TrackingINTACTConfig:
    return TrackingINTACTConfig(
        observation_dim=8,
        proprio_dim=6,
        action_dim=2,
        effect_steps=2,
        context_chunk_steps=2,
        context_tokens=16,
        embed_dim=16,
        encoder_hidden_dim=32,
        context_depth=1,
        context_heads=4,
        forward_history=3,
        forward_depth=2,
        forward_heads=4,
        forward_mlp_dim=32,
        actor_hidden_dim=32,
        actor_depth=2,
    )


def _batch(config: TrackingINTACTConfig, batch_size: int = 3, horizon: int = 5):
    return {
        "observation": torch.randn(batch_size, horizon + 1, config.observation_dim),
        "goal_observation": torch.randn(batch_size, horizon, config.observation_dim),
        "forward_action": torch.randn(batch_size, horizon, config.forward_action_dim),
        "action": torch.randn(batch_size, horizon, config.action_dim),
        "previous_action": torch.randn(batch_size, horizon, config.action_dim),
        "context": torch.randn(batch_size, config.context_tokens, config.context_token_dim),
        "context_mask": torch.ones(batch_size, config.context_tokens, dtype=torch.bool),
        "transition_mask": torch.ones(batch_size, horizon, dtype=torch.bool),
        "physical_mask": torch.ones(batch_size, horizon, dtype=torch.bool),
        "goal_mask": torch.ones(batch_size, horizon, dtype=torch.bool),
    }


def test_paired_objective_is_finite_and_updates_shared_components() -> None:
    config = _config()
    model = TrackingINTACT(config)
    output = intact_objective(model, _batch(config), sigreg=SIGReg(num_proj=16))
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert any(
        parameter.grad is not None for parameter in model.forward_action_encoder.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.previous_action_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.intent_actor.parameters())
    # There is exactly one actor object serving both physical and goal calls.
    actor_modules = [
        module for module in model.modules() if module.__class__.__name__ == "IntentActionActor"
    ]
    assert actor_modules == [model.intent_actor]


def test_forward_diagnostics_are_observational_and_action_scaled() -> None:
    config = _config()
    model = TrackingINTACT(config)
    batch = _batch(config)
    torch.manual_seed(123)
    unit_output = intact_objective(model, batch, sigreg=SIGReg(num_proj=16))

    scaled_batch = dict(batch)
    scaled_batch["action_scale"] = torch.full(
        (batch["action"].size(0), config.action_dim), 3.0
    )
    torch.manual_seed(123)
    scaled_output = intact_objective(model, scaled_batch, sigreg=SIGReg(num_proj=16))

    diagnostic_names = (
        "forward_copy_mse",
        "forward_vs_copy_ratio",
        "forward_state_cosine_similarity",
        "forward_decoded_action_mae",
        "forward_action_consistency_mae",
        "forward_decoded_action_mae_env",
        "forward_decoded_action_rmse_env",
        "forward_action_consistency_mae_env",
        "physical_action_mae_env",
        "goal_action_mae_env",
    )
    assert all(torch.isfinite(unit_output[name]) for name in diagnostic_names)
    assert all(not unit_output[name].requires_grad for name in diagnostic_names)
    assert torch.allclose(unit_output["loss"], scaled_output["loss"])
    assert torch.allclose(
        scaled_output["forward_decoded_action_mae_env"],
        3 * unit_output["forward_decoded_action_mae"],
    )
    assert torch.allclose(
        scaled_output["forward_action_consistency_mae_env"],
        3 * unit_output["forward_action_consistency_mae"],
    )
    assert torch.allclose(
        scaled_output["physical_action_mae_env"], 3 * unit_output["physical_mae"]
    )
    assert torch.allclose(
        scaled_output["goal_action_mae_env"], 3 * unit_output["goal_mae"]
    )


def test_goal_endpoint_is_detached_but_physical_successor_is_attached() -> None:
    embeddings = torch.randn(2, 6, 4, requires_grad=True)
    goal = torch.randn(2, 5, 4, requires_grad=True)
    physical, deployment = construct_intents(embeddings, goal)
    loss = physical.square().mean() + deployment.square().mean()
    loss.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum() > 0
    assert goal.grad is None


def test_four_slot_grammar_has_expected_width() -> None:
    config = _config()
    model = TrackingINTACT(config)
    z = torch.randn(2, config.embed_dim)
    intent = torch.randn_like(z)
    previous = torch.randn(2, config.embed_dim)
    features = model.intent_actor.actor_features(z, intent, previous)
    assert features.shape == (2, 4 * config.embed_dim)
    assert torch.equal(features[:, : config.embed_dim], z)
    assert torch.equal(features[:, 2 * config.embed_dim : 3 * config.embed_dim], z * intent)


def test_context_count_is_fixed_at_sixteen() -> None:
    config = _config()
    model = TrackingINTACT(config)
    with pytest.raises(ValueError, match="Context contract"):
        model.encode_context(
            torch.randn(2, 15, config.context_token_dim),
            torch.ones(2, 15, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="fixed at 16"):
        TrackingINTACTConfig(context_tokens=15)


def test_direct_control_returns_one_action_without_forward_rollout() -> None:
    config = _config()
    model = TrackingINTACT(config).eval()
    forward_calls: list[bool] = []
    hook = model.predictor.register_forward_pre_hook(
        lambda _module, _inputs: forward_calls.append(True)
    )
    plan = model.direct_plan(
        observation=torch.randn(2, config.observation_dim),
        goal_observation=torch.randn(2, config.observation_dim),
        previous_action=torch.zeros(2, config.action_dim),
        context=torch.randn(2, 16, config.context_token_dim),
        context_mask=torch.ones(2, 16, dtype=torch.bool),
        horizon=1,
    )
    hook.remove()
    assert plan.shape == (2, 1, config.action_dim)
    assert torch.isfinite(plan).all()
    assert not forward_calls
    with pytest.raises(ValueError, match="horizon=1"):
        model.direct_plan(
            observation=torch.randn(2, config.observation_dim),
            goal_observation=torch.randn(2, config.observation_dim),
            previous_action=torch.zeros(2, config.action_dim),
            context=torch.randn(2, 16, config.context_token_dim),
            horizon=2,
        )
