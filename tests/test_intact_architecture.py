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
        action_block_size=2,
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
        "goal_observation": torch.randn(batch_size, config.observation_dim),
        "action": torch.randn(batch_size, horizon, config.action_block_dim),
        "previous_action": torch.randn(batch_size, horizon, config.action_block_dim),
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
    assert any(parameter.grad is not None for parameter in model.intent_actor.parameters())
    # There is exactly one actor object serving both physical and goal calls.
    actor_modules = [
        module for module in model.modules() if module.__class__.__name__ == "IntentActionActor"
    ]
    assert actor_modules == [model.intent_actor]


def test_goal_endpoint_is_detached_but_physical_successor_is_attached() -> None:
    embeddings = torch.randn(2, 6, 4, requires_grad=True)
    goal = torch.randn(2, 4, requires_grad=True)
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


def test_direct_plan_returns_action_blocks_without_search() -> None:
    config = _config()
    model = TrackingINTACT(config).eval()
    plan = model.direct_plan(
        observation=torch.randn(2, config.observation_dim),
        goal_observation=torch.randn(2, config.observation_dim),
        previous_action=torch.zeros(2, config.action_block_dim),
        context=torch.randn(2, 16, config.context_token_dim),
        context_mask=torch.ones(2, 16, dtype=torch.bool),
        horizon=3,
    )
    assert plan.shape == (2, 3, config.action_block_dim)
    assert torch.isfinite(plan).all()
