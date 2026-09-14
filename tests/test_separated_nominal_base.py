from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from intact_tracking.adaptation_policy import (
    ContextAdaptationActor,
    _base_with_optional_update,
    base_action_from_features,
)
from intact_tracking.context_export import _ContextInferenceCore


def test_oracle_features_never_change_separated_nominal_action_branch():
    torch.manual_seed(501)
    raw = torch.randn(4, 3)
    obs = {"height": torch.randn(4, 1), "contacts": torch.randn(4, 2), "reference": torch.randn(4, 3)}
    tracker = SimpleNamespace(
        estimator_target_group="height", foot_contact_target_group="contacts",
        reference_encoder_target_group="reference", policy_normalizer=nn.Identity(),
        _spv5_2_features=lambda inputs, h, c, r: r + h + c.sum(-1, keepdim=True),
        get_latent=lambda inputs: raw, mlp=nn.Linear(3, 2),
        distribution=SimpleNamespace(deterministic_output=lambda value: value),
    )
    actor = SimpleNamespace(
        tracker=tracker, oracle_tracking_features=True, train_base_policy=False,
        frozen_nominal_prior_deployable_base=True,
        frozen_nominal_prior_mlp=nn.Linear(3, 2), frozen_nominal_prior_scale=0.25,
    )
    features, action = _base_with_optional_update(actor, obs)
    expected = base_action_from_features(actor, raw)
    torch.testing.assert_close(action, expected, atol=0, rtol=0)
    for key in obs:
        obs[key] = obs[key] + 10
    changed, same_action = _base_with_optional_update(actor, obs)
    assert not torch.equal(changed, features)
    torch.testing.assert_close(same_action, expected, atol=0, rtol=0)


def test_separated_nominal_base_with_physics_gating_survives_residual_learning():
    from rsl_rl.modules import MLP

    from intact_tracking.adaptation_modulation import PhysicsLowRankResidual

    torch.manual_seed(503)
    raw = torch.randn(6, 3)
    oracle = torch.randn(6, 3)
    tracker = SimpleNamespace(
        mlp=MLP(3, 2, (8, 6), "elu").requires_grad_(False),
        distribution=SimpleNamespace(deterministic_output=lambda value: value),
    )
    actor = SimpleNamespace(
        tracker=tracker,
        frozen_nominal_prior_mlp=nn.Linear(3, 2).requires_grad_(False),
        frozen_nominal_prior_scale=0.25,
    )
    residual = PhysicsLowRankResidual(tracker.mlp, 7, 2, shared_rank=0)
    code = torch.randn(6, 7)
    source = base_action_from_features(actor, raw)
    source_weights = {key: value.clone() for key, value in tracker.mlp.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in residual.parameters() if p.requires_grad], lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        correction = residual(torch.cat((oracle, code), -1)).tanh()
        (source + correction - 1).square().mean().backward()
        optimizer.step()
    assert residual(torch.cat((oracle, code), -1)).count_nonzero() > 0
    for changed_oracle in (oracle, oracle + 10):
        nominal_correction = residual(torch.cat((changed_oracle, torch.zeros_like(code)), -1)).tanh()
        torch.testing.assert_close(source + nominal_correction, source, atol=0, rtol=0)
    assert all(torch.equal(value, source_weights[key]) for key, value in tracker.mlp.state_dict().items())


@pytest.mark.parametrize("separated", [False, True])
def test_student_and_export_use_identical_base_and_adapter_paths(separated):
    torch.manual_seed(502)
    actor = SimpleNamespace(
        tracker=SimpleNamespace(
            mlp=nn.Linear(3, 2),
            distribution=SimpleNamespace(deterministic_output=lambda value: value),
        ),
        frozen_nominal_prior_mlp=nn.Linear(3, 2), frozen_nominal_prior_scale=0.25,
        frozen_nominal_prior_deployable_base=separated,
    )
    core = _ContextInferenceCore.__new__(_ContextInferenceCore)
    nn.Module.__init__(core)
    core.base_mlp = actor.tracker.mlp
    core.base_output = nn.Identity()
    core.frozen_nominal_prior_mlp = actor.frozen_nominal_prior_mlp
    core.frozen_nominal_prior_scale = actor.frozen_nominal_prior_scale
    core.frozen_nominal_prior_deployable_base = separated
    core.feature_adapter = nn.Linear(5, 3)
    core.residual_mlp = nn.Linear(5, 2)
    core.residual_scale = 0.5
    core.refinement_mlp = None
    core.refinement_scale = 0.0
    core.context_residual_log_gain = None
    core.correction_mlp = None
    raw, latent = torch.randn(4, 3), torch.randn(4, 2)
    adapted = raw + core.feature_adapter(torch.cat((raw, latent), -1))
    base = ContextAdaptationActor.base_action_from_adapted_features(actor, raw, adapted)
    action_features = raw if separated else adapted
    torch.testing.assert_close(base, base_action_from_features(actor, action_features), atol=0, rtol=0)
    residual = 0.5 * core.residual_mlp(torch.cat((adapted, latent), -1)).tanh()
    torch.testing.assert_close(core.action(raw, latent), base + residual, atol=0, rtol=0)
    base_gradient = torch.autograd.grad(base.sum(), core.feature_adapter.weight, allow_unused=True, retain_graph=True)[0]
    assert (base_gradient is None) == separated
    # The adapter still receives action gradients through the NEW residual.
    gradient = torch.autograd.grad(residual.sum(), core.feature_adapter.weight)[0]
    assert gradient.abs().max() > 0
