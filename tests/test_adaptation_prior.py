from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from rsl_rl.modules import MLP

import intact_tracking.adaptation_prior as prior_module


def test_arbitrary_legacy_prior_is_rejected_before_loading(tmp_path):
    path = tmp_path / "not_the_audited_nominal.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="exact independently audited"):
        prior_module.load_frozen_nominal_prior(path, None, 3)


def test_prior_checks_every_tracker_tensor_and_freezes_only_exact_head(tmp_path, monkeypatch):
    tracker = torch.nn.Linear(5, 3)
    tracker.policy_input_dim = 5
    source_prior = MLP(5, 3, (8, 6), "elu")
    state = {"tracker." + k: v.clone() for k, v in tracker.state_dict().items()}
    state.update({"residual_mlp." + k: v.clone() for k, v in source_prior.state_dict().items()})
    cfg = OmegaConf.create({"agent": {"actor": {
        "class_name": "intact_tracking.residual_policy:FrozenTrackerResidualActor",
        "use_dynamics_latent": False, "residual_hidden_dims": [8, 6], "residual_scale": 0.25,
    }}})
    metadata = {"version": "spv52a_frozen_tracker_residual_v1", "baseline": "no-latent",
                "tracker_sha256": prior_module.TRACKER_SHA, "tracker_frozen": True,
                "nominal_physics": {"enabled": True}, "payload": {"enabled": False}}
    path = tmp_path / "fixture.pt"
    torch.save({"actor_state_dict": state, "cfg": cfg, "residual_policy": metadata}, path)
    monkeypatch.setattr(prior_module, "_sha256", lambda _: prior_module.NOMINAL_SHA)
    prior, scale, audit = prior_module.load_frozen_nominal_prior(path, tracker, 3)
    assert scale == 0.25
    assert not any(p.requires_grad for p in prior.parameters())
    sample = torch.randn(4, 5)
    torch.testing.assert_close(prior(sample), source_prior(sample), atol=0, rtol=0)
    assert audit["tracker_and_preprocessing_tensors_bitwise_equal"] == 2
    with torch.no_grad():
        tracker.bias.add_(0.01)
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        prior_module.load_frozen_nominal_prior(path, tracker, 3)
