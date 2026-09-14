"""Exactly audited original-reward nominal controller as a frozen action prior."""

from __future__ import annotations

from pathlib import Path

import torch
from rsl_rl.modules import MLP

from intact_tracking.rollout.mjlab_adapter import _sha256

NOMINAL_SHA = "800da8c40016bba3263e685e9694e51e82b83a5e1c3df3e2e80543bd47ad8f1f"
TRACKER_SHA = "fd7bd90d5552e573bbbce1417e9b415c64bb487a76b683ba3c20503b5ec77635"
TRAINING_COMMIT = "692f90bf9dd9900229f8a5d201fb20e1db2c314d"


def load_frozen_nominal_prior(path, tracker, output_dim):
    if path is None:
        return None, 0.0, None
    path = Path(path).resolve()
    if _sha256(path) != NOMINAL_SHA:
        raise ValueError("Only the exact independently audited original-reward nominal prior is permitted")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta, cfg = checkpoint["residual_policy"], checkpoint["cfg"].agent.actor
    if (
        meta.get("version") != "spv52a_frozen_tracker_residual_v1"
        or meta.get("baseline") != "no-latent"
        or meta.get("tracker_sha256") != TRACKER_SHA
        or not meta.get("tracker_frozen")
        or not meta.get("nominal_physics", {}).get("enabled")
        or meta.get("payload", {}).get("enabled")
        or meta.get("reward_changes")
        or not cfg.class_name.endswith(":FrozenTrackerResidualActor")
        or cfg.get("use_dynamics_latent")
    ):
        raise ValueError("Nominal prior provenance or input contract mismatch")
    state = checkpoint["actor_state_dict"]
    for name, value in tracker.state_dict().items():
        source = state.get("tracker." + name)
        if source is None or not torch.equal(source, value.detach().cpu()):
            raise ValueError(f"Nominal prior tracker/preprocessing mismatch: {name}")
    prior = MLP(tracker.policy_input_dim, output_dim, cfg.residual_hidden_dims, "elu")
    prior.load_state_dict({k.removeprefix("residual_mlp."): v for k, v in state.items() if k.startswith("residual_mlp.")}, strict=True)
    prior.requires_grad_(False).eval()
    return prior, float(cfg.residual_scale), {
        "checkpoint": str(path), "sha256": NOMINAL_SHA,
        "saved_clean_training_commit": TRAINING_COMMIT,
        "tracker_and_preprocessing_tensors_bitwise_equal": len(tracker.state_dict()),
        "reward_provenance": "saved clean git snapshot; original environment/runtime unchanged since training commit; original residual training entrypoint never changes rewards; see docs/adaptation_nominal_prior_audit.md",
        "scope": "frozen source action prior, not a general exception for legacy initialization or shaped teachers",
    }
