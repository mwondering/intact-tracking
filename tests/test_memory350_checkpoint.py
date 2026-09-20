from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from intact_tracking.limb_context_sampling import state_digest
from intact_tracking.memory350_checkpoint import (
    BUNDLE_KEYS, dependencies_from_legacy_checkpoint, embed_checkpoint_file,
    embedded_tracker, load_policy_context,
)
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_model import HierarchicalContextEncoder
from intact_tracking.memory350_proprio_inputs import INPUT_CONTRACT, ProprioMemory350Config
from intact_tracking.residual_policy import FrozenTrackerResidualActor
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import _sha256
from test_residual_policy import _FakeTracker


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    import intact_tracking.residual_policy as policy

    torch.set_num_threads(1)
    monkeypatch.setattr(policy, "SPV52HeightContactEstimatorActor", _FakeTracker)
    tracker = _FakeTracker(None, {"actor": ["features"]}, "actor", 2)
    tracker_path = tmp_path / "tracker.pt"
    torch.save({"actor_state_dict": tracker.state_dict(), "cfg": {"task": {}, "agent": {}}},
               tracker_path)
    config = ProprioMemory350Config(context_dim=16, context_heads=4, context_depth=1,
                                   chunk_depth=1, memory_depth=1)
    encoder = HierarchicalContextEncoder(config).eval()
    context_path = tmp_path / "encoder.pt"
    torch.save({
        "architecture_version": config.architecture_version, "model_config": asdict(config),
        "model": {**{f"context_encoder.{k}": v for k, v in encoder.state_dict().items()},
                  "predictor.unused": torch.ones(3)},
        "optimizer": {"unneeded": torch.ones(3)},
        "normalization": {"context_state_mean": [.1] * 122, "context_state_std": [.8] * 122,
                          "context_action_mean": [.2] * 29, "context_action_std": [.9] * 29},
        "context_input_contract": deepcopy(INPUT_CONTRACT),
        "tracker": {"checkpoint_sha256": _sha256(tracker_path)},
    }, context_path)
    obs = TensorDict({"features": torch.randn(4, 3), "dynamics_latent": torch.randn(4, 64)}, [4])
    kwargs = dict(tracker_checkpoint=str(tracker_path), tracker_actor_kwargs={},
                  tracker_obs_groups={"actor": ["features"]}, use_dynamics_latent=True,
                  residual_hidden_dims=(8, 4), residual_output_mode="unbounded", residual_scale=1.)
    actor = FrozenTrackerResidualActor(obs, {"actor": ["features"]}, "actor", 2, **kwargs)
    optimizer = torch.optim.Adam(actor.parameters())
    actor(obs).square().mean().backward()
    optimizer.step()
    weights = actor.state_dict()
    rsl = {"actor_state_dict": weights, "optimizer_state_dict": optimizer.state_dict()}
    checkpoint = {
        **rsl, "policy": weights, "rsl_rl": rsl, "completed_updates": 101,
        "iter": 100, "cfg": OmegaConf.create({"agent": {"actor": kwargs}}),
        "env": {"common_step_counter": 2424},
        "motion_sampling_state": {"visits": torch.arange(32), "fails": torch.arange(32) % 3},
        "residual_policy": {"context_checkpoint": str(context_path),
                            "context_sha256": _sha256(context_path),
                            "tracker_checkpoint": str(tracker_path),
                            "tracker_sha256": _sha256(tracker_path)},
    }
    return checkpoint, actor, obs, kwargs, context_path, tracker_path


def test_portable_context_and_actor_match_after_original_files_are_deleted(legacy):
    state, actor, obs, kwargs, context_path, tracker_path = legacy
    original = load_memory350_checkpoint(context_path, device="cpu")
    state.update(dependencies_from_legacy_checkpoint(state))
    context_path.unlink()
    tracker_path.unlink()
    restored = load_policy_context(state, device="cpu")
    tracker = embedded_tracker(state)
    loaded_actor = FrozenTrackerResidualActor(
        obs.clone(), {"actor": ["features"]}, "actor", 2,
        tracker_state_dict=tracker["actor_state_dict"], **kwargs)
    loaded_actor.load_state_dict(state["actor_state_dict"], strict=True)
    torch.testing.assert_close(loaded_actor(obs), actor(obs), atol=0, rtol=0)
    for key in ("state_mean", "state_std", "action_mean", "action_std"):
        torch.testing.assert_close(getattr(restored, key), getattr(original, key), atol=0, rtol=0)
    inputs = (torch.randn(2, 50, 122), torch.randn(2, 50, 29), torch.randn(2, 50, 122),
              torch.ones(2, 50, dtype=torch.bool), torch.randn(2, 30, 10, 273),
              torch.ones(2, 30, dtype=torch.bool))
    with torch.no_grad():
        torch.testing.assert_close(restored.encoder(*inputs), original.encoder(*inputs), atol=0, rtol=0)
    assert not restored.encoder.training
    assert not any(p.requires_grad for p in restored.encoder.parameters())
    assert not any(p.requires_grad for p in loaded_actor.tracker.parameters())
    assert "optimizer" not in state["frozen_context"]["payload"]
    assert all(k.startswith("context_encoder.") for k in state["frozen_context"]["payload"]["model"])


def test_legacy_loading_remains_available_and_checks_source_hash(legacy):
    state, _, _, _, context_path, _ = legacy
    assert load_policy_context(state, device="cpu").state_mean.numel() == 122
    assert embedded_tracker(state) is None
    source = torch.load(context_path, weights_only=False)
    source["normalization"]["context_state_mean"][0] = .3
    torch.save(source, context_path)
    with pytest.raises(ValueError, match="changed"):
        load_policy_context(state, device="cpu")


@pytest.mark.parametrize("target", ["encoder", "tracker", "context_source", "config", "contract", "missing"])
def test_corrupt_embedded_dependencies_fail_without_external_fallback(legacy, target):
    state = legacy[0]
    state.update(dependencies_from_legacy_checkpoint(state))
    if target == "encoder":
        next(iter(state["frozen_context"]["payload"]["model"].values())).add_(1)
    elif target == "tracker":
        state["actor_state_dict"]["tracker.mlp.weight"].add_(1)
    elif target == "context_source":
        state["frozen_context"]["source_sha256"] = "wrong"
    elif target == "config":
        state["frozen_tracker"]["cfg"]["task"]["changed"] = True
    elif target == "contract":
        bundle = state["frozen_context"]
        bundle["payload"]["context_input_contract"]["action_source"] = "wrong"
        bundle["payload_sha256"] = state_digest(bundle["payload"])
    else:
        del state["frozen_context"]
    with pytest.raises(ValueError):
        embedded_tracker(state)
        load_policy_context(state, device="cpu")


def test_atomic_embedding_preserves_optimizer_sampler_and_all_original_state(legacy, tmp_path):
    state = legacy[0]
    path = tmp_path / "checkpoint_100.pt"
    torch.save(state, path)
    result = embed_checkpoint_file(path)
    restored = torch.load(path, weights_only=False)
    assert result["original_state_preserved"]
    assert state_digest({k: v for k, v in restored.items() if k not in BUNDLE_KEYS}) == state_digest(state)
    assert embed_checkpoint_file(path)["already_embedded"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_packaging_leaves_original_file_untouched(legacy, tmp_path, monkeypatch):
    state = legacy[0]
    path = tmp_path / "checkpoint_100.pt"
    torch.save(state, path)
    digest = _sha256(path)
    dependencies = dependencies_from_legacy_checkpoint(state)

    def fail(_state, stream):
        stream.write(b"partial")
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        embed_checkpoint_file(path, dependencies)
    assert _sha256(path) == digest
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_trainer_replacement_is_never_overwritten(legacy, tmp_path, monkeypatch):
    state = legacy[0]
    path = tmp_path / "checkpoint_final.pt"
    torch.save(state, path)
    dependencies = dependencies_from_legacy_checkpoint(state)
    save = torch.save

    def replace_during_save(value, stream):
        save(value, stream)
        replacement = tmp_path / "new.pt"
        save({**state, "completed_updates": 102}, replacement)
        replacement.replace(path)

    monkeypatch.setattr(torch, "save", replace_during_save)
    with pytest.raises(RuntimeError, match="changed during packaging"):
        embed_checkpoint_file(path, dependencies)
    assert torch.load(path, weights_only=False)["completed_updates"] == 102


def test_runner_saves_cached_dependencies_without_reopening_sources(legacy, tmp_path):
    state, _, _, _, context_path, tracker_path = legacy
    dependencies = dependencies_from_legacy_checkpoint(state)
    context_path.unlink()
    tracker_path.unlink()
    runner = object.__new__(ResidualOnPolicyRunner)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=2424))
    runner.alg = SimpleNamespace(save=lambda: state["rsl_rl"])
    runner.checkpoint_cfg, runner.residual_metadata = state["cfg"], state["residual_policy"]
    runner.current_learning_iteration, runner.completed_learning_updates = 100, 101
    runner.frozen_dependencies, runner.cfg = dependencies, {}
    path = tmp_path / "checkpoint.pt"
    runner.save(str(path))
    restored = torch.load(path, weights_only=False)
    assert load_policy_context(restored, device="cpu").state_mean.numel() == 122
    assert embedded_tracker(restored) is not None
    assert state_digest(restored["rsl_rl"]) == state_digest(state["rsl_rl"])
