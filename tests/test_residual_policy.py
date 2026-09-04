from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

import intact_tracking.residual_policy as residual_policy_module
from intact_tracking.cli.residual_policy_train import (
    _build_train_configuration,
    _validate_arguments,
    build_parser,
)
from intact_tracking.forward_predictor import ForwardDynamicsTransformer, ForwardPredictorConfig
from intact_tracking.residual_context import (
    DynamicsContextInference,
    FrozenContextCheckpoint,
    load_frozen_context_checkpoint,
)
from intact_tracking.residual_policy import FrozenTrackerResidualActor


class _FakeTracker(nn.Module):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__()
        del obs, kwargs
        self.obs_groups = list(obs_groups[obs_set])
        self.policy_input_dim = 3
        self.mlp = nn.Linear(3, output_dim)
        self.distribution = GaussianDistribution(output_dim, init_std=0.7)

    def populate_policy_context_cache(self, obs):
        del obs

    def get_latent(self, obs):
        return obs["features"]


def _fake_actor_checkpoint(path: Path) -> None:
    tracker = _FakeTracker(None, {"actor": ["features"]}, "actor", 2)
    tracker.distribution.std_param.data.copy_(torch.tensor([0.2, 0.4]))
    torch.save({"actor_state_dict": tracker.state_dict()}, path)


def test_residual_actor_starts_as_the_exact_frozen_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residual_policy_module,
        "SPV52HeightContactEstimatorActor",
        _FakeTracker,
    )
    checkpoint = tmp_path / "tracker.pt"
    _fake_actor_checkpoint(checkpoint)
    obs = TensorDict(
        {
            "features": torch.randn(4, 3),
            "dynamics_latent": torch.randn(4, 5),
        },
        batch_size=[4],
    )
    actor = FrozenTrackerResidualActor(
        obs,
        {"actor": ["features"]},
        "actor",
        2,
        tracker_checkpoint=str(checkpoint),
        tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["features"]},
        use_dynamics_latent=True,
        dynamics_latent_dim=5,
        residual_hidden_dims=(8, 4),
        residual_scale=0.25,
    )

    with torch.no_grad():
        expected = actor.tracker.mlp(actor.tracker.get_latent(obs))
        actual = actor(obs)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actor.distribution.std_param, torch.tensor([0.2, 0.4]))
    assert not any(parameter.requires_grad for parameter in actor.tracker.parameters())
    assert actor.policy_metrics(obs)["residual_action_rms"] == 0.0


class _CaptureEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.inputs = None

    def forward(self, history_state, history_action, current_state, history_valid):
        self.inputs = tuple(
            value.detach().clone()
            for value in (history_state, history_action, current_state, history_valid)
        )
        return history_state.new_zeros((history_state.size(0), self.latent_dim))


def test_context_ring_is_causal_and_clears_only_boundary_worlds() -> None:
    config = ForwardPredictorConfig(
        context_history_steps=10,
        transformer_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        context_dim=16,
        context_depth=1,
        context_heads=4,
        dynamics_latent_dim=8,
    )
    encoder = _CaptureEncoder(config.dynamics_latent_dim)
    checkpoint = FrozenContextCheckpoint(
        encoder=encoder,  # type: ignore[arg-type]
        config=config,
        state_mean=torch.zeros(71),
        state_std=torch.ones(71),
        action_mean=torch.zeros(29),
        action_std=torch.ones(29),
        path="unused",
        sha256="unused",
        tracker_sha256=None,
    )
    context = DynamicsContextInference(
        checkpoint,
        num_envs=2,
        device="cpu",
        use_bfloat16=False,
    )
    state = torch.zeros(2, 71)
    action = torch.zeros(2, 29)
    state[:, 0] = 1.0
    action[:, 0] = 10.0
    context.append(state, action, torch.tensor([False, False]))
    state[:, 0] = 2.0
    action[:, 0] = 20.0
    context.append(state, action, torch.tensor([True, False]))
    context.encode(torch.zeros(2, 71))

    assert encoder.inputs is not None
    ordered_state, ordered_action, _, ordered_valid = encoder.inputs
    assert ordered_valid[0].sum() == 0
    assert ordered_valid[1].sum() == 2
    torch.testing.assert_close(ordered_state[1, -2:, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(ordered_action[1, -2:, 0], torch.tensor([10.0, 20.0]))
    assert context.metrics["context_valid_fraction"] == pytest.approx(0.1)


def test_forward_checkpoint_loader_extracts_only_frozen_context_encoder(tmp_path: Path) -> None:
    config = ForwardPredictorConfig(
        context_history_steps=10,
        transformer_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        context_dim=16,
        context_depth=1,
        context_heads=4,
        dynamics_latent_dim=8,
    )
    model = ForwardDynamicsTransformer(config)
    path = tmp_path / "forward.pt"
    torch.save(
        {
            "architecture_version": config.architecture_version,
            "model_config": asdict(config),
            "model": model.state_dict(),
            "normalization": {
                "state_mean": [0.0] * 71,
                "state_std": [1.0] * 71,
                "action_mean": [0.0] * 29,
                "action_std": [1.0] * 29,
            },
            "tracker": {"checkpoint_sha256": "tracker-hash"},
        },
        path,
    )

    loaded = load_frozen_context_checkpoint(
        path,
        device="cpu",
        expected_tracker_sha256="tracker-hash",
    )

    assert loaded.config.dynamics_latent_dim == 8
    assert loaded.tracker_sha256 == "tracker-hash"
    assert not loaded.encoder.training
    assert not any(parameter.requires_grad for parameter in loaded.encoder.parameters())
    expected = model.context_encoder.state_dict()
    for name, value in loaded.encoder.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_residual_cli_enforces_two_unambiguous_baselines() -> None:
    parser = build_parser()
    common = [
        "--tracker-checkpoint",
        "tracker.pt",
        "--motion-file",
        "motion.npz",
        "--output-dir",
        "run",
    ]
    with pytest.raises(ValueError, match="requires --forward-checkpoint"):
        _validate_arguments(parser.parse_args([*common, "--baseline", "latent"]))
    with pytest.raises(ValueError, match="must not receive"):
        _validate_arguments(
            parser.parse_args(
                [
                    *common,
                    "--baseline",
                    "no-latent",
                    "--forward-checkpoint",
                    "forward.pt",
                ]
            )
        )


def test_train_configuration_reuses_spv52_observations_and_ppo_hyperparameters() -> None:
    source = OmegaConf.create(
        {
            "agent": {
                "num_steps_per_env": 24,
                "actor": {
                    "class_name": "old:Actor",
                    "distribution_cfg": {
                        "class_name": "GaussianDistribution",
                        "init_std": 1.0,
                    },
                },
                "critic": {"class_name": "old:Critic", "hidden_dims": [4, 3]},
                "algorithm": {
                    "class_name": "old:PPO",
                    "clip_param": 0.2,
                    "gamma": 0.99,
                    "lam": 0.95,
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                    "actor_learning_rate": 1.0e-3,
                    "critic_learning_rate": 5.0e-4,
                    "estimator_learning_rate": 5.0e-5,
                },
                "obs_groups": {"actor": ["base"], "critic": ["policy", "priv"]},
            },
            "task": {"agent_overrides": {"check_for_nan": False}},
        }
    )
    train = _build_train_configuration(
        source,
        tracker_checkpoint=Path("tracker.pt"),
        tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["base"], "critic": ["policy", "priv"]},
        baseline="latent",
        dynamics_latent_dim=64,
        residual_hidden_dims=(512, 256, 128),
        residual_scale=0.25,
        iterations=10,
        num_steps_per_env=24,
        save_interval=5,
        seed=42,
        logger="tensorboard",
        wandb_project="test",
        actor_learning_rate=None,
        critic_learning_rate=None,
        check_for_nan=False,
    )

    assert train["obs_groups"] == {
        "actor": ["base"],
        "critic": ["policy", "priv"],
    }
    assert train["actor"]["use_dynamics_latent"] is True
    assert train["critic"]["initial_checkpoint"] == "tracker.pt"
    assert train["algorithm"]["clip_param"] == 0.2
    assert train["algorithm"]["actor_learning_rate"] == 1.0e-3
    assert train["algorithm"]["critic_learning_rate"] == 5.0e-4
    assert "estimator_learning_rate" not in train["algorithm"]
