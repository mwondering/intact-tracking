from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

import intact_tracking.residual_policy as residual_module
from intact_tracking.cli.simulator_preview_train import (
    build_parser,
    validate_arguments,
    validate_resume,
)
from intact_tracking.residual_policy import FrozenTrackerResidualActor
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.simulator_preview import preview_features
from intact_tracking.simulator_preview_experiment import (
    CRITIC_HIDDEN_DIMS,
    EXPERIMENT_VERSION,
    PREVIEW_DIM,
    PREVIEW_GROUP,
    PreviewObservationWrapper,
    PreviewResidualActor,
    ScratchHeftCritic,
    configure_preview_comparison,
)


def configuration():
    groups = {"actor": ["features"], "critic": ["policy", "priv"]}
    return {
        "actor": {
            "class_name": "intact_tracking.residual_policy:FrozenTrackerResidualActor",
            "tracker_obs_groups": groups, "use_dynamics_latent": False,
            "residual_input_mode": "tracker_features",
        },
        "critic": {"initial_checkpoint": "MUST_NOT_LOAD", "hidden_dims": [1024, 512, 512]},
        "obs_groups": {key: list(value) for key, value in groups.items()},
        "algorithm": {"learning_rate": 1e-4},
    }


def test_default_4096_and_user_must_choose_training_budget():
    parser = build_parser()
    args = parser.parse_args(["--variant", "preview", "--output-dir", "unused", "--iterations", "9000"])
    assert args.num_envs == 4096 and args.iterations == 9000
    validate_arguments(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "baseline", "--output-dir", "unused"])
    args.iterations = 0
    with pytest.raises(ValueError, match="iterations"):
        validate_arguments(args)


def test_v2_features_omit_both_distance_scalars():
    state = torch.zeros(2, 5, 71)
    state[..., 3] = 1
    target = state.clone()
    state[..., 13] = 0.4
    reference, outcome = preview_features(state, target, state[:, 0])
    assert reference.shape == (2, 355) and outcome.shape == (2, 710)
    assert PREVIEW_DIM == 1065
    torch.testing.assert_close(outcome.reshape(2, 5, 142)[..., 71 + 13], torch.full((2, 5), 0.4))


@pytest.mark.parametrize("variant", ["preview", "baseline"])
def test_comparison_has_exact_original_groups_and_single_scratch_critic(variant):
    original = configuration()
    train = configure_preview_comparison(original, variant)
    extra = [PREVIEW_GROUP] if variant == "preview" else []
    assert train["obs_groups"]["actor"] == ["features", *extra]
    assert train["obs_groups"]["critic"] == ["policy", "priv", *extra]
    assert "initial_checkpoint" not in train["critic"]
    assert train["critic"]["hidden_dims"] == [1024, 512, 512, 256]
    assert train["critic"]["class_name"].endswith(":ScratchHeftCritic")
    assert original == configuration()  # no mutation of the source configuration


@pytest.mark.parametrize("extra", [0, PREVIEW_DIM])
def test_critic_uses_requested_widths_random_init_and_all_inputs(monkeypatch, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError("Scratch critic must not load any checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    obs = TensorDict({"policy": torch.randn(2, 1728), "priv": torch.randn(2, 4602)}, [2])
    groups = {"critic": ["policy", "priv"]}
    if extra:
        obs[PREVIEW_GROUP] = torch.randn(2, extra, requires_grad=True)
        groups["critic"].append(PREVIEW_GROUP)
    critic = ScratchHeftCritic(obs, groups, "critic", 1)
    linear = [module for module in critic.mlp if isinstance(module, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (6330 + extra, 1024), (1024, 512), (512, 512), (512, 256), (256, 1),
    ]
    assert tuple(critic._modules) == ("obs_normalizer", "mlp")
    assert all(parameter.requires_grad for parameter in critic.parameters())
    assert linear[-1].weight.count_nonzero() > 0  # not zero-output value warm start
    value = critic(obs)
    assert value.shape == (2, 1) and torch.isfinite(value).all()
    if extra:
        gradient = torch.autograd.grad(value.sum(), obs[PREVIEW_GROUP])[0]
        assert gradient.abs().sum() > 0
    with pytest.raises(TypeError, match="initial_checkpoint"):
        ScratchHeftCritic(obs, groups, "critic", 1, initial_checkpoint="forbidden")
    with pytest.raises(ValueError, match="widths"):
        ScratchHeftCritic(obs, groups, "critic", 1, hidden_dims=(256, 128))


def test_wrapper_adds_only_future_tensor_without_current_privileges():
    reference, outcome = torch.randn(2, 355), torch.randn(2, 710)
    wrapper = SimpleNamespace(
        preview=SimpleNamespace(query=lambda tracker, obs: (reference, outcome)),
        tracker=object(), preview_mode="true", num_envs=2,
    )
    original = torch.randn(2, 3)
    obs = TensorDict({"features": original.clone()}, [2])
    result = PreviewObservationWrapper.attach(wrapper, obs)
    assert set(result.keys()) == {"features", PREVIEW_GROUP}
    torch.testing.assert_close(result["features"], original, atol=0, rtol=0)
    torch.testing.assert_close(result[PREVIEW_GROUP], torch.cat((reference, outcome), -1))
    wrapper.preview_mode = "zero"
    assert PreviewObservationWrapper.attach(wrapper, obs)[PREVIEW_GROUP].count_nonzero() == 0


class FakeTracker(torch.nn.Module):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__()
        self.obs_groups = list(obs_groups[obs_set])
        self.policy_input_dim = 3
        self.mlp = torch.nn.Linear(3, output_dim)
        self.distribution = GaussianDistribution(output_dim, init_std=0.1)

    def populate_policy_context_cache(self, obs):
        pass

    def get_latent(self, obs):
        return obs["features"]


def test_actor_uses_raw_preview_and_freezes_original_tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(residual_module, "SPV52HeightContactEstimatorActor", FakeTracker)
    path = tmp_path / "tracker.pt"
    tracker = FakeTracker(None, {"actor": ["features"]}, "actor", 2)
    torch.save({"actor_state_dict": tracker.state_dict()}, path)
    config = configure_preview_comparison(configuration(), "preview")
    kwargs = {key: value for key, value in config["actor"].items() if key != "class_name"}
    kwargs.update(tracker_checkpoint=str(path), tracker_actor_kwargs={}, residual_hidden_dims=(16, 8))
    obs = TensorDict({"features": torch.randn(4, 3), PREVIEW_GROUP: torch.randn(4, PREVIEW_DIM)}, [4])
    actor = PreviewResidualActor(obs, config["obs_groups"], "actor", 2, **kwargs)
    actor.train()
    assert not actor.tracker.training
    assert all(not p.requires_grad for p in actor.tracker.parameters())
    assert actor.residual_mlp[0].in_features == 3 + PREVIEW_DIM
    assert actor.obs_groups == ["features", PREVIEW_GROUP]
    torch.testing.assert_close(actor(obs), tracker.mlp(obs["features"]), atol=0, rtol=0)
    before = {key: value.clone() for key, value in actor.tracker.state_dict().items()}
    actor.update_normalization(obs)
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    actor(obs).square().mean().backward()
    optimizer.step()
    altered = obs.clone(recurse=True)
    altered[PREVIEW_GROUP] = obs[PREVIEW_GROUP].roll(1, 0)
    assert not torch.equal(actor(obs), actor(altered))
    for key, value in actor.tracker.state_dict().items():
        assert torch.equal(value, before[key])
    # The control truly has no preview input, not an unused padded latent slot.
    baseline_kwargs = dict(kwargs, use_dynamics_latent=False, dynamics_latent_dim=0)
    baseline = FrozenTrackerResidualActor(obs, {"actor": ["features"]}, "actor", 2, **baseline_kwargs)
    assert baseline.residual_mlp[0].in_features == 3
    torch.testing.assert_close(baseline(obs), baseline(altered), atol=0, rtol=0)


def test_resume_rejects_old_architecture_and_mismatched_variant():
    train = configure_preview_comparison(configuration(), "preview")
    metadata = {
        "version": EXPERIMENT_VERSION, "variant": "preview", "tracker_sha256": "tracker",
        "motion_sha256": "motion", "physics_mode": "dr",
        "arguments": {"num_envs": 4096, "seed": 121, "rollout_steps": 24, "initial_action_std": 0.1},
    }
    checkpoint = {"residual_policy": metadata, "cfg": OmegaConf.create({"agent": train})}
    validate_resume(checkpoint, metadata, train)
    with pytest.raises(ValueError, match="v2"):
        validate_resume({"residual_policy": {"version": "old"}}, metadata, train)
    with pytest.raises(ValueError, match="variant"):
        validate_resume(checkpoint, {**metadata, "variant": "baseline"}, train)


def test_stop_signal_finishes_update_before_checkpoint():
    obs = TensorDict({"x": torch.zeros(2, 1)}, [2])
    calls = []
    runner = object.__new__(ResidualOnPolicyRunner)
    runner.stop_requested = False
    runner.completed_learning_updates = 0
    runner.current_learning_iteration = 0
    runner.device = "cpu"
    runner.is_distributed = False
    runner.gpu_global_rank = 0
    runner.cfg = {"num_steps_per_env": 2, "save_interval": 1, "check_for_nan": True}

    def step(actions):
        calls.append("step")
        runner.request_stop()
        return obs, torch.ones(2), torch.zeros(2, dtype=torch.bool), {}

    runner.env = SimpleNamespace(get_observations=lambda: obs, step=step, device="cpu")
    runner.alg = SimpleNamespace(
        train_mode=lambda: None, act=lambda value: torch.zeros(2, 1),
        process_env_step=lambda *args: None, compute_returns=lambda value: None,
        update=lambda: calls.append("update") or {}, learning_rate=1e-4,
        get_policy=lambda: SimpleNamespace(output_std=torch.ones(1)),
    )
    runner.logger = SimpleNamespace(
        init_logging_writer=lambda: None, process_env_step=lambda *args: None,
        log=lambda **kwargs: None, writer=object(), log_dir="unused",
        stop_logging_writer=lambda: calls.append("close_writer"),
    )
    runner._begin_adaptive_sampling_iteration = lambda iteration: None
    runner._record_policy_action_mean = lambda: None
    runner._policy_diagnostics = lambda value: {}
    runner.save = lambda path, **kwargs: calls.append((path, runner.completed_learning_updates))
    runner.learn(100)
    assert calls[:3] == ["step", "step", "update"]
    assert calls.count("update") == 1
    assert ("unused/checkpoint_interrupted.pt", 1) in calls
    assert ("unused/checkpoint_final.pt", 1) in calls
    assert calls[-1] == "close_writer"


def test_critic_width_constant_is_not_legacy_bottleneck():
    assert CRITIC_HIDDEN_DIMS == (1024, 512, 512, 256)


def test_runner_isolates_serializable_configuration_from_rsl_factory(monkeypatch):
    from mjlab.rl.runner import MjlabOnPolicyRunner

    config = configuration()

    def mutating_factory(self, env, cfg, log_dir, device):
        cfg["actor"].pop("class_name")
        cfg["critic"].clear()
        cfg["algorithm"].clear()

    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", mutating_factory)
    ResidualOnPolicyRunner(object(), config, "unused", "cpu",
                           checkpoint_cfg=OmegaConf.create({"agent": config}), residual_metadata={})
    assert config == configuration()
