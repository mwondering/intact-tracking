from copy import deepcopy

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage

from intact_tracking.memory350_tracker_action_policy import (
    ACTION_GROUP, LATENT_HISTORY_DIM, TrackerActionResidualActor, TrackerActionCritic,
    TrackerActionPPO, configure_tracker_action_models, audit_tracker_action_models,
)


class FakeTracker(nn.Module):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.policy_input_dim = 3
        self.mlp = nn.Linear(3, output_dim)
        self.distribution = GaussianDistribution(output_dim, init_std=.7)

    def populate_policy_context_cache(self, obs):
        pass

    def get_latent(self, obs):
        return obs["features"]

    def forward(self, obs):
        return self.mlp(self.get_latent(obs))


def observation(offset=0):
    return TensorDict({"features": torch.randn(8, 3) + offset, "priv": torch.randn(8, 7),
                       "dynamics_latent": torch.randn(8, LATENT_HISTORY_DIM)}, [8])


@pytest.fixture
def models(tmp_path, monkeypatch):
    import intact_tracking.residual_policy as policy
    monkeypatch.setattr(policy, "SPV52HeightContactEstimatorActor", FakeTracker)
    torch.manual_seed(123)
    tracker = FakeTracker(None, {"actor": ["features"]}, "actor", 29)
    path = tmp_path / "tracker.pt"
    torch.save({"actor_state_dict": tracker.state_dict()}, path)

    def make(**actor_options):
        obs = observation()
        groups = {"actor": ["features"], "critic": ["priv"]}
        actor = TrackerActionResidualActor(
            obs, groups, "actor", 29, tracker_checkpoint=str(path), tracker_actor_kwargs={},
            tracker_obs_groups=groups, residual_hidden_dims=(16, 8), initialization_seed=10128,
            **{"initial_action_std": .25, **actor_options})
        critic = TrackerActionCritic(obs, groups, "critic", 1, initial_checkpoint=None,
                                    initialization_seed=20124, hidden_dims=(16, 8))
        return actor, critic, obs

    return make


def test_initial_models_receive_same_current_tracker_output_and_zero_residual(models):
    from intact_tracking.cli.memory350_policy_train import audit_initial_models
    actor, critic, obs = models()
    critic.update_normalization(obs)
    expected = actor.tracker(obs).detach()
    torch.testing.assert_close(obs[ACTION_GROUP], expected, rtol=0, atol=0)
    torch.testing.assert_close(actor(obs), expected, rtol=0, atol=0)
    features, base = actor._base_features_and_action(obs)
    torch.testing.assert_close(actor._residual_input(obs, features, base),
                               torch.cat((features, obs["dynamics_latent"], expected), -1))
    torch.testing.assert_close(critic.value_input(obs), torch.cat((
        critic.obs_normalizer(obs["priv"]), obs["dynamics_latent"], expected), -1))
    actor.train()
    assert not actor.tracker.training and not any(p.requires_grad for p in actor.tracker.parameters())
    report = audit_tracker_action_models(actor, critic, obs, "concat", original_audit=audit_initial_models)
    assert report["latent_dimensions"] == 320 and report["latent_history_frames"] == 5
    assert report["actor_input_dimensions"] == 3 + 320 + 29
    assert report["critic_input_dimensions"] == 7 + 320 + 29


def test_both_heads_learn_from_action_and_all_five_latents_without_encoder_tracker_gradients(models):
    actor, critic, obs = models()
    obs["dynamics_latent"].requires_grad_()
    obs[ACTION_GROUP].requires_grad_()
    obs["features"].requires_grad_()
    # Model a learned nonzero residual output. At scratch initialization the
    # zero output head deliberately blocks first-step gradients to its trunk.
    with torch.no_grad():
        actor.residual_mlp.base[-1].weight.normal_(std=.05)
        actor.residual_mlp.latent_input.weight.normal_(std=.05)
        critic.mlp.latent_input.weight.normal_(std=.05)
    altered = obs.clone()
    altered[ACTION_GROUP] = obs[ACTION_GROUP] + 1
    features, action = actor._base_features_and_action(obs)
    residual = actor._residual(actor._residual_input(obs, features, action))
    changed_residual = actor._residual(actor._residual_input(obs, features, altered[ACTION_GROUP]))
    assert not torch.allclose(residual, changed_residual)
    assert not torch.allclose(critic(obs), critic(altered))
    (actor(obs).square().mean() + critic(obs).square().mean()).backward()
    for name in ("dynamics_latent", ACTION_GROUP, "features"):
        assert obs[name].grad is None
    assert all(p.grad is None for p in actor.tracker.parameters())
    for mlp in (actor.residual_mlp, critic.mlp):
        gradient = mlp.latent_input.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert all(gradient[:, i * 64:(i + 1) * 64].abs().sum() > 0 for i in range(5))
        assert gradient[:, 320:].abs().sum() > 0


def _algorithm(actor, critic, obs, **kwargs):
    storage = RolloutStorage("rl", 8, 2, obs, [29])
    return TrackerActionPPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2,
                            schedule="fixed", actor_learning_rate=1e-3, critic_learning_rate=1e-3,
                            **kwargs)


def test_native_exploration_and_unbounded_residual_keep_gradients_and_legacy_checkpoints(models):
    from intact_tracking.cli.memory350_policy_train import audit_initial_models
    from intact_tracking.cli.memory350_proprio_native_policy_train import build_parser

    args = build_parser().parse_args([
        "--fusion", "concat", "--context-checkpoint", "new.pt", "--output-dir", "unused"])
    assert args.entropy_coef == .005
    assert args.initial_action_std == 1.
    assert args.residual_output_mode == "unbounded" and args.residual_scale == 1.
    actor, critic, obs = models(initial_action_std=args.initial_action_std,
                                residual_output_mode=args.residual_output_mode,
                                residual_scale=args.residual_scale)
    critic.update_normalization(obs)
    audit = audit_tracker_action_models(actor, critic, obs, "concat", original_audit=audit_initial_models)
    assert audit["action_std_initialization"] == 1.
    algorithm = _algorithm(actor, critic, obs, entropy_coef=args.entropy_coef)
    assert algorithm.entropy_coef == .005
    torch.testing.assert_close(actor.output_std, torch.ones_like(actor.output_std), rtol=0, atol=0)

    # Large corrections of both signs must reach the command unchanged and
    # retain unit output derivative even where the old tanh was saturated.
    output = actor.residual_mlp.base[-1]
    correction = torch.linspace(-10, 10, 29)
    with torch.no_grad():
        output.bias.copy_(correction)
    action = actor(obs)
    torch.testing.assert_close(action - actor.last_base_action, correction.expand_as(action))
    action.sum().backward()
    torch.testing.assert_close(output.bias.grad, torch.full_like(output.bias, 8.), rtol=0, atol=0)
    assert all(p.grad is None for p in actor.tracker.parameters())
    # A stochastic forward cannot turn exploration noise into mean diagnostics.
    actor(obs, stochastic_output=True)
    metrics = actor.policy_metrics(obs)
    assert metrics["residual_output_bounded"] == 0.
    assert "residual_saturation_fraction" not in metrics
    assert metrics["residual_action_abs_max"] == 10.
    assert metrics["residual_action_rms"] == pytest.approx(float(correction.square().mean().sqrt()))
    assert metrics["residual_action_abs_mean"] == pytest.approx(float(correction.abs().mean()))

    # Old serialized constructor configs have no output-mode key. Loading the
    # same weights must still reproduce their original 0.25*tanh action.
    legacy, _, _ = models()
    legacy.load_state_dict(actor.state_dict(), strict=True)
    legacy(obs)
    torch.testing.assert_close(legacy.last_residual_mean,
                               (.25 * correction.tanh()).expand_as(action), rtol=0, atol=0)
    assert legacy.policy_metrics(obs)["residual_output_bounded"] == 1.


@pytest.mark.parametrize("output_mode", ["bounded", "unbounded"])
def test_rollout_minibatch_bootstrap_and_resume_keep_actions_aligned(models, tmp_path, output_mode):
    options = {"residual_output_mode": output_mode,
               "residual_scale": 1. if output_mode == "unbounded" else .25,
               "initial_action_std": 1. if output_mode == "unbounded" else .25}
    actor, critic, initial = models(**options)
    algorithm = _algorithm(actor, critic, initial, entropy_coef=.005)
    frozen = deepcopy(actor.tracker.state_dict())
    expected_actions = []
    seen_by_critic = []
    handle = critic.register_forward_pre_hook(
        lambda _, args: seen_by_critic.append(args[0][ACTION_GROUP].clone()))
    with torch.inference_mode():
        obs = observation()  # First rollout need not reuse the constructor's obs.
        for step in range(2):
            expected_actions.append(actor.tracker(obs).clone())
            sampled = algorithm.act(obs)
            torch.testing.assert_close(seen_by_critic[-1], expected_actions[-1], rtol=0, atol=0)
            assert not torch.equal(sampled, expected_actions[-1])  # Never store exploration as tracker output.
            obs = observation(step + 1)
            dones = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]) if step == 0 else torch.zeros(8)
            algorithm.process_env_step(obs, torch.randn(8), dones,
                                       {"time_outs": dones.clone(),
                                        "motion_resample_boundary": torch.zeros(8, dtype=torch.bool)})
        stored = algorithm.storage.observations[ACTION_GROUP]
        torch.testing.assert_close(stored, torch.stack(expected_actions), rtol=0, atol=0)
        # Explicitly exercise bootstrap without actor.act or process_env_step on
        # this next state. A mutable last_base_action would be incorrect here.
        final = observation(10)
        actor.last_base_action = torch.full((8, 29), -999.)
        algorithm.compute_returns(final)
        torch.testing.assert_close(seen_by_critic[-1], actor.tracker(final), rtol=0, atol=0)
        assert not torch.equal(seen_by_critic[-1], expected_actions[-1])
    handle.remove()
    for batch in algorithm.storage.mini_batch_generator(2, 1):
        with torch.no_grad():
            torch.testing.assert_close(batch.observations[ACTION_GROUP], actor.tracker(batch.observations))
    before = deepcopy(critic.state_dict())
    losses = algorithm.update()
    assert losses and all(torch.isfinite(torch.tensor(x)) for x in losses.values())
    assert losses["motion_boundary_fraction"] == 0
    assert algorithm._motion_boundary_counts.tolist() == [0, 0, 0, 0]
    assert not torch.equal(critic.mlp.latent_input.weight, before["mlp.latent_input.weight"])
    for name, value in actor.tracker.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    checkpoint = tmp_path / "ppo.pt"
    torch.save(algorithm.save(), checkpoint)
    new_actor, new_critic, new_obs = models(**options)
    restored = _algorithm(new_actor, new_critic, new_obs, entropy_coef=.005)
    restored.load(torch.load(checkpoint, weights_only=False), load_cfg=None, strict=True)
    for first, second in ((actor, new_actor), (critic, new_critic)):
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert len(restored.optimizer.state) == len(algorithm.optimizer.state) > 0


def test_standalone_critic_rejects_missing_action_instead_of_using_previous_step(models):
    actor, critic, _ = models()
    obs = observation()
    with pytest.raises(ValueError, match=ACTION_GROUP):
        critic(obs)
    actor.populate_tracker_cache(obs)
    assert torch.isfinite(critic(obs)).all()
    obs["dynamics_latent"] = torch.zeros(8, 64)
    with pytest.raises(ValueError, match="dynamics_latent"):
        critic(obs)


def test_formal_configuration_uses_history5_both_actions_and_correct_encoder_command(monkeypatch):
    from intact_tracking.cli import memory350_proprio_native_policy_train as entry
    args = entry.build_parser().parse_args([
        "--fusion", "concat", "--context-checkpoint", "new.pt", "--output-dir", "unused"])
    assert args.latent_history_frames == 5 and args.tracker_action_observed
    assert args.num_envs == 8192 and args.training_ranks == 8 and args.until_user_stop
    assert args.motion_sampling == "adaptive" and args.adaptive_after_update == 0
    config = configure_tracker_action_models({"actor": {}, "critic": {}}, "concat", scratch_seed=121)
    for role in ("actor", "critic"):
        assert config[role]["dynamics_latent_dim"] == 320
        assert config[role]["tracker_action_dim"] == 29
    assert config["critic"]["initial_checkpoint"] is None
    monkeypatch.setattr(entry.native, "configure_physics", lambda *args, **kwargs: {})
    physics = entry.configure_proprio_physics(None)
    assert physics["encoder_action_input"] == "joint_pos action term.raw_action"
    assert "cached" in physics["encoder_state_input"]


def test_motion_boundary_pulse_reaches_ppo_without_mutating_episode_dones(monkeypatch):
    from types import SimpleNamespace
    from mjlab.rl import RslRlVecEnvWrapper
    from intact_tracking.memory350_policy_env import Memory350PolicyWrapper

    wrapped = object.__new__(Memory350PolicyWrapper)
    wrapped.context = None
    wrapped.episode_ids = torch.zeros(8, dtype=torch.long)
    pulse = torch.tensor([True, False, False, False, False, False, False, False])
    wrapped.motion_command = SimpleNamespace(motion_resample_boundary=pulse)
    wrapped._attach = lambda obs: obs
    dones = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    extras = {"time_outs": torch.zeros(8, dtype=torch.bool)}
    monkeypatch.setattr(RslRlVecEnvWrapper, "step",
                        lambda *_: (observation(), torch.ones(8), dones, extras))
    _, _, returned_dones, returned_extras = wrapped.step(torch.zeros(8, 29))
    assert returned_dones is dones
    assert dones.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert "motion_resample_boundary" not in extras
    assert returned_extras["motion_resample_boundary"].tolist() == pulse.tolist()
    assert wrapped.episode_ids.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    pulse.zero_()
    assert returned_extras["motion_resample_boundary"][0]


@pytest.mark.parametrize("with_timeouts", [False, True])
def test_spv53_boundary_bootstraps_old_value_once_and_preserves_episode_inputs(models, with_timeouts):
    actor, critic, obs = models()
    algorithm = _algorithm(actor, critic, obs)
    with torch.no_grad():
        algorithm.act(obs)
        old_values = algorithm.transition.values.squeeze(-1).clone()
        # Motion only; ordinary timeout; motion plus timeout; true failure;
        # then four continuing worlds. Environment signals remain untouched.
        boundary = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
        dones = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0])
        extras = {"motion_resample_boundary": boundary}
        if with_timeouts:
            extras["time_outs"] = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
        saved_dones = dones.clone()
        saved_extras = {k: v.clone() for k, v in extras.items()}
        rewards = torch.ones(8)
        algorithm.process_env_step(observation(100), rewards, dones, extras)
    assert algorithm.storage.dones[0, :, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    bootstrap = boundary | (saved_extras["time_outs"] if with_timeouts else False)
    expected = rewards + algorithm.gamma * old_values * bootstrap
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected)
    torch.testing.assert_close(dones, saved_dones)
    torch.testing.assert_close(rewards, torch.ones(8))
    for key in extras:
        torch.testing.assert_close(extras[key], saved_extras[key])
    assert algorithm._motion_boundary_counts.tolist() == [2, 1, 1 if with_timeouts else 0, 8]


def test_spv53_segments_24_step_returns_including_boundary_at_rollout_end(models):
    actor, critic, obs = models()
    algorithm = _algorithm(actor, critic, obs)
    algorithm.storage = RolloutStorage("rl", 8, 24, obs, [29])
    st = algorithm.storage
    st.rewards.fill_(1)
    st.values.fill_(10)
    st.dones.zero_()
    st.dones[7] = st.dones[15] = 1
    st.dones[23, 0] = 1
    # SPV5-3 carries current-state value in the boundary reward, not the new
    # motion's value. gamma=lambda=1 makes expected segmented returns exact.
    st.rewards += st.values * st.dones
    algorithm.gamma = algorithm.lam = 1.0
    algorithm.normalize_advantage_per_mini_batch = True
    critic.forward = lambda *_args, **_kwargs: torch.full((8, 1), 100.)
    algorithm.compute_returns(obs)
    expected = torch.tensor(list(range(18, 10, -1)) * 2 + list(range(108, 100, -1)))
    torch.testing.assert_close(st.returns[:, 1, 0], expected.float())
    torch.testing.assert_close(st.returns[-8:, 0, 0], torch.arange(18, 10, -1).float())
    old_returns, old_advantages = st.returns[:16].clone(), st.advantages[:16].clone()
    st.rewards[16:] *= 1000
    algorithm.compute_returns(obs)
    torch.testing.assert_close(st.returns[:16], old_returns, atol=0, rtol=0)
    torch.testing.assert_close(st.advantages[:16], old_advantages, atol=0, rtol=0)


def test_motion_boundary_missing_or_wrong_shape_is_rejected(models):
    actor, critic, obs = models()
    algorithm = _algorithm(actor, critic, obs)
    dones, rewards = torch.zeros(8), torch.ones(8)
    with pytest.raises(KeyError, match="motion_resample_boundary"):
        algorithm.process_env_step(obs, rewards, dones, {})
    with pytest.raises(ValueError, match="shape"):
        algorithm.process_env_step(obs, rewards, dones,
                                   {"motion_resample_boundary": torch.zeros(8, 1)})
    with pytest.raises(ValueError, match="time_outs shape"):
        algorithm.process_env_step(obs, rewards, dones,
                                   {"motion_resample_boundary": torch.zeros(8),
                                    "time_outs": torch.zeros(8, 1)})
