from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage

from intact_tracking.heavy_rma_teacher import RMATeacherActor, RMATeacherCritic, PHYSICS_GROUP, audit_initial_models
from intact_tracking.heavy_anyadapter import AnyAdapterActor, AnyAdapterCritic, LayerAdapters
from intact_tracking.anyadapter_world_model import (
    HistoryEncoder, WorldModel, CausalHistory, RolloutHistoryBank, append_frame,
    component_losses, autoregressive_loss, STATE_GROUP, WORLD_GROUP, COUNT_GROUP,
)
from intact_tracking.anyadapter_training import AnyAdapterPPO
from intact_tracking.memory350_tracker_action_policy import TrackerActionPPO
from intact_tracking.limb_context_sampling import state_digest


class Tracker(nn.Module):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__()
        self.obs_groups, self.policy_input_dim = obs_groups[obs_set], 3
        self.mlp = nn.Sequential(nn.Linear(3, 16), nn.ELU(), nn.Linear(16, 12), nn.ELU(), nn.Linear(12, output_dim))
        self.distribution = GaussianDistribution(output_dim, init_std=.7)

    def populate_policy_context_cache(self, obs):
        pass

    def get_latent(self, obs):
        return obs['features']

    def forward(self, obs):
        return self.mlp(self.get_latent(obs))


def observation():
    state = torch.randn(8, 65) * .01
    state[:, 3:6] = torch.tensor([0., 0., -1.])
    state[:, 64] = .8
    return TensorDict({'features': torch.randn(8, 3), 'priv': torch.randn(8, 7),
                       PHYSICS_GROUP: torch.rand(8, 108)*2-1, 'dynamics_latent': torch.randn(8, 128),
                       STATE_GROUP: state[:, :64].clone(), WORLD_GROUP: state,
                       COUNT_GROUP: torch.zeros(8, 1, dtype=torch.long)}, [8])


@pytest.fixture
def models(tmp_path, monkeypatch):
    import intact_tracking.residual_policy as policy
    monkeypatch.setattr(policy, 'SPV52HeightContactEstimatorActor', Tracker)
    torch.manual_seed(121)
    groups = {'actor': ['features'], 'critic': ['priv']}
    path = tmp_path/'tracker.pt'
    torch.save({'actor_state_dict': Tracker(None, groups, 'actor', 29).state_dict()}, path)

    def make(method):
        obs = observation()
        actor_cls, critic_cls = ((RMATeacherActor, RMATeacherCritic) if method == 'rma_teacher'
                                 else (AnyAdapterActor, AnyAdapterCritic))
        actor = actor_cls(obs, groups, 'actor', 29, tracker_checkpoint=str(path), tracker_actor_kwargs={},
                          tracker_obs_groups=groups, residual_hidden_dims=(16, 8), initialization_seed=10128,
                          residual_output_mode='unbounded', residual_scale=1.)
        critic = critic_cls(obs, groups, 'critic', 1, initial_checkpoint=None, hidden_dims=(16, 8),
                            initialization_seed=20124)
        critic.update_normalization(obs)
        return actor, critic, obs
    return make


@pytest.mark.parametrize('method', ['rma_teacher', 'any2track'])
def test_initial_identity_final_gaussian_and_frozen_backbone(models, method):
    actor, critic, obs = models(method)
    assert audit_initial_models(actor, critic, obs, 'concat')['identical_initial_action']
    saved = state_digest(actor.tracker.state_dict())
    actor.train()
    assert not actor.tracker.training
    actor(obs, stochastic_output=True)
    expected = torch.distributions.Normal(actor.tracker(obs), torch.ones(8, 29))
    command = torch.randn(8, 29)
    torch.testing.assert_close(actor.get_output_log_prob(command), expected.log_prob(command).sum(-1))
    (actor(obs).square().mean()+critic(obs).square().mean()).backward()
    assert all(p.grad is None for p in actor.tracker.parameters())
    assert state_digest(actor.tracker.state_dict()) == saved


def test_teacher_raw_theta_recomputed_and_both_encoders_receive_gradients(models):
    actor, critic, obs = models('rma_teacher')
    with torch.no_grad():
        actor.residual_mlp[-1].weight.normal_(std=.2)
    obs[PHYSICS_GROUP].requires_grad_()
    (actor(obs).square().mean()+critic(obs).square().mean()).backward()
    assert obs[PHYSICS_GROUP].grad is None
    for encoder in (actor.dr_encoder, critic.dr_encoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
        assert encoder[0].weight.grad.abs().sum() > 0
    assert not set(map(id, actor.dr_encoder.parameters())) & set(map(id, critic.dr_encoder.parameters()))
    altered = obs.clone()
    altered[PHYSICS_GROUP] = -obs[PHYSICS_GROUP]
    assert not torch.allclose(actor(obs), actor(altered))
    assert not torch.allclose(critic(obs), critic(altered))


def test_every_adapter_has_gradients_through_frozen_layers_and_encoder_has_none(models):
    actor, critic, obs = models('any2track')
    obs['dynamics_latent'].requires_grad_()
    (actor(obs).square().mean()+critic(obs).square().mean()).backward()
    assert obs['dynamics_latent'].grad is None
    assert all(p.grad is None for p in actor.history_encoder.parameters())
    for layer in actor.adapters.layers:
        assert layer.weight.grad.abs().sum() > 0
    assert not hasattr(actor, 'residual_mlp')


def test_reference_parameter_counts_and_chronological_history():
    widths = [1645, 2048, 2048, 1024, 1024, 512, 256, 128, 29]
    mlp = nn.Sequential(*[layer for i in range(len(widths)-1)
                          for layer in ([nn.Linear(widths[i], widths[i+1]), nn.ELU()]
                                        if i < len(widths)-2 else [nn.Linear(widths[i], widths[i+1])])])
    assert sum(p.numel() for p in LayerAdapters(mlp).parameters()) == 8301085
    assert sum(p.numel() for p in HistoryEncoder().parameters()) == 111168
    assert sum(p.numel() for p in WorldModel().parameters()) == 676897
    live = CausalHistory(3, 'cpu')
    bank = RolloutHistoryBank(3, 5, 'cpu')
    for t in range(100):
        live.append(torch.full((3, 64), float(t)), torch.full((3, 29), -float(t)), torch.zeros(3, dtype=torch.bool))
    bank.start(live)
    for t in range(5):
        torch.testing.assert_close(bank.history(torch.arange(3), t), live.frames, rtol=0, atol=0)
        state, action = torch.full((3, 64), 100.+t), torch.full((3, 29), -100.-t)
        boundary = torch.tensor([t == 1, t == 3, False])
        live.append(state, action, boundary)
        bank.append(t, state, action, live.count)
    # Samples at different timesteps/envs use only their own completed past.
    assert bank.history(torch.tensor([0, 1, 2]), torch.tensor([2, 4, 5]))[:2].eq(0).all()
    torch.testing.assert_close(bank.history(torch.tensor([2]), 5)[0, -1, :64], torch.full((64,), 104.))
    assert bank.frames.numel() == 3 * (79+5) * 93


def test_integrator_loss_units_boundary_mask_and_full_frame_shift():
    model = WorldModel(hidden_dims=(8,))
    for p in model.parameters():
        nn.init.zeros_(p)
    state = torch.zeros(2, 65)
    state[:, 5], state[:, 64], state[:, 35:64] = -1, 1, .05
    predicted = model(state, torch.zeros(2, 128), torch.zeros(2, 29))
    torch.testing.assert_close(predicted[:, 6:35], torch.full((2, 29), .02))
    torch.testing.assert_close(predicted[:, 3:6].norm(dim=-1), torch.ones(2))
    target = state.clone()
    target[:, :3] = .1
    target[:, 64] += .1
    parts = component_losses(predicted, target, torch.tensor([True, False]))
    torch.testing.assert_close(parts, torch.tensor([75., 0., .29, 0., 25.]), atol=1e-4, rtol=1e-5)
    assert component_losses(predicted, target, torch.zeros(2, dtype=torch.bool)).eq(0).all()
    history = torch.randn(2, 79, 93)
    action = torch.randn(2, 29)
    shifted = append_frame(history, state, action)
    torch.testing.assert_close(shifted[:, :-1], history[:, 1:])
    torch.testing.assert_close(shifted[:, -1], torch.cat((state[:, :64], action), -1))


def test_prediction_updates_encoder_only_and_reanchors_after_boundary(models):
    actor, _, _ = models('any2track')
    encoder = actor.history_encoder.requires_grad_(True)
    model = WorldModel(hidden_dims=(16,))
    states = torch.stack([observation()[WORLD_GROUP] for _ in range(4)], 1)
    history, action = torch.randn(8, 79, 93)*.01, torch.randn(8, 3, 29)*.01
    boundaries = torch.zeros(8, 3, dtype=torch.bool)
    boundaries[:, 0] = True
    observed = []
    encoder.register_forward_pre_hook(lambda m, a: observed.append(a[0].detach().clone()))
    reanchors = [torch.full_like(history, .01*i) for i in range(4)]
    loss, _ = autoregressive_loss(encoder, model, history, states, action, boundaries,
                                   reanchor=lambda i: reanchors[i])
    loss.backward()
    torch.testing.assert_close(observed[1], reanchors[1], rtol=0, atol=0)
    assert sum(p.grad.abs().sum() for p in encoder.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
    assert all(p.grad is None for p in actor.adapters.parameters())
    assert all(p.grad is None for p in actor.tracker.parameters())


@pytest.mark.parametrize('method', ['rma_teacher', 'any2track'])
def test_real_ppo_update_refresh_and_full_optimizer_resume(models, method):
    def build():
        actor, critic, obs = models(method)
        storage = RolloutStorage('rl', 8, 3, obs, [29])
        options = dict(num_learning_epochs=2, num_mini_batches=2, schedule='fixed',
                       actor_learning_rate=1e-3, critic_learning_rate=1e-3, entropy_coef=.005)
        if method == 'any2track':
            alg = AnyAdapterPPO(actor, critic, storage, prediction_steps=2, world_model_epochs=1,
                                world_model_mini_batches=2, world_model_hidden_dims=(16,), **options)
            alg.bind_environment(SimpleNamespace(history=CausalHistory(8, 'cpu')))
        else:
            alg = TrackerActionPPO(actor, critic, storage, **options)
        return alg, obs
    alg, obs = build()
    frozen = state_digest(alg.actor.tracker.state_dict())
    encoders_before = ([state_digest(model.dr_encoder.state_dict()) for model in (alg.actor, alg.critic)]
                       if method == 'rma_teacher' else None)
    with torch.inference_mode():
        for step in range(3):
            if method == 'any2track':
                obs['dynamics_latent'] = alg.actor.history_encoder(alg.live_history.frames)
            sampled = alg.act(obs)
            boundary = torch.arange(8) == step
            following = observation()
            if method == 'any2track':
                alg.live_history.append(obs[STATE_GROUP], sampled, boundary)
                following[COUNT_GROUP] = alg.live_history.count.clone()
                following['dynamics_latent'] = alg.actor.history_encoder(alg.live_history.frames)
            alg.process_env_step(following, torch.randn(8), torch.zeros(8), {'motion_resample_boundary': boundary})
            obs = following
        alg.compute_returns(obs)
    old_logprob = alg.storage.actions_log_prob.clone()
    if method == 'any2track':
        # Inspect the exact observations the PPO optimizer will consume.
        original = alg._refresh_embeddings
        def refreshed():
            stats = original()
            for t in range(3):
                want = alg.actor.history_encoder(alg.history_bank.history(torch.arange(8), t))
                torch.testing.assert_close(alg.storage.observations['dynamics_latent'][t], want)
            torch.testing.assert_close(alg.storage.actions_log_prob, old_logprob, rtol=0, atol=0)
            return stats
        alg._refresh_embeddings = refreshed
    result = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert state_digest(alg.actor.tracker.state_dict()) == frozen
    if method == 'rma_teacher':
        assert all(state_digest(model.dr_encoder.state_dict()) != before for model, before
                   in zip((alg.actor, alg.critic), encoders_before, strict=True))
    if method == 'any2track':
        assert result['WorldModel/optimizer_steps'] == 2
        assert result['WorldModel/embedding_drift_rms'] > 0
        assert all(p.grad is None and not p.requires_grad for p in alg.actor.history_encoder.parameters())
        ppo_ids = {id(p) for group in alg.optimizer.param_groups for p in group['params']}
        assert not ppo_ids & set(map(id, alg.wm_parameters))
    restored, _ = build()
    restored.load(deepcopy(alg.save()), None, True)
    assert state_digest(restored.save()) == state_digest(alg.save())


def test_production_stage_and_sampling_reset_are_separate_from_smoke(monkeypatch):
    from intact_tracking.cli import heavy_baseline_train as cli
    args = cli.build_parser().parse_args(['--method', 'rma_teacher', '--output-dir', 'unused',
                                         '--stage', 'stage1', '--motion-sampling', 'adaptive'])
    assert not args.bounded_smoke and args.iterations == 2001
    assert args.entropy_coef == .005 and args.initial_action_std == 1.
    args.bounded_smoke = True
    args.stage, args.reset_adaptive_sampling = 'resume_sampling_reset', True
    previous = {'completed_updates': 2001, 'residual_policy': {'method': 'rma_teacher', 'baseline_stage': 'stage1'}}
    metadata = {'research_source_sha256': {}}
    cli.adapt_metadata(metadata, args, previous)
    assert metadata['sampling_reset_generation'] == 1
    previous['residual_policy'] = metadata
    with pytest.raises(ValueError, match='single planned'):
        cli.adapt_metadata({'research_source_sha256': {}}, args, previous)
    args.reset_adaptive_sampling = False
    second = {'research_source_sha256': {}}
    cli.adapt_metadata(second, args, previous)
    assert second['sampling_reset_generation'] == 1


def test_deployment_history_is_causal_and_teacher_never_defaults_to_zero():
    import numpy as np
    from collections import deque
    from intact_tracking.heavy_baseline_deploy import HeavyBaselinePolicy, proprio_state as numpy_state
    from intact_tracking.heavy_baseline_env import proprio_state as torch_state

    class Session:
        calls = []
        def run(self, names, inputs):
            self.calls.append({key: value.copy() for key, value in inputs.items()})
            return np.ones((1, 29), np.float32), np.ones((1, 128), np.float32)
    policy = object.__new__(HeavyBaselinePolicy)
    policy.session, policy.method, policy.history = Session(), 'any2track', deque(maxlen=79)
    policy.metadata = {'in_keys': ['tracker_observation', 'history'], 'out_keys': ['action', 'embedding']}
    policy.previous_state = policy.previous_action = None
    x = np.random.default_rng(12).normal(size=8199).astype(np.float32)
    expected = torch_state({'estimator_history': torch.from_numpy(x[4:6104])[None]})
    np.testing.assert_array_equal(expected.numpy()[0], numpy_state(x))
    policy.step(x)
    assert not policy.session.calls[-1]['history'].any()
    policy.step(x+1, command_applied=np.full(29, 2., np.float32))
    history = policy.session.calls[-1]['history'].reshape(79, 93)
    assert not history[:-1].any()
    np.testing.assert_array_equal(history[-1, :64], numpy_state(x))
    np.testing.assert_array_equal(history[-1, 64:], np.full(29, 2., np.float32))
    policy.step(x, reset_boundary=True)
    assert not policy.session.calls[-1]['history'].any()
    policy.method = 'rma_teacher'
    with pytest.raises(ValueError, match='requires real physical'):
        policy.step(x)


@pytest.mark.parametrize('fraction', [0., 1.])
def test_full_evaluation_physics_does_not_use_the_ten_percent_subset(fraction):
    from intact_tracking.memory350_heavy_dr import stratified_payload_samples
    from intact_tracking.memory350_native_dr import native_nominal_ids
    mass, offsets, groups = stratified_payload_samples(512, fraction, 0, 121)
    assert len(native_nominal_ids(512, fraction, allow_endpoints=True)) == int(512*fraction)
    if fraction == 1.:
        assert mass.eq(0).all() and offsets.eq(0).all() and groups.eq(-1).all()
    else:
        assert (groups >= 0).all() and mass.gt(0).all()
        assert torch.bincount(groups, minlength=256).tolist() == [2]*256
    with pytest.raises(ValueError):
        native_nominal_ids(512, fraction)  # Training helper retains its strict default.


def test_controller_selects_boundary_by_contents_and_never_refreshes_stage_b_twice(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_144000_heavy_baselines as controller
    assert controller.select_stage(tmp_path) == ('stage1', None, False)
    first = tmp_path/'stage1'
    first.mkdir()
    path = first/'checkpoint_2000.pt'
    torch.save({'completed_updates': 2000}, path)
    assert controller.select_stage(tmp_path) == ('stage1', path, False)
    torch.save({'completed_updates': 2001}, path)
    assert controller.select_stage(tmp_path) == ('resume_sampling_reset', path, True)
    second = tmp_path/'resume_sampling_reset'
    second.mkdir()
    restored = second/'checkpoint_resume.pt'
    torch.save({'completed_updates': 2001, 'residual_policy': {'sampling_reset_generation': 1}}, restored)
    assert controller.select_stage(tmp_path) == ('resume_sampling_reset', restored, False)
    command = controller.command_for('rma_teacher', 'stage1', first)
    assert '--bounded-smoke' not in command and '--num-envs' in command
    assert command[command.index('--num-envs')+1] == '8192'
    assert '--nproc-per-node=4' in command


def test_batched_evaluation_uses_episode_weights_and_validates_model_identity(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from evaluate_heavy_baselines import aggregate
    first = {'episodes': 2, 'motions': 1, 'metric_names': ['error'], 'checkpoint_sha256': 'a',
             'completed_training_updates': 2001, 'mean': {'error': 1.}, 'failed': [False, True],
             'episode_lengths': [5, 1], 'horizons': [5, 5], 'episode_returns': [1., 2.], 'coverage_fraction': .6}
    second = {**first, 'episodes': 4, 'motions': 2, 'mean': {'error': 4.}, 'failed': [False]*4,
              'episode_lengths': [5]*4, 'horizons': [5]*4, 'episode_returns': [1.]*4, 'coverage_fraction': 1.}
    summary = aggregate([first, second])
    assert summary['mean']['error'] == 3.
    assert summary['failure_rate'] == pytest.approx(1/6)
    assert summary['coverage_fraction'] == pytest.approx((2*.6+4)/6)
    with pytest.raises(ValueError, match='different'):
        aggregate([first, {**second, 'checkpoint_sha256': 'b'}])
