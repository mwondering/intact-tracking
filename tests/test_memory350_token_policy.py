import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage

from intact_tracking.memory350_token_inputs import (
    BASE_ACTION, FRAME_DIM, FRAME_DIMS, FUTURE, HISTORY, LATENT_SLICE, VALID,
    SPV52TokenSplitter, TokenHistory, intervene_latent_history,
)
from intact_tracking.memory350_token_policy import (
    TemporalTokenActor, TemporalTokenBackbone, TemporalTokenCritic, configure_models,
)


def token_obs(n=6, critic_dim=12):
    history = torch.randn(n, 5, FRAME_DIM)
    return TensorDict({HISTORY: history, VALID: torch.ones(n, 5, dtype=torch.bool),
                      FUTURE: torch.randn(n, 4, 77), BASE_ACTION: history[:, -1, -29:].clone(),
                      "dynamics_latent": history[:, -1, LATENT_SLICE].clone(),
                      "critic": torch.randn(n, critic_dim)}, [n])


def test_splitter_exactly_partitions_current_and_future_reference_with_no_future_state():
    features = torch.arange(1645).float()[None].repeat(2, 1).requires_grad_()
    action = torch.randn(2, 29, requires_grad=True)
    latent = torch.randn(2, 64, requires_grad=True)
    splitter = SPV52TokenSplitter()
    frame, future = splitter(features, action, latent)
    p, r, e, z, a = frame.split(FRAME_DIMS, -1)
    assert frame.shape == (2, 942) and future.shape == (2, 4, 77)
    # Latest frame of each term, not the last 122 entries of a term-major history.
    torch.testing.assert_close(p[0, :29], torch.arange(116, 145).float())
    torch.testing.assert_close(p[0, 29:58], torch.arange(261, 290).float())
    torch.testing.assert_close(p[0, -2:], torch.tensor([1643., 1644.]))
    # 269 current + 4*77 future cover exactly all 577 original reference values.
    reference_indices = torch.cat((r[0], future[0].flatten())).sort().values
    torch.testing.assert_close(reference_indices, torch.arange(806, 1383).float())
    torch.testing.assert_close(future[0, :, :3], torch.arange(806, 818).float().reshape(4, 3))
    torch.testing.assert_close(future[0, :, 3:9], torch.arange(824, 848).float().reshape(4, 6))
    torch.testing.assert_close(e[0], torch.arange(1383, 1643).float())
    torch.testing.assert_close(z, latent)
    torch.testing.assert_close(a, action)
    assert not frame.requires_grad and not future.requires_grad


def test_history_snapshots_are_causal_idempotent_and_reset_only_discontinuous_worlds():
    bank = TokenHistory(3, "cpu")
    stamp = torch.tensor([[0, 10, 0], [0, 11, 0], [0, 12, 0]])
    old = None
    for step in range(7):
        stamp[:, 2] = step
        frames, valid = bank.observe(torch.full((3, FRAME_DIM), float(step)), stamp)
        assert valid.sum(1).tolist() == [min(step + 1, 5)] * 3
        again, again_valid = bank.observe(torch.full((3, FRAME_DIM), float(step)), stamp)
        torch.testing.assert_close(again, frames, atol=0, rtol=0)
        assert torch.equal(again_valid, valid)
        if step == 2:
            old = frames
    torch.testing.assert_close(old[0, :, 0], torch.tensor([0., 0., 0., 1., 2.]))
    torch.testing.assert_close(frames[0, :, 0], torch.arange(2, 7).float())
    snapshot = frames.clone()
    stamp[0] = torch.tensor([1, 10, 7])  # Episode reset despite continuing target time.
    stamp[1] = torch.tensor([0, 99, 7])  # Motion changed without a time rewind.
    stamp[2, 2] = 7
    frames, valid = bank.observe(torch.full((3, FRAME_DIM), 7.), stamp)
    assert valid.sum(1).tolist() == [1, 1, 5]
    torch.testing.assert_close(frames[2, :-1], snapshot[2, 1:])
    bank.clear(torch.tensor([2]))
    assert frames[2].abs().sum() > 0  # Published tensors were not modified.
    assert not bank.valid[2].any()
    stamp[0, 2] += 3
    _, valid = bank.observe(torch.ones(3, FRAME_DIM), stamp)
    assert valid[0].sum() == 1


def test_environment_attaches_aligned_frozen_actions_and_latents_without_model_side_effects():
    from intact_tracking.memory350_token_env import TokenPolicyWrapper
    wrapper = TokenPolicyWrapper.__new__(TokenPolicyWrapper)
    wrapper.num_envs, wrapper.device = 2, torch.device("cpu")
    wrapper.token_history = TokenHistory(2, "cpu")
    wrapper.splitter = SPV52TokenSplitter()
    wrapper.episode_ids = torch.zeros(2, dtype=torch.long)
    wrapper.motion_command = SimpleNamespace(motion_idx=torch.tensor([7, 8]),
                                             time_steps=torch.zeros(2, dtype=torch.long))
    wrapper._latent = torch.randn(2, 64)
    wrapper._policy = None
    # This is the schema seen by the actor/critic factory before binding.
    empty = wrapper._attach(TensorDict({"features": torch.zeros(2, 1645)}, [2]))
    assert empty[HISTORY].shape == (2, 5, 942)
    assert not empty[VALID].any()
    policy = SimpleNamespace(architecture="transformer", populate_tracker_cache=lambda obs: None,
        _base_features_and_action=lambda obs: (obs["features"], obs["features"][:, :29] + 3))
    wrapper.bind_policy(policy)
    observed = []
    for step in range(6):
        wrapper.motion_command.time_steps[:] = step
        wrapper._latent = torch.full((2, 64), float(step + 100))
        obs = wrapper._attach(TensorDict({"features": torch.full((2, 1645), float(step))}, [2]))
        observed.append(obs)
        torch.testing.assert_close(obs[HISTORY][:, -1, LATENT_SLICE], wrapper._latent)
        torch.testing.assert_close(obs[HISTORY][:, -1, -29:], obs[BASE_ACTION])
    torch.testing.assert_close(observed[-1][HISTORY][0, :, -1], torch.arange(4, 9).float())
    torch.testing.assert_close(observed[0][HISTORY][0, -1, -1], torch.tensor(3.))
    assert observed[-1][VALID].all()
    # The policy's repeated forward calls never operate on/mutate this bank.
    frozen_history = observed[-1][HISTORY].clone()
    network = TemporalTokenBackbone()
    for _ in range(2):
        network(obs[HISTORY], obs[FUTURE], obs[VALID])
    torch.testing.assert_close(wrapper.token_history.frames, frozen_history, atol=0, rtol=0)


def test_padding_cannot_change_outputs_and_minibatch_order_is_preserved():
    torch.manual_seed(4)
    model = TemporalTokenBackbone()
    obs = token_obs()
    obs[VALID][:, :3] = False
    clean = model(obs[HISTORY], obs[FUTURE], obs[VALID])
    poisoned = obs[HISTORY].clone()
    poisoned[:, :3] = float("nan")
    torch.testing.assert_close(model(poisoned, obs[FUTURE], obs[VALID]), clean, atol=0, rtol=0)
    order = torch.tensor([5, 2, 0, 3, 1, 4])
    torch.testing.assert_close(model(obs[HISTORY][order], obs[FUTURE][order], obs[VALID][order]),
                               clean[order], atol=2e-6, rtol=2e-6)
    assert model.type_ids.numel() == 29
    assert model.time_ids[-4:].tolist() == [5, 6, 7, 8]
    assert model.time_ids[:5].tolist() == [0] * 5


def test_future_and_each_history_modality_receive_gradients_without_training_encoder_inputs():
    torch.manual_seed(81)
    model = TemporalTokenBackbone()
    obs = token_obs()
    obs[HISTORY].requires_grad_()
    obs[FUTURE].requires_grad_()
    output = model(obs[HISTORY], obs[FUTURE], obs[VALID])
    # A nonuniform linear objective avoids LayerNorm's constant squared norm.
    (output * torch.linspace(-1, 2, 128)).sum().backward()
    for name, projection in model.projections.items():
        assert sum(p.grad.abs().sum() for p in projection.parameters()) > 0, name
    assert obs[HISTORY].grad is None and obs[FUTURE].grad is None
    for block in model.blocks:
        q, k, v = block.qkv.weight.grad.chunk(3)
        assert all(x.abs().sum() > 0 for x in (q, k, v))


def test_latent_interventions_replace_the_whole_stream_without_mutating_original_observations():
    obs = token_obs()
    original = obs.clone()
    zero = intervene_latent_history(obs, "zero")
    assert not zero[HISTORY][..., LATENT_SLICE].any()
    assert not zero["dynamics_latent"].any()
    donor = torch.arange(6).roll(3)
    eligible = torch.tensor([True, False, True, True, False, True])
    swapped = intervene_latent_history(obs, "paired-swap", donors=donor, eligible=eligible)
    torch.testing.assert_close(swapped[HISTORY][eligible, :, LATENT_SLICE],
                               obs[HISTORY][donor[eligible], :, LATENT_SLICE])
    for key in obs.keys():
        torch.testing.assert_close(obs[key], original[key], atol=0, rtol=0)
    for changed in (zero, swapped):
        torch.testing.assert_close(changed[HISTORY][..., :849], obs[HISTORY][..., :849])
        torch.testing.assert_close(changed[HISTORY][..., -29:], obs[HISTORY][..., -29:])
        torch.testing.assert_close(changed[FUTURE], obs[FUTURE])


def test_rollout_storage_copies_history_and_shuffles_whole_windows():
    obs = token_obs(4)
    storage = RolloutStorage("rl", 4, 2, obs, [29], "cpu")
    expected = []
    for step in range(2):
        sample = token_obs(4)
        sample[HISTORY][:, -1, 0] = torch.arange(4) + 10 * step
        sample[BASE_ACTION][:, 0] = sample[HISTORY][:, -1, 0]
        expected.append(sample[HISTORY].clone())
        transition = RolloutStorage.Transition()
        transition.observations = sample
        transition.actions = sample[BASE_ACTION]
        transition.rewards = torch.zeros(4)
        transition.dones = torch.zeros(4)
        transition.values = torch.zeros(4, 1)
        transition.actions_log_prob = torch.zeros(4)
        transition.distribution_params = (torch.zeros(4, 29), torch.ones(4, 29))
        storage.add_transition(transition)
        sample[HISTORY].zero_()
    torch.testing.assert_close(storage.observations[HISTORY], torch.stack(expected))
    for batch in storage.mini_batch_generator(2, 1):
        torch.testing.assert_close(batch.observations[HISTORY][:, -1, 0], batch.actions[:, 0])


class FakeTracker(nn.Module):
    def __init__(self, obs, groups, obs_set, output_dim, **kwargs):
        super().__init__()
        self.policy_input_dim = 1645
        self.mlp = nn.Linear(1645, output_dim)
        self.distribution = GaussianDistribution(output_dim)

    def get_latent(self, obs):
        return obs["features"]

    def populate_policy_context_cache(self, obs):
        pass

    def forward(self, obs):
        return self.mlp(self.get_latent(obs))


def make_actor(tmp_path, monkeypatch, architecture, obs):
    import intact_tracking.residual_policy as parent
    monkeypatch.setattr(parent, "SPV52HeightContactEstimatorActor", FakeTracker)
    path = tmp_path / "tracker.pt"
    source = FakeTracker(None, None, None, 29)
    torch.save({"actor_state_dict": source.state_dict()}, path)
    return TemporalTokenActor(obs, {"actor": ["features"]}, "actor", 29,
        tracker_checkpoint=str(path), tracker_actor_kwargs={},
        tracker_obs_groups={"actor": ["features"]}, architecture=architecture)


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_actor_starts_at_tracker_ppo_logprob_unbounded_output_and_frozen_gradients(
        tmp_path, monkeypatch, architecture):
    obs = token_obs()
    obs.set("features", torch.randn(6, 1645))
    actor = make_actor(tmp_path, monkeypatch, architecture, obs)
    with torch.no_grad():
        base = actor.tracker(obs)
        obs.set(BASE_ACTION, base)
        obs[HISTORY][:, -1, -29:] = base
        torch.testing.assert_close(actor(obs), base, atol=0, rtol=0)
        actor.head[-1].bias.fill_(3.0)
        torch.testing.assert_close(actor(obs) - base, torch.full_like(base, 3.), atol=1e-6, rtol=0)
        actor.head[-1].weight.normal_(std=.05)
    mean = actor(obs)
    sample = actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.get_output_log_prob(sample), Normal(mean, .25).log_prob(sample).sum(-1))
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad()
    (-actor.get_output_log_prob(sample.detach()).mean() + mean.square().mean()).backward()
    if architecture == "transformer":
        assert actor.backbone.projections["latent"].weight.grad.abs().sum() > 0
        assert actor.backbone.projections["future"].weight.grad.abs().sum() > 0
    else:
        ordinary = obs.select("features")
        torch.testing.assert_close(actor(ordinary), mean, atol=0, rtol=0)
        assert not hasattr(actor, "backbone")
        assert [x.in_features for x in actor.head if isinstance(x, nn.Linear)] == [1645, 512, 256, 128]
    optimizer.step()
    actor.assert_tracker_frozen()


@pytest.mark.parametrize("privileged", [True, False])
def test_critic_has_own_transformer_and_uses_latent(privileged):
    obs = token_obs()
    critic = TemporalTokenCritic(obs, {"critic": ["critic"]}, "critic", 1,
        initial_checkpoint=None, architecture="transformer", critic_privileged_token=privileged)
    critic.update_normalization(obs)
    value = critic(obs)
    assert value.shape == (6, 1)
    assert critic.backbone.type_ids.numel() == 29 + int(privileged)
    assert not torch.allclose(value, critic(intervene_latent_history(obs, "zero")))
    value.square().mean().backward()
    assert critic.backbone.projections["latent"].weight.grad.abs().sum() > 0
    if privileged:
        assert critic.backbone.privileged[0].weight.grad.abs().sum() > 0
    restored = copy.deepcopy(critic)
    restored.load_state_dict(critic.state_dict(), strict=True)
    torch.testing.assert_close(restored(obs), value, atol=0, rtol=0)


def test_baseline_critic_is_a_plain_mlp_and_cannot_read_any_latent_or_token():
    obs = token_obs()
    critic = TemporalTokenCritic(obs, {"critic": ["critic"]}, "critic", 1,
        initial_checkpoint=None, architecture="mlp")
    critic.update_normalization(obs)
    expected = critic(obs.select("critic"))
    for key in (HISTORY, FUTURE, BASE_ACTION, "dynamics_latent"):
        obs[key].fill_(float("nan"))
    torch.testing.assert_close(critic(obs), expected, atol=0, rtol=0)
    assert not hasattr(critic, "backbone")


def test_configuration_and_cli_keep_plain_mlp_uniform_cold_unbounded_defaults():
    from intact_tracking.cli.memory350_token_policy_train import build_parser
    for architecture in ("mlp", "transformer"):
        args = build_parser().parse_args(["--architecture", architecture, "--output-dir", "runs/test"])
        assert args.iterations is None and args.num_envs == 8192 and args.training_ranks == 4
        assert args.motion_sampling == "uniform" and args.training_terminations == "original"
        assert args.residual_scale == 1 and args.policy_precision == "fp32"
        source = {"actor": {}, "critic": {}}
        result = configure_models(source, "concat" if architecture == "transformer" else "baseline",
                                  scratch_seed=121, architecture=architecture)
        assert result["actor"]["architecture"] == result["critic"]["architecture"] == architecture
        assert result["actor"]["initialization_seed"] != result["critic"]["initialization_seed"]
        assert source == {"actor": {}, "critic": {}}
