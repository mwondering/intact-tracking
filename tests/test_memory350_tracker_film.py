import copy

import pytest
import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict
from rsl_rl.modules.distribution import GaussianDistribution

from intact_tracking.memory350_tracker_film import (
    ObservationContextFiLM, TrackerFiLMActor, TrackerFiLMCritic, configure_models,
)


def backbone():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(813)
        result = nn.Sequential(nn.Linear(12, 24), nn.ELU(), nn.Linear(24, 16), nn.ELU(),
                               nn.Linear(16, 8), nn.ELU(), nn.Linear(8, 3))
    return result.requires_grad_(False)


def adapter(conditioning="latent"):
    return ObservationContextFiLM((24, 16, 8), observation_dim=12,
        compression_dims=(16, 8), condition_width=12, layer_indices=(0, 1, 2),
        conditioning=conditioning, seed=19)


def nonidentity(module):
    with torch.no_grad():
        for head in module.heads.values():
            head.weight.normal_(std=.08)
            head.bias.normal_(std=.02)


def test_identity_adapters_preserve_exact_tracker_and_matching_initial_parameters():
    original = backbone()
    learned, constant = adapter(), adapter("constant")
    features, z = torch.randn(7, 12), torch.randn(7, 64)
    for name, value in learned.state_dict().items():
        torch.testing.assert_close(value, constant.state_dict()[name], atol=0, rtol=0)
    for model in (learned, constant):
        torch.testing.assert_close(model(original, features, z), original(features), atol=0, rtol=0)
    assert set(learned.state_dict()) == set(constant.state_dict())
    assert not any("tracker" in key for key in learned.state_dict())


def test_ppo_gradient_traverses_frozen_layers_and_trains_observation_conditioning():
    original, film = backbone(), adapter()
    snapshot = copy.deepcopy(original.state_dict())
    features, z = torch.randn(16, 12, requires_grad=True), torch.randn(16, 64, requires_grad=True)
    target = torch.randn(16, 3)
    optimizer = torch.optim.Adam(film.parameters(), lr=.005)
    before = copy.deepcopy(film.state_dict())
    for step in range(3):
        mean = film(original, features, z)
        loss = -Normal(mean, .25).log_prob(target).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in film.parameters())
        assert all(head.weight.grad.abs().sum() > 0 for head in film.heads.values())
        if step:
            assert film.observation[0][0].weight.grad.abs().sum() > 0
            assert film.condition[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert features.grad is None and z.grad is None
    assert all(p.grad is None and not p.requires_grad for p in original.parameters())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], atol=0, rtol=0)
    assert not torch.equal(before["observation.0.0.weight"], film.state_dict()["observation.0.0.weight"])


def test_baseline_modulation_depends_on_observation_and_cannot_read_latent():
    film, original = adapter("constant"), backbone()
    nonidentity(film)
    features, z = torch.randn(8, 12), torch.randn(8, 64)
    before = film.coefficients(features, z)
    after = film.coefficients(features + 3, z)
    assert any(not torch.equal(before[i][0], after[i][0]) for i in before)
    torch.testing.assert_close(film(original, features, z),
                               film(original, features, torch.randn_like(z) * 100), atol=0, rtol=0)
    # It must ignore invalid/NaN donor codes as well, not multiply them by zero.
    torch.testing.assert_close(film(original, features, z),
                               film(original, features, torch.full_like(z, float("nan"))), atol=0, rtol=0)
    film(original, features, z).square().mean().backward()
    assert film.observation[0][0].weight.grad.abs().sum() > 0


def test_real_latent_can_change_actions_and_minibatch_reordering_is_exact():
    film, original = adapter(), backbone()
    nonidentity(film)
    features, z = torch.randn(8, 12), torch.randn(8, 64)
    normal = film(original, features, z)
    assert not torch.allclose(normal, film(original, features, z.roll(1, 0)))
    order = torch.tensor([7, 0, 2, 5, 4, 3, 1, 6])
    torch.testing.assert_close(film(original, features[order], z[order]), normal[order])
    restored = adapter()
    restored.load_state_dict(copy.deepcopy(film.state_dict()), strict=True)
    torch.testing.assert_close(restored(original, features, z), normal, atol=0, rtol=0)


class SmallTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = backbone()
        self.distribution = GaussianDistribution(3)
        self.requires_grad_(False)

    def get_latent(self, obs):
        return obs["features"]


def test_actor_ppo_probability_is_over_the_modulated_action_not_a_detached_tracker():
    actor = TrackerFiLMActor.__new__(TrackerFiLMActor)
    nn.Module.__init__(actor)
    actor.tracker, actor.film = SmallTracker(), adapter()
    actor.distribution = GaussianDistribution(3, init_std=.25)
    actor.dynamics_latent_group = "dynamics_latent"
    nonidentity(actor.film)
    obs = TensorDict({"features": torch.randn(6, 12), "dynamics_latent": torch.randn(6, 64)}, [6])
    mean = actor(obs)
    sampled = actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.output_mean, mean)
    torch.testing.assert_close(actor.get_output_log_prob(sampled), Normal(mean, .25).log_prob(sampled).sum(-1))
    old = tuple(p.detach().clone() for p in actor.output_distribution_params)
    order = torch.tensor([3, 5, 0, 2, 1, 4])
    actor(obs[order], stochastic_output=True)
    torch.testing.assert_close(actor.get_kl_divergence(tuple(p[order] for p in old),
        actor.output_distribution_params), torch.zeros(6), atol=1e-6, rtol=0)
    actor.get_output_log_prob(sampled.detach()[order]).mean().backward()
    assert actor.film.heads["0"].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in actor.tracker.parameters())


@pytest.mark.parametrize("fusion", ["concat", "baseline"])
def test_critic_compresses_observations_before_latent_and_never_uses_film(fusion):
    z = torch.randn(8, 64, requires_grad=True)
    obs = TensorDict({"critic": torch.randn(8, 12), "dynamics_latent": z}, [8])
    critic = TrackerFiLMCritic(obs, {"critic": ["critic"]}, "critic", 1,
        initial_checkpoint=None, initialization_seed=901, fusion_mode=fusion,
        compression_dims=(32, 128), head_dims=(32, 16))
    critic.update_normalization(obs)
    assert critic.mlp.observation[0][0].in_features == 12
    assert critic.mlp.head[0].in_features == 192
    assert not any(isinstance(m, ObservationContextFiLM) for m in critic.modules())
    with torch.no_grad():
        critic.mlp.head[0].weight[:, 128:].normal_(std=.1)
    alternate = obs.clone()
    alternate["dynamics_latent"] = z.detach().roll(1, 0)
    if fusion == "baseline":
        torch.testing.assert_close(critic(obs), critic(alternate), atol=0, rtol=0)
        # A baseline critic does not even require the latent observation field.
        torch.testing.assert_close(critic(obs), critic(obs.exclude("dynamics_latent")), atol=0, rtol=0)
    else:
        assert not torch.allclose(critic(obs), critic(alternate))
    critic(obs).square().mean().backward()
    assert z.grad is None
    assert critic.mlp.observation[0][0].weight.grad.abs().sum() > 0
    assert bool(critic.mlp.head[0].weight.grad[:, 128:].abs().sum() > 0) == (fusion == "concat")


def test_arms_have_matching_train_config_except_actor_and_critic_latent_access():
    source = {"actor": {}, "critic": {}}
    a = configure_models(source, "film", scratch_seed=121, actor_conditioning="latent")
    b = configure_models(source, "film", scratch_seed=121, actor_conditioning="constant")
    assert a["actor"].pop("actor_conditioning") == "latent"
    assert b["actor"].pop("actor_conditioning") == "constant"
    assert a["critic"].pop("fusion_mode") == "concat"
    assert b["critic"].pop("fusion_mode") == "baseline"
    assert a == b and source == {"actor": {}, "critic": {}}


def test_training_cli_requires_context_and_retains_uniform_unbounded_protocol():
    from intact_tracking.cli.memory350_tracker_film_train import build_parser
    parser = build_parser()
    for conditioning in ("latent", "constant"):
        args = parser.parse_args(["--conditioning", conditioning, "--context-checkpoint", "context.pt",
                                  "--output-dir", "runs/test", "--training-ranks", "8"])
        assert args.actor_conditioning == conditioning
        assert args.fusion == "film" and args.iterations is None
        assert args.motion_sampling == "uniform" and args.training_terminations == "original"
        assert args.tracker_warmup_steps == 0 and args.num_envs == 8192
    with pytest.raises(SystemExit):
        parser.parse_args(["--conditioning", "constant", "--output-dir", "runs/test"])


@pytest.mark.parametrize("indices", [(), (1, 1), (2, 0), (-1,), (3,)])
def test_film_rejects_invalid_or_action_layer_modulation(indices):
    with pytest.raises(ValueError):
        ObservationContextFiLM((24, 16, 8), layer_indices=indices)
