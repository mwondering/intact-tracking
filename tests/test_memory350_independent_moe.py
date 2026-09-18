import copy

import pytest
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from intact_tracking.memory350_independent_moe_policy import (
    IndependentHardRoutedMLP, ExpertGaussianDistribution, IndependentMoEActor,
    IndependentMoECritic, configure_independent_models,
)
from intact_tracking.memory350_moe_policy import HardRoutedMLP
from intact_tracking.memory350_moe_critic_action import ACTION_GROUP


def network(output_dim=3, **kwargs):
    result = IndependentHardRoutedMLP(12, output_dim, compression_dims=(32, 128), seed=81, **kwargs)
    result.router.centers.copy_(torch.eye(16, 64))
    return result


def inputs(ids):
    ids = torch.as_tensor(ids)
    return torch.cat((torch.randn(len(ids), 41), torch.eye(16, 64)[ids]), -1)


def test_independent_encoders_match_shared_initial_function_without_shared_storage():
    options = dict(feature_dim=12, output_dim=3, compression_dims=(32, 128),
                   fusion="concat", tracker_action_dim=29, seed=81)
    shared, separate = HardRoutedMLP(**options), IndependentHardRoutedMLP(**options)
    for m in (shared, separate):
        m.router.centers.copy_(torch.eye(16, 64))
    value = inputs([15, 1, 7, 0, 3, 3, 14])
    torch.testing.assert_close(shared(value), separate(value), atol=1e-6, rtol=1e-6)
    storages = set()
    for i, expert in enumerate(separate.experts):
        for key, tensor in expert.observation.state_dict().items():
            torch.testing.assert_close(tensor, shared.observation.state_dict()[key], atol=0, rtol=0)
        for key, tensor in expert.head.state_dict().items():
            torch.testing.assert_close(tensor, shared.heads[i].state_dict()[key], atol=0, rtol=0)
        for p in expert.parameters():
            assert p.data_ptr() not in storages
            storages.add(p.data_ptr())


def test_dispatch_precedes_encoder_and_only_selected_complete_expert_learns():
    model = network()
    seen = {}
    handles = [e.observation.register_forward_pre_hook(
        lambda module, args, expert=i: seen.update({expert: args[0].shape[0]}))
        for i, e in enumerate(model.experts)]
    value = inputs([3] * 9).requires_grad_()
    model(value).square().mean().backward()
    for handle in handles:
        handle.remove()
    assert seen == {i: 9 if i == 3 else 0 for i in range(16)}
    for i, expert in enumerate(model.experts):
        assert all(p.grad is not None for p in expert.parameters())
        if i == 3:
            assert expert.observation[0][0].weight.grad.abs().sum() > 0
            assert expert.head[0].weight.grad[:, 128:].abs().sum() > 0
        else:
            assert all(p.grad.count_nonzero() == 0 for p in expert.parameters())
    assert value.grad[:, 12:].count_nonzero() == 0  # Raw tracker action and router latent are detached.


def test_routes_preserve_order_and_checkpointed_expert_identity():
    model = network()
    with torch.no_grad():
        for i, expert in enumerate(model.experts):
            expert.head[-1].weight.zero_()
            expert.head[-1].bias.fill_(i)
    ids = torch.tensor([15, 2, 7, 0, 2, 14, 7])
    value = inputs(ids).reshape(1, 7, -1)
    expected = ids.reshape(1, 7, 1).expand(-1, -1, 3).float()
    output, actual_ids = model(value, return_routes=True)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_ids, ids[None], atol=0, rtol=0)
    restored = network()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    torch.testing.assert_close(restored(value), expected, atol=0, rtol=0)


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_exploration_parameters_follow_current_routes_and_ppo_logprob(std_type):
    distribution = ExpertGaussianDistribution(GaussianDistribution(3, init_std=.25, std_type=std_type), 16)
    with torch.no_grad():
        for i, p in enumerate(distribution.expert_std_parameters):
            value = torch.tensor(.2 + .01 * i)
            p.fill_(value.log() if std_type == "log" else value)
    ids = torch.tensor([15, 2, 7, 0, 2])
    mean = torch.randn(5, 3, requires_grad=True)
    action = torch.randn(5, 3)
    distribution.update(mean, ids)
    expected_std = (.2 + .01 * ids[:, None]).expand_as(mean)
    torch.testing.assert_close(distribution.std, expected_std)
    old = tuple(p.detach().clone() for p in distribution.params)
    expected_log_prob = Normal(mean, expected_std).log_prob(action).sum(-1)
    torch.testing.assert_close(distribution.log_prob(action), expected_log_prob)
    order = torch.tensor([4, 1, 0, 2, 3])
    distribution.update(mean[order], ids[order])
    torch.testing.assert_close(distribution.log_prob(action[order]), expected_log_prob[order])
    torch.testing.assert_close(distribution.kl_divergence(tuple(p[order] for p in old), distribution.params), torch.zeros(5))
    distribution.log_prob(action[order]).sum().backward()
    for i, p in enumerate(distribution.expert_std_parameters):
        assert p.grad is not None
        assert bool(p.grad.abs().sum() > 0) == bool((ids == i).any())
    restored = ExpertGaussianDistribution(GaussianDistribution(3, std_type=std_type), 16)
    restored.load_state_dict(distribution.state_dict(), strict=True)
    restored.update(mean[order], ids[order])
    torch.testing.assert_close(restored.std, expected_std[order])
    with pytest.raises(ValueError, match="current mean"):
        distribution.update(mean)


def test_actor_mean_and_stochastic_policy_use_the_same_current_routes():
    class SmallActor(IndependentMoEActor):
        def _base_features_and_action(self, obs):
            return obs["features"], obs[ACTION_GROUP]

        def _residual_input(self, obs, features, action):
            return torch.cat((features, action, obs["dynamics_latent"]), -1)

    actor = SmallActor.__new__(SmallActor)
    nn.Module.__init__(actor)
    actor.residual_mlp = network(29, zero_output=True)
    actor.distribution = ExpertGaussianDistribution(GaussianDistribution(29, init_std=.25), 16)
    ids = torch.tensor([15, 0, 7, 3])
    obs = TensorDict({"features": torch.randn(4, 12), ACTION_GROUP: torch.randn(4, 29),
                      "dynamics_latent": torch.eye(16, 64)[ids]}, [4])
    torch.testing.assert_close(actor(obs), obs[ACTION_GROUP], atol=0, rtol=0)
    with torch.no_grad():
        for i, expert in enumerate(actor.residual_mlp.experts):
            expert.head[-1].bias.fill_(i)
            actor.distribution.expert_std_parameters[i].fill_(.2 + .01 * i)
    actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.output_mean, obs[ACTION_GROUP] + ids[:, None])
    torch.testing.assert_close(actor.output_std, (.2 + .01 * ids[:, None]).expand(4, 29))
    torch.testing.assert_close(actor.last_base_action, obs[ACTION_GROUP], atol=0, rtol=0)
    assert actor.last_residual_mean.abs().max() == 15  # No residual output clamp.


def test_independent_critic_retains_current_raw_tracker_action():
    critic = IndependentMoECritic.__new__(IndependentMoECritic)
    nn.Module.__init__(critic)
    critic.obs_dim, critic.obs_groups, critic.fusion_mode = 12, ["features"], "concat"
    critic.obs_normalizer = nn.Identity()
    critic.mlp = network(1)
    action = torch.randn(16, 29, requires_grad=True)
    latent = torch.eye(16, 64).requires_grad_()
    obs = TensorDict({"features": torch.randn(16, 12), ACTION_GROUP: action, "dynamics_latent": latent}, [16])
    seen = {}
    handles = [e.head.register_forward_pre_hook(
        lambda module, args, expert=i: seen.update({expert: args[0].detach().clone()}))
        for i, e in enumerate(critic.mlp.experts)]
    critic(obs).square().mean().backward()
    for h in handles:
        h.remove()
    for i, value in seen.items():
        torch.testing.assert_close(value[:, 128:], action.detach()[i:i + 1], atol=0, rtol=0)
    assert action.grad is None and latent.grad is None
    assert all(e.observation[0][0].weight.grad.abs().sum() > 0 for e in critic.mlp.experts)


def test_configuration_changes_only_architecture_classes():
    from intact_tracking.residual_uniform_protocol import configure_models
    old = configure_models({"actor": {}, "critic": {}}, "concat", scratch_seed=121)
    new = configure_independent_models({"actor": {}, "critic": {}}, "concat", scratch_seed=121)
    for side in ("actor", "critic"):
        assert "IndependentMoE" in new[side].pop("class_name")
        old[side].pop("class_name")
        assert old[side] == new[side]
    with pytest.raises(ValueError):
        configure_independent_models({"actor": {}, "critic": {}}, "baseline")
