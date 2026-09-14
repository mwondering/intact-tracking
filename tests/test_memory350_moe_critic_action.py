from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from intact_tracking.memory350_moe_critic_action import (
    ACTION_GROUP, TrackerActionMoEPPO, TrackerActionMoECritic, TrackerActionSlotMixin,
    reserve_tracker_action, attach_tracker_action, configure_critic_action_models,
)
from intact_tracking.memory350_moe_policy import HardRoutedMLP
from intact_tracking.memory350_moe_training import OnlineMoEPPO


class FakeActor:
    def __init__(self):
        self.forward_calls = 0
        self.bootstrap_calls = 0

    def get_hidden_state(self):
        return None

    def __call__(self, obs, stochastic_output=False):
        self.forward_calls += 1
        self.last_base_action = 2 * obs['state'] + 7
        self.output_distribution_params = (self.last_base_action + 100, torch.ones_like(self.last_base_action))
        return self.last_base_action + 100

    def _base_features_and_action(self, obs):
        self.bootstrap_calls += 1
        return obs['state'], 2 * obs['state'] + 7

    def get_output_log_prob(self, action):
        return action.new_zeros(len(action))


def test_collector_captures_current_raw_mean_once_before_critic_and_storage():
    actor = FakeActor()
    observations = reserve_tracker_action(TensorDict({'state':torch.randn(5, 29)}, [5]))
    seen = []
    class Critic:
        def get_hidden_state(self):
            return None
        def __call__(self, obs):
            seen.append(obs[ACTION_GROUP].clone())
            return obs[ACTION_GROUP].mean(-1, keepdim=True)
    ppo = TrackerActionMoEPPO.__new__(TrackerActionMoEPPO)
    ppo.actor, ppo.critic, ppo.transition = actor, Critic(), RolloutStorage.Transition()
    actions = ppo.act(observations)
    expected = 2 * observations['state'] + 7
    assert actor.forward_calls == 1 and actor.bootstrap_calls == 0
    torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(ppo.transition.observations[ACTION_GROUP], expected, atol=0, rtol=0)
    assert not torch.equal(actions, expected)
    actor.last_base_action.zero_()
    torch.testing.assert_close(ppo.transition.observations[ACTION_GROUP], expected, atol=0, rtol=0)


def test_next_state_bootstrap_replaces_previous_tracker_action(monkeypatch):
    ppo = TrackerActionMoEPPO.__new__(TrackerActionMoEPPO)
    ppo.actor = FakeActor()
    obs = reserve_tracker_action(TensorDict({'state':torch.randn(3,29)}, [3]))
    obs[ACTION_GROUP].fill_(-999)
    seen = []
    def compute(self, value):
        seen.append(value[ACTION_GROUP].clone())
    monkeypatch.setattr(OnlineMoEPPO, 'compute_returns', compute)
    ppo.compute_returns(obs)
    torch.testing.assert_close(seen[0], 2 * obs['state'] + 7, atol=0, rtol=0)
    assert ppo.actor.bootstrap_calls == 1 and ppo.actor.forward_calls == 0


def test_saved_action_stays_paired_with_its_state_after_minibatch_shuffle():
    template = reserve_tracker_action(TensorDict({'state':torch.zeros(4,29)}, [4]))
    storage = RolloutStorage('rl', 4, 2, template, [29], device='cpu')
    for step in range(2):
        obs = reserve_tracker_action(TensorDict({'state':torch.arange(4.)[:,None].expand(-1,29)+10*step}, [4]))
        action = 2 * obs['state'] + 7
        attach_tracker_action(obs, action)
        tr = RolloutStorage.Transition()
        tr.observations, tr.actions = obs, action + 100
        tr.rewards, tr.dones = torch.ones(4), torch.zeros(4)
        tr.values, tr.actions_log_prob = torch.zeros(4,1), torch.zeros(4)
        tr.distribution_params = (action+100, torch.ones_like(action))
        storage.add_transition(tr)
        action.zero_()
    for batch in storage.mini_batch_generator(2, 1):
        torch.testing.assert_close(batch.observations[ACTION_GROUP], 2*batch.observations['state']+7, atol=0, rtol=0)
        assert not torch.equal(batch.observations[ACTION_GROUP], batch.actions)


@pytest.mark.parametrize('fusion', ['baseline','concat'])
def test_critic_heads_receive_tracker_action_and_learn_action_columns(fusion):
    critic = TrackerActionMoECritic.__new__(TrackerActionMoECritic)
    nn.Module.__init__(critic)
    critic.obs_dim, critic.obs_groups, critic.fusion_mode = 12, ['critic_state'], fusion
    critic.obs_normalizer = nn.Identity()
    critic.mlp = HardRoutedMLP(12,1,compression_dims=(32,128),fusion=fusion,tracker_action_dim=29,seed=81)
    n=16
    z=torch.eye(16,64).requires_grad_()
    if fusion=='concat':critic.mlp.router.centers.copy_(torch.eye(16,64))
    action=torch.randn(n,29,requires_grad=True)
    obs=TensorDict({'critic_state':torch.randn(n,12),ACTION_GROUP:action,'dynamics_latent':z},[n])
    seen={}
    handles=[]
    for i,head in enumerate(critic.mlp.heads):
        handles.append(head.register_forward_pre_hook(lambda module, inputs, expert=i: seen.update({expert:inputs[0].detach().clone()})))
    value=critic(obs)
    for h in handles:h.remove()
    for i,inputs in seen.items():
        expected=action.detach() if fusion=='baseline' else action.detach()[i:i+1]
        torch.testing.assert_close(inputs[:,128:],expected,atol=0,rtol=0)
    changed=obs.clone();changed[ACTION_GROUP]=action.detach()+1
    assert not torch.allclose(value,critic(changed))
    value.square().mean().backward()
    assert action.grad is None and z.grad is None
    assert critic.mlp.observation[0][0].weight.grad.abs().sum()>0
    assert all(head[0].weight.grad[:,128:].abs().sum()>0 for head in critic.mlp.heads)


def test_new_observation_slots_do_not_mutate_previous_steps():
    class Env:
        def get_observations(self):
            return TensorDict({'state':torch.randn(2,29)},[2])
        def reset(self):
            return self.get_observations(),{}
        def step(self, actions):
            return self.get_observations(),torch.zeros(2),torch.zeros(2),{}
    class Wrapped(TrackerActionSlotMixin,Env):
        pass
    env=Wrapped();old=env.get_observations()
    attach_tracker_action(old,torch.ones(2,29))
    new,*_=env.step(torch.zeros(2,29))
    assert not new[ACTION_GROUP].any()
    assert old[ACTION_GROUP].eq(1).all()


def test_configuration_changes_only_critic_implementation():
    from intact_tracking.memory350_moe_policy import configure_moe_models
    old=configure_moe_models({'actor':{},'critic':{}},'concat',scratch_seed=121)
    new=configure_critic_action_models({'actor':{},'critic':{}},'concat',scratch_seed=121)
    assert old['actor']==new['actor']
    assert new['critic'].pop('class_name').endswith(':TrackerActionMoECritic')
    old['critic'].pop('class_name')
    assert old['critic']==new['critic']
