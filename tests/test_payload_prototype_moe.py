import copy

import pytest
import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict
from rsl_rl.modules.distribution import GaussianDistribution

from intact_tracking.payload_prototype_moe import (
    SparseExperts, PrototypeGate, PayloadMoEActor, PayloadMoECritic, mlp, select_top5,
    ROUTE_IDS, ROUTE_WEIGHTS, TRACKER_ACTION,
)
from intact_tracking.payload_prototype_physics import load_grid, load_bank


def test_balanced_anchor_grid_and_continuous_loads_with_asymmetric_caps():
    grid = load_grid()
    assert grid.shape == (256,4) and len(grid.unique(dim=0)) == 256
    torch.testing.assert_close(grid.max(0).values,torch.tensor([2.5,2.5,4.,4.]))
    mass,ids = load_bank(8192,121)
    torch.testing.assert_close(torch.bincount(ids[ids>=0]),torch.full((256,),16))
    torch.testing.assert_close(mass[:4096],grid[ids[:4096]],atol=0,rtol=0)
    assert (mass>=0).all() and (mass<=grid.max(0).values).all()
    other,_ = load_bank(8192,122)
    torch.testing.assert_close(mass[:4096],other[:4096],atol=0,rtol=0)
    assert not torch.equal(mass[4096:],other[4096:])


def test_sparse_dispatch_matches_dense_reference_outputs_and_gradients_including_unused_experts():
    torch.manual_seed(80)
    sparse = SparseExperts(3,2,experts=7)
    dense = copy.deepcopy(sparse)
    x,extra = torch.randn(11,512),torch.randn(11,3)
    logits = torch.randn(11,6,requires_grad=True)
    other = logits.detach().clone().requires_grad_()
    ids,weights = select_top5(logits)
    _,dw = select_top5(other)
    actual = sparse(x,extra,ids,weights)
    all_outputs = torch.stack([m(x,extra) for m in dense.experts],1)
    expected = (all_outputs.gather(1,ids[...,None].expand(-1,-1,2))*dw[...,None]).sum(1)
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for p,q in zip(sparse.parameters(),dense.parameters(),strict=True):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=1e-4)
    torch.testing.assert_close(logits.grad,other.grad,atol=2e-7,rtol=1e-4)
    assert all(p.grad.count_nonzero()==0 for p in sparse.experts[6].parameters())


def test_prototype_routing_has_no_gradients_and_preserves_nearest_centroid_metric():
    centers = torch.nn.functional.normalize(torch.randn(256,64),dim=-1)*0.95
    gate = PrototypeGate(centers,0.02)
    z = centers[:19].clone().requires_grad_()
    ids,w = gate(z)
    assert torch.equal(ids[:,0],torch.arange(19))
    torch.testing.assert_close(w.sum(-1),torch.ones(19))
    assert not w.requires_grad and list(gate.parameters()) == []
    restored = PrototypeGate(torch.zeros_like(centers),1.)
    restored.load_state_dict(gate.state_dict())
    torch.testing.assert_close(restored(z)[1],w,atol=0,rtol=0)


class SmallActor(PayloadMoEActor):
    def _base_features_and_action(self,obs):
        return obs['features'].detach(),obs['raw_action'].detach()


def small_actor():
    actor = SmallActor.__new__(SmallActor)
    nn.Module.__init__(actor)
    actor.route_mode = 'learned'
    actor.obs_encoder = nn.Sequential(nn.Linear(12,512),nn.ELU())
    actor.expert_bank = SparseExperts(29,29,experts=7,zero_output=True)
    actor.gate = mlp([541,16,7])
    actor.distribution = GaussianDistribution(29,init_std=.25)
    return actor


def small_critic(mode='learned'):
    critic = PayloadMoECritic.__new__(PayloadMoECritic)
    nn.Module.__init__(critic)
    critic.obs_groups,critic.obs_dim,critic.route_mode = ['critic'],10,mode
    critic.obs_normalizer = nn.Identity()
    critic.obs_encoder = nn.Sequential(nn.Linear(10,512),nn.ELU())
    critic.expert_bank = SparseExperts(93,1,experts=7)
    return critic


def test_baseline_routes_and_critic_ignore_latent_and_value_loss_cannot_train_actor():
    actor,critic = small_actor(),small_critic()
    obs = TensorDict({'features':torch.randn(8,12),'raw_action':torch.randn(8,29),
                      'critic':torch.randn(8,10),'dynamics_latent':torch.randn(8,64)},[8])
    actor(obs)
    ids,weights,values = obs[ROUTE_IDS].clone(),obs[ROUTE_WEIGHTS].clone(),critic(obs)
    obs['dynamics_latent'] = torch.full((8,64),float('nan'))
    actor(obs)
    torch.testing.assert_close(obs[ROUTE_IDS],ids,atol=0,rtol=0)
    torch.testing.assert_close(obs[ROUTE_WEIGHTS],weights,atol=0,rtol=0)
    torch.testing.assert_close(critic(obs),values,atol=0,rtol=0)
    critic(obs).square().sum().backward()
    assert all(p.grad is None for p in actor.parameters())
    assert critic.obs_encoder[0].weight.grad.abs().sum()>0


def test_ppo_gaussian_matches_mixed_mean_and_router_learns_after_zero_head_step():
    actor = small_actor()
    obs = TensorDict({'features':torch.randn(9,12),'raw_action':torch.randn(9,29)},[9])
    target = torch.randn(9,29)
    optimizer = torch.optim.Adam(actor.parameters(),lr=.003)
    for step in range(3):
        actor(obs,stochastic_output=True)
        if step == 0:
            torch.testing.assert_close(actor.output_mean,obs['raw_action'],atol=0,rtol=0)
        expected = Normal(actor.output_mean,actor.output_std).log_prob(target).sum(-1)
        torch.testing.assert_close(actor.get_output_log_prob(target),expected)
        optimizer.zero_grad()
        (-actor.get_output_log_prob(target).mean()).backward()
        assert all(p.grad is not None for p in actor.parameters())
        if step: assert actor.gate[-1].weight.grad.abs().sum()>0
        optimizer.step()
    original = actor(obs).detach()
    perm = torch.randperm(9)
    torch.testing.assert_close(actor(obs[perm]),original[perm],atol=1e-6,rtol=1e-6)


def test_latent_critic_reads_detached_latent_after_private_encoder():
    critic = small_critic('latent')
    z = torch.randn(8,64,requires_grad=True)
    obs = TensorDict({'critic':torch.randn(8,10),'dynamics_latent':z,
                      TRACKER_ACTION:torch.randn(8,29),ROUTE_IDS:torch.arange(5).expand(8,-1),
                      ROUTE_WEIGHTS:torch.full((8,5),.2)},[8])
    a = critic(obs)
    other = obs.clone()
    other['dynamics_latent'] = z.roll(1,0)
    assert not torch.allclose(a,critic(other))
    a.square().mean().backward()
    assert z.grad is None


def test_cli_keeps_cold_start_uniform_no_update_cap():
    from intact_tracking.cli.payload_prototype_train import build_parser
    args = build_parser().parse_args(['--route-mode','learned','--prototype-file','p.pt','--output-dir','runs/test'])
    assert args.iterations is None and args.num_envs==8192
    assert args.motion_sampling=='uniform' and args.training_terminations=='original'
    assert args.context_checkpoint is None and args.residual_scale==1.
