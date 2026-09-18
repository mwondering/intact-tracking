import copy

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from intact_tracking.memory350_metric_router import FrozenMetricRouter, mix_expert_outputs
from intact_tracking.router_information_probe import fit_projection, project


def make_router(*,interpolation=False,tau=.5):
    matrix=torch.zeros(64,8);matrix[:8]=torch.eye(8)
    centers=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]).repeat(4,1,1)
    return FrozenMetricRouter(matrix,torch.full((8,),.5),centers,torch.full((4,),.3),
                              group_top_k=3,time_constant=tau,interpolation=interpolation)


def test_interpolation_keeps_all_eight_factor_coordinates():
    router=make_router(interpolation=True)
    x=torch.rand(127,8)
    x[:3]=torch.tensor([[0.]*8,[1.]*8,[.5]*8])
    weight=router.weights_from_features(x)
    groups=weight.reshape(-1,4,4)*4
    reconstructed=torch.stack((groups[...,1]+groups[...,3],groups[...,2]+groups[...,3]),-1).reshape(-1,8)
    torch.testing.assert_close(reconstructed,x,atol=2e-7,rtol=2e-7)
    torch.testing.assert_close(weight.sum(-1),torch.ones(len(x)))
    assert (weight>=0).all() and ((weight>0).sum(-1)<=12).all()


def test_router_has_no_ppo_gradient_and_replay_is_pure():
    router=make_router();z=torch.randn(7,64,requires_grad=True)
    recorded=router.advance(z,torch.ones(7,dtype=torch.bool),dt=.02)
    assert not list(router.parameters()) and not recorded.requires_grad
    state=copy.deepcopy(router.state_dict());history=router.filtered_features.clone()
    for _ in range(3):router(z[torch.randperm(7)])
    for key,value in router.state_dict().items():torch.testing.assert_close(value,state[key],atol=0,rtol=0)
    torch.testing.assert_close(router.filtered_features,history,atol=0,rtol=0)
    outputs=torch.randn(7,16,29,requires_grad=True)
    mix_expert_outputs(outputs,recorded).square().sum().backward()
    assert outputs.grad is not None and z.grad is None
    # Cached old-policy weights are ordinary immutable rollout values.
    saved=recorded.clone()
    router.update_centers_from_features(router.project(torch.randn(100,64)),center_rate=.5)
    torch.testing.assert_close(recorded,saved,atol=0,rtol=0)


def test_causal_filter_holds_short_reset_but_clears_changed_physics():
    router=make_router();old=torch.zeros(2,64);old[:,0]=-1
    new=old.clone();new[:,0]=1
    router.advance(old,torch.ones(2,dtype=torch.bool),dt=.02)
    before=router.filtered_features.clone()
    router.advance(new,torch.zeros(2,dtype=torch.bool),dt=.02)
    torch.testing.assert_close(router.filtered_features,before,atol=0,rtol=0)
    router.advance(new,torch.ones(2,dtype=torch.bool),dt=.02,parameters_changed=torch.tensor([True,False]))
    instant=router.project(new)
    torch.testing.assert_close(router.filtered_features[0],instant[0],atol=0,rtol=0)
    assert before[1,0]<router.filtered_features[1,0]<instant[1,0]
    state=copy.deepcopy(router.state_dict());restored=make_router();restored.load_state_dict(state)
    assert len(restored.history_valid)==0
    torch.testing.assert_close(router(new),restored(new))


def test_center_updates_respect_weight_change_cap():
    torch.manual_seed(17);router=make_router()
    features=torch.rand(101,8)*.7+.3
    previous=router.weights_from_features(features)
    result=router.update_centers_from_features(features,center_rate=1,max_mean_weight_tv=.001)
    actual=.5*(router.weights_from_features(features)-previous).abs().sum(-1).mean()
    assert float(actual)<=.001+1e-7
    assert result['samples']==101 and int(router.center_updates)==1


def test_same_world_metric_separates_persistent_signal_from_motion_noise():
    rng=np.random.default_rng(123);worlds=np.repeat(np.arange(80),32)
    environment=np.repeat(rng.normal(size=80),32)
    x=np.stack((.01*environment+rng.normal(scale=.0001,size=len(worlds)),rng.normal(size=len(worlds))),-1)
    metric=fit_projection(x,worlds,'reliability',dimension=1,floor=1e-6)
    z=project(x,metric).reshape(80,32)
    assert z.var(1).mean()/z.mean(1).var()<.01


def _distributed_worker(rank,path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        generator=torch.Generator().manual_seed(41)
        features=torch.rand(80,8,generator=generator)
        actual=make_router();reference=make_router()
        reference._pool=lambda value:None
        reference.update_centers_from_features(features,center_rate=.25,max_mean_weight_tv=.01)
        # A rank with zero eligible samples must still join all collectives.
        local=features[:0] if rank==0 else features
        actual.update_centers_from_features(local,center_rate=.25,max_mean_weight_tv=.01)
        torch.testing.assert_close(actual.centers,reference.centers,atol=1e-6,rtol=1e-6)
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_update_matches_pooled_worlds_including_empty_rank(tmp_path):
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)
