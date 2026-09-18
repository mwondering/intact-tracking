"""Frozen latent metric and continuous gates for offline-validated router prototypes.

Advance causal state once per environment observation outside PPO forwards.
Store returned weights in rollout observations and replay those exact weights
in actor/critic minibatches. Update KMeans centers only after PPO finishes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrozenMetricRouter(nn.Module):
    def __init__(self, matrix, offset, centers, temperatures, *, group_top_k,
                 time_constant=0., clip_unit_range=True, interpolation=False):
        super().__init__()
        matrix=torch.as_tensor(matrix,dtype=torch.float32)
        offset=torch.as_tensor(offset,dtype=torch.float32)
        centers=torch.as_tensor(centers,dtype=torch.float32)
        temperatures=torch.as_tensor(temperatures,dtype=torch.float32).reshape(-1)
        if matrix.ndim!=2 or matrix.shape[0]!=64 or centers.ndim!=3:
            raise ValueError('Expected a64-D input metric and [groups,centers,dimensions] prototypes')
        self.groups,self.per_group,self.group_dimension=centers.shape
        if matrix.shape[1]!=self.groups*self.group_dimension or offset.shape!=(matrix.shape[1],):
            raise ValueError('Metric and group dimensions disagree')
        if temperatures.shape!=(self.groups,) or not bool((temperatures>0).all()):
            raise ValueError('One positive temperature per group is required')
        if not 1<=group_top_k<=self.per_group or time_constant<0:
            raise ValueError('Invalid sparsity or time constant')
        if interpolation and (self.groups,self.per_group,self.group_dimension)!=(4,4,2):
            raise ValueError('Triangular interpolation requires four2-D groups with four corners')
        for name,value in [('matrix',matrix),('offset',offset),('centers',centers),('temperatures',temperatures)]:
            if not torch.isfinite(value).all():raise ValueError(f'Nonfinite{name}')
            self.register_buffer(name,value.clone())
        self.group_top_k=int(group_top_k)
        self.time_constant=float(time_constant)
        self.clip_unit_range=bool(clip_unit_range)
        self.interpolation=bool(interpolation)
        self.register_buffer('center_updates',torch.zeros((),dtype=torch.long))
        # World identities/physics are recreated on a new rollout, so these
        # caches intentionally are not restored from a policy checkpoint.
        self.register_buffer('filtered_features',torch.empty(0,matrix.shape[1]),persistent=False)
        self.register_buffer('history_valid',torch.empty(0,dtype=torch.bool),persistent=False)

    @torch.no_grad()
    def project(self,latent):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            features=F.normalize(latent.detach().float(),dim=-1,eps=1e-8)@self.matrix+self.offset
            return features.clamp(0,1) if self.clip_unit_range else features

    @torch.no_grad()
    def weights_from_features(self,features,*,centers=None):
        centers=self.centers if centers is None else centers
        shape=features.shape[:-1]
        x=features.detach().float().reshape(-1,self.groups,self.group_dimension)
        if self.interpolation:
            u,v=x.clamp(0,1).unbind(-1)
            low=u+v<=1
            weights=torch.stack(((1-u-v).clamp_min(0),torch.where(low,u,1-v),
                                 torch.where(low,v,1-u),(u+v-1).clamp_min(0)),-1)
        else:
            distance=(x[:,:,None,:]-centers[None]).square().sum(-1)
            if self.group_top_k==1:
                weights=F.one_hot(distance.argmin(-1),self.per_group).float()
            else:
                scores=-distance/self.temperatures[None,:,None]
                if self.group_top_k<self.per_group:
                    ids=scores.topk(self.group_top_k,dim=-1).indices
                    keep=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,ids,True)
                    scores=scores.masked_fill(~keep,-torch.inf)
                weights=scores.softmax(-1)
        return (weights/self.groups).reshape(*shape,self.groups*self.per_group)

    @torch.no_grad()
    def forward(self,latent):
        """Instantaneous pure function. Never advances EMA or KMeans state."""
        return self.weights_from_features(self.project(latent))

    @torch.no_grad()
    def advance(self,latent,full_memory_valid,*,dt,parameters_changed=None):
        """Once per environment step; episode reset alone does not clear DR memory."""
        if latent.ndim!=2 or dt<=0:
            raise ValueError('advance expects [world,64] latent and positive dt')
        features=self.project(latent)
        valid=torch.as_tensor(full_memory_valid,device=features.device,dtype=torch.bool)
        if valid.shape!=(len(features),):raise ValueError('One validity bit per world is required')
        if self.filtered_features.shape!=features.shape:
            self.filtered_features=features.clone()
            self.history_valid=torch.zeros(len(features),device=features.device,dtype=torch.bool)
        if parameters_changed is not None:
            changed=torch.as_tensor(parameters_changed,device=features.device,dtype=torch.bool)
            if changed.shape!=valid.shape:raise ValueError('One physics-change bit per world is required')
            self.history_valid[changed]=False
            self.filtered_features[changed]=features[changed]
        first=valid&~self.history_valid
        continuing=valid&self.history_valid
        alpha=1. if self.time_constant==0 else -math.expm1(-dt/self.time_constant)
        self.filtered_features[first]=features[first]
        self.filtered_features[continuing]+=alpha*(features[continuing]-self.filtered_features[continuing])
        self.history_valid|=valid
        return self.weights_from_features(self.filtered_features)

    @torch.no_grad()
    def clear_history(self):
        self.filtered_features=self.filtered_features[:0]
        self.history_valid=self.history_valid[:0]

    @staticmethod
    def _pool(value):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value)

    @torch.no_grad()
    def update_centers_from_features(self,features,*,center_rate=.01,max_mean_weight_tv=.02):
        """Explicit post-PPO groupwise KMeans update with pooled weight-change cap."""
        if self.interpolation:raise RuntimeError('Fixed interpolation corners are not KMeans prototypes')
        if not 0<center_rate<=1 or not 0<max_mean_weight_tv<=1:
            raise ValueError('Invalid update bounds')
        x=features.detach().float().reshape(-1,self.groups,self.group_dimension)
        old=self.centers.clone()
        ids=(x[:,:,None,:]-old[None]).square().sum(-1).argmin(-1)
        sums=torch.zeros_like(old)
        counts=old.new_zeros(self.groups,self.per_group)
        for g in range(self.groups):
            sums[g].index_add_(0,ids[:,g],x[:,g])
            counts[g]=torch.bincount(ids[:,g],minlength=self.per_group)
        packed=torch.cat((sums.flatten(),counts.flatten()))
        self._pool(packed)
        sums=packed[:old.numel()].reshape_as(old)
        counts=packed[old.numel():].reshape_as(counts)
        total=float(counts.sum()/self.groups)
        if total==0:return {'updated':False,'samples':0,'mean_weight_tv':0.}
        delta=center_rate*(sums/counts.clamp_min(1)[...,None]-old)*(counts>0)[...,None]
        previous=self.weights_from_features(features,centers=old)
        scale=1.
        for _ in range(12):
            candidate=old+scale*delta
            current=self.weights_from_features(features,centers=candidate)
            stats=old.new_tensor([float((.5*(current-previous).abs().sum(-1)).sum()),len(x)],dtype=torch.float64)
            self._pool(stats)
            tv=float(stats[0]/stats[1].clamp_min(1))
            if tv<=max_mean_weight_tv:break
            scale*=.5
        else:
            candidate,scale,tv=old,0.,0.
        self.centers.copy_(candidate)
        self.center_updates.add_(1)
        return {'updated':scale>0,'samples':int(total),'mean_weight_tv':tv,'effective_center_rate':center_rate*scale}


def mix_expert_outputs(outputs,weights):
    """Mix residual means or values; retain the policy's single Gaussian log-prob."""
    if outputs.shape[:-1]!=weights.shape:raise ValueError('Expected [...,experts,output] and [...,experts]')
    return (outputs*weights.detach()[...,None]).sum(-2)
