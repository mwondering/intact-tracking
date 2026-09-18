"""Matched observation-routed and fixed-latent-prototype Top-5 residual MoEs."""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.limb_context_policy import LimbContextCritic, configure_context_models
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.residual_policy import FrozenTrackerResidualActor

VERSION = "payload_prototype256_top5_residual_moe_v1"
INITIALIZATION = "independent_expert_hidden_zero_actor_heads_fresh_critic_v1"
ROUTE_IDS, ROUTE_WEIGHTS, TRACKER_ACTION = "payload_route_ids", "payload_route_weights", "frozen_tracker_action"


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def select_top5(logits, k=5):
    values, ids = logits.topk(k, dim=-1)
    return ids, values.softmax(-1)


class PrototypeGate(nn.Module):
    def __init__(self, centers, temperature):
        super().__init__()
        if centers.shape != (256,64) or not torch.isfinite(centers).all() or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Expected 256 finite 64-D prototypes and positive temperature")
        self.register_buffer("centers", centers.detach().float().clone())
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, latent):
        z = F.normalize(latent.detach().float(), dim=-1, eps=1e-8)
        d2 = (z.square().sum(-1,keepdim=True) + self.centers.square().sum(-1)[None] - 2*z@self.centers.T).clamp_min(0)
        return select_top5(-d2/self.temperature)


class PrivateExpert(nn.Module):
    def __init__(self, extra_dim, output_dim, *, zero_output=False):
        super().__init__()
        self.features = nn.Sequential(mlp([512,256,128]), nn.ELU())
        self.head = mlp([128+extra_dim,256,128,output_dim])
        if zero_output:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, extra):
        return self.head(torch.cat((self.features(x), extra), dim=-1))


class SparseExperts(nn.Module):
    """Sort once, run contiguous expert batches, then mix outputs.

    One counts transfer per dispatch avoids 256 CUDA nonzero synchronizations.
    Empty batches retain zero gradients on unused experts on every DDP rank.
    No per-token weight expansion and no capacity overflow/drop are allowed.
    """
    def __init__(self, extra_dim, output_dim, *, experts=256, zero_output=False):
        super().__init__()
        self.experts = nn.ModuleList([PrivateExpert(extra_dim,output_dim,zero_output=zero_output) for _ in range(experts)])

    def forward(self, x, extra, ids, weights):
        n, k = ids.shape
        order = ids.flatten().argsort(stable=True)
        counts = torch.bincount(ids.flatten(),minlength=len(self.experts)).tolist()
        token = torch.div(order,k,rounding_mode="floor")
        xs, es = x[token].split(counts), extra[token].split(counts)
        outputs = torch.cat([expert(a,b) for expert,a,b in zip(self.experts,xs,es,strict=True)],0)
        restored = torch.empty_like(outputs).index_copy(0,order,outputs).reshape(n,k,-1)
        return (restored*weights[...,None]).sum(1)


def reserve_slots(obs):
    n, device = obs.batch_size[0], obs.device
    if device is None:
        device = next(iter(obs.values())).device
    obs.set(TRACKER_ACTION, torch.zeros(n,29,device=device))
    obs.set(ROUTE_IDS, torch.zeros(n,5,dtype=torch.long,device=device))
    obs.set(ROUTE_WEIGHTS, torch.zeros(n,5,device=device))
    return obs


class PayloadMoEWrapper(Memory350PolicyWrapper):
    def get_observations(self):
        return reserve_slots(super().get_observations())

    def reset(self):
        obs, extras = super().reset()
        return reserve_slots(obs), extras

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        return reserve_slots(obs), rewards, dones, extras


class PayloadMoEActor(FrozenTrackerResidualActor):
    def __init__(self, *args, route_mode, prototype_file=None, initialization_seed=10128,
                 initial_action_std=0.25, **kwargs):
        kwargs.pop("fusion_mode",None)
        kwargs["use_dynamics_latent"] = False
        super().__init__(*args, **kwargs)
        if route_mode not in ("learned","latent"):
            raise ValueError(route_mode)
        self.route_mode, self.initialization_seed = route_mode, initialization_seed
        self.fusion_mode = "baseline" if route_mode == "learned" else "concat"
        self.use_dynamics_latent = route_mode == "latent"
        del self.residual_mlp
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.obs_encoder = nn.Sequential(nn.Linear(self.tracker.policy_input_dim,512),nn.ELU())
            self.expert_bank = SparseExperts(29,29,zero_output=True)
        if route_mode == "latent":
            data = torch.load(prototype_file,map_location="cpu",weights_only=False)
            self.gate = PrototypeGate(data["centers"],data["temperature"])
            self.prototype_sha256 = hashlib.sha256(Path(prototype_file).read_bytes()).hexdigest()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(initialization_seed+17001)
                self.gate = mlp([512+29,128,256])
            self.prototype_sha256 = None
        cfg = dict(kwargs.get("distribution_cfg") or {})
        cfg.pop("class_name",None)
        cfg["init_std"] = initial_action_std
        self.distribution = GaussianDistribution(self.distribution.output_dim,**cfg)
        self.tracker_digest = tensor_digest(self.tracker.state_dict().items())
        self.register_buffer("rollout_usage",torch.zeros(256),persistent=False)
        self.register_buffer("rollout_switch",torch.zeros(2),persistent=False)
        self.previous_top1 = None

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        del hidden_state
        obs = unpad_trajectories(obs,masks) if masks is not None else obs
        features, base = self._base_features_and_action(obs)
        common = self.obs_encoder(features)
        if self.route_mode == "latent":
            ids, weights = self.gate(obs["dynamics_latent"])
        else:
            ids, weights = select_top5(self.gate(torch.cat((common,base),-1)))
        residual = self.expert_bank(common,base,ids,weights)
        mean = base+residual
        # Critic uses the current actor's assignments, with no gradient into it.
        obs.set(TRACKER_ACTION,base.detach())
        obs.set(ROUTE_IDS,ids.detach())
        obs.set(ROUTE_WEIGHTS,weights.detach())
        self.last_base_action, self.last_residual_mean = base,residual.detach()
        self.last_route_ids, self.last_route_weights = ids.detach(),weights.detach()
        if torch.is_inference_mode_enabled() and stochastic_output:
            self.rollout_usage.scatter_add_(0,ids.flatten(),weights.flatten())
            if self.previous_top1 is not None and self.previous_top1.shape == ids[:,0].shape:
                self.rollout_switch[0] += (self.previous_top1 != ids[:,0]).sum()
                self.rollout_switch[1] += ids.shape[0]
            self.previous_top1 = ids[:,0].clone()
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def assert_tracker_frozen(self):
        if self.tracker.training or any(p.requires_grad or p.grad is not None for p in self.tracker.parameters()):
            raise RuntimeError("Frozen tracker acquired training state/gradients")
        if tensor_digest(self.tracker.state_dict().items()) != self.tracker_digest:
            raise RuntimeError("Frozen tracker weights/normalizers changed")

    @torch.no_grad()
    def policy_metrics(self,obs):
        self(obs)
        usage, switches = self.rollout_usage.clone(), self.rollout_switch.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(usage)
            torch.distributed.all_reduce(switches)
        p = usage / usage.sum().clamp_min(1)
        w = self.last_route_weights
        result = {"base_action_rms":float(self.last_base_action.square().mean().sqrt()),
                  "residual_action_rms":float(self.last_residual_mean.square().mean().sqrt()),
                  "residual_action_abs_max":float(self.last_residual_mean.abs().max()),
                  "residual_output_bounded":0., "route_mean_max_weight":float(w.max(-1).values.mean()),
                  "route_per_sample_effective_experts":float((1/w.square().sum(-1)).mean()),
                  "route_population_effective_experts":float((-(p*p.clamp_min(1e-12).log()).sum()).exp()),
                  "route_population_used_experts":float((p>0).sum()),
                  "route_step_top1_switch_fraction":float(switches[0]/switches[1].clamp_min(1))}
        for i, value in enumerate(p.tolist()):
            result[f"expert_{i:03d}_weight_fraction"] = value
        self.rollout_usage.zero_()
        self.rollout_switch.zero_()
        return result


class PayloadMoECritic(LimbContextCritic):
    def __init__(self,*args,route_mode,initialization_seed=20124,**kwargs):
        kwargs.pop("fusion_mode",None)
        super().__init__(*args,fusion_mode="baseline",initialization_seed=initialization_seed,**kwargs)
        self.route_mode = route_mode
        del self.mlp
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.obs_encoder = nn.Sequential(mlp([self.obs_dim,1024,512]),nn.ELU())
            self.expert_bank = SparseExperts(29+64,1)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        del hidden_state,stochastic_output
        obs = unpad_trajectories(obs,masks) if masks is not None else obs
        x = self.obs_encoder(self.obs_normalizer(self._flat_obs(obs)))
        z = (F.normalize(obs["dynamics_latent"].detach(),dim=-1,eps=1e-8)
             if self.route_mode == "latent" else x.new_zeros((x.shape[0],64)))
        extra = torch.cat((obs[TRACKER_ACTION].detach(),z),-1)
        return self.expert_bank(x,extra,obs[ROUTE_IDS].detach(),obs[ROUTE_WEIGHTS].detach())


def configure_models(train,fusion,*,scratch_seed,route_mode,prototype_file=None):
    result = configure_context_models(train,"baseline",scratch_seed=scratch_seed)
    result["actor"].update(class_name="intact_tracking.payload_prototype_moe:PayloadMoEActor",
                           route_mode=route_mode,prototype_file=prototype_file,residual_scale=1.)
    result["critic"].update(class_name="intact_tracking.payload_prototype_moe:PayloadMoECritic",route_mode=route_mode)
    return result


@torch.no_grad()
def audit_initial_models(actor,critic,obs,fusion):
    features,base = actor._base_features_and_action(obs)
    actual = actor(obs)
    torch.testing.assert_close(actual,base,atol=0,rtol=0)
    torch.testing.assert_close(actual,actor.tracker(obs),atol=0,rtol=0)
    actor.distribution.update(actual)
    torch.testing.assert_close(actor.output_std,torch.full_like(actual,0.25),atol=0,rtol=0)
    values = critic(obs)
    if not torch.isfinite(values).all(): raise RuntimeError("Nonfinite initial value")
    if set(map(id,actor.parameters())) & set(map(id,critic.parameters())):
        raise RuntimeError("Actor and critic cannot share trainable parameters")
    actor.assert_tracker_frozen()
    return {"actor_original_features":features.shape[-1],"critic_original_features":critic.obs_dim,
            "route_mode":actor.route_mode,"experts":256,"top_k":5,"identical_initial_action":True,
            "action_std_initialization":0.25,"actor_critic_parameters_disjoint":True,
            "baseline_has_no_latent_input_or_route":actor.route_mode=="learned",
            "prototype_sha256":actor.prototype_sha256,
            "actor_expert_sha256":tensor_digest(actor.expert_bank.state_dict().items()),
            "critic_expert_sha256":tensor_digest(critic.expert_bank.state_dict().items()),
            "actor_trainable_parameters":sum(p.numel() for p in actor.parameters() if p.requires_grad),
            "critic_trainable_parameters":sum(p.numel() for p in critic.parameters() if p.requires_grad)}
