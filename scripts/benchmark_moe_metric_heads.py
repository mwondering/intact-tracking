"""FP32 shared-encoder actor+critic microbenchmark; no environment or PPO update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from intact_tracking.memory350_moe_policy import HardRoutedMLP
from intact_tracking.memory350_metric_router import FrozenMetricRouter


def restore_router(path,device):
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    s=artifact['state_dict']
    router=FrozenMetricRouter(s['matrix'],s['offset'],s['centers'],s['temperatures'],**artifact['configuration'])
    router.load_state_dict(s,strict=True)
    return router.to(device)


def mixed(module,observation,weights,*,dense):
    h=module.observation(observation[:,:module.feature_dim])
    h=torch.cat((h,observation[:,module.feature_dim:module.feature_dim+29]),-1)
    if dense:
        return sum(head(h)*weights[:,i:i+1] for i,head in enumerate(module.heads))
    result=h.new_zeros(len(h),module.output_dim)
    for i,head in enumerate(module.heads):
        ids=(weights[:,i]>0).nonzero().flatten()
        result=result.index_add(0,ids,head(h[ids])*weights[ids,i:i+1])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output-name',default='head_benchmark.json');args=p.parse_args();root=Path(args.root)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(67)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    actor=HardRoutedMLP(1645,29,compression_dims=(512,256,128),fusion='concat',tracker_action_dim=29)
    critic=HardRoutedMLP(6330,1,compression_dims=(1024,512,256,128),fusion='concat',tracker_action_dim=29)
    for module,key,prefix in [(actor,'actor_state_dict','residual_mlp.'),(critic,'critic_state_dict','mlp.')]:
        module.load_state_dict({k.removeprefix(prefix):v for k,v in state[key].items() if k.startswith(prefix)},strict=True)
        module.to(args.device)
    del state
    with np.load(root/'heldout_sample_seed30403/traces.npz') as f:z=torch.from_numpy(f['unit_latent_samples'][50]).to(args.device)
    n=len(z);actor_obs=torch.cat((torch.randn(n,1645+29,device=args.device),z),-1)
    critic_obs=torch.cat((torch.randn(n,6330+29,device=args.device),z),-1)
    names=['decoded_dr8_k8_t1','product_dr8_gk3_t1','product_dr8_gk4_t1','factorized_triangular']
    weights={name:restore_router(root/'exports'/f'{name}.pt',args.device)(z).detach() for name in names}
    hard=torch.nn.functional.one_hot(actor.router(z),16).float()
    weights['dense_hard_control']=hard
    with torch.no_grad():
        torch.testing.assert_close(mixed(actor,actor_obs,hard,dense=False),actor(actor_obs),atol=2e-6,rtol=2e-6)
    def run(name,backward):
        if backward:actor.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            if name=='original_hard16':a,v=actor(actor_obs),critic(critic_obs)
            else:
                dense=name in ('product_dr8_gk4_t1','dense_hard_control')
                a,v=mixed(actor,actor_obs,weights[name],dense=dense),mixed(critic,critic_obs,weights[name],dense=dense)
            if backward:(a.square().mean()+v.square().mean()).backward()
    cases=['original_hard16','dense_hard_control']+names;results={}
    for backward in (False,True):
        for name in cases:
            for _ in range(3):run(name,backward)
        torch.cuda.synchronize()
        samples={name:[] for name in cases}
        for repeat in range(15):
            for name in np.random.default_rng(repeat).permutation(cases):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();run(name,backward);end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end))
        label='forward_backward_ms' if backward else 'forward_ms'
        results[label]={name:{'median':float(np.median(v)),'p10':float(np.quantile(v,.1)),'p90':float(np.quantile(v,.9)),'samples':v} for name,v in samples.items()}
    report={'scope':'Actor+critic MLPs only, FP32,1024 synthetic normalized observations and actual heldout latents. Gate weights cached; no encoder/tracker/simulation/DDP/optimizer timing. Concurrent GPU users can affect timings; candidate order alternated. Original policy weights never updated.',
            'device':torch.cuda.get_device_name(),'results':results,
            'source_checkpoint':args.checkpoint,'active_heads':{name:float((w>0).sum(-1).float().mean()) for name,w in weights.items()}}
    (root/args.output_name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:{name:values['median'] for name,values in v.items()}for key,v in results.items()}))


if __name__=='__main__':main()
