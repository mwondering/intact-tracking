"""Paired frozen-checkpoint evaluation of independently trained Short50/Memory350."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from intact_tracking.forward_predictor_objective import _normalized_state_error
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.short50_model import Short50Config, Short50Predictor
from analyze_memory350 import cluster_ci, error_metrics


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def run(args):
    torch.set_num_threads(4)
    started=time.time()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    paths={'short50':Path(args.short_checkpoint),'memory350':Path(args.memory_checkpoint)}
    checkpoints={k:torch.load(p,map_location='cpu',weights_only=False,mmap=True) for k,p in paths.items()}
    updates={c['update'] for c in checkpoints.values()}
    assert len(updates)==1,'Compare the same optimizer budget'
    assert checkpoints['short50']['normalization']==checkpoints['memory350']['normalization']
    assert checkpoints['short50']['loss_config']==checkpoints['memory350']['loss_config']
    assert checkpoints['short50']['tracker']==checkpoints['memory350']['tracker']
    assert checkpoints['short50']['long_memory_in_model'] is False
    models={'short50':Short50Predictor(Short50Config(**checkpoints['short50']['model_config'])),
            'memory350':Memory350Predictor(Memory350Config(**checkpoints['memory350']['model_config']))}
    for name,model in models.items():
        model.load_state_dict(checkpoints[name]['model'],strict=True)
        model.to('cuda:0').eval().requires_grad_(False)
    parts=[];file_hashes={}
    for rank in range(4):
        filename=f'validation_broad_rank_{rank}.pt'
        p=paths['memory350'].parent/filename
        reference_hash=digest(p)
        assert digest(paths['short50'].parent/filename)==reference_hash
        file_hashes[filename]=reference_hash
        batch={k:v.to('cuda:0') for k,v in torch.load(p,map_location='cpu',weights_only=False).items()}
        part={'rank':np.full(len(batch['state']),rank), 'world_id':batch['world_id'].cpu().numpy(),
              'short_steps':batch['history_valid'].sum(1).cpu().numpy(),
              'long_chunks':batch['memory_valid'].sum(1).cpu().numpy()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            target=batch['state'][:,1:]
            unchanged=batch['state'][:,:1].expand_as(target)
            error=_normalized_state_error(unchanged,target,batch['state_mean'],batch['state_std'],batch['delta_std'])
            part['unchanged_mse']=error.float().square().mean(-1).cpu().numpy()
            for name,model in models.items():
                latent=model.encode_context(batch['history_state'],batch['history_action'],batch['state'][:,0],
                    batch['history_valid'],history_next_state=batch['history_next_state'],
                    memory_interactions=batch['memory_interactions'],memory_valid=batch['memory_valid'])
                part[name+'_mse']=error_metrics(model,batch,latent)
        parts.append(part)
    samples={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
    np.savez_compressed(output/'paired_samples.npz',**samples)
    a=samples['memory350_mse'][:,-1];b=samples['short50_mse'][:,-1]
    denominator=samples['unchanged_mse'][:,-1];short=samples['short_steps'];long=samples['long_chunks']
    groups={'all':np.ones(len(a),dtype=bool),
            'short_incomplete_with_long':(short<50)&(long>0),
            'short_under10_with_long':(short<10)&(long>0),
            'short_full_with_long':(short==50)&(long>0),
            'no_long_in_reference':long==0, 'full_30_chunks':long==30}
    result={'update':updates.pop(),'optimizer_steps':checkpoints['short50']['optimizer_steps'],
            'checkpoints':{k:{'path':str(p.resolve()),'sha256':digest(p)} for k,p in paths.items()},
            'validation_sha256':file_hashes,'same_normalization':True,'same_validation':True,
            'comparison':'independently trained matched Short50 vs Memory350, identical held-out windows',
            'groups':{},'timestamp':time.time()}
    for group,mask in groups.items():
        if not mask.any():continue
        result['groups'][group]={'samples':int(mask.sum()),'worlds':len(np.unique(samples['world_id'][mask])),
            'memory350_nmse':float(a[mask].sum()/denominator[mask].sum()),
            'short50_nmse':float(b[mask].sum()/denominator[mask].sum()),
            'short50_to_memory350_error_ratio':float(b[mask].sum()/a[mask].sum()),
            'paired_world_bootstrap_ratio_ci95':cluster_ci(samples['world_id'],a,b,mask),
            'fraction_windows_memory_better':float((a[mask]<b[mask]).mean())}
    result['rank_mean_nmse']={name:float(np.mean([samples[name+'_mse'][samples['rank']==rank,-1].sum()/denominator[samples['rank']==rank].sum() for rank in range(4)])) for name in models}
    result['elapsed_seconds']=time.time()-started
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.wandb:
        import wandb
        run=wandb.init(project='intact-forward-predictor',entity='2486344338-zhejiang-university',
            group=args.group,name=f'short50_vs_memory350_u{result["update"]:06d}',dir=str(output),
            config={'comparison_update':result['update'],'checkpoints':result['checkpoints'],
                    'same_normalization':True,'same_validation':True})
        payload={'update':result['update'],'optimizer_steps':result['optimizer_steps']}
        for group,values in result['groups'].items():
            for key,value in values.items():
                if isinstance(value,(float,int)):payload[f'{group}/{key}']=value
            lo,hi=values['paired_world_bootstrap_ratio_ci95']
            payload[f'{group}/error_ratio_ci95_low']=lo;payload[f'{group}/error_ratio_ci95_high']=hi
        run.log(payload)
        (output/'wandb_run.json').write_text(json.dumps({'id':run.id,'url':run.url},indent=2)+'\n')
        run.finish()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-checkpoint',required=True)
    parser.add_argument('--memory-checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--group',default='short50-memory350-paired-evaluation')
    run(parser.parse_args())
