"""Confirm fixed selected partitions on a fresh independent bank of motion states."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from intact_tracking.forward_predictor import physical_state_delta
from evaluate_partition_dynamics import BLOCKS,pair_weights,squared_distances
from explore_latent_environment_partitions import dump

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')
DYN=ROOT/'dynamics_evaluation'
OUT=ROOT/'fresh_validation'


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ema',action='store_true');parser.add_argument('--confirmation',action='store_true');args=parser.parse_args()
    if args.ema:OUT=ROOT/'fresh_ema_validation'
    if args.confirmation:OUT=ROOT/'confirmation'
    OUT.mkdir(exist_ok=False);torch.set_num_threads(3)
    raw=ROOT/('dynamics_confirmation' if args.confirmation else 'dynamics_fresh');assert json.loads((raw/'audit.json').read_text())['complete']
    selection=json.loads((ROOT/'blends/selection.json').read_text())
    a=dict(np.load(SOURCE/'selected_worlds.npz'));indices=np.load(raw/'worlds.npz')['indices']
    records=json.loads((raw/'anchors.json').read_text());families=np.array([x['family'] for x in records])
    protocol=json.loads((DYN/'protocol.json').read_text());scales=np.array(protocol['scales']);sigma=np.array(protocol['old_scales'])
    assert not set(families)&set(protocol['fit_families'])
    original=json.loads((ROOT/'dynamics/anchors.json').read_text())
    identity=lambda x:(x['world'],x['query_step'],x['motion_file'])
    assert not set(map(identity,records))&set(map(identity,original))
    if args.confirmation:
        assert not set(map(identity,records))&set(map(identity,json.loads((ROOT/'dynamics_fresh/anchors.json').read_text())))
        assert not set(indices)&set(np.load(ROOT/'dynamics/worlds.npz')['indices'])
    s=np.load(raw/'state.npy',mmap_mode='r');f=np.load(raw/'feature.npy',mmap_mode='r')
    models=[]
    for fold in (0,1):
        for k in (16,64):
            for method in ('window_latent','true_dr'):
                models.append(dict(method=method,fold=fold,k=k,path=str(SOURCE/'models'/f'fold{fold}_{method}_n8192_k{k}.npz')))
            models.append(dict(method='response_metric',fold=fold,k=k,path=str(DYN/'models'/f'fold{fold}_response_metric_k{k}.npz')))
            chosen=next(x for x in selection['selected'] if x['fold']==fold and x['k']==k)['selected']
            if chosen is not None:
                models.append(dict(method='selected_blend',fold=fold,k=k,path=str(ROOT/'blends/models'/f"{chosen['key']}.npz")))
        models.append(dict(method='latent_radius_fixed',fold=fold,k=16,path=str(ROOT/'models'/f'fold{fold}_latent_radius_fixed_k16.npz')))
    if args.ema:
        assert json.loads((ROOT/'causal_ema/complete.json').read_text())['complete']
        models=[dict(method='causal200',fold=fold,k=64,path=str(ROOT/'causal_ema/models'/f'fold{fold}_causal200_k64.npz')) for fold in (0,1)]
    if args.confirmation:
        models=[dict(method='causal200',fold=fold,k=64,path=str(ROOT/'causal_ema/models'/f'fold{fold}_causal200_k64.npz')) for fold in (0,1)]
        models += [dict(method='window_latent',fold=fold,k=64,path=str(SOURCE/'models'/f'fold{fold}_window_latent_n8192_k64.npz')) for fold in (0,1)]
    results=[];class_rows=[];verification=[]
    with threadpool_limits(limits=12):
        for fold in (0,1):
            local=np.flatnonzero(a['fold'][indices]!=fold);worlds=indices[local];n=len(worlds)
            random=np.ones((n,n));np.fill_diagonal(random,0);random/=random.sum()
            fitted=[]
            for row in models:
                if row['fold']!=fold:continue
                m=np.load(row['path']);lookup={int(w):i for i,w in enumerate(m['test_dr'])}
                labels=m['dr_labels'][[lookup[int(w)] for w in worlds]]
                w,p,coverage=pair_weights(labels,row['k']);fitted.append((row,w,p,coverage))
            for family in np.unique(families):
                mask=families==family
                ss=torch.from_numpy(np.array(s[mask,:,1:]))
                delta=physical_state_delta(ss[:,0:1].expand_as(ss),ss).numpy()
                fd=f[mask,:,1:]-f[mask,0:1,1:]
                response=np.concatenate([fd[...,:24],delta[...,:3],delta[...,6:9],delta[...,3:6],delta[...,9:12],fd[...,24:]],axis=-1)
                for i,(_,lo,hi,_) in enumerate(BLOCKS):response[...,lo:hi]/=scales[i]*np.sqrt(8*(hi-lo))
                blocks=[]
                for _,lo,hi,_ in BLOCKS:
                    values=response[:,2+local,:,lo:hi].transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
                    d=squared_distances(values);blocks.append(d)
                    # Independent direct distance spot checks of the Gram shortcut.
                    left=np.arange(32);right=(left*17+33)%n
                    exact=((values[left]-values[right])**2).sum(1)
                    np.testing.assert_allclose(d[left,right],exact,atol=1e-5,rtol=3e-5)
                blocks=np.stack(blocks);total=blocks.sum(0)
                ov=(delta[:,2+local]/sigma/np.sqrt(70)).transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
                old=squared_distances(ov)
                baseline=(blocks*random).sum((1,2));baseline_old=float((old*random).sum())
                for row,w,p,coverage in fitted:
                    inside=(blocks*w).sum((1,2));io=float((old*w).sum())
                    results.append(dict(method=row['method'],fold=fold,k=row['k'],family=str(family),anchors=int(mask.sum()),
                                        within=inside.tolist(),random=baseline.tolist(),within_old=io,random_old=baseline_old,
                                        reduction=float(1-np.sqrt(inside.sum()/baseline.sum())),coverage=coverage))
                    if row['method'] in ('selected_blend','causal200') and row['k']==64:
                        numerator=np.diag(p.T@total@p)
                        denominator=p.sum(0)**2-(p*p).sum(0)
                        for c in range(row['k']):
                            class_rows.append(dict(fold=fold,cluster=c,family=str(family),
                                                   effective_windows=float(p[:,c].sum()*16),worlds=int((p[:,c]>0).sum()),
                                                   pair_weight=float(denominator[c]),squared_distance=float(numerator[c]/denominator[c]) if denominator[c]>1e-10 else None,
                                                   random_squared_distance=float(baseline.sum())))
                verification.append(dict(fold=fold,family=str(family),direct_distance_spotcheck=True,
                                          nominal_repeat_rms=float(np.sqrt(((response[:,0]-response[:,1])**2).sum(-1).mean())),
                                          dr_repeat_rms=float(np.sqrt(((response[:,2:18]-response[:,-16:])**2).sum(-1).mean()))))
                print(json.dumps(dict(fold=fold,family=str(family),complete=True)),flush=True)
    dump(OUT/'per_family.json',results);dump(OUT/'per_class.json',class_rows)
    aggregate=[]
    for row in models:
        rr=[x for x in results if x['fold']==row['fold'] and x['method']==row['method'] and x['k']==row['k']]
        w=np.array([x['anchors'] for x in rr]);w=w/w.sum()
        inside=w@np.array([x['within'] for x in rr]);ref=w@np.array([x['random'] for x in rr])
        io=w@np.array([x['within_old'] for x in rr]);ro=w@np.array([x['random_old'] for x in rr])
        aggregate.append(dict(method=row['method'],fold=row['fold'],k=row['k'],
                              reduction=float(1-np.sqrt(inside.sum()/ref.sum())),
                              old_state_reduction=float(1-np.sqrt(io/ro)),
                              block_reduction={name:float(1-np.sqrt(inside[i]/ref[i])) for i,(name,*_) in enumerate(BLOCKS)},
                              within_rms=float(np.sqrt(inside.sum())),random_rms=float(np.sqrt(ref.sum()))))
    dump(OUT/'results.json',aggregate)
    dump(OUT/'verification.json',dict(passed=True,fresh_states_disjoint=True,fit_response_motion_families_disjoint=True,
                                     model_selection=str(ROOT/'blends/selection.json'),states=64,worlds=2048,
                                     numerical=verification,scope='Fixed models, new motion states; no coefficient or center refitting.'))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
