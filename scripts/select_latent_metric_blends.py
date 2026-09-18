"""Select blend weights using FIT worlds and FIT response-motion families only."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate_partition_dynamics import BLOCKS,pair_weights,squared_distances
from explore_latent_environment_partitions import classify,dump
from analyze_dr16384_clusters import pair_quality

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')
DYN=ROOT/'dynamics_evaluation'
OUT=ROOT/'blends'


def main():
    while True:
        path=OUT/'physical_results.json'
        if path.exists():
            rows=json.loads(path.read_text())
            if len(rows)==24:break
        time.sleep(5)
    a=dict(np.load(SOURCE/'selected_worlds.npz'))
    indices=np.load(ROOT/'dynamics/worlds.npz')['indices']
    split=np.load(DYN/'motion_split.npz');train=split['train'];families=split['families']
    response=np.load(DYN/'response.npy',mmap_mode='r')
    measurements=[];baselines=[]
    with threadpool_limits(limits=12):
        for fold in (0,1):
            local=np.flatnonzero(a['fold'][indices]==fold);worlds=indices[local];n=len(worlds)
            uniform=np.ones((n,n));np.fill_diagonal(uniform,0);uniform/=uniform.sum()
            models=[]
            for row in rows:
                if row['fold']!=fold:continue
                m=np.load(OUT/'models'/f"{row['key']}.npz")
                lookup={int(w):i for i,w in enumerate(m['train_dr'])}
                labels=m['fit_dr_labels'][[lookup[int(w)] for w in worlds]]
                w,_,_=pair_weights(labels,row['k']);models.append((row,w))
            for k in (16,64):
                m=np.load(SOURCE/'models'/f'fold{fold}_window_latent_n8192_k{k}.npz')
                labels=classify(m['centers'],a['z'][m['train_dr']])
                lookup={int(w):i for i,w in enumerate(m['train_dr'])}
                sample=labels[[lookup[int(w)] for w in worlds]]
                w,_,_=pair_weights(sample,k)
                q=pair_quality(a,m['train_dr'],labels)
                row=dict(key=f'fold{fold}_raw_k{k}',method='raw',fold=fold,k=k,fit_cluster_pair_quality=q)
                models.append((row,w));baselines.append(row)
            for family in np.unique(families[train]):
                mask=(families==family)&train
                chunk=np.asarray(response[mask][:,2+local]);squared=[]
                for _,lo,hi,_ in BLOCKS:
                    f=chunk[...,lo:hi].transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
                    squared.append(squared_distances(f))
                total=np.stack(squared).sum(0)
                ref=float((total*uniform).sum())
                for row,w in models:
                    inside=float((total*w).sum())
                    measurements.append(dict(key=row['key'],fold=fold,k=row['k'],family=str(family),
                                             anchors=int(mask.sum()),within=inside,random=ref))
                print(json.dumps(dict(event='strict_fit_family',fold=fold,family=str(family))),flush=True)
    dump(OUT/'strict_fit_dynamics.json',measurements)
    summary=[]
    for row in rows+baselines:
        rr=[x for x in measurements if x['key']==row['key']]
        w=np.array([x['anchors'] for x in rr]);w=w/w.sum()
        reduction=1-np.sqrt((w@np.array([x['within'] for x in rr]))/(w@np.array([x['random'] for x in rr])))
        q=row['fit_cluster_pair_quality']['reduction']
        summary.append(dict(key=row['key'],method=row['method'],fold=row['fold'],k=row['k'],weight=row.get('weight'),
                            fit_response_reduction=float(reduction),fit_physical_reduction=q['dr_distance_mean'],
                            left_shin_reduction=q['left_shin_kg'],right_shin_reduction=q['right_shin_kg']))
    selected=[]
    for fold in (0,1):
        for k in (16,64):
            rr=[x for x in summary if x['fold']==fold and x['k']==k]
            baseline=next(x for x in rr if x['method']=='raw')
            eligible=[x for x in rr if x['method']!='raw' and min(x['left_shin_reduction'],x['right_shin_reduction'])>=.08
                      and x['fit_physical_reduction']>=baseline['fit_physical_reduction']]
            best=max(eligible,key=lambda x:x['fit_response_reduction']) if eligible else None
            selected.append(dict(fold=fold,k=k,baseline=baseline,eligible=len(eligible),selected=best,
                                 also_beats_raw_response=(best is not None and best['fit_response_reduction']>baseline['fit_response_reduction'])))
    dump(OUT/'selection.json',dict(rule='Per outer fold independently; fit-world physical and fit-world + fit-motion response scores only. At least 8% left/right shin improvement and no overall physical regression versus raw, then maximize response similarity.',
                                 candidates=summary,selected=selected))
    print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()
