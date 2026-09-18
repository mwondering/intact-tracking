"""Explore response/physical metric tradeoffs with fit-world-only transforms."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from explore_latent_environment_partitions import predict_linear,fit_kmeans,classify,dump
from analyze_dr16384_clusters import pair_quality
from evaluate_partition_dynamics import BLOCKS,pair_weights,squared_distances

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')
DYN=ROOT/'dynamics_evaluation'
OUT=ROOT/'blends'
LAMBDAS=(.1,.2,.35,.5,.65,.8)


def main():
    OUT.mkdir(exist_ok=False);(OUT/'models').mkdir()
    a=dict(np.load(SOURCE/'selected_worlds.npz'));records=[]
    dump(OUT/'protocol.json',dict(weights=list(LAMBDAS),response='response-calibrated latent distance',
             physical='refined affine latent-to-DR readout, with physical bounds',
             normalization='Each component divided by sqrt(trace covariance) across fit DR windows only',
             selection_rule='Use fit dynamics and fit physical similarity only. Prefer maximal fit response improvement among candidates with both shins at least 8% closer than random and overall physical distance reduction at least raw latent. If none meet, report no feasible candidate; do not hide tradeoffs.',
             followup='Selected candidates will also be checked on fresh nominal motion states absent from these 96 probes.'))
    with threadpool_limits(limits=12):
        for fold in (0,1):
            response=np.load(DYN/'models'/f'fold{fold}_response_metric.npz')
            physical=dict(np.load(ROOT/'extensions/models'/f'fold{fold}_readout_refined.npz'))
            tr=np.flatnonzero((a['fold']==fold)&~a['nominal']);tn=np.flatnonzero((a['fold']==fold)&a['nominal'])
            te=np.flatnonzero((a['fold']!=fold)&~a['nominal']);ne=np.flatnonzero((a['fold']!=fold)&a['nominal'])
            rv=(a['z']-response['mean'])@response['transform']
            pv=np.clip(predict_linear(physical,a['z'].reshape(-1,64)),physical['lower'],physical['upper']).reshape(-1,16,38)
            rs=float(np.sqrt(rv[tr].reshape(-1,64).var(0).sum()));ps=float(np.sqrt(pv[tr].reshape(-1,38).var(0).sum()))
            np.savez_compressed(OUT/'models'/f'fold{fold}_transform.npz',response_mean=response['mean'],response_transform=response['transform'],
                                physical_mean=physical['mean'],physical_scale=physical['scale'],physical_coef=physical['coef'],
                                physical_intercept=physical['intercept'],lower=physical['lower'],upper=physical['upper'],
                                response_scale=rs,physical_scale_total=ps)
            for weight in LAMBDAS:
                value=np.concatenate([rv*np.sqrt(1-weight)/rs,pv*np.sqrt(weight)/ps],axis=-1)
                for k in (16,64):
                    method=f'blend_{str(weight).replace(".","p")}'
                    key=f'fold{fold}_{method}_k{k}'
                    model=fit_kmeans(value,tr,tn,k)
                    ld=classify(model.cluster_centers_,value[te]);ln=classify(model.cluster_centers_,value[ne])
                    fitlabels=classify(model.cluster_centers_,value[tr])
                    q=pair_quality(a,te,ld);fitq=pair_quality(a,tr,fitlabels)
                    records.append(dict(key=key,method=method,fold=fold,k=k,weight=weight,
                                        cluster_pair_quality=q,fit_cluster_pair_quality=fitq))
                    np.savez_compressed(OUT/'models'/f'{key}.npz',centers=model.cluster_centers_,test_dr=te,test_nom=ne,
                                        train_dr=tr,train_nom=tn,dr_labels=ld,nominal_labels=ln,fit_dr_labels=fitlabels)
                    dump(OUT/'physical_results.json',records)
                    print(json.dumps(dict(event='fit',key=key,physical_reduction=q['reduction']['dr_distance_mean'],
                                          shin=[q['within_cluster']['left_shin_kg'],q['within_cluster']['right_shin_kg']])),flush=True)
        evaluate(a,records)
    dump(OUT/'complete.json',dict(complete=True,models=len(records)))


def evaluate(a,records):
    indices=np.load(ROOT/'dynamics/worlds.npz')['indices']
    split=np.load(DYN/'motion_split.npz');train=split['train'];families=split['families']
    new=np.load(DYN/'response.npy',mmap_mode='r');old=np.load(DYN/'state_response.npy',mmap_mode='r')
    family_results=[]
    for fold in (0,1):
        local=np.flatnonzero(a['fold'][indices]!=fold);test=indices[local];n=len(test)
        random=np.ones((n,n));np.fill_diagonal(random,0);random/=random.sum()
        models=[]
        for row in records:
            if row['fold']!=fold:continue
            m=np.load(OUT/'models'/f"{row['key']}.npz")
            lookup={int(w):i for i,w in enumerate(m['test_dr'])};pos=np.array([lookup[int(w)] for w in test])
            w,p,coverage=pair_weights(m['dr_labels'][pos],row['k'])
            models.append((row,w))
        for family in np.unique(families):
            mask=families==family
            chunk=np.asarray(new[mask][:,2+local]);sq=[]
            for _,lo,hi,_ in BLOCKS:
                vector=chunk[...,lo:hi].transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
                sq.append(squared_distances(vector))
            blocks=np.stack(sq)
            oc=np.asarray(old[mask][:,2+local]).transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
            od=squared_distances(oc)
            ref=(blocks*random).sum((1,2));oref=float((od*random).sum())
            for row,w in models:
                inside=(blocks*w).sum((1,2));ow=float((od*w).sum())
                family_results.append(dict(key=row['key'],method=row['method'],k=row['k'],weight=row['weight'],fold=fold,
                                           family=str(family),split='fit' if train[mask][0] else 'test',anchors=int(mask.sum()),
                                           within=inside.tolist(),random=ref.tolist(),within_old=ow,random_old=oref,
                                           reduction=float(1-np.sqrt(inside.sum()/ref.sum()))))
            print(json.dumps(dict(event='family',fold=fold,family=str(family))),flush=True)
        dump(OUT/'per_family.json',family_results)
    aggregate=[]
    for row in records:
        for mode in ('fit','test'):
            parts=[r for r in family_results if r['key']==row['key'] and r['split']==mode]
            w=np.array([r['anchors'] for r in parts]);w=w/w.sum()
            inside=w@np.array([r['within'] for r in parts]);ref=w@np.array([r['random'] for r in parts])
            oldw=float(w@np.array([r['within_old'] for r in parts]));oldr=float(w@np.array([r['random_old'] for r in parts]))
            aggregate.append(dict(key=row['key'],method=row['method'],k=row['k'],weight=row['weight'],fold=row['fold'],split=mode,
                                  response_distance_reduction=float(1-np.sqrt(inside.sum()/ref.sum())),
                                  old_state_distance_reduction=float(1-np.sqrt(oldw/oldr)),
                                  block_reduction={name:float(1-np.sqrt(inside[i]/ref[i])) for i,(name,*_) in enumerate(BLOCKS)}))
    dump(OUT/'dynamics_results.json',aggregate)


if __name__=='__main__':main()
