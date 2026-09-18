"""Fit-only refinement of readout regularization and severity/profile partitions."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from explore_latent_environment_partitions import fit_linear,predict_linear,fit_kmeans,classify,dump
from analyze_dr16384_clusters import pair_quality

SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')
ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
OUT=ROOT/'extensions'


def main():
    OUT.mkdir(exist_ok=False);(OUT/'models').mkdir()
    a=dict(np.load(SOURCE/'selected_worlds.npz'));results=[];readouts=[]
    with threadpool_limits(limits=12):
        for fold in (0,1):
            tr=np.flatnonzero((a['fold']==fold)&~a['nominal']);tn=np.flatnonzero((a['fold']==fold)&a['nominal'])
            te=np.flatnonzero((a['fold']!=fold)&~a['nominal']);ne=np.flatnonzero((a['fold']!=fold)&a['nominal'])
            reference=dict(np.load(ROOT/'models'/f'fold{fold}_readout.npz'))
            lo,hi=reference['lower'],reference['upper']
            rng=np.random.default_rng(38517+fold);td=rng.permutation(tr);nd=rng.permutation(tn)
            inner=np.r_[td[:6144],nd[:768]];valid=np.r_[td[6144:],nd[768:]]
            choices=[]
            for alpha in (1e-8,1e-7,1e-6,1e-5,1e-4):
                m=fit_linear(a['z'][inner].reshape(-1,64),np.repeat(a['x'][inner],16,axis=0),np.repeat(a['nominal'][inner],16),alpha)
                v=np.clip(predict_linear(m,a['z'][valid].reshape(-1,64)),lo,hi)
                err=((v-np.repeat(a['x'][valid],16,axis=0))**2).sum(1);nom=np.repeat(a['nominal'][valid],16)
                choices.append(dict(alpha=alpha,score=float(.5*(err[nom].mean()+err[~nom].mean()))))
            alpha=min(choices,key=lambda r:r['score'])['alpha']
            fit=np.r_[tr,tn]
            m=fit_linear(a['z'][fit].reshape(-1,64),np.repeat(a['x'][fit],16,axis=0),np.repeat(a['nominal'][fit],16),alpha)
            pred=np.clip(predict_linear(m,a['z'].reshape(-1,64)),lo,hi).reshape(-1,16,38)
            error=pred[te]-a['x'][te,None,:]
            rr=dict(fold=fold,choices=choices,alpha=alpha,
                    per_parameter_r2=(1-(error**2).mean((0,1))/a['x'][te].var(0)).tolist())
            readouts.append(rr);dump(OUT/'readout.json',readouts)
            np.savez_compressed(OUT/'models'/f'fold{fold}_readout_refined.npz',**m,lower=lo,upper=hi,
                                train_worlds=fit,physics_span=a['span'],physics_weights=a['weights'],physics_nominal=reference['physics_nominal'])
            np.savez_compressed(OUT/'models'/f'fold{fold}_estimated_dr.npz',x=pred)

            def record(method,k,centers,ld,ln,extra=None):
                key=f'fold{fold}_{method}_k{k}'
                q=pair_quality(a,te,ld)
                results.append(dict(key=key,fold=fold,method=method,k=k,cluster_pair_quality=q,extra=extra or {}))
                np.savez_compressed(OUT/'models'/f'{key}.npz',centers=centers,dr_labels=ld,nominal_labels=ln,
                                    train_dr=tr,train_nom=tn,test_dr=te,test_nom=ne)
                dump(OUT/'results.json',results)
                print(json.dumps(dict(key=key,reduction=q['reduction']['dr_distance_mean'],left_shin=q['within_cluster']['left_shin_kg'],torso=q['within_cluster']['torso_mass_difference_kg'])),flush=True)

            for k in (16,64):
                km=fit_kmeans(pred,tr,tn,k)
                record('readout_refined',k,km.cluster_centers_,classify(km.cluster_centers_,pred[te]),classify(km.cluster_centers_,pred[ne]))
                oracle=np.load(SOURCE/'models'/f'fold{fold}_true_dr_n8192_k{k}.npz')['centers']
                record('readout_true_refined',k,oracle,classify(oracle,pred[te]),classify(oracle,pred[ne]))

            # One explicit nominal cell plus 3 severity tiers with 5 or 21
            # direction/profile cells: total 16 or 64. True DR defines expert
            # ranges; inference uses calibrated latent radius and readout only.
            radius_model=np.load(ROOT/'models'/f'fold{fold}_radial.npz')
            radius=np.linalg.norm(a['z']-radius_model['nominal_ref'],axis=-1)
            est=np.interp(radius.ravel(),radius_model['isotonic_x'],radius_model['isotonic_y']).reshape(radius.shape)
            edges=np.quantile(a['radius'][tr],[1/3,2/3])
            true_tier=np.digitize(a['radius'],edges)
            tier=np.digitize(est,edges)
            for subk in (5,21):
                k=1+3*subk
                centers=[np.zeros((1,38))]
                for t in range(3):
                    km=KMeans(n_clusters=subk,n_init=5,max_iter=300,random_state=731).fit(a['x'][tr[true_tier[tr]==t]])
                    centers.append(km.cluster_centers_)
                centers=np.concatenate(centers)
                def route(ids):
                    labels=np.zeros((len(ids),16),np.int64)
                    for t in range(3):
                        local=classify(centers[1+t*subk:1+(t+1)*subk],pred[ids])+1+t*subk
                        choose=(tier[ids]==t)&(est[ids]>=.05)
                        labels[choose]=local[choose]
                    return labels
                record('severity_profile',k,centers,route(te),route(ne),dict(severity_edges=edges.tolist(),nominal_max_distance=.05,subclusters_per_tier=subk))
                np.savez_compressed(OUT/'models'/f'fold{fold}_severity_profile_k{k}_routing.npz',centers=centers,edges=edges,
                                    nominal_ref=radius_model['nominal_ref'],isotonic_x=radius_model['isotonic_x'],isotonic_y=radius_model['isotonic_y'])
    dump(OUT/'complete.json',dict(complete=True,models=len(results)))


if __name__=='__main__':main()
