"""Compare radial bins and calibrated latent metrics on world-held-out DR.

Encoder and policies are frozen. All fitted statistics, readout selection and
cluster centers use outer-fit worlds only; a nested world split selects ridge.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

from analyze_dr16384_clusters import pair_quality, cluster_rows, distribution

SOURCE = Path('runs/limb_context_20260916_dr16384/analysis')
OUT = Path('runs/limb_context_20260916_latent_environment_partitions')


def dump(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fit_linear(z, y, is_nominal, alpha):
    z=z.astype(np.float64);y=y.astype(np.float64)
    weight=np.where(is_nominal,(~is_nominal).sum()/is_nominal.sum(),1.)
    weight/=weight.sum()
    mean=weight@z
    scale=np.sqrt(weight@((z-mean)**2)).clip(1e-5)
    design=(z-mean)/scale
    ym=weight@y
    gram=design.T@(design*weight[:,None])
    rhs=design.T@((y-ym)*weight[:,None])
    coef=np.linalg.solve(gram+alpha*np.eye(gram.shape[0]),rhs)
    return dict(mean=mean,scale=scale,coef=coef,intercept=ym,alpha=alpha)


def predict_linear(model,z):
    return ((z-model['mean'])/model['scale'])@model['coef']+model['intercept']


def fit_kmeans(value, train_dr, train_nom, k):
    dim=value.shape[-1]
    d=value[train_dr].reshape(-1,dim);n=value[train_nom].reshape(-1,dim)
    fit=np.r_[d,n];weights=np.r_[np.ones(len(d)),np.full(len(n),len(d)/len(n))]
    return KMeans(n_clusters=k,n_init=5,max_iter=300,random_state=731).fit(fit,sample_weight=weights)


def classify(centers,z):
    shape=z.shape[:-1]
    flat=z.reshape(-1,z.shape[-1])
    # Chunking also makes exported inference suitable for large vectorized worlds.
    labels=[]
    for part in np.array_split(flat,max(1,int(np.ceil(len(flat)/8192)))):
        d=(part*part).sum(1)[:,None]+(centers*centers).sum(1)[None,:]-2*part@centers.T
        labels.append(d.argmin(1))
    return np.concatenate(labels).reshape(shape)


def main():
    args=argparse.ArgumentParser(description=__doc__);args.add_argument('--threads',type=int,default=12);args=args.parse_args()
    OUT.mkdir(exist_ok=False);(OUT/'models').mkdir()
    a=dict(np.load(SOURCE/'selected_worlds.npz'))
    assert json.loads((SOURCE/'model_verification.json').read_text())['passed']
    source_hash=hashlib.sha256((SOURCE/'selected_worlds.npz').read_bytes()).hexdigest()
    dump(OUT/'protocol.json',dict(source=str(SOURCE),source_sha256=source_hash,
          checkpoint='DR-supervised u35857, frozen',outer_splits='same two world folds as 16384 audit',
          nominal_fit_weight=.5,k=[16,64],readout='Affine ridge on unit latent; nested world holdout chooses alpha; parameter bounds clip predictions',
          metrics='Identical other-world within-cluster physical comparisons, all parameters retained; actual response replay evaluated separately',
          radial='fixed DR edges .05,.10,...,.75 (16 bins) and nominal-first + 15 fit-DR quantile bins; raw monotone target inverse, plus fit-only isotonic calibration',
          no_policy_or_encoder_training=True))
    results=[];readouts=[];rows_all=[]
    base=np.zeros(38);base[4]=.6;base[5:34]=1.
    audit=json.loads((SOURCE/'collection_audit.json').read_text());schema=audit['schema']
    lower=(np.asarray(schema['lower'])-base)/a['span']*np.sqrt(a['weights'])
    upper=(np.asarray(schema['upper'])-base)/a['span']*np.sqrt(a['weights'])
    with threadpool_limits(limits=args.threads):
        for fold in (0,1):
            started=time.monotonic()
            tr=np.flatnonzero((a['fold']==fold)&~a['nominal']);tn=np.flatnonzero((a['fold']==fold)&a['nominal'])
            te=np.flatnonzero((a['fold']!=fold)&~a['nominal']);ne=np.flatnonzero((a['fold']!=fold)&a['nominal'])
            nomref=a['zmean'][tn].mean(0)
            fitworld=np.r_[tr,tn]
            rng=np.random.default_rng(38517+fold)
            td=rng.permutation(tr);nd=rng.permutation(tn)
            inner=np.r_[td[:6144],nd[:768]];valid=np.r_[td[6144:],nd[768:]]
            iz=a['z'][inner].reshape(-1,64);iy=np.repeat(a['x'][inner],16,axis=0);inom=np.repeat(a['nominal'][inner],16)
            vz=a['z'][valid].reshape(-1,64);vy=np.repeat(a['x'][valid],16,axis=0);vnom=np.repeat(a['nominal'][valid],16)
            choice=[]
            for alpha in (.0001,.001,.01,.1,1.):
                model=fit_linear(iz,iy,inom,alpha)
                pred=np.clip(predict_linear(model,vz),lower,upper)
                err=((pred-vy)**2).sum(1)
                score=.5*(err[~vnom].mean()+err[vnom].mean())
                choice.append(dict(alpha=alpha,balanced_validation_mse=float(score),dr_mse=float(err[~vnom].mean())))
            best=min(choice,key=lambda r:r['balanced_validation_mse'])['alpha']
            readout=fit_linear(a['z'][fitworld].reshape(-1,64),np.repeat(a['x'][fitworld],16,axis=0),np.repeat(a['nominal'][fitworld],16),best)
            pred=np.clip(predict_linear(readout,a['z'].reshape(-1,64)),lower,upper).reshape(-1,16,38)
            rawpred=pred*a['span']/np.sqrt(a['weights'])+base
            error=rawpred[te]-a['physics'][te,None,:]
            rr=dict(fold=fold,alpha=best,inner_selection=choice,per_parameter=[dict(name=name,
                    mae=float(np.abs(error[...,j]).mean()),rmse=float(np.sqrt((error[...,j]**2).mean())),
                    r2=float(1-(error[...,j]**2).mean()/a['physics'][te,j].var())) for j,name in enumerate(a['names'])])
            readouts.append(rr);dump(OUT/'readout.json',readouts)
            np.savez_compressed(OUT/'models'/f'fold{fold}_readout.npz',**readout,lower=lower,upper=upper,
                                train_worlds=fitworld,nominal_ref=nomref,physics_span=a['span'],physics_weights=a['weights'],physics_nominal=base)
            np.savez_compressed(OUT/'models'/f'fold{fold}_estimated_dr.npz',x=pred)
            # Whitening is fit only to fit-world mean latents; use a noise floor
            # to avoid blindly amplifying nearly constant directions.
            fitmean=a['zmean'][fitworld].astype(float)
            ww=np.where(a['nominal'][fitworld],len(tr)/len(tn),1.);ww/=ww.sum()
            wm=ww@fitmean;cov=(fitmean-wm).T@((fitmean-wm)*ww[:,None])
            eig,vec=eigh(cov);transform=vec@np.diag(1/np.sqrt(np.maximum(eig,.01*eig.max())))@vec.T
            white=(a['z']-wm)@transform
            np.savez_compressed(OUT/'models'/f'fold{fold}_whitening.npz',mean=wm,transform=transform,eigenvalues=eig)

            def record(method,k,value,centers,ld,ln,ltrain,extra=None):
                key=f'fold{fold}_{method}_k{k}'
                quality=pair_quality(a,te,ld)
                nominal_ref=value[tn].reshape(-1,value.shape[-1]).mean(0)
                cr,nominal=cluster_rows(a,te,ne,ld,ln,centers,nominal_ref,ltrain,key)
                rows_all.extend(cr)
                hist=np.stack([(ld==c).sum(1) for c in range(k)],axis=-1)
                counts=np.bincount(ld.ravel(),minlength=k)
                output=dict(key=key,fold=fold,method=method,k=k,cluster_pair_quality=quality,nominal=nominal,
                            occupied_dr_classes=int((counts>0).sum()),dr_class_window_counts=counts.tolist(),
                            modal_window_fraction=float((hist.max(1)/ld.shape[1]).mean()),
                            extra=extra or {})
                results.append(output)
                np.savez_compressed(OUT/'models'/f'{key}.npz',centers=centers,test_dr=te,test_nom=ne,
                                    dr_labels=ld,nominal_labels=ln,train_dr=tr,train_nom=tn,nominal_ref=nominal_ref)
                dump(OUT/'results.json',results)
                print(json.dumps(dict(key=key,physical_reduction=quality['reduction']['dr_distance_mean'],
                      left_shin_difference=quality['within_cluster']['left_shin_kg'],torso_difference=quality['within_cluster']['torso_mass_difference_kg'],
                      seconds=time.monotonic()-started)),flush=True)

            for method,value in [('whitened_latent',white),('readout_metric',pred)]:
                for k in (16,64):
                    km=fit_kmeans(value,tr,tn,k)
                    ld=classify(km.cluster_centers_,value[te]);ln=classify(km.cluster_centers_,value[ne]);lt=classify(km.cluster_centers_,value[tn])
                    record(method,k,value,km.cluster_centers_,ld,ln,lt)
            for k in (16,64):
                oracle=np.load(SOURCE/'models'/f'fold{fold}_true_dr_n8192_k{k}.npz')
                centers=oracle['centers']
                record('readout_true_centers',k,pred,centers,classify(centers,pred[te]),classify(centers,pred[ne]),classify(centers,pred[tn]))

            # The radial classification candidates use the same ground-truth
            # boundaries for truth and latent. Isotonic calibration is optional
            # and never looks at evaluation-world distances.
            radius=np.linalg.norm(a['z']-nomref,axis=-1)
            inv=.2*radius/np.maximum(2-radius,1e-6)
            inverse=np.minimum(inv,float(np.linalg.norm(np.maximum(np.abs(lower),np.abs(upper)))))
            iso=IsotonicRegression(out_of_bounds='clip')
            fitrad=radius[fitworld].ravel();fity=np.repeat(a['radius'][fitworld],16)
            weights=np.where(np.repeat(a['nominal'][fitworld],16),len(tr)/len(tn),1.)
            iso.fit(fitrad,fity,sample_weight=weights)
            calibrated=iso.predict(radius.ravel()).reshape(radius.shape)
            np.savez_compressed(OUT/'models'/f'fold{fold}_radial.npz',nominal_ref=nomref,
                                isotonic_x=iso.X_thresholds_,isotonic_y=iso.y_thresholds_)
            for bins,edges in [('fixed',np.arange(.05,.8,.05)),('quantile',np.r_[.05,np.quantile(a['radius'][tr],np.arange(1,15)/15)])]:
                assert len(edges)==15 and np.all(np.diff(edges)>0)
                actual=np.digitize(a['radius'],edges)
                for method,estimate in [('oracle_radius',np.broadcast_to(a['radius'][:,None],radius.shape)),('latent_radius',inverse),('calibrated_radius',calibrated)]:
                    label=np.digitize(estimate,edges)
                    centers=np.r_[0.,(edges[:-1]+edges[1:])/2,edges[-1]+.025][:,None]
                    delta=label[te]-actual[te,None]
                    extra=dict(edges=edges.tolist(),dr_bin_accuracy=float((delta==0).mean()),
                               dr_within_one_bin=float((np.abs(delta)<=1).mean()),
                               dr_mean_absolute_bin_error=float(np.abs(delta).mean()),
                               dr_distance_mae=float(np.abs(estimate[te]-a['radius'][te,None]).mean()),
                               nominal_distance_mae=float(np.abs(estimate[ne]).mean()))
                    record(method+'_'+bins,16,estimate[...,None],centers,label[te],label[ne],label[tn],extra)
                    np.savez_compressed(OUT/'models'/f'fold{fold}_{method}_{bins}_detail.npz',
                                        edges=edges,truth=actual[te],prediction=label[te],estimated_distance=estimate[te])
    import csv
    fields=list(dict.fromkeys(k for row in rows_all for k in row))
    with (OUT/'cluster_parameters.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows_all)
    dump(OUT/'complete.json',dict(complete=True,models=len(results),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    main()
