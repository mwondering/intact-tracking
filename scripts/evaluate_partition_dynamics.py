"""Validate latent partitions using matched mjwarp motion responses.

Includes a compact supervised metric: a linear map of frozen latent fitted to
response fingerprints on fit worlds and fit motion families. Test worlds AND
test response-motion families are excluded from that fit.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from intact_tracking.forward_predictor import physical_state_delta
from explore_latent_environment_partitions import dump,fit_kmeans,classify
from analyze_dr16384_clusters import pair_quality

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')
OUT=ROOT/'dynamics_evaluation'
BLOCKS=[('limb_position',0,12,.001),('limb_velocity',12,24,.01),
        ('root_position',24,27,.001),('root_velocity',27,30,.01),
        ('root_rotation',30,33,.001),('root_angular_velocity',33,36,.01),
        ('foot_xy_position',36,40,.001),('foot_xy_velocity',40,44,.01)]


def prepare_responses():
    raw=ROOT/'dynamics'
    assert json.loads((raw/'audit.json').read_text())['complete']
    s=np.load(raw/'state.npy',mmap_mode='r');f=np.load(raw/'feature.npy',mmap_mode='r')
    records=json.loads((raw/'anchors.json').read_text())
    families=np.array([r['family'] for r in records]);unique=np.unique(families)
    fit_families=np.random.default_rng(972).permutation(unique)[:len(unique)//2]
    train=np.isin(families,fit_families)
    nominal=torch.from_numpy(np.array(s[:,0]))
    nd=physical_state_delta(nominal[:,:-1],nominal[:,1:]).numpy()
    nf=np.diff(f[:,0],axis=1)
    increments=np.concatenate([nf[...,:24],nd[...,:3],nd[...,6:9],nd[...,3:6],nd[...,9:12],nf[...,24:]],axis=-1)
    scales=np.array([max(float(np.sqrt(np.mean(increments[train,:,lo:hi]**2))),floor) for _,lo,hi,floor in BLOCKS])
    sigma=np.maximum(np.sqrt((nd[train]**2).mean((0,1))),np.r_[np.full(6,.001),np.full(6,.01),np.full(29,.001),np.full(29,.01)])
    shape=(len(records),s.shape[1],10)
    response=np.lib.format.open_memmap(OUT/'response.npy',mode='w+',dtype=np.float32,shape=shape+(44,))
    old=np.lib.format.open_memmap(OUT/'state_response.npy',mode='w+',dtype=np.float32,shape=shape+(70,))
    for begin in range(0,len(records),4):
        end=min(begin+4,len(records))
        part=torch.from_numpy(np.array(s[begin:end,:,1:]))
        delta=physical_state_delta(part[:,0:1].expand_as(part),part).numpy()
        fd=f[begin:end,:,1:]-f[begin:end,0:1,1:]
        new=np.concatenate([fd[...,:24],delta[...,:3],delta[...,6:9],delta[...,3:6],delta[...,9:12],fd[...,24:]],axis=-1)
        for i,(_,lo,hi,_) in enumerate(BLOCKS):
            new[...,lo:hi]/=scales[i]*np.sqrt(8*(hi-lo))
        response[begin:end]=new
        old[begin:end]=delta/sigma/np.sqrt(70)
    response.flush();old.flush()
    repeat_noise={}
    for label,mask in [('fit',train),('test',~train)]:
        r=np.asarray(response[mask])
        repeat_noise[label]=dict(nominal_rms=float(np.sqrt(((r[:,0]-r[:,1])**2).sum(-1).mean())),
                                repeated_dr_rms=float(np.sqrt(((r[:,2:18]-r[:,-16:])**2).sum(-1).mean())))
    protocol=dict(fit_families=fit_families.tolist(),test_families=unique[~np.isin(unique,fit_families)].tolist(),
                  train_anchors=int(train.sum()),test_anchors=int((~train).sum()),scales=scales.tolist(),old_scales=sigma.tolist(),
                  blocks=[x[0] for x in BLOCKS],repeat_noise=repeat_noise,
                  primary='RMS difference of matched H10 response vectors, 8 equally weighted physical blocks, scales from fit nominal motion increments only',
                  secondary='70D original physical state deltas, nominal increment scale from the same fit motions',
                  routing='Average over the 16 independent-history routing decisions available for each physical world; different worlds only. Probe actions/initial states are identical across worlds.',
                  fitting='Response-calibrated metric uses only outer-fit worlds in the 2048 simulation subset and fit motion families; cluster centers use outer-fit latent observations only.',
                  no_policy_or_encoder_updates=True)
    dump(OUT/'protocol.json',protocol)
    np.savez_compressed(OUT/'motion_split.npz',train=train,families=families)
    print(json.dumps(dict(event='responses_ready',**protocol)),flush=True)


def fit_response_linear(dr_z,nom_z,r,alpha):
    z=np.concatenate([dr_z,nom_z]).astype(float).reshape(-1,64)
    # Equal DR and nominal weight regardless of number of calibration worlds.
    weight=np.r_[np.full(dr_z.shape[0]*16,.5/(dr_z.shape[0]*16)),np.full(nom_z.shape[0]*16,.5/(nom_z.shape[0]*16))]
    mean=weight@z;scale=np.sqrt(weight@((z-mean)**2)).clip(1e-5)
    x=(z-mean)/scale
    gram=x.T@(x*weight[:,None])
    target_mean=r.mean(0)*.5
    cross=((dr_z.mean(1)-mean)/scale).T@r*(.5/len(dr_z))
    coef=np.linalg.solve(gram+alpha*np.eye(64),cross)
    return dict(mean=mean,scale=scale,coef=coef,intercept=target_mean)


def response_mse(model,z,truth):
    x=(z-model['mean'])/model['scale'];b=model['coef'];y=model['intercept']
    gram=b@b.T;by=b@y
    quadratic=np.einsum('nwi,ij,nwj->nw',x,gram,x,optimize=True)+2*x@by+float(y@y)
    cross=np.einsum('nwi,ni->nw',x,truth@b.T,optimize=True)+truth@y[:,None]
    return float(np.mean(quadratic-2*cross+(truth*truth).sum(1)[:,None]))


def train_response_metric(a):
    worlds=np.load(ROOT/'dynamics/worlds.npz')['indices']
    train_motion=np.load(OUT/'motion_split.npz')['train']
    r=np.load(OUT/'response.npy',mmap_mode='r')
    fit_response=np.asarray(r[train_motion,2:2050]).transpose(1,0,2,3).reshape(2048,-1).astype(float)
    fit_response/=np.sqrt(int(train_motion.sum())*10)
    results=[]
    (OUT/'models').mkdir(exist_ok=True)
    for fold in (0,1):
        dr=np.flatnonzero((a['fold']==fold)&~a['nominal']);nom=np.flatnonzero((a['fold']==fold)&a['nominal'])
        test=np.flatnonzero((a['fold']!=fold)&~a['nominal']);ntest=np.flatnonzero((a['fold']!=fold)&a['nominal'])
        response_idx=np.flatnonzero(a['fold'][worlds]==fold);fitworld=worlds[response_idx]
        rng=np.random.default_rng(5827+fold)
        order=rng.permutation(len(fitworld));no=rng.permutation(nom)
        fit,valid=order[:768],order[768:]
        choices=[]
        for alpha in (1e-6,1e-4,.01,.1,1.):
            model=fit_response_linear(a['z'][fitworld[fit]],a['z'][no[:768]],fit_response[response_idx[fit]],alpha)
            err=response_mse(model,a['z'][fitworld[valid]],fit_response[response_idx[valid]])
            nomerr=response_mse(model,a['z'][no[768:]],np.zeros((256,fit_response.shape[1])))
            choices.append(dict(alpha=alpha,score=.5*(err+nomerr),dr_response_mse=err))
        alpha=min(choices,key=lambda r:r['score'])['alpha']
        model=fit_response_linear(a['z'][fitworld],a['z'][nom],fit_response[response_idx],alpha)
        matrix=model['coef']@model['coef'].T
        eig,vec=eigh(matrix)
        transform=(vec*np.sqrt(np.maximum(eig,0))[None,:])/model['scale'][:,None]
        value=(a['z']-model['mean'])@transform
        np.savez_compressed(OUT/'models'/f'fold{fold}_response_metric.npz',mean=model['mean'],transform=transform,
                            alpha=alpha,fit_response_worlds=fitworld,fit_nominal_worlds=nom,fit_motion_mask=train_motion)
        for k in (16,64):
            km=fit_kmeans(value,dr,nom,k)
            ld=classify(km.cluster_centers_,value[test]);ln=classify(km.cluster_centers_,value[ntest])
            key=f'fold{fold}_response_metric_k{k}'
            quality=pair_quality(a,test,ld)
            record=dict(key=key,fold=fold,method='response_metric',k=k,cluster_pair_quality=quality,alpha=alpha,selection=choices)
            results.append(record);dump(OUT/'response_metric_results.json',results)
            np.savez_compressed(OUT/'models'/f'{key}.npz',centers=km.cluster_centers_,test_dr=test,test_nom=ntest,
                                train_dr=dr,train_nom=nom,dr_labels=ld,nominal_labels=ln)
            print(json.dumps(dict(event='response_metric',key=key,alpha=alpha,physical_reduction=quality['reduction']['dr_distance_mean'])),flush=True)


def pair_weights(labels,k):
    p=np.stack([(labels==c).mean(1) for c in range(k)],axis=1)
    total=p.sum(0)
    denominator=total[None,:]-p
    factor=np.divide(p,denominator,out=np.zeros_like(p),where=denominator>1e-10)
    w=factor@p.T
    np.fill_diagonal(w,0)
    coverage=w.sum()/len(p)
    assert coverage>.99,coverage
    w/=w.sum()
    return w,p,float(coverage)


def squared_distances(value):
    value=np.ascontiguousarray(value,dtype=np.float32)
    norm=(value*value).sum(1)
    d=np.maximum(norm[:,None]+norm[None,:]-2*value@value.T,0)
    np.fill_diagonal(d,0)
    return d


def evaluate(a):
    indices=np.load(ROOT/'dynamics/worlds.npz')['indices']
    train_motion=np.load(OUT/'motion_split.npz')['train'];families=np.load(OUT/'motion_split.npz')['families']
    new=np.load(OUT/'response.npy',mmap_mode='r');old=np.load(OUT/'state_response.npy',mmap_mode='r')
    all_models=[]
    baseline=json.loads((SOURCE/'results.json').read_text())
    for r in baseline:
        if r['fit_dr_worlds']==8192 and r['representation'] in ('window_latent','true_dr'):
            all_models.append((dict(r,method=r['representation']),SOURCE/'models'/f"{r['key']}.npz"))
    for folder,file in [(ROOT,'results.json'),(ROOT/'extensions','results.json'),(OUT,'response_metric_results.json')]:
        for r in json.loads((folder/file).read_text()):
            all_models.append((r,folder/'models'/f"{r['key']}.npz"))
    reports=[];by_family=[]
    for fold in (0,1):
        local=np.flatnonzero(a['fold'][indices]!=fold);test=indices[local];n=len(test)
        pairs=np.ones((n,n),np.float64);np.fill_diagonal(pairs,0);pairs/=pairs.sum()
        models=[]
        for record,path in all_models:
            if record['fold']!=fold:continue
            saved=np.load(path)
            lookup={int(w):i for i,w in enumerate(saved['test_dr'])}
            pos=np.array([lookup[int(w)] for w in test]);labels=saved['dr_labels'][pos]
            w,p,coverage=pair_weights(labels,record['k'])
            models.append((record,w,p,coverage))
        # Keep per-family evidence; each family is evaluated with the same
        # environments, probe states and pair-weighting rules for every method.
        for family in np.unique(families):
            mask=families==family
            chunk=np.asarray(new[mask][:,2+local])
            sq=[]
            for _,lo,hi,_ in BLOCKS:
                feature=chunk[...,lo:hi].transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
                sq.append(squared_distances(feature))
            blocks=np.stack(sq);total=blocks.sum(0)
            ochunk=np.asarray(old[mask][:,2+local]).transpose(1,0,2,3).reshape(n,-1)/np.sqrt(mask.sum()*10)
            od=squared_distances(ochunk)
            random=(blocks*pairs).sum((1,2));random_old=float((od*pairs).sum())
            for rec,w,p,coverage in models:
                inside=(blocks*w).sum((1,2));inside_old=float((od*w).sum())
                by_family.append(dict(key=rec['key'],method=rec['method'],k=rec['k'],fold=fold,
                                      family=str(family),split='fit' if train_motion[mask][0] else 'test',anchors=int(mask.sum()),
                                      within=inside.tolist(),random=random.tolist(),within_old=inside_old,random_old=random_old,
                                      reduction=float(1-np.sqrt(inside.sum()/random.sum())),
                                      old_reduction=float(1-np.sqrt(inside_old/random_old)),coverage=coverage))
            print(json.dumps(dict(event='family',fold=fold,family=str(family),models=len(models))),flush=True)
        dump(OUT/'per_family.json',by_family)
    for record,path in all_models:
        for split in ('fit','test'):
            rows=[r for r in by_family if r['key']==record['key'] and r['split']==split]
            weight=np.array([r['anchors'] for r in rows]);weight=weight/weight.sum()
            inside=weight@np.array([r['within'] for r in rows]);random=weight@np.array([r['random'] for r in rows])
            wo=float(weight@np.array([r['within_old'] for r in rows]));ro=float(weight@np.array([r['random_old'] for r in rows]))
            reports.append(dict(key=record['key'],method=record['method'],fold=record['fold'],k=record['k'],split=split,
                                anchors=sum(r['anchors'] for r in rows),dr_worlds=1024,
                                within_response_rms=float(np.sqrt(inside.sum())),random_response_rms=float(np.sqrt(random.sum())),
                                response_distance_reduction=float(1-np.sqrt(inside.sum()/random.sum())),
                                old_state_distance_reduction=float(1-np.sqrt(wo/ro)),
                                block_reduction={name:float(1-np.sqrt(inside[i]/random[i])) for i,(name,*_) in enumerate(BLOCKS)},
                                families_improved=sum(r['reduction']>0 for r in rows),families=len(rows)))
    dump(OUT/'results.json',reports)
    dump(OUT/'complete.json',dict(complete=True,models=len(all_models),family_evaluations=len(by_family)))


def main():
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=False)
    a=dict(np.load(SOURCE/'selected_worlds.npz'))
    with threadpool_limits(limits=12):
        prepare_responses()
        train_response_metric(a)
        evaluate(a)


if __name__=='__main__':main()
