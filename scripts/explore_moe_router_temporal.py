"""Select causal smoothing on validation worlds, then seal routing settings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from intact_tracking.router_information_probe import (
    affine_predict, causal_ema, gate_weights, gate_summary, project,
    regression_metrics, select_readout, squared_distances,
)
from explore_moe_factorized_router import factorized_weights
from explore_moe_product_router import product_weights


def load_trace(directory):
    meta = json.loads((directory/'result.json').read_text())
    assert meta['completed_steps']==3000 and meta['static_DR_unchanged_verified'] and meta['router_state_unchanged_verified']
    with np.load(directory/'traces.npz') as f:
        trace = {key:f[key] for key in ('unit_latent_samples','sampled_steps','short_count','long_count',
                                      'episode_ids','motion_ids','motion_steps','raw_dr')}
    steps=trace['sampled_steps']
    trace['full']=(trace['short_count'][steps]==50)&(trace['long_count'][steps]==30)
    rng=np.random.default_rng(981);rows=[];worlds=[]
    for world in range(trace['raw_dr'].shape[0]):
        candidates=np.flatnonzero((steps>=500)&trace['full'][:,world])
        assert len(candidates)>=32
        selected=rng.choice(candidates,32,replace=False)
        rows.extend(selected);worlds.extend([world]*32)
    trace['point_rows'],trace['point_worlds']=np.asarray(rows),np.asarray(worlds)
    trace['dt']=float(np.diff(steps)[0])*meta['control_dt_seconds']
    return trace,meta


def load_projection(root,name):
    with np.load(root/'models'/f'{name}.npz') as f:
        return {key:f[key] for key in f.files}


def route_features(features,model,candidate):
    shape=features.shape[:-1];flat=features.reshape(-1,features.shape[-1])
    if candidate.get('kind')=='factorized':
        weights=factorized_weights(flat,candidate['mode'])
    elif candidate.get('kind')=='product_kmeans':
        weights=product_weights(flat,model['centers'],candidate['temperatures'],candidate['group_top_k'])
    else:
        weights=gate_weights(squared_distances(flat,model['centers']),top_k=candidate['top_k'],temperature=candidate['temperature'])
    return weights.reshape(*shape,16)


def stability(weights,trace):
    steps=trace['sampled_steps'];steady=steps>=500
    ep,motion,frame=[trace[key][steps] for key in ('episode_ids','motion_ids','motion_steps')]
    valid=trace['full']&steady[:,None]
    same=(ep[1:]==ep[:-1])&(motion[1:]==motion[:-1])&(frame[1:]-frame[:-1]==np.diff(steps)[:,None])
    mask=valid[1:]&valid[:-1]&same
    top=weights.argmax(-1)
    tv=.5*np.abs(weights[1:]-weights[:-1]).sum(-1)
    within=tv[mask]
    cross=[];agreements=[]
    for world in range(weights.shape[1]):
        edges=np.r_[0,np.flatnonzero((ep[1:,world]!=ep[:-1,world])|(motion[1:,world]!=motion[:-1,world]))+1,len(weights)]
        means=[];motions=[]
        for left,right in zip(edges[:-1],edges[1:]):
            keep=valid[left:right,world]
            if keep.sum()>=5:
                means.append(weights[left:right,world][keep].mean(0));motions.append(motion[left,world])
        if len(means)<2:continue
        means=np.asarray(means);motions=np.asarray(motions)
        a,b=np.triu_indices(len(means),1);different=motions[a]!=motions[b]
        a,b=a[different],b[different]
        if len(a):
            cross.append(float((.5*np.abs(means[a]-means[b]).sum(-1)).mean()))
            agreements.append(float((means[a].argmax(-1)==means[b].argmax(-1)).mean()))
    return {'observed_interval_seconds':trace['dt'],
            'warning':'Latent recorded every10 control steps; these are 0.2s changes, not per-control-step switch rates. Global argmax is not an adequate stability measure for grouped/soft mixtures.',
            'full_memory_same_motion_intervals':int(mask.sum()),
            'mean_TV_change_per_interval':float(within.mean()),
            'p95_TV_change_per_interval':float(np.quantile(within,.95)),
            'p99_TV_change_per_interval':float(np.quantile(within,.99)),
            'global_argmax_change_fraction_per_interval':float((top[1:]!=top[:-1])[mask].mean()),
            'cross_motion_world_mean_TV':float(np.mean(cross)),
            'cross_motion_global_dominant_agreement':float(np.mean(agreements)),
            'gate':gate_summary(weights[valid])}


def candidate_catalog(root):
    chosen=json.loads((root/'selected.json').read_text())['candidates']
    keep={'saved_k1_t1','saved_k4_t0.3','whiten_f0.01_k4_t1','whiten_f0.01_k8_t1','whiten_f0.01_k16_t3',
          'decoded_dr8_k4_t1','decoded_dr8_k8_t1','decoded_dr8_k16_t3'}
    candidates={row['name']:row for row in chosen if row['name'] in keep}
    for row in json.loads((root/'product.json').read_text())['selected']:
        candidates[row['name']]=row
    for mode in ('hard','top2','triangular','bilinear'):
        name=f'factorized_{mode}'
        candidates[name]={'name':name,'kind':'factorized','metric':'decoded_dr8','mode':mode}
    return list(candidates.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--root',required=True)
    args=parser.parse_args();source,root=Path(args.source),Path(args.root)
    (root/'temporal_readouts').mkdir(exist_ok=True)
    a,meta=load_trace(source/'sample_seed30402');b,other=load_trace(source/'sample_seed30401')
    assert meta['context_sha256']==other['context_sha256'] and meta['checkpoint_sha256']==other['checkpoint_sha256']
    schema=meta['dr_schema'];lower=np.asarray(schema['lower']);width=np.asarray(schema['upper'])-lower
    order=np.random.default_rng(887).permutation(len(a['raw_dr']))
    fit=np.isin(a['point_worlds'],order[:768]);val=~fit
    y=(a['raw_dr'][a['point_worlds']]-lower)/width;test_y=(b['raw_dr'][b['point_worlds']]-lower)/width
    rows=[];selected=[]
    for candidate in candidate_catalog(root):
        model=load_projection(root,candidate['metric'])
        projected=[]
        for trace in (a,b):
            x=trace['unit_latent_samples'];features=project(x.reshape(-1,64),model).reshape(x.shape[0],x.shape[1],-1)
            projected.append(features)
        options=[]
        for tau in (0.,.2,.5,1.,2.):
            gates=[];points=[]
            for trace,features in zip((a,b),projected):
                smoothed=causal_ema(features,trace['full'],dt=trace['dt'],time_constant=tau)
                weights=route_features(smoothed,model,candidate)
                gates.append(weights)
                points.append(weights[trace['point_rows'],trace['point_worlds']])
            readout,validation,grid=select_readout(points[0][fit],y[fit],points[0][val],y[val],schema['groups'])
            name=f"{candidate['name']}_ema{tau:g}"
            row={'name':name,'base':candidate['name'],'time_constant_seconds':tau,'validation':validation,
                 'exploratory_test':regression_metrics(test_y,affine_predict(points[1],readout),schema['groups']),
                 'readout_grid':grid}
            rows.append(row);options.append(row)
            np.savez(root/'temporal_readouts'/f'{name}.npz',**readout)
        best=max(row['validation']['strong8_r2'] for row in options)
        choice=min([row for row in options if row['validation']['strong8_r2']>=best-.01],key=lambda row:row['time_constant_seconds'])
        for row in [options[0],choice]:
            weights=route_features(causal_ema(projected[1],b['full'],dt=b['dt'],time_constant=row['time_constant_seconds']),model,candidate)
            row['exploratory_stability']=stability(weights,b)
        selected.append({**candidate,'evaluation_name':choice['name'],'time_constant_seconds':choice['time_constant_seconds']})
        (root/'temporal_sweep.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'candidate':candidate['name'],'tau':choice['time_constant_seconds'],
                          'test_R2':choice['exploratory_test']['strong8_r2'],
                          'loads':choice['exploratory_test']['r2'][:4],
                          'TV_0p2s':choice['exploratory_stability']['mean_TV_change_per_interval'],
                          'cross_motion_TV':choice['exploratory_stability']['cross_motion_world_mean_TV']}),flush=True)
    decision={'rule':'For each candidate choose smallest EMA time constant within0.01 of best validation strong8 R2. EMA advances only on full-memory observations and is held across motion/episode changes because DR stays fixed.',
              'scope':'Replay at recorded 0.2s cadence. A50Hz deployment must use dt=0.02; high-frequency behavior requires separate runtime measurement.',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'candidates':selected,'heldout_used_for_selection':False,
              'recommendation_slots':{'preserve_joint_online_KMeans':'decoded_dr8_k8_t1',
                                      'retain_online_KMeans_group_factors':'product_dr8_gk3_t1',
                                      'exact_factor_information_reference':'factorized_triangular'}}
    (root/'final_selection.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
