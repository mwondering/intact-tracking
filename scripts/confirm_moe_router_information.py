"""Apply sealed routers/readouts to independent DR traces without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from intact_tracking.memory350_metric_router import FrozenMetricRouter
from intact_tracking.router_information_probe import (
    STRONG_COORDINATES, affine_predict, causal_ema, project, regression_metrics,
)
from explore_moe_router_temporal import load_projection, route_features, stability


def read_trace(directory):
    meta=json.loads((directory/'result.json').read_text())
    assert meta['completed_steps']>=1000 and meta['static_DR_unchanged_verified'] and meta['router_state_unchanged_verified']
    with np.load(directory/'traces.npz') as f:
        trace={key:f[key] for key in ('unit_latent_samples','sampled_steps','short_count','long_count',
                                     'episode_ids','motion_ids','motion_steps','raw_dr')}
    steps=trace['sampled_steps'];trace['full']=(trace['short_count'][steps]==50)&(trace['long_count'][steps]==30)
    rng=np.random.default_rng(981);rows=[];worlds=[]
    for world in range(len(trace['raw_dr'])):
        candidates=np.flatnonzero((steps>=500)&trace['full'][:,world]);assert len(candidates)>=32
        rows.extend(rng.choice(candidates,32,replace=False));worlds.extend([world]*32)
    trace.update(point_rows=np.asarray(rows),point_worlds=np.asarray(worlds),dt=float(np.diff(steps)[0])*meta['control_dt_seconds'])
    return trace,meta


def make_prototype(model,candidate):
    configuration={'time_constant':candidate['time_constant_seconds'],
                   'clip_unit_range':bool(model.get('clip_unit_range',False))}
    if candidate.get('kind')=='factorized':
        if candidate['mode']!='triangular':return None,None
        centers=np.tile(np.asarray([[0,0],[1,0],[0,1],[1,1]],np.float32)[None],(4,1,1))
        temperatures=np.ones(4,np.float32)
        configuration.update(group_top_k=3,interpolation=True)
    elif candidate.get('kind')=='product_kmeans':
        centers=model['centers'];temperatures=candidate['temperatures']
        configuration['group_top_k']=candidate['group_top_k']
    else:
        centers=model['centers'][None];temperatures=[candidate['temperature']]
        configuration['group_top_k']=candidate['top_k']
    router=FrozenMetricRouter(model['matrix'],model['offset'],centers,temperatures,**configuration)
    return router,configuration


def fine_stability(weights,trace,readout):
    steps=trace['sampled_steps'];full=trace['full']&(steps>=500)[:,None]
    ep,motion,frame=[trace[key][steps] for key in ('episode_ids','motion_ids','motion_steps')]
    same=(ep[1:]==ep[:-1])&(motion[1:]==motion[:-1])&(frame[1:]-frame[:-1]==np.diff(steps)[:,None])
    mask=full[1:]&full[:-1]&same
    changes=weights[1:]-weights[:-1]
    tv=.5*np.abs(changes[mask]).sum(-1)
    # Compensates for routes that hide all information in tiny weight changes.
    decoded_change=changes[mask]@readout['matrix'][:,STRONG_COORDINATES]
    rms=np.sqrt(np.square(decoded_change).mean(-1))
    return {'interval_seconds':trace['dt'],'eligible_intervals':int(mask.sum()),
            'mean_gate_TV':float(tv.mean()),'p95_gate_TV':float(np.quantile(tv,.95)),
            'p99_gate_TV':float(np.quantile(tv,.99)),
            'mean_decoded_DR_change_RMS_in_full_ranges':float(rms.mean()),
            'p95_decoded_DR_change_RMS_in_full_ranges':float(np.quantile(rms,.95)),
            'mean_global_argmax_change_fraction':float((weights[1:].argmax(-1)!=weights[:-1].argmax(-1))[mask].mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--trace',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--runtime-device',default='cpu')
    args=parser.parse_args();root,directory=Path(args.root),Path(args.trace)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    selection=json.loads((root/'final_selection.json').read_text())
    seal=json.loads((root/'selection_seal.json').read_text())
    current_sha=hashlib.sha256((root/'final_selection.json').read_bytes()).hexdigest()
    assert current_sha==seal['final_selection_sha256']
    trace,meta=read_trace(directory)
    protocol=json.loads((root/'protocol.json').read_text())
    assert meta['context_sha256']==protocol['context_sha256'] and meta['checkpoint_sha256']==protocol['policy_checkpoint_sha256']
    assert meta['arguments']['seed'] not in [protocol['fit_seed'],protocol['exploratory_test_seed']]
    schema=meta['dr_schema'];lower=np.asarray(schema['lower']);width=np.asarray(schema['upper'])-lower
    y=(trace['raw_dr'][trace['point_worlds']]-lower)/width
    x=trace['unit_latent_samples'];point_x=x[trace['point_rows'],trace['point_worlds']]
    with np.load(root/'full_latent_decoder.npz') as f:full_decoder={k:f[k] for k in f.files}
    full=regression_metrics(y,affine_predict(point_x,full_decoder),schema['groups'])
    exports=root/'exports';exports.mkdir(exist_ok=True)
    output=root/args.tag;output.mkdir(exist_ok=True)
    predictions={};results=[]
    for candidate in selection['candidates']:
        name=candidate['name'];model=load_projection(root,candidate['metric'])
        features=project(x.reshape(-1,64),model).reshape(*x.shape[:-1],-1)
        filtered=causal_ema(features,trace['full'],dt=trace['dt'],time_constant=candidate['time_constant_seconds'])
        weights=route_features(filtered,model,candidate)
        with np.load(root/'temporal_readouts'/f"{candidate['evaluation_name']}.npz") as f:readout={k:f[k] for k in f.files}
        predicted=affine_predict(weights[trace['point_rows'],trace['point_worlds']],readout)
        predictions[name]=predicted[:,STRONG_COORDINATES].astype(np.float32)
        metrics=regression_metrics(y,predicted,schema['groups'])
        # Compare across motions at the same 0.2s cadence used during selection.
        coarse_stride=max(1,int(round(.2/trace['dt'])))
        coarse={**trace,'sampled_steps':trace['sampled_steps'][::coarse_stride],
                'full':trace['full'][::coarse_stride],'dt':trace['dt']*coarse_stride}
        temporal=stability(weights[::coarse_stride],coarse)
        row={'name':name,'settings':candidate,'metrics':metrics,'coarse_stability':temporal,
             'native_cadence_stability':fine_stability(weights,trace,readout)}
        prototype,configuration=make_prototype(model,candidate)
        if prototype is not None:
            artifact={'format_version':'frozen_metric_router_research_v1','configuration':configuration,
                      'state_dict':prototype.state_dict(),'candidate':candidate,
                      'source_policy_sha256':meta['checkpoint_sha256'],'context_sha256':meta['context_sha256'],
                      'selection_sha256':current_sha,'policy_integration':'Advance once in rollout wrapper and store weights; replay exact weights in PPO. This artifact does not contain trained new expert heads.'}
            path=exports/f'{name}.pt'
            if not path.exists():torch.save(artifact,path)
            existing=torch.load(path,map_location='cpu',weights_only=True)
            assert existing['selection_sha256']==current_sha and existing['candidate']==candidate
            prototype.load_state_dict(existing['state_dict'],strict=True)
            prototype=prototype.to(args.runtime_device)
            maximum=0.;l1=0.;count=0
            with torch.inference_mode():
                for step in range(len(x)):
                    actual=prototype.advance(torch.from_numpy(x[step]).to(args.runtime_device),
                                             torch.from_numpy(trace['full'][step]).to(args.runtime_device),dt=trace['dt']).cpu().numpy()
                    error=np.abs(actual-weights[step]);maximum=max(maximum,float(error.max()))
                    l1+=float(error.sum());count+=error.size
            # Sparse top-k may differ at exact ties across BLAS backends. Keep
            # the full discrepancy visible instead of silently accepting it.
            row['prototype_replay']={'device':args.runtime_device,'maximum_absolute_weight_error':maximum,
                                      'mean_absolute_weight_error':l1/count,'passed':maximum<5e-4,
                                      'has_trainable_parameters':bool(list(prototype.parameters()))}
            assert not row['prototype_replay']['has_trainable_parameters']
            if candidate.get('kind')=='factorized':
                w=weights.reshape(*weights.shape[:-1],4,4)*4
                restored=np.stack((w[...,1]+w[...,3],w[...,2]+w[...,3]),-1).reshape(*weights.shape[:-1],8)
                row['factor_coordinate_reconstruction_max_error']=float(np.abs(restored-filtered).max())
        results.append(row)
        print(json.dumps({'name':name,'R2':metrics['strong8_r2'],'loads':metrics['r2'][:4],
                          'native_TV':row['native_cadence_stability']['mean_gate_TV'],
                          'prototype':row.get('prototype_replay')}),flush=True)
        (output/'summary.partial.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'predictions.npz',worlds=trace['point_worlds'],targets=y[:,STRONG_COORDINATES].astype(np.float32),**predictions)
    report={'trace':str(directory.resolve()),'seed':meta['arguments']['seed'],'worlds':len(trace['raw_dr']),
            'control_dt':meta['control_dt_seconds'],'latent_recording_dt':trace['dt'],'steps':meta['completed_steps'],
            'selection_sha256':current_sha,'parameters_fitted_on_this_trace':False,
            'full_latent_reference':full,'rows':results,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
