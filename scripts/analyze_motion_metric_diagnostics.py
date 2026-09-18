"""Per-factor and unseen joint-DR tests of response-label clustering; no encoder training."""
import json,argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
P=Path('runs/limb_context_20260916_motion_metric_probe')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--reverse',action='store_true');args=parser.parse_args();suffix='_reverse' if args.reverse else '';
 meta=json.load(open(P/'simulation.json'));d=np.load(P/'responses.npz');vs=meta['variants'];params=d['physical_parameters'];tr=d['train_anchors'];te=~tr
 metrics={'old':d['old_checkpoint_scale'],'new':d['new_grouped'],'old_recalibrated':d['old_matched_calibration']};span=np.array([2.5,2.5,4,4,2,.15,.15,.15,1.7]);groups=[[0,1,2,3],[4],[5,6,7],[8]]
 if args.reverse:
  import torch
  from intact_tracking.forward_predictor import physical_state_delta
  tr,te=te,tr
  sim=np.load(P/'simulation.npz');state=torch.from_numpy(sim['state']);f=sim['feature'].astype(float)
  nd=physical_state_delta(state[:,0,:-1],state[:,0,1:]).numpy().astype(float);nf=np.diff(f[:,0],axis=1)
  inc=np.concatenate([nf[...,:24],nd[...,:3],nd[...,6:9],nd[...,3:6],nd[...,9:12],nf[...,24:]],axis=-1)
  blocks=[(0,12,.001),(12,24,.01),(24,27,.001),(27,30,.01),(30,33,.001),(33,36,.01),(36,40,.001),(40,44,.01)]
  metrics['new']=metrics['new'].copy()
  for i,(a,b,floor) in enumerate(blocks):metrics['new'][...,a:b]*=d['new_scales'][i]/max(float(np.sqrt(np.mean(inc[tr,:,a:b]**2))),floor)
  floors=np.r_[np.full(6,.001),np.full(6,.01),np.full(29,.001),np.full(29,.01)]
  sig=np.maximum(np.sqrt(np.mean(nd[tr]**2,axis=(0,1))),floors)
  metrics['old_recalibrated']=metrics['old_recalibrated']*d['matched_old_std']/sig
 out={'protocol':'Targeted factor scans: train/test motion families disjoint, same parameter levels. Random scenarios: also split 64 physical configurations into disjoint32/32 train/test; no shared motion families or DR parameter tuples. Both include nominal as 1 condition. All KMeans fit inputs are response vectors only. Strong load = total>=8kg for random; scan level>=0.8 for single factors. Metrics compare actual parameter differences, not response inertia.','results':[],'signal_audit':{}}
 archived={}
 scenarios=[]
 for family,sl in [('load',slice(0,24)),('mass',slice(24,30)),('com',slice(30,36)),('friction',slice(36,44))]:
  ids=np.array([0]+[i for i,v in enumerate(vs) if v['family']==family]);scenarios.append((family,ids,ids,sl))
 for family in ['random_full','random_load']:
  ids=np.array([i for i,v in enumerate(vs) if v['family']==family]);np.random.default_rng(824).shuffle(ids);scenarios.append((family,np.r_[0,ids[32:] if args.reverse else ids[:32]],np.r_[0,ids[:32] if args.reverse else ids[32:]],None))
 for scenario,fi,qi,sl in scenarios:
  methods={**metrics}
  if sl is not None:methods['new_targeted']=metrics['new'][...,sl]
  for name,response in methods.items():
   amp=np.sqrt(np.mean(np.sum(response**2,axis=-1),axis=-1));noise=amp[te,1];heavyids=np.array([i for i in qi if i!=0 and ((params[i,:4].sum()>=8) if scenario.startswith('random') else vs[i]['level']>=.8)])
   ampheavy=amp[te][:,heavyids]
   out['signal_audit'][scenario+'_'+name]={'noise_median_p95_max':np.quantile(noise,[.5,.95,1]).tolist(),'heavy_amplitude_median_p10':np.quantile(ampheavy,[.5,.1]).tolist(),'heavy_above_5x_repeat_noise_fraction':float((ampheavy>5*np.maximum(noise[:,None],1e-8)).mean())}
   fit=response[tr][:,fi].reshape(tr.sum()*len(fi),-1);query=response[te][:,qi].reshape(te.sum()*len(qi),-1);scale=max(float(np.sqrt((fit**2).mean())),1e-12);fit/=scale;query/=scale
   testids=np.tile(qi,te.sum());trainids=np.tile(fi,tr.sum())
   for k in [8,16,32]:
    km=KMeans(n_clusters=k,n_init=5,random_state=731,max_iter=300).fit(fit);labels=km.predict(query);nom=int(km.predict(np.zeros((1,fit.shape[1])))[0]);rng=np.random.default_rng(614)
    anchors=rng.integers(len(query),size=30000);pools=[np.flatnonzero(labels==c) for c in range(k)];partners=np.array([rng.choice(pools[labels[i]]) for i in anchors]);ref=rng.integers(len(query),size=len(anchors))
    def diff(a,b):
     v=(params[testids[a]]-params[testids[b]])/span;return np.stack([(v[:,g]**2).mean(1) for g in groups],1)
    intra=diff(anchors,partners);random=diff(anchors,ref);heavy=np.isin(testids,heavyids);mask=labels==nom
    result={'scenario':scenario,'method':name,'k':k,'overall_distance_reduction':float(1-np.sqrt(intra.mean()/max(random.mean(),1e-20))),'heavy_nominal_fraction':float((labels[heavy]==nom).mean()),'nominal_cluster_max_total_load':float(params[testids[mask],:4].sum(1).max()) if mask.any() else None,'family_reductions':{g:float(1-np.sqrt(intra[:,i].mean()/random[:,i].mean())) if random[:,i].mean()>1e-20 else None for i,g in enumerate(['load','mass','com','friction'])},'test_samples':len(query),'test_motion_anchors':int(te.sum()),'train_variants':fi.tolist(),'test_variants':qi.tolist()}
    out['results'].append(result);key=f'{scenario}_{name}_{k}';archived[key+'_labels']=labels;archived[key+'_centers']=km.cluster_centers_;archived[key+'_scale']=scale
    if k==16:print(json.dumps(result),flush=True)
 (P/('diagnostics'+suffix+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');np.savez_compressed(P/('diagnostic_clusters'+suffix+'.npz'),**archived)
 lines=['# 按 DR 项目与未见随机 DR 的补充诊断','',out['protocol'],'',
 '|场景|指标|K|同簇参数距离缩小|强扰动进入nominal类|nominal类最大总负载kg|','|---|---|---:|---:|---:|---:|']
 for r in out['results']:
  if r['k']==16:lines.append(f'|{r["scenario"]}|{r["method"]}|16|{r["overall_distance_reduction"]:.1%}|{r["heavy_nominal_fraction"]:.1%}|{r["nominal_cluster_max_total_load"]:.2f}|')
 lines+=['','new_targeted只用于诊断单项指标可辨识性，需要已知测试哪一类DR；不能假装部署时已知DR类型。随机DR测试只使用完整new指标。']
 (P/('DIAGNOSTICS'+suffix+'.md')).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
