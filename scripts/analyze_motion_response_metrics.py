"""No encoder training: assess motion-response label geometry directly on held-out motions."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from intact_tracking.forward_predictor import physical_state_delta
P=Path('runs/limb_context_20260916_motion_metric_probe')
CK=Path('runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/update_015000.pt')

def main():
 info=json.loads((P/'simulation.json').read_text());d=np.load(P/'simulation.npz');s=torch.from_numpy(d['state']);f=d['feature'].astype(float);variants=info['variants'];na,nv,_,_=s.shape
 # All geometry retains directions and time, without latent normalization.
 delta=physical_state_delta(s[:,0:1,1:].expand_as(s[:,:,1:]),s[:,:,1:]).numpy().astype(float)
 fd=f[:,:,1:]-f[:,0:1,1:]
 new=np.concatenate([fd[...,:24],delta[...,:3],delta[...,6:9],delta[...,3:6],delta[...,9:12],fd[...,24:]],axis=-1)
 nominaldelta=physical_state_delta(s[:,0,:-1],s[:,0,1:]).numpy().astype(float);nominalfd=np.diff(f[:,0],axis=1)
 increments=np.concatenate([nominalfd[...,:24],nominaldelta[...,:3],nominaldelta[...,6:9],nominaldelta[...,3:6],nominaldelta[...,9:12],nominalfd[...,24:]],axis=-1)
 families=np.array([r['family'] for r in info['anchors']]);unique=np.unique(families);shuffled=np.random.default_rng(972).permutation(unique);fitfamilies=shuffled[:len(unique)//2];train=np.isin(families,fitfamilies);test=~train
 assert train.any() and test.any()
 blocks=[(0,12,.001),(12,24,.01),(24,27,.001),(27,30,.01),(30,33,.001),(33,36,.01),(36,40,.001),(40,44,.01)]
 scaled=new.copy();scales=[]
 for a,b,floor in blocks:
  scale=max(float(np.sqrt(np.mean(increments[train,:,a:b]**2))),floor);scales.append(scale);scaled[...,a:b]/=scale*np.sqrt(8*(b-a))
 ck=torch.load(CK,map_location='cpu',weights_only=False,mmap=True);std=np.array(ck['normalization']['delta_std']);old=delta/std/np.sqrt(70)
 # Matched-calibration control distinguishes changed signals from changed normalization.
 floors=np.r_[np.full(3,.001),np.full(3,.001),np.full(6,.01),np.full(29,.001),np.full(29,.01)]
 sigma=np.maximum(np.sqrt(np.mean(nominaldelta[train]**2,axis=(0,1))),floors)
 oldcal=delta/sigma/np.sqrt(70)
 labels={'old_checkpoint_scale':old,'old_matched_calibration':oldcal,'new_grouped':scaled}
 params=np.array([v['load']+[v['mass']]+v['com']+[v['friction']] for v in variants],float);span=np.array([2.5,2.5,4,4,2,.15,.15,.15,1.7]);groups={'load':list(range(4)),'mass':[4],'com':[5,6,7],'friction':[8]}
 metrics={'protocol':'Same simulated trajectories, H10 at 50Hz. Four groups: root-relative limb body position/velocity; root position/velocity; relative root rotation/angular velocity; ungated foot XY displacement/velocity. No force/contact conditions, no encoder/policy training. Fit normalization and KMeans on one set of motion families, evaluate other families. KMeans labels are response-feature clusters, not trained latent clusters.',
 'normalization':'Original: actual u15000 single-step delta_std. New: one RMS nominal step increment scale per physical block, fitted on training motions only; position/rotation floor0.001, velocity/angular velocity floor0.01; each of 8 blocks weight1/8. Matched-calibration control uses coordinate RMS nominal increments and same unit floors for old70D features.',
 'train_families':fitfamilies.tolist(),'test_families':unique[~np.isin(unique,fitfamilies)].tolist(),'train_anchors':int(train.sum()),'test_anchors':int(test.sum()),'new_block_scales':scales,'simulation_audit':info['audit'],'sensitivities':{},'clustering':[], 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 np.savez_compressed(P/'responses.npz',**labels,physical_parameters=params,train_anchors=train,old_std=std,matched_old_std=sigma,new_scales=scales)
 active=np.array([i for i,v in enumerate(variants) if i!=1 and not v['family'].startswith('random_')]);train_ids=np.tile(active,train.sum());test_ids=np.tile(active,test.sum())
 for name,response in labels.items():
  amplitudes=np.sqrt(np.mean(np.sum(response**2,axis=-1),axis=-1))
  noise=amplitudes[test,1];bybranch={}
  for branch in sorted(set(v['branch'] for v in variants)-{'nominal','random_full','random_load'}):
   ids=np.array([i for i,v in enumerate(variants) if v['branch']==branch]);ids=ids[np.argsort([variants[i]['level'] for i in ids])];level=np.array([variants[i]['level'] for i in ids]);vals=amplitudes[test][:,ids]
   correlations=[float(spearmanr(level,row).statistic) if np.ptp(row)>1e-12 else 0. for row in vals]
   bybranch[branch]={'family':variants[ids[0]]['family'],'levels':level.tolist(),'median_amplitude':np.median(vals,axis=0).tolist(),'spearman_median':float(np.median(correlations)),'monotone_fraction':float(np.mean(np.all(np.diff(vals,axis=1)>=-1e-10,axis=1))), 'heavy_over_light_median':float(np.median(vals[:,-1]/np.maximum(vals[:,0],1e-12))),'heavy_above_5x_repeat_noise_fraction':float(np.mean(vals[:,-1]>5*np.maximum(noise,1e-8)))}
  metrics['sensitivities'][name]={'nominal_duplicate_noise_median_p95_max':np.quantile(noise,[.5,.95,1]).tolist(),'branches':bybranch}
  fit=response[train][:,active].reshape(train.sum()*len(active),-1);query=response[test][:,active].reshape(test.sum()*len(active),-1)
  # Remove irrelevant overall scalar between methods, leaving actual within-method geometry.
  global_scale=max(np.sqrt(np.mean(fit**2)),1e-12);fit=fit/global_scale;query=query/global_scale
  for k in [8,16,32]:
   km=KMeans(n_clusters=k,n_init=10,random_state=731,max_iter=300).fit(fit);ql=km.predict(query);nomlabel=int(km.predict(np.zeros((1,fit.shape[1])))[0]);count=np.bincount(ql,minlength=k)
   rng=np.random.default_rng(385);ix=rng.integers(len(query),size=50000);pools=[np.flatnonzero(ql==c) for c in range(k)];partners=np.array([rng.choice(pools[ql[i]]) for i in ix]);rand=rng.integers(len(query),size=len(ix))
   def dist(left,right):
    diff=(params[test_ids[left]]-params[test_ids[right]])/span
    return np.stack([(diff[:,g]**2).mean(1) for g in groups.values()],1)
   within=dist(ix,partners);ref=dist(ix,rand);ratios=np.sqrt(within.mean(0)/np.maximum(ref.mean(0),1e-12))
   result={'method':name,'k':k,'nominal_cluster':nomlabel,'overall_distance_reduction':float(1-np.sqrt(within.mean()/ref.mean())),'family_distance_reduction':dict(zip(groups,(1-ratios).tolist())),'nominal_leak':{},'occupied_test_clusters':int((count>0).sum())}
   for family in groups:
    severe=np.array([v['family']==family and v['level']>=.8 for v in variants])[test_ids];mild=np.array([v['family']==family and v['level']<=.4 for v in variants])[test_ids]
    result['nominal_leak'][family]={'heavy_fraction':float(np.mean(ql[severe]==nomlabel)),'light_fraction':float(np.mean(ql[mild]==nomlabel)) if mild.any() else None}
   metrics['clustering'].append(result);np.savez_compressed(P/f'clusters_{name}_k{k}.npz',centers=km.cluster_centers_,test_labels=ql,train_labels=km.labels_,test_variant_ids=test_ids,train_variant_ids=train_ids)
   print(json.dumps(result),flush=True)
 (P/'analysis.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
 lines=['# 新旧动力学响应标签：真实 motion GPU 测试','',metrics['protocol'],'',metrics['normalization'],'',
  '同簇物理距离相对随机缩小越多越好。强扰动落入nominal类越少越好。此处尚未训练encoder；这些数字是响应标签空间的可聚类性。','',
  '|指标|K|总体参数距离缩小|强负载进nominal|大质量变化进nominal|大COM偏移进nominal|大摩擦变化进nominal|','|---|---:|---:|---:|---:|---:|---:|']
 for r in metrics['clustering']:lines.append('|'+r['method']+'|'+str(r['k'])+'|'+f'{r["overall_distance_reduction"]:.1%}'+'|'+'|'.join(f'{r["nominal_leak"][g]["heavy_fraction"]:.1%}' for g in groups)+'|')
 lines+=['','强扰动：单项扫描达到配置幅度的80%以上；四肢负载含单肢与四肢共同增重；质量负向与正向分别扫描，摩擦相对nominal0.6向低/高分别扫描。不是随机全DR混合分布。正/负变化保留方向，没有把参数输入聚类。','',
  '原尺度与新尺度数值大小不能直接比较。逐项标签响应的单调性、强/弱幅度比、重复nominal噪声见analysis.json。新指标四组都可能受多种DR影响；不能把分组当作参数唯一对应的证明。']
 (P/'README.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(12,4.4))
 for name in labels:
  rr=[r for r in metrics['clustering'] if r['method']==name];axs[0].plot([r['k'] for r in rr],[100*r['overall_distance_reduction'] for r in rr],'o-',label=name);axs[1].plot([r['k'] for r in rr],[100*r['nominal_leak']['load']['heavy_fraction'] for r in rr],'o-',label=name)
 axs[0].set_title('Physical distance reduction within clusters');axs[1].set_title('Heavy-load samples assigned to nominal cluster')
 for ax in axs:ax.set_xlabel('K');ax.set_ylabel('%');ax.set_xticks([8,16,32]);ax.legend(fontsize=7);ax.grid(alpha=.3)
 fig.tight_layout();fig.savefig(P/'comparison.png',dpi=170)

if __name__=='__main__':main()
