"""Locate nominal cluster and report observed physical extrema of collocated held-out DR."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from sklearn.cluster import KMeans
from intact_tracking.memory350_inference import load_memory350_checkpoint
from probe_memory350_dr_readout import encode

BASE=Path('runs/limb_context_20260916_paired_encoder_dr_clustering')
COMMON=Path('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/cluster_check_030750/common')
OUT=Path('runs/limb_context_20260916_nominal_cluster_contamination')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def assign(x,c):return np.concatenate([np.square(b[:,None]-c[None]).sum(-1).argmin(1) for b in np.array_split(x,32)])

def transform(x,mu,W):
 y=(x-mu)@W;return y/np.linalg.norm(y,axis=1,keepdims=True)

def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
 old=json.loads((BASE/'summary.json').read_text());a=np.load(BASE/'labels_and_inputs.npz');meta=json.loads((COMMON/'metadata.json').read_text());d=np.load(COMMON/'latents.npz')
 eligible=d['nominal']&(d['short_steps']==50)&(d['long_chunks']==30);worlds=np.unique(d['world'][eligible]);assert len(worlds)==512
 rng=np.random.default_rng(731);rows=np.concatenate([rng.choice(np.flatnonzero(eligible&(d['world']==w)),16,replace=False) for w in worlds]);nw=d['world'][rows];cur=np.full((len(rows),64),np.nan,np.float32)
 ck=load_memory350_checkpoint(old['checkpoints']['dr_label']['path'],device='cpu');assert ck.sha256==old['checkpoints']['dr_label']['sha256']
 for i,info in enumerate(meta['queries']):
  selected=np.flatnonzero(d['step'][rows]==info['step'])
  if not len(selected):continue
  assert sha(info['path'])==info['sha256'];query=torch.load(info['path'],map_location='cpu',weights_only=False,mmap=True)
  cached=np.flatnonzero(d['step']==info['step']);np.testing.assert_array_equal(query['world'].numpy(),d['world'][cached])
  lookup={int(w):j for j,w in enumerate(query['world'])};ids=torch.tensor([lookup[int(w)] for w in nw[selected]])
  q={k:query[k][ids] for k in ('world','short','long','short_valid','long_valid')};assert q['short_valid'].all() and q['long_valid'].all()
  cur[selected]=encode(ck,q)
  if i%4==3:print(json.dumps({'encoded_query':i+1,'seconds':time.monotonic()-start}),flush=True)
 assert np.isfinite(cur).all()
 nom={'response10':d['latent_response10'][rows].astype(float),'dr_label':cur.astype(float)}
 for z in nom.values():z/=np.linalg.norm(z,axis=1,keepdims=True)
 np.savez_compressed(OUT/'nominal_latents.npz',source_rows=d['source_row'][rows],worlds=nw,**nom)
 order=np.random.default_rng(1731).permutation(worlds);nomhalf=order[:256];w=a['worlds'];raw=a['physics'];rawnom=np.zeros(38);rawnom[4]=.6;rawnom[5:34]=1.
 report={'checkpoints':old['checkpoints'],'nominal_source':str(COMMON),'nominal_source_sha256':sha(COMMON/'latents.npz'),
  'script_sha256':sha(Path(__file__)),'protocol':'512 pure nominal worlds, 16 full-history actual latents each. Current u35857 re-encoded CPU FP32; response10 matching cached rows. Same 512 random-DR worlds/16 windows used in previous paired comparison. Disjoint 256-world train/test halves for nominal and DR, reverse. Dominant nominal class selected by TRAIN nominal votes; report held-out nominal coverage and actual held-out DR extrema. First use existing DR-only clusters unchanged. Second refit KMeans on 50% nominal + 50% DR training samples, allowing a dedicated nominal class; regularized whitening is refit on that mixture when enabled. This is not a scalar-severity clustering.',
  'limits':'Maxima are observed finite-sample extrema, not certified bounds. Per-limb maxima can belong to DIFFERENT environments; max-total row is one actual environment. Nominal histories and DR histories are different trajectories within the same cached tracker protocol. No inference that static mass alone measures actual response severity. Bias missing in saved physics. Class IDs arbitrary per fit.', 'results':[]}
 archive={}
 for fold in (0,1):
  tr=np.isin(w,a['fit_half0_worlds']);nt=np.isin(nw,nomhalf)
  if fold:tr=~tr;nt=~nt
  tw=w[~tr]
  for enc in nom:
   dz=a['latent_'+enc];nz=nom[enc]
   for metric in ('raw','whiten01'):
    for k in (16,64,128):
     for mode in ('existing_dr_only','refit_with_nominal50'):
      key=f'fold{fold}_{enc}_{metric}_k{k}'
      if mode=='existing_dr_only':
       c=a[key+'_centers'];dl=a[key+'_labels']
       if metric=='whiten01':
        mu=a[key+'_mean'];W=a[key+'_transform'];nfit=transform(nz[nt],mu,W);nquery=transform(nz[~nt],mu,W)
       else:nfit,nquery=nz[nt],nz[~nt]
      else:
       fit=np.concatenate([dz[tr],nz[nt]])
       if metric=='whiten01':
        mu=fit.mean(0);ev,V=np.linalg.eigh(np.cov(fit.T));W=V/np.sqrt(np.maximum(ev,ev[-1]*.01))[None,:];fit=transform(fit,mu,W);nfit=transform(nz[nt],mu,W);nquery=transform(nz[~nt],mu,W);dq=transform(dz[~tr],mu,W)
       else:nfit,nquery,dq=nz[nt],nz[~nt],dz[~tr]
       model=KMeans(n_clusters=k,n_init=5,random_state=731,max_iter=300).fit(fit);c=model.cluster_centers_;dl=assign(dq,c)
      nfl=assign(nfit,c);nql=assign(nquery,c);modal=int(np.bincount(nfl,minlength=k).argmax());ncounts=np.bincount(nql,minlength=k)
      def detail(cluster):
       mask=dl==cluster;ids=np.unique(tw[mask]);r={'cluster':int(cluster),'nominal_test_fraction':float((nql==cluster).mean()),'dr_samples':int(mask.sum()),'dr_unique_worlds':len(ids)}
       if not len(ids):return r
       vals=raw[ids];load=vals[:,34:38];total=load.sum(1);hi=int(total.argmax());dev=np.abs(vals-rawnom)
       r.update(load_max_kg=load.max(0).tolist(),load_q95_kg=np.quantile(load,.95,axis=0).tolist(),total_load_max_kg=float(total[hi]),total_load_q95_kg=float(np.quantile(total,.95)),
         max_total_world_id=int(ids[hi]),max_total_load_vector_kg=load[hi].tolist(),max_total_world_raw_parameters=vals[hi].tolist(),
         com_max_abs_cm=(dev[:,:3].max(0)*100).tolist(),torso_mass_max_abs_kg=float(dev[:,3].max()/0.12790995401552643),
         friction_min_max=[float(vals[:,4].min()),float(vals[:,4].max())],friction_max_abs_change=float(dev[:,4].max()),armature_max_abs_scale_change=float(dev[:,5:34].max()),
         total_load_ge8kg_world_count=int((total>=8).sum()),total_load_ge8kg_sample_fraction=float((raw[tw[mask],34:38].sum(1)>=8).mean()),
         world_ids=ids.tolist(),max_abs_all_coordinates=dev.max(0).tolist())
       return r
      result={'fold':fold,'encoder':enc,'metric':metric,'k':k,'mode':mode,'modal_nominal_cluster':modal,'train_nominal_modal_fraction':float((nfl==modal).mean()),
       'modal_cluster':detail(modal),'all_nominal_occupied_clusters':[detail(int(c)) for c in np.flatnonzero(ncounts)]}
      report['results'].append(result);prefix=key+'_'+mode;archive[prefix+'_centers']=c;archive[prefix+'_dr_labels']=dl;archive[prefix+'_nominal_labels']=nql
      if mode=='refit_with_nominal50' and metric=='whiten01':archive[prefix+'_mean']=mu;archive[prefix+'_transform']=W
      print(json.dumps({key:result[key] for key in ('fold','encoder','metric','k','mode')}|{'nominal_fraction':result['modal_cluster']['nominal_test_fraction'],'DR_worlds':result['modal_cluster']['dr_unique_worlds'],'max_total_load':result['modal_cluster'].get('total_load_max_kg')}),flush=True)
  (OUT/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(OUT/'models_and_labels.npz',**archive)
 lines=['# Nominal 所在簇中的最大 DR 偏离','',report['protocol'],'',report['limits'],'',
 '|聚类数据|encoder|距离|K|fold|nominal覆盖|混入DR环境数|左手最大kg|右手最大kg|左小腿最大kg|右小腿最大kg|单环境总负载最大kg|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in report['results']:
  c=r['modal_cluster'];loads=c.get('load_max_kg',[0]*4);total=c.get('total_load_max_kg',0)
  lines.append(f'|{r["mode"]}|{r["encoder"]}|{r["metric"]}|{r["k"]}|{r["fold"]}|{c["nominal_test_fraction"]:.1%}|{c["dr_unique_worlds"]}|'+ '|'.join(f'{v:.3f}' for v in loads)+f'|{total:.3f}|')
 lines+=['','没有混入DR时表中最大负载记0，表示本批测试未观察到混入，不是全空间保证。完整参数极值和所有 nominal 访问类别见summary.json。']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
