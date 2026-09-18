"""Paired-trajectory encoder comparison using held-out within-cluster physical distances."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.cluster import KMeans

ROOT=Path('runs/limb_context_20260916_dr_readout_u35857')
OUT=Path('runs/limb_context_20260916_paired_encoder_dr_clustering')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    provenance=json.loads((ROOT/'summary.json').read_text());source=Path(provenance['source']);meta=json.loads((source/'metadata.json').read_text())
    d=np.load(source/'latents.npz');pred=np.load(ROOT/'predictions.npz');physics=np.load(provenance['physics']['path'])
    assert np.array_equal(d['source_row'],pred['all_source_rows']);assert sha(provenance['physics']['path'])==provenance['physics']['sha256']
    names=physics['names'].tolist();raw=physics['values'].astype(float)
    assert names[0]=='base_com/com_offset/torso_link/x' and 'relative_mass' in names[3] and 'friction' in names[4]
    assert all('armature' in x for x in names[5:34]) and all('added_mass_kg' in x for x in names[34:38])
    # Fixed DR schema stored in the u35857 checkpoint; torso coordinate is relative delta mass.
    mass_half_range=0.12790995401552643
    span=np.array([.15]*3+[2*mass_half_range,1.7]+[.4]*29+[2.5,2.5,4,4])
    groups={'limb_load':list(range(34,38)),'torso_mass':[3],'torso_com':[0,1,2],'foot_friction':[4],'armature':list(range(5,34))}
    eligible=(~d['nominal'])&(d['short_steps']==50)&(d['long_chunks']==30)
    worlds=np.unique(d['world'][eligible]);assert len(worlds)==512
    rng=np.random.default_rng(731);rows=np.concatenate([rng.choice(np.flatnonzero(eligible&(d['world']==w)),16,replace=False) for w in worlds])
    w=d['world'][rows];encoders={'response10':d['latent_response10'][rows].astype(float),'dr_label':pred['latent_current'][rows].astype(float)}
    for z in encoders.values():z/=np.linalg.norm(z,axis=1,keepdims=True)
    order=np.random.default_rng(1731).permutation(worlds);parts=[order[:256],order[256:]]
    report={'checkpoints':{'response10':meta['checkpoints']['response10'],'dr_label':provenance['checkpoint']},
      'source':str(source),'source_hashes':{'latents.npz':sha(source/'latents.npz'),'current_predictions':sha(ROOT/'predictions.npz'),'physics':sha(provenance['physics']['path'])},
      'script_sha256':sha(Path(__file__)),'names':names,'span':span.tolist(),'groups':groups,
      'protocol':'Same 512 continuous random DR worlds and exact same frozen-tracker histories for both encoders. 16 actual full-history windows/world, no per-world prototypes. Two disjoint 256-world halves: fit on one, evaluate other, reverse. KMeans n_init5 seed731 max_iter300, K16/32/64/128. Unit latent or train-only covariance whitening (eigenvalue floor 1% max), then unit normalize. No DR labels used to cluster.',
      'distance':'100000 anchor draws/variant/fold. Uniform sample anchor; partner uniformly from same held-out cluster excluding same physical world. Reference partner uniform among OTHER held-out worlds. Per-coordinate full-range normalization, equal-weight RMS over FIVE saved DR families. Encoder bias absent; do not compare absolute totals with previous SIX-family MoE trajectory analysis. Report eligible anchor coverage: singleton-world clusters cannot establish cross-world similarity.',
      'limits':'Two checkpoints differ in training updates, load ranges and auxiliary losses: comparison of available models, not an isolated causal loss ablation. Cached LaFAN frozen-tracker histories; no new simulation or policy changes.', 'folds':[]}
    artifacts={'source_rows':d['source_row'][rows],'worlds':w,'physics':raw,'names':np.array(names),'fit_half0_worlds':parts[0]}
    for name,z in encoders.items():artifacts['latent_'+name]=z
    for fold in (0,1):
      train=np.isin(w,parts[fold]);test=~train;tw=w[test];testworlds=np.unique(tw)
      for encoder,z in encoders.items():
       x,t=z[train],z[test];mu=x.mean(0);ev,v=np.linalg.eigh(np.cov(x.T));transform=v/np.sqrt(np.maximum(ev,ev[-1]*.01))[None,:]
       wx=(x-mu)@transform;wt=(t-mu)@transform;wx/=np.linalg.norm(wx,axis=1,keepdims=True);wt/=np.linalg.norm(wt,axis=1,keepdims=True)
       for metric,fit,query in [('raw',x,t),('whiten01',wx,wt)]:
        for k in (16,32,64,128):
         model=KMeans(n_clusters=k,n_init=5,random_state=731,max_iter=300).fit(fit);labels=model.predict(query)
         pools=[np.flatnonzero(labels==c) for c in range(k)];nworld=np.array([len(np.unique(tw[p])) for p in pools]);valid=nworld[labels]>=2;assert valid.any()
         rng=np.random.default_rng(1729);a=rng.choice(np.flatnonzero(valid),100000);partner=np.empty_like(a)
         for c,pool in enumerate(pools):
          selected=np.flatnonzero(labels[a]==c)
          if not len(selected):continue
          b=rng.choice(pool,len(selected));bad=tw[b]==tw[a[selected]]
          while bad.any():
           b[bad]=rng.choice(pool,bad.sum());bad=tw[b]==tw[a[selected]]
          partner[selected]=b
         aw=tw[a];bw=tw[partner];assert np.all(aw!=bw);assert np.all(labels[a]==labels[partner])
         wi=np.searchsorted(testworlds,aw);j=rng.integers(len(testworlds)-1,size=len(a));j+=j>=wi;rw=testworlds[j]
         delta=np.abs(raw[aw]-raw[bw]);reference=np.abs(raw[aw]-raw[rw]);dn=delta/span;rn=reference/span
         fam=np.stack([np.mean(dn[:,ids]**2,axis=1) for ids in groups.values()],1);ref=np.stack([np.mean(rn[:,ids]**2,axis=1) for ids in groups.values()],1)
         result={'fold':fold,'encoder':encoder,'metric':metric,'k':k,'overall_ratio':float(np.sqrt(fam.mean()/ref.mean())),
           'family_ratios':dict(zip(groups,np.sqrt(fam.mean(0)/ref.mean(0)).tolist())),
           'load_mae_kg':delta[:,34:38].mean(0).tolist(),'random_load_mae_kg':reference[:,34:38].mean(0).tolist(),
           'com_mae_cm':(100*delta[:,:3].mean(0)).tolist(),'random_com_mae_cm':(100*reference[:,:3].mean(0)).tolist(),
           'mass_mae_kg':float(delta[:,3].mean()/mass_half_range),'random_mass_mae_kg':float(reference[:,3].mean()/mass_half_range),
           'friction_mae':float(delta[:,4].mean()),'random_friction_mae':float(reference[:,4].mean()),
           'eligible_anchor_fraction':float(valid.mean()),'empty_test_clusters':int((nworld==0).sum()),'singleton_world_clusters':int((nworld==1).sum()),
           'test_unique_worlds_per_cluster':nworld.tolist(),'test_sample_counts':np.bincount(labels,minlength=k).tolist()}
         report['folds'].append(result);key=f'fold{fold}_{encoder}_{metric}_k{k}'
         artifacts[key+'_labels']=labels;artifacts[key+'_centers']=model.cluster_centers_
         artifacts[key+'_audit_anchor']=a[:1000];artifacts[key+'_audit_partner']=partner[:1000]
         if metric=='whiten01':artifacts[key+'_mean']=mu;artifacts[key+'_transform']=transform
         print(json.dumps({key:result[key] for key in ('fold','encoder','metric','k','overall_ratio','eligible_anchor_fraction')}),flush=True)
      (OUT/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    averages=[]
    for encoder in encoders:
     for metric in ('raw','whiten01'):
      for k in (16,32,64,128):
       rr=[r for r in report['folds'] if (r['encoder'],r['metric'],r['k'])==(encoder,metric,k)];row={'encoder':encoder,'metric':metric,'k':k}
       for key in ('overall_ratio','load_mae_kg','random_load_mae_kg','com_mae_cm','random_com_mae_cm','mass_mae_kg','random_mass_mae_kg','friction_mae','random_friction_mae','eligible_anchor_fraction'):
        row[key]=np.mean([r[key] for r in rr],axis=0).tolist()
       row['family_ratios']={g:float(np.mean([r['family_ratios'][g] for r in rr])) for g in groups};averages.append(row)
    report['averages']=averages
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(OUT/'labels_and_inputs.npz',**artifacts)
    lines=['# 同一历史上的 response10 与 DR 参数监督聚类比较','',report['protocol'],'',report['distance'],'',report['limits'],'',
      '百分比是同簇距离相对随机缩小多少，不是准确率；负值表示反而更远。','',
      '|encoder|metric|K|总体缩小|负载缩小|质量缩小|COM缩小|摩擦缩小|armature缩小|有效anchor覆盖|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in averages:lines.append('|'+r['encoder']+'|'+r['metric']+'|'+str(r['k'])+'|'+'|'.join(f'{100*(1-v):.1f}%' for v in [r['overall_ratio']]+list(r['family_ratios'].values()))+f'|{r["eligible_anchor_fraction"]:.1%}|')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for encoder in encoders:
     for metric,style in [('raw','o-'),('whiten01','s--')]:
      rr=[r for r in averages if r['encoder']==encoder and r['metric']==metric]
      for ax,key in zip(axes,['overall_ratio','limb_load']):
       vals=[r['overall_ratio'] if key=='overall_ratio' else r['family_ratios'][key] for r in rr]
       ax.plot([r['k'] for r in rr],[(1-v)*100 for v in vals],style,label=encoder+' '+metric)
    for ax,title in zip(axes,['All five families','Four limb loads']):
     ax.set_title(title);ax.set_xlabel('K');ax.set_ylabel('Within-cluster distance reduction vs random (%)');ax.set_xscale('log',base=2);ax.set_xticks([16,32,64,128],[16,32,64,128]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Matched histories, held-out DR worlds');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170)

if __name__=='__main__':main()
