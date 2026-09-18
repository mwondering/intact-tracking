"""Held-out random-DR KMeans sweep on raw and regularized-whitened response10 latents."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from sklearn.cluster import KMeans

SOURCE=Path('runs/limb_context_20260916_response10_k16_dr_severity')
OUT=Path('runs/limb_context_20260916_response10_whiten_k_sweep')

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    meta=json.loads((SOURCE/'summary.json').read_text());schema=meta['schema']
    data=np.load(SOURCE/'samples_and_labels.npz');cases={}
    for seed in (30401,30402):
        x=data[f'latent_test{seed}'];w=data[f'worlds_test{seed}'];dr=data[f'raw_dr_seed{seed}']
        assert np.all(np.bincount(w)==32)
        np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
        diff=(dr-np.array(meta['nominal']))/(np.array(schema['upper'])-schema['lower'])
        family=np.stack([np.sqrt(np.mean(diff[:,idx]**2,axis=1)) for idx in schema['groups'].values()],axis=1)
        severity=np.sqrt(np.mean(family**2,axis=1))
        cases[seed]={'x':x,'w':w,'dr':dr,'target':np.column_stack([dr,severity,family])}
    report={'encoder_sha256':meta['checkpoint_sha256'],'source':str(SOURCE),'source_npz_sha256':digest(SOURCE/'samples_and_labels.npz'),
      'script_sha256':digest(Path(__file__)), 'schema':schema,
      'protocol':'Two independent random DR seeds, 1024 worlds each, 32 actual full-memory latents/world. Fit centers and whitening on one complete seed, test on the other, reverse. No environment prototypes or DR labels in clustering. Cached response10/u15000 sample-action MoE trajectories. KMeans 16/32/64/128, n_init=5, random_state=731, max_iter=300. Same settings for both metrics.',
      'whitening':'Training unit-latent mean and covariance only. W=V diag(1/sqrt(max(eigenvalue,0.01*max_eigenvalue))); transform (z-mean)W, then L2 normalize. No dropped dimensions.',
      'readout':'Only training-cluster parameter means predict held-out DR. R2 is instantaneous prediction, world-balanced via 32 samples/world. Negative R2 means worse than held-out population mean; not classification accuracy. Family signed R2 averages coordinate R2; severity uses 6 equal families and full-range normalization.',
      'folds':[]}
    archive={}
    for train,test in ((30401,30402),(30402,30401)):
        a,b=cases[train],cases[test];x,t=a['x'],b['x']
        mu=x.mean(0);ev,v=np.linalg.eigh(np.cov(x.T));floor=ev[-1]*.01;transform=v/np.sqrt(np.maximum(ev,floor))[None,:]
        wx=(x-mu)@transform;wt=(t-mu)@transform
        wx/=np.linalg.norm(wx,axis=1,keepdims=True);wt/=np.linalg.norm(wt,axis=1,keepdims=True)
        archive[f'mean_train{train}']=mu;archive[f'whitening_train{train}']=transform
        fold={'train_seed':train,'test_seed':test,'eigenvalues':ev.tolist(),'floored_dimensions':int((ev<floor).sum()),'results':[]}
        ya=a['target'][a['w']];yb=b['target'][b['w']]
        for name,fit,query in (('raw',x,t),('whiten01',wx,wt)):
            for k in (16,32,64,128):
                model=KMeans(n_clusters=k,n_init=5,random_state=731,max_iter=300).fit(fit)
                ids=model.labels_;labels=model.predict(query)
                counts=np.bincount(ids,minlength=k);testcounts=np.bincount(labels,minlength=k)
                assert counts.min()>0
                means=np.stack([ya[ids==i].mean(0) for i in range(k)])
                error=(means[labels]-yb)**2;r2=1-error.mean(0)/yb.var(0)
                mae=np.abs(means[labels]-yb).mean(0)
                perworld=np.zeros((len(b['dr']),yb.shape[1]));np.add.at(perworld,b['w'],error);perworld/=32
                trainworlds=np.array([len(np.unique(a['w'][ids==i])) for i in range(k)])
                testworlds=np.array([len(np.unique(b['w'][labels==i])) for i in range(k)])
                memberships=np.zeros((len(b['dr']),k),dtype=int);np.add.at(memberships,(b['w'],labels),1)
                result={'metric':name,'k':k,'r2_loads':r2[:4].tolist(),'mae_loads_kg':mae[:4].tolist(),'r2_parameters':r2[:67].tolist(),
                  'r2_parameter_families':{g:float(r2[idx].mean()) for g,idx in schema['groups'].items()},
                  'r2_severity':float(r2[67]),'r2_family_deviations':dict(zip(schema['groups'],r2[68:].tolist())),
                  'train_counts':counts.tolist(),'test_counts':testcounts.tolist(),'train_unique_worlds_per_cluster':trainworlds.tolist(),
                  'test_unique_worlds_per_cluster':testworlds.tolist(),'test_empty_clusters':int((testcounts==0).sum()),
                  'test_worlds_per_cluster_min_median_max':np.quantile(testworlds,[0,.5,1]).tolist(),
                  'train_worlds_per_cluster_min_median_max':np.quantile(trainworlds,[0,.5,1]).tolist(),
                  'test_dominant_class_fraction_mean':float((memberships.max(1)/32).mean()),
                  'test_unique_classes_per_world_mean':float((memberships>0).sum(1).mean()),
                  'fit_iterations':int(model.n_iter_),'train_inertia_per_sample':float(model.inertia_/len(x))}
                prefix=f'train{train}_{name}_k{k}'
                archive[prefix+'_centers']=model.cluster_centers_;archive[prefix+'_train_labels']=ids;archive[prefix+'_test_labels']=labels
                archive[prefix+'_train_parameter_means']=means;archive[prefix+'_test_world_mse']=perworld
                fold['results'].append(result)
                print(json.dumps({'test_seed':test,**{key:result[key] for key in ('metric','k','r2_loads','r2_severity','test_worlds_per_cluster_min_median_max')}}),flush=True)
        report['folds'].append(fold)
        (OUT/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    averages=[]
    for i in range(8):
        rs=[f['results'][i] for f in report['folds']];r={'metric':rs[0]['metric'],'k':rs[0]['k']}
        for key in ('r2_loads','mae_loads_kg','r2_severity','test_dominant_class_fraction_mean','test_unique_classes_per_world_mean'):
            r[key]=np.mean([v[key] for v in rs],axis=0).tolist()
        r['r2_parameter_families']={g:float(np.mean([v['r2_parameter_families'][g] for v in rs])) for g in schema['groups']}
        averages.append(r)
    report['averages']=averages
    np.savez_compressed(OUT/'models_and_labels.npz',**archive)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['metric','k','left_hand_r2','right_hand_r2','left_shin_r2','right_shin_r2','severity_r2','mean_unique_classes_per_world'])
        for r in averages:w.writerow([r['metric'],r['k']]+r['r2_loads']+[r['r2_severity'],r['test_unique_classes_per_world_mean']])
    lines=['# Response10 u15000：白化与类别数对照','',report['protocol'],'',report['whitening'],'',report['readout'],'',
      '|距离|K|左手R²|右手R²|左小腿R²|右小腿R²|总体偏离R²|每world访问类别数|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in averages:lines.append('|'+r['metric']+'|'+str(r['k'])+'|'+'|'.join(f'{v:.3f}' for v in r['r2_loads']+[r['r2_severity'],r['test_unique_classes_per_world_mean']])+'|')
    lines+=['','均为两次交换训练/测试方向的平均；簇均值仅由训练侧参数估计。参数未参与白化或聚类。每world访问类别数取32个离散历史窗口，不是逐步路由切换率。增加K会机械地降低训练簇内距离，因此以独立环境读出指标为主。','',
      '这些离线结果不代表专家控制收益。DR参数读取与响应类别效用不是同一指标。白化改变距离定义，不直接比较不同度量的inertia。完整每折/每簇占用、参数读出见 summary.json，变换/中心/标签见 models_and_labels.npz。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for name,style in (('raw','o-'),('whiten01','s-')):
        rr=[r for r in averages if r['metric']==name];ks=[r['k'] for r in rr]
        axes[0].plot(ks,[np.mean(r['r2_loads'][:2]) for r in rr],style,label=name)
        axes[1].plot(ks,[np.mean(r['r2_loads'][2:]) for r in rr],style,label=name)
        axes[2].plot(ks,[r['r2_severity'] for r in rr],style,label=name)
    for ax,title in zip(axes,['Hand mass R2 (two hands mean)','Shin mass R2 (two shins mean)','Overall DR distance-to-nominal R2']):
        ax.set_title(title);ax.set_xscale('log',base=2);ax.set_xticks([16,32,64,128],[16,32,64,128]);ax.set_xlabel('K');ax.axhline(0,color='gray',lw=.8);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('response10/u15000 | held-out DR | mean of two seed directions');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    lines+=['','![Comparison](comparison.png)'];(OUT/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
