"""Fresh 16-way clustering of response10 latents and held-out DR severity audit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from probe_moe_latent_payload import load_case

ROOT=Path('runs/limb_context_20260914_moe_routing_diagnostic')
OUT=Path('runs/limb_context_20260916_response10_k16_dr_severity')
SHA='db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac'

def eta(y, labels):
    counts=np.bincount(labels,minlength=16)
    means=np.array([y[labels==c].mean(0) if counts[c] else y.mean(0) for c in range(16)])
    return ((means-y.mean(0))**2*counts[:,None]).sum(0)/np.maximum(((y-y.mean(0))**2).sum(0),1e-15)

def main():
    OUT.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'sample_seed30401/result.json').read_text())
    schema=meta['dr_schema'];groups=schema['groups']; names=list(groups)
    nominal=np.zeros(67);nominal[8]=0.6;nominal[9:38]=1.
    span=np.array(schema['upper'])-schema['lower']
    maxdev=np.maximum(np.abs(np.array(schema['upper'])-nominal),np.abs(np.array(schema['lower'])-nominal))
    cases={s:load_case(ROOT,'sample',s) for s in (30401,30402)}
    for a in cases.values():
        assert a['context_sha256']==SHA
        a['latent']=a['latent'].astype(np.float64)
        a['latent']/=np.linalg.norm(a['latent'],axis=1,keepdims=True)
        delta=(a['raw_dr']-nominal)/span
        a['family']=np.stack([np.sqrt((delta[:,ids]**2).mean(1)) for ids in groups.values()],1)
        a['severity']=np.sqrt((a['family']**2).mean(1))
        dmax=(a['raw_dr']-nominal)/maxdev
        a['maxdev_severity']=np.sqrt(np.stack([(dmax[:,ids]**2).mean(1) for ids in groups.values()],1).mean(1))
        # Alternative matches the current 10-factor, 38-coordinate metric, excluding encoder bias.
        a['tenfactor_severity']=np.sqrt((np.sum(delta[:,:9]**2,axis=1)+(delta[:,9:38]**2).mean(1))/10)
    report={'checkpoint_sha256':SHA,'source':str(ROOT),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'protocol':'Fresh KMeans K=16 n_init=10 seed731, unit raw 64D latents. 32 actual full-history samples/world, no world prototypes, no PCA/whitening/parameter labels. Fit on one 1024-world random DR seed, test on independent 1024-world seed; reverse as robustness check. Cached sample-action closed-loop MoE trajectories, not newly simulated or current PPO.',
      'distance':'RMS over 6 equal-weight family RMSs; coordinates (parameter-nominal)/full sampling span. Nominal foot friction=0.6 (FULL_COLLISION robot config), armature=1, others=0. External force not included. Sensitivity checks: max deviation normalization and 10-factor 38-coordinate metric.',
      'nominal':nominal.tolist(),'schema':schema,'folds':[]}
    artifacts={}
    for trainseed,testseed in ((30401,30402),(30402,30401)):
        a,b=cases[trainseed],cases[testseed];w=b['worlds'];wa=a['worlds']
        model=KMeans(n_clusters=16,n_init=10,random_state=731,max_iter=300).fit(a['latent'])
        tr=model.labels_; labels=model.predict(b['latent']); counts=np.bincount(labels,minlength=16)
        scores=np.column_stack((b['severity'],b['family']))
        held_eta=eta(scores[w],labels)
        rng=np.random.default_rng(935)
        null=np.array([eta(scores[rng.permutation(len(scores))][w],labels) for _ in range(200)])
        trainmeans=np.array([a['severity'][wa][tr==c].mean() for c in range(16)])
        testmeans=np.array([b['severity'][w][labels==c].mean() for c in range(16)])
        predicted=trainmeans[labels]
        r2=1-np.mean((predicted-b['severity'][w])**2)/np.var(b['severity'])
        qlow,qhigh=np.quantile(b['severity'],[.25,.75])
        clusters=[]
        for c in range(16):
            mask=labels==c;vals=b['severity'][w[mask]];raw=b['raw_dr'][w[mask]];fam=b['family'][w[mask]]
            clusters.append({'cluster':c,'samples':int(mask.sum()),'worlds':int(len(np.unique(w[mask]))),
               'severity_mean':float(vals.mean()),'severity_std':float(vals.std()),'severity_q10_q50_q90':np.quantile(vals,[.1,.5,.9]).tolist(),
               'low_quartile_fraction':float(np.mean(vals<=qlow)),'high_quartile_fraction':float(np.mean(vals>=qhigh)),
               'family_rms_mean':fam.mean(0).tolist(),'family_ratio_to_global':(fam.mean(0)/b['family'].mean(0)).tolist(),
               'raw_mean':raw.mean(0).tolist(),'raw_std':raw.std(0).tolist(),'raw_q10_q50_q90':np.quantile(raw,[.1,.5,.9],axis=0).tolist(),
               'maxdev_severity_mean':float(b['maxdev_severity'][w[mask]].mean()),
               'tenfactor_severity_mean':float(b['tenfactor_severity'][w[mask]].mean())})
        fold={'train_seed':trainseed,'test_seed':testseed,'global_severity_mean':float(b['severity'].mean()),
          'global_severity_std':float(b['severity'].std()),'global_severity_q10_q25_q50_q75_q90':np.quantile(b['severity'],[.1,.25,.5,.75,.9]).tolist(),
          'global_family_means':b['family'].mean(0).tolist(),'heldout_severity_r2':float(r2),
          'train_test_cluster_severity_spearman':float(spearmanr(trainmeans,testmeans).statistic),
          'eta2_keys':['overall']+names,'heldout_eta2':held_eta.tolist(),'world_permutation_eta2_mean':null.mean(0).tolist(),
          'world_permutation_p':((1+(null>=held_eta).sum(0))/201).tolist(),
          'maxdev_eta2':float(eta(b['maxdev_severity'][w,None],labels)[0]),
          'tenfactor_eta2':float(eta(b['tenfactor_severity'][w,None],labels)[0]),
          'signed_parameter_eta2':eta(b['raw_dr'][w],labels).tolist(),'clusters':clusters}
        report['folds'].append(fold)
        artifacts[f'centers_fit{trainseed}']=model.cluster_centers_
        artifacts[f'labels_test{testseed}']=labels;artifacts[f'worlds_test{testseed}']=w
        artifacts[f'latent_test{testseed}']=b['latent'];artifacts[f'raw_dr_seed{testseed}']=b['raw_dr']
        print(json.dumps({k:v for k,v in fold.items() if k!='clusters' and k!='signed_parameter_eta2'}),flush=True)
    np.savez_compressed(OUT/'samples_and_labels.npz',**artifacts)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/'cluster_parameters.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['test_seed','cluster','samples','worlds','severity_mean','severity_p10','severity_p90','low_quartile_fraction','high_quartile_fraction']+[f'{n}_deviation' for n in names]+schema['names'])
        for fold in report['folds']:
            for c in fold['clusters']:
                writer.writerow([fold['test_seed'],c['cluster'],c['samples'],c['worlds'],c['severity_mean'],c['severity_q10_q50_q90'][0],c['severity_q10_q50_q90'][2],c['low_quartile_fraction'],c['high_quartile_fraction']]+c['family_rms_mean']+c['raw_mean'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fold=report['folds'][0];ordered=sorted(fold['clusters'],key=lambda c:c['severity_mean'])
    fig,axes=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[1,1.3]})
    y=np.arange(16);means=np.array([c['severity_mean'] for c in ordered]);qs=np.array([c['severity_q10_q50_q90'] for c in ordered])
    axes[0].hlines(y,qs[:,0],qs[:,2],color='lightsteelblue',lw=6);axes[0].plot(means,y,'o',color='navy')
    axes[0].axvline(fold['global_severity_mean'],ls='--',color='gray');axes[0].set_yticks(y,[f'C{c["cluster"]} (n={c["samples"]})' for c in ordered]);axes[0].set_xlabel('Distance to nominal: mean and 10-90%');axes[0].set_title('Held-out random DR environments')
    heat=np.array([c['family_ratio_to_global'] for c in ordered]);im=axes[1].imshow(heat,aspect='auto',vmin=.4,vmax=1.6,cmap='coolwarm',origin='lower')
    axes[1].set_xticks(np.arange(6),names,rotation=35,ha='right');axes[1].set_yticks(y,[f'C{c["cluster"]}' for c in ordered]);axes[1].set_title('Family deviation / global family mean')
    fig.colorbar(im,ax=axes[1]);fig.suptitle('response10 u15000 | fresh KMeans-16 | fit seed30401, test seed30402');fig.tight_layout();fig.savefig(OUT/'cluster_severity.png',dpi=170);plt.close(fig)
    lines=['# Response10 u15000：16 类 latent 的 DR 偏离分析','',report['protocol'],'',report['distance'],'',
       '簇编号仅对本次重新拟合有效。主表是 seed30401 拟合、seed30402 测试。近/远指静态 DR 参数距离，不是实测 10-step 响应幅度。每个环境取 32 条真实历史 latent，不做环境平均。','',
       '|簇|样本数|总体距离均值|P10–P90|最近25%占比|最远25%占比|负载偏离|质量偏离|COM偏离|摩擦偏离|armature偏离|bias偏离|','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in ordered:
        q=c['severity_q10_q50_q90'];lines.append(f'|{c["cluster"]}|{c["samples"]}|{c["severity_mean"]:.4f}|{q[0]:.3f}–{q[2]:.3f}|{c["low_quartile_fraction"]:.1%}|{c["high_quartile_fraction"]:.1%}|'+ '|'.join(f'{v:.3f}' for v in c['family_rms_mean'])+'|')
    for f in report['folds']:
        lines+=['',f'Test seed {f["test_seed"]}: severity eta²={f["heldout_eta2"][0]:.4f}; held-out R²={f["heldout_severity_r2"]:.4f}; training/test cluster mean rank correlation={f["train_test_cluster_severity_spearman"]:.4f}.',f'Family deviation eta²: {dict(zip(names,f["heldout_eta2"][1:]))}']
    lines+=['','![Cluster severity](cluster_severity.png)','','完整各坐标均值、标准差、P10/P50/P90 在 summary.json；CSV 保留均值；NPZ 保存实际 latent、新簇标签、中心和物理参数。原数据来自该 encoder 的闭环 MoE 交互，不是本次新跑的模拟器。各参数独立均匀随机，样本不包含专门构造的全 nominal 或全弱 DR 环境；不能从缺少全弱簇推断模型不能识别全弱环境。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
