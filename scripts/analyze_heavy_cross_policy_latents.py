"""World-disjoint cross-policy probes and joint plots for heavy context latents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ARMS = ('tracker_v0','tracker_v1','residual_v0','residual_v1')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or not np.isfinite(x).all():
        raise ValueError('Invalid or collapsed-to-zero latent')
    return x/norm


def retrieval(query, gallery):
    scores = query @ gallery.T
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = 1+np.argmax(order == np.arange(len(query))[:,None], axis=1)
    # Independent Euclidean implementation checks direction, pairing and sorting.
    np.testing.assert_array_equal(order[:,0], cdist(query, gallery).argmin(axis=1))
    return {'top1':float((ranks==1).mean()), 'top5':float((ranks<=5).mean()),
        'median_rank':float(np.median(ranks)), 'worlds':len(query)}, ranks


def interval(per_world, seed=921):
    rng=np.random.default_rng(seed)
    values=np.asarray(per_world)
    boots=np.asarray([values[rng.integers(len(values),size=len(values))].mean() for _ in range(2000)])
    return np.quantile(boots,[.025,.975]).tolist()


def groups(names):
    result={}
    for i,name in enumerate(names):
        if name.startswith('base_com/'):
            group='Torso COM '+name.rsplit('/',1)[1]
        elif name.startswith('base_mass/'):
            group='Torso mass'
        elif name.startswith('foot_friction/'):
            group='Friction'
        elif name.startswith('motor_params_implicit/'):
            group={'kp_scale':'Kp','kd_scale':'Kd','armature_scale':'Armature'}[name.split('/')[1]]
        elif '/added_mass_kg/' in name:
            group=name.rsplit('/',1)[1].replace('_',' ').capitalize()+' mass'
        elif '/payload_com_offset/' in name:
            group=name.split('/')[-2].replace('_',' ').capitalize()+' COM'
        else:
            raise ValueError(name)
        result.setdefault(group,[]).append(i)
    return result


def probes(features, truth, split, physical_ranges, parameter_groups):
    n=len(truth)
    predictions={f'{a}_to_{b}':np.zeros_like(truth,dtype=np.float64)
                 for a in ('tracker','residual') for b in ('tracker','residual')}
    chosen_alphas={}
    for fold,(train,test) in enumerate(split):
        for source in ('tracker','residual'):
            # One snapshot per world, view 0 only. Every view of test worlds is excluded.
            xtrain=features[f'{source}_v0'][train]
            scaler=StandardScaler().fit(xtrain)
            model=RidgeCV(alphas=(.1,1.,10.,100.,1000.),alpha_per_target=True)
            model.fit(scaler.transform(xtrain),truth[train])
            chosen_alphas[f'{fold}_{source}']=np.asarray(model.alpha_).tolist()
            for target in ('tracker','residual'):
                predictions[f'{source}_to_{target}'][test]=model.predict(
                    scaler.transform(features[f'{target}_v1'][test]))
    metrics={}
    variance=truth.var(0)
    for key,prediction in predictions.items():
        error=prediction-truth
        r2=1-np.mean(error**2,axis=0)/np.maximum(variance,1e-12)
        mae=np.mean(np.abs(error),axis=0)*physical_ranges
        metrics[key]={'r2':r2.tolist(),'mae_physical':mae.tolist(),
            'groups':{name:{'r2_mean':float(r2[ids].mean()),
                           'mae_physical_mean':float(mae[ids].mean())}
                      for name,ids in parameter_groups.items()}}
    return metrics,predictions,chosen_alphas


def policy_probe(features, split):
    n=len(features['tracker_v0'])
    probability=np.zeros((2,n))
    labels=np.repeat([0,1],n)
    for train,test in split:
        train_x=np.concatenate([features[f'{p}_v0'][train] for p in ('tracker','residual')])
        test_x=np.concatenate([features[f'{p}_v1'][test] for p in ('tracker','residual')])
        y=np.repeat([0,1],len(train))
        scaler=StandardScaler().fit(train_x)
        clf=LogisticRegression(C=1.,max_iter=2000,random_state=921)
        clf.fit(scaler.transform(train_x),y)
        probability[:,test]=clf.predict_proba(scaler.transform(test_x))[:,1].reshape(2,-1)
    correct=((probability >= .5)==np.asarray([0,1])[:,None]).mean(0)
    return {'accuracy':float(correct.mean()), 'accuracy_world_bootstrap_95':interval(correct),
        'roc_auc':float(roc_auc_score(labels,probability.ravel())), 'chance':.5,
        'protocol':'5-fold held-out worlds; train view 0, test view 1, balanced policies, fixed linear logistic C=1; no per-row splits'},probability


def save_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def plot_joint(coords, z, truth, names, root, title):
    n=len(truth)
    labels=np.repeat(['Tracker / motion A','Tracker / motion B','Residual / motion A','Residual / motion B'],n)
    colors=['#1766b3','#1766b3','#e17a25','#e17a25']
    markers=['o','x','o','x']
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for i in range(4):
        part=coords[i*n:(i+1)*n]
        axes[0,0].scatter(*part.T,s=9,c=colors[i],marker=markers[i],alpha=.55,label=labels[i*n])
    axes[0,0].legend(fontsize=8,markerscale=2,frameon=False)
    axes[0,0].set_title('Same embedding, colored by policy')
    axes[0,1].scatter(*coords.T,s=4,c='#c8cdd3',alpha=.25)
    # This subset is selected by RNG before inspecting coordinates or results.
    chosen=np.random.default_rng(921).choice(n,min(24,n),replace=False)
    cmap=plt.get_cmap('turbo')
    for j,world in enumerate(chosen):
        pts=coords[world+np.arange(4)*n]
        color=cmap(j/max(len(chosen)-1,1))
        axes[0,1].plot(*pts[[0,2]].T,c=color,lw=.8,alpha=.8)
        axes[0,1].plot(*pts[[1,3]].T,c=color,lw=.8,alpha=.8,ls='--')
        axes[0,1].scatter(*pts[:2].T,c=[color],s=21,marker='o')
        axes[0,1].scatter(*pts[2:].T,c=[color],s=25,marker='^')
    axes[0,1].set_title('24 random worlds; lines join policies')
    factors=[(0,'Torso COM x'),(92,'Left hand mass'),(94,'Left shin mass'),(4,'Friction')]
    for ax,(index,label) in zip([axes[0,2],*axes[1]],factors):
        values=np.tile(truth[:,index],4)
        dots=ax.scatter(*coords.T,c=values,s=8,cmap='viridis',alpha=.7,linewidths=0)
        fig.colorbar(dots,ax=ax,shrink=.8,label='Physical value')
        ax.set_title(label)
    for ax in axes.ravel():
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(title+'\nOne full-history snapshot per world / policy / motion; common joint fit',fontsize=14)
    fig.tight_layout()
    save_figure(fig,root/'joint_tsne')


def main(args):
    root=Path(args.root).resolve()
    output=root/'analysis'
    output.mkdir(exist_ok=False)
    metadata={arm:json.loads((root/arm/'metadata.json').read_text()) for arm in ARMS}
    data={arm:dict(np.load(root/arm/'latents.npz',allow_pickle=False)) for arm in ARMS}
    ref=metadata[ARMS[0]]
    for arm in ARMS:
        m=metadata[arm]
        if not m['complete'] or not m['actor_and_encoder_unchanged']:
            raise ValueError('Incomplete or mutable collection')
        for key in ('context_sha256','tracker_sha256','checkpoint_sha256','schema','manifest_sha256'):
            assert m[key]==ref[key],key
        assert m['world_metadata']['physics_world_fingerprints']==ref['world_metadata']['physics_world_fingerprints']
        np.testing.assert_array_equal(data[arm]['step'],data[ARMS[0]]['step'])
        np.testing.assert_array_equal(data[arm]['truth_normalized'],data[ARMS[0]]['truth_normalized'])
    for view in (0,1):
        a,b=metadata[f'tracker_v{view}'],metadata[f'residual_v{view}']
        assert a['initial_state_sha256']==b['initial_state_sha256']
        assert a['force_diagnostics']['query_force_sha256']==b['force_diagnostics']['query_force_sha256']
    full=np.logical_and.reduce([data[arm]['full'] for arm in ARMS])
    full &= data[ARMS[0]]['step'][:,None]>=args.minimum_step
    worlds=np.flatnonzero(full.any(0))
    rows=full[:,worlds].argmax(0)
    if len(worlds)<64:
        raise ValueError('Too few matched full-history worlds')
    def take(arm,key):
        return data[arm][key][rows,worlds]
    z={arm:unit(take(arm,'z')) for arm in ARMS}
    motion={arm:take(arm,'motion') for arm in ARMS}
    assert np.all(motion['tracker_v0']!=motion['tracker_v1'])
    for view in (0,1):
        np.testing.assert_array_equal(motion[f'tracker_v{view}'],motion[f'residual_v{view}'])
    history_duplicates={}
    for i,a in enumerate(ARMS):
        for b in ARMS[i+1:]:
            count=int(np.sum(take(a,'history_sha256')==take(b,'history_sha256')))
            history_duplicates[f'{a}__{b}']=count
            assert count==0,'Exact duplicated interaction histories'
    truth=data[ARMS[0]]['truth_normalized'][worlds].astype(np.float64)
    schema=ref['schema']
    lower,upper=np.asarray(schema['lower']),np.asarray(schema['upper'])
    physical=lower+truth*(upper-lower)
    parameter_groups=groups(schema['names'])
    n=len(worlds)
    splits=list(KFold(5,shuffle=True,random_state=921).split(worlds))
    fold=np.zeros(n,dtype=int)
    for i,(_,test) in enumerate(splits):fold[test]=i
    report={'complete':False,'matched_worlds':n,'requested_worlds':len(full[0]),
        'excluded_worlds':np.flatnonzero(~full.any(0)).tolist(),
        'primary_selection':'First simultaneous full 50+300 history in all four arms at or after minimum_step; same step within each world; one snapshot per arm/world, no world averaging',
        'minimum_step':args.minimum_step,'sampling_steps':data[ARMS[0]]['step'].tolist(),
        'paired_physics_initial_states_and_forces_verified':True,'exact_history_duplicates':history_duplicates,
        'random_retrieval_top1':1/n,'random_retrieval_top5':5/n,
        'retrieval':{},'geometry':{},'parameters':{},'policy_decodability':{},
        'parameter_groups':parameter_groups,'parameter_names':schema['names'],
        'scope':'One encoder / one residual checkpoint / one physics seed. New HDR worlds; training-catalog motion IDs. World bootstrap is conditional on this motion set and these checkpoints, not training-seed uncertainty.'}
    report['coverage']={arm:{'worlds_with_full_history_after_minimum_step':int(np.any(
        data[arm]['full'][data[arm]['step']>=args.minimum_step],axis=0).sum()),
        'failure_events':metadata[arm]['failure_events'],'failure_worlds':metadata[arm]['failure_worlds'],
        'resets':metadata[arm]['resets']} for arm in ARMS}
    report['policy_action_rms']={arm:{'action_rms':float(np.sqrt(np.mean(data[arm]['action_rms']**2))),
        'residual_rms':float(np.sqrt(np.mean(data[arm]['residual_rms']**2)))} for arm in ARMS}
    ranks_out={}
    for a in ARMS:
        for b in ARMS:
            if a==b:continue
            key=f'{a}__{b}'
            metric,ranks=retrieval(z[a],z[b])
            metric['top1_world_bootstrap_95']=interval(ranks==1)
            report['retrieval'][key]=metric
            ranks_out[key]=ranks
            dist=np.linalg.norm(z[a]-z[b],axis=1)
            between=pdist((z[a]+z[b])/2)
            report['geometry'][key]={'same_world_cosine':float(np.mean(np.sum(z[a]*z[b],axis=1))),
                'same_world_distance_rms':float(np.sqrt(np.mean(dist**2))),
                'between_world_center_distance_rms':float(np.sqrt(np.mean(between**2))),
                'same_over_between_rms':float(np.sqrt(np.mean(dist**2)/np.mean(between**2)))}
    feature_sets={'latent':z, 'raw_latent':{arm:take(arm,'z').astype(np.float64) for arm in ARMS},
        'history_mean_std':{arm:take(arm,'history_features').astype(np.float64) for arm in ARMS}}
    report['latent_norm']={arm:{'mean':float(np.linalg.norm(take(arm,'z'),axis=1).mean()),
        'std':float(np.linalg.norm(take(arm,'z'),axis=1).std())} for arm in ARMS}
    report['history_statistics_retrieval']={}
    for a in ('tracker','residual'):
        for b in ('tracker','residual'):
            result,_=retrieval(unit(feature_sets['history_mean_std'][f'{a}_v0']),
                               unit(feature_sets['history_mean_std'][f'{b}_v1']))
            report['history_statistics_retrieval'][f'{a}_to_{b}']=result
    saved={}
    for feature_name,features in feature_sets.items():
        metrics,predictions,alphas=probes(features,truth,splits,upper-lower,parameter_groups)
        report['parameters'][feature_name]={'metrics':metrics,'ridge_alphas':alphas,
            'protocol':'5-fold unseen-world evaluation. Fit per-target RidgeCV and scaling on source policy / motion view 0 training worlds only; evaluate motion view 1 of held-out worlds. No residual auxiliary head is used.'}
        decodable,probability=policy_probe(features,splits)
        report['policy_decodability'][feature_name]=decodable
        saved[feature_name+'_policy_probability']=probability
        saved.update({feature_name+'_'+key:value for key,value in predictions.items()})
    # Row-shuffled target correspondence is a retrieval negative control, not a refitted encoder.
    rng=np.random.default_rng(921)
    shuffled=[]
    for _ in range(200):
        order=rng.permutation(n)
        predictions=np.argmax(z['tracker_v0'] @ z['residual_v1'][order].T,axis=1)
        shuffled.append(float(np.mean(predictions==np.arange(n))))
    report['shuffled_world_retrieval_top1']={'mean':float(np.mean(shuffled)),'range_95':np.quantile(shuffled,[.025,.975]).tolist()}
    # Repeated time windows are descriptive; they are never treated as independent test rows.
    report['time_sensitivity']=[]
    for t,step in enumerate(data[ARMS[0]]['step']):
        eligible=np.flatnonzero(np.logical_and.reduce([data[a]['full'][t] for a in ARMS]))
        if len(eligible)<10:continue
        a=unit(data['tracker_v0']['z'][t,eligible]);b=unit(data['residual_v1']['z'][t,eligible])
        metrics,_=retrieval(a,b)
        report['time_sensitivity'].append({'step':int(step),**metrics})
    raw=np.concatenate([z[a] for a in ARMS])
    embeddings={}
    report['embeddings']={}
    for perplexity,seed in ((30,921),(50,922)):
        estimator=TSNE(n_components=2,perplexity=min(perplexity,(len(raw)-1)/3),
            init='pca',learning_rate='auto',random_state=seed,max_iter=1000)
        coords=estimator.fit_transform(raw)
        key=f'tsne_p{perplexity}_s{seed}'
        embeddings[key]=coords
        report['embeddings'][key]={'kl_divergence':float(estimator.kl_divergence_),
            'trustworthiness_k10':float(trustworthiness(raw,coords,n_neighbors=10)),
            'joint_fit':True,'source':'raw unit 64D latent; no per-policy alignment or whitening'}
    pca=PCA(2).fit(raw)
    embeddings['pca']=pca.transform(raw)
    report['embeddings']['pca_explained_variance']=pca.explained_variance_ratio_.tolist()
    plot_joint(embeddings['tsne_p30_s921'],z,physical,schema['names'],output,
        f'Heavy encoder u8816 | {n} paired HDR worlds | residual update {ref["completed_updates"]}')
    # Four cross-motion retrieval directions, all computed in the original 64D space.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    table=np.asarray([[report['retrieval'][f'{a}_v0__{b}_v1']['top1'] for b in ('tracker','residual')]
                     for a in ('tracker','residual')])
    im=axes[0].imshow(table,vmin=0,vmax=1,cmap='Blues')
    axes[0].set(xticks=[0,1],xticklabels=['Tracker gallery B','Residual gallery B'],
        yticks=[0,1],yticklabels=['Tracker query A','Residual query A'],title='Environment retrieval Top-1 (64D)')
    for i in range(2):
        for j in range(2):axes[0].text(j,i,f'{table[i,j]:.1%}',ha='center',va='center',fontsize=18,color='white' if table[i,j]>.55 else 'black')
    feature_names=['latent','history_mean_std']
    acc=[report['policy_decodability'][f]['accuracy'] for f in feature_names]
    axes[1].bar(['Encoder latent','History mean/std'],acc,color=['#1766b3','#9ba8b6'])
    axes[1].axhline(.5,color='black',ls='--',lw=1,label='Chance')
    for i,value in enumerate(acc):axes[1].text(i,value+.02,f'{value:.1%}',ha='center')
    axes[1].set(ylim=(0,1.05),ylabel='Accuracy',title='Can a linear probe identify the policy?')
    axes[1].legend(frameon=False)
    fig.tight_layout();save_figure(fig,output/'retrieval_and_policy_probe')
    keys=('tracker_to_tracker','tracker_to_residual','residual_to_tracker','residual_to_residual')
    groups_order=list(parameter_groups)
    values=np.asarray([[report['parameters']['latent']['metrics'][key]['groups'][g]['r2_mean'] for key in keys] for g in groups_order])
    fig,ax=plt.subplots(figsize=(9,10))
    im=ax.imshow(values,vmin=-.25,vmax=1,cmap='RdYlGn',aspect='auto')
    ax.set(xticks=np.arange(4),xticklabels=['Tracker → Tracker','Tracker → Residual','Residual → Tracker','Residual → Residual'],
        yticks=np.arange(len(groups_order)),yticklabels=groups_order,title='Latent-only linear physical probes: unseen worlds + different motion\nR²; scaling and decoder fit on training worlds only')
    plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
    for i in range(len(groups_order)):
        for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,shrink=.65,label='R²');fig.tight_layout();save_figure(fig,output/'physical_parameter_transfer')
    # Export alternative projections so users can inspect t-SNE sensitivity.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,key in zip(axes,('tsne_p50_s922','pca')):
        for index,policy in enumerate(('tracker','residual')):
            coords=embeddings[key][index*2*n:(index+1)*2*n]
            ax.scatter(*coords.T,s=7,alpha=.45,label=policy)
        ax.set_title(key);ax.legend(frameon=False);ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();save_figure(fig,output/'projection_sensitivity')
    np.savez_compressed(output/'analysis_arrays.npz',worlds=worlds,rows=rows,world_fold=fold,
        truth_normalized=truth,**{a+'_z':z[a] for a in ARMS},**embeddings,**saved,
        **{key+'_rank':value for key,value in ranks_out.items()})
    report.update(complete=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),Path('scripts/probe_heavy_cross_policy_latents.py'))})
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'matched_worlds':n,'retrieval':report['retrieval'],
        'policy_decodability':report['policy_decodability']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--minimum-step',type=int,default=500)
    main(parser.parse_args())
