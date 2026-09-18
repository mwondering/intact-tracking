"""Fixed causal 200-control-step EMA, with unchanged fitted metric and centers.

Use every available full-history query in time order, score only the same 16
selected queries per world as before. Never use future latent or test physics.
"""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from explore_latent_environment_partitions import classify,dump
from analyze_dr16384_clusters import pair_quality,cluster_rows

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384')
OUT=ROOT/'causal_ema'


def main():
    OUT.mkdir(exist_ok=False);(OUT/'models').mkdir()
    a=dict(np.load(SOURCE/'analysis/selected_worlds.npz'))
    z=np.empty((len(a['world']),32,64),np.float32);valid=np.empty((len(z),32),bool);query=np.empty((len(z),16),int)
    for shard in range(8):
        q=dict(np.load(SOURCE/f'shard_{shard:02d}'/'latents.npz'))
        chosen=np.flatnonzero(a['shard']==shard);n=len(chosen)
        assert np.array_equal(q['world'][:n],a['world'][chosen])
        zz=q['latent'].reshape(32,n,64).transpose(1,0,2)
        z[chosen]=zz/np.linalg.norm(zz,axis=-1,keepdims=True)
        valid[chosen]=((q['short_steps']==50)&(q['long_chunks']==30)).reshape(32,n).T
        query[chosen]=a['source_rows'][chosen]//n
        np.testing.assert_allclose(z[chosen[:,None],query[chosen]],a['z'][chosen],atol=0,rtol=0)
    dump(OUT/'protocol.json',dict(tau_control_steps=200,control_dt=.02,query_interval_control_steps=100,
                                 rule='Feature-space EMA: alpha=exp(-elapsed_control_steps/200); first full-history query initializes state. Invalid histories do not update state. No future windows, no refitting.',
                                 deployment_scope='Static DR per physics session; reset smoother when physics session changes. Actual evaluated update cadence is every 100 control steps (2 seconds).'))
    results=[]
    with threadpool_limits(limits=12):
        for fold in (0,1):
            t=dict(np.load(ROOT/'blends/models'/f'fold{fold}_transform.npz'))
            original=np.load(ROOT/'blends/models'/f'fold{fold}_blend_0p35_k64.npz')
            centers=original['centers'];weight=.35
            state=np.zeros((len(z),102),float);last=np.full(len(z),-1,int)
            outputs=np.empty((len(z),16,102),float)
            all_labels=np.full((len(z),32),-1,int)
            for step in range(32):
                zz=z[:,step]
                r=(zz-t['response_mean'])@t['response_transform']*np.sqrt(1-weight)/t['response_scale']
                p=np.clip(((zz-t['physical_mean'])/t['physical_scale'])@t['physical_coef']+t['physical_intercept'],t['lower'],t['upper'])
                f=np.concatenate([r,p*np.sqrt(weight)/t['physical_scale_total']],axis=1)
                current=(step+1)*100;ok=valid[:,step];fresh=ok&(last<0);continuing=ok&~fresh
                state[fresh]=f[fresh]
                alpha=np.exp(-(current-last[continuing])/200.)[:,None]
                state[continuing]=alpha*state[continuing]+(1-alpha)*f[continuing]
                last[ok]=current
                all_labels[ok,step]=classify(centers,state[ok])
                ii,jj=np.where(query==step)
                assert ok[ii].all()
                outputs[ii,jj]=state[ii]
            labels=classify(centers,outputs)
            assert np.array_equal(labels,all_labels[np.arange(len(z))[:,None],query])
            tr,tn,te,ne=[original[name] for name in ('train_dr','train_nom','test_dr','test_nom')]
            ld,ln=labels[te],labels[ne]
            q=pair_quality(a,te,ld)
            rows,nominal=cluster_rows(a,te,ne,ld,ln,centers,outputs[tn].mean((0,1)),labels[tn],f'fold{fold}_causal200_k64')
            hist=np.stack([(ld==c).mean(1) for c in range(64)],axis=1)
            record=dict(key=f'fold{fold}_causal200_k64',method='causal200',fold=fold,k=64,
                        cluster_pair_quality=q,nominal=nominal,modal_window_fraction=float(hist.max(1).mean()),
                        causality_verified=True)
            results.append(record)
            np.savez_compressed(OUT/'models'/f"{record['key']}.npz",centers=centers,dr_labels=ld,nominal_labels=ln,
                                train_dr=tr,train_nom=tn,test_dr=te,test_nom=ne,all_world_labels=labels,
                                all_query_labels=all_labels,all_query_valid=valid,selected_query_index=query)
            dump(OUT/'results.json',results)
            print(json.dumps(record),flush=True)
    dump(OUT/'complete.json',dict(complete=True,models=2))


if __name__=='__main__':main()
