"""Export and GPU-audit the selected frozen metric without changing a policy."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from intact_tracking.latent_environment_partition import LatentEnvironmentPartition
from explore_latent_environment_partitions import dump

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384')
OUT=ROOT/'export'
SHA='3c489cf5457889bf88441175ec4f10f4448104f44f7914eaaf762a9c3659d8f7'


@torch.inference_mode()
def main():
    torch.set_num_threads(3);torch.set_float32_matmul_precision('highest')
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'runtime_verification.json').exists(), 'Do not overwrite a completed export audit'
    a=dict(np.load(SOURCE/'analysis/selected_worlds.npz'))
    selected=json.loads((ROOT/'blends/selection.json').read_text())
    assert next(x for x in selected['selected'] if x['fold']==0 and x['k']==64)['selected']['weight']==.35
    transform=dict(np.load(ROOT/'blends/models/fold0_transform.npz'))
    fit=np.load(ROOT/'blends/models/fold0_blend_0p35_k64.npz')
    ema=np.load(ROOT/'causal_ema/models/fold0_causal200_k64.npz')
    nominal=int(np.bincount(ema['nominal_labels'].ravel(),minlength=64).argmax())
    artifact=OUT/'partition_k64.npz'
    np.savez_compressed(artifact,**transform,centers=fit['centers'],format_version='latent_environment_blend_v1',
                        encoder_sha256=SHA,physical_weight=.35,tau_control_steps=200.,nominal_class=nominal)
    metadata=dict(encoder_sha256=SHA,model='fold0_blend_0p35_k64',classes=64,nominal_class=nominal,
                  physical_weight=.35,response_weight=.65,query_stride_control_steps=100,tau_control_steps=200,
                  normalized_input='Raw 64D latent is unit-normalized inside the runtime; normalized=True accepts pre-normalized latent.',
                  requirements='Full 350-step history; fixed row identity for causal mode; reset causal state on physics-session change.',
                  scope='Original frozen tracker distribution, hand loads0..2.5/shin0..4 and original background DR; no policy modification.',
                  selection='Fold0 FIT worlds and FIT response motion families only; both folds independently chose0.35 at K64.',
                  artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    dump(OUT/'metadata.json',metadata)
    labels=ema['all_world_labels'];prob=np.stack([(labels==c).mean(1) for c in range(64)],axis=1)
    modal=prob.argmax(1)
    np.savez_compressed(OUT/'environment_bank.npz',world=a['world'],physics=a['physics'],names=a['names'],
                        nominal=a['nominal'],fit_world=(a['fold']==0),class_id=modal,class_probabilities=prob,
                        observed_query_labels=labels,observed_query_steps=a['step'])
    class_stats=[]
    for c in range(64):
        use=(modal==c)&~a['nominal'];p=a['physics'][use]
        class_stats.append(dict(cluster=c,dr_worlds=int(use.sum()),nominal_worlds=int(((modal==c)&a['nominal']).sum()),
                                mean=p.mean(0).tolist() if len(p) else None,
                                p10=np.quantile(p,.1,axis=0).tolist() if len(p) else None,
                                p90=np.quantile(p,.9,axis=0).tolist() if len(p) else None))
    dump(OUT/'environment_bank_summary.json',dict(names=a['names'].tolist(),classes=class_stats,
         note='World label is modal causal label across selected observations. This bank is an explicit DR inventory, not a guarantee that arbitrary resampling around its means stays in class.'))

    runtime=LatentEnvironmentPartition(artifact,encoder_sha256=SHA).to('cuda:0')
    # Full held-out single-window inference audit against stored NumPy labels.
    z=a['z'][fit['test_dr']].reshape(-1,64)
    labels_cuda=[]
    for start in range(0,len(z),4096):
        result=runtime(torch.from_numpy(z[start:start+4096]).to('cuda:0'),normalized=True)
        labels_cuda.append(result['class_id'].cpu().numpy())
    predicted=np.concatenate(labels_cuda);expected=fit['dr_labels'].ravel()
    mismatch=float((predicted!=expected).mean())
    assert mismatch<.001,mismatch
    # Causal replay of all 32 queries for 4096 worlds spanning every shard.
    pick=[];source_z=[];source_valid=[]
    rng=np.random.default_rng(52733)
    for shard in range(8):
        ids=np.flatnonzero(a['shard']==shard);chosen=rng.choice(len(ids),512,replace=False);pick.extend(ids[chosen])
        q=dict(np.load(SOURCE/f'shard_{shard:02d}'/'latents.npz'));n=len(ids)
        zz=q['latent'].reshape(32,n,64).transpose(1,0,2)[chosen]
        zz=zz/np.linalg.norm(zz,axis=-1,keepdims=True)
        source_z.append(zz);source_valid.append(((q['short_steps']==50)&(q['long_chunks']==30)).reshape(32,n).T[chosen])
    pick=np.array(pick);zz=np.concatenate(source_z);valid=np.concatenate(source_valid)
    runtime.clear_history();actual=[]
    for step in range(32):
        result=runtime.route_causal(torch.from_numpy(zz[:,step]).to('cuda:0'),elapsed_control_steps=100,
                                    valid=torch.from_numpy(valid[:,step]).to('cuda:0'),normalized=True)
        actual.append(result['class_id'].cpu().numpy())
    actual=np.stack(actual,axis=1);truth=ema['all_query_labels'][pick]
    causal_mismatch_all=float((actual!=truth).mean())
    causal_mismatch=float((actual[valid]!=truth[valid]).mean())
    # The offline archive deliberately has -1 at every invalid query; runtime
    # instead retains the last usable decision after first initialization.
    never_ready=np.cumsum(valid,axis=1)==0
    assert np.all(actual[never_ready]==-1)
    print(json.dumps(dict(event='causal_comparison',valid_queries=int(valid.sum()),
                          mismatch_valid=causal_mismatch,mismatch_all_including_invalid=causal_mismatch_all)),flush=True)
    assert causal_mismatch<.001,causal_mismatch
    # A physics-session reset must discard old history, not mix old/new DR.
    latest=torch.from_numpy(zz[:,-1]).to('cuda:0');mask=torch.ones(len(zz),device='cuda:0',dtype=torch.bool)
    expected_reset=runtime(latest,normalized=True)['class_id']
    reset_output=runtime.route_causal(latest,elapsed_control_steps=100,valid=mask,reset=mask,normalized=True)['class_id']
    assert torch.equal(expected_reset,reset_output)
    # An invalid post-reset row has no class, even with a numeric latent supplied.
    invalid=runtime.route_causal(latest,elapsed_control_steps=100,valid=~mask,reset=mask,normalized=True)
    assert (invalid['class_id']==-1).all() and not invalid['ready'].any()
    try:
        LatentEnvironmentPartition(artifact,encoder_sha256='wrong-checkpoint')
    except ValueError:
        pass
    else:
        raise AssertionError('Checkpoint mismatch was accepted')
    x=torch.from_numpy(z[:8192]).to('cuda:0')
    for _ in range(5):runtime(x,normalized=True)
    torch.cuda.synchronize();started=time.perf_counter()
    for _ in range(20):runtime(x,normalized=True)
    torch.cuda.synchronize();ms=(time.perf_counter()-started)*1000/20
    report=dict(passed=True,device='cuda:0',precision='float32; TF32 disabled',heldout_stateless_windows=len(z),
                stateless_label_difference_fraction=mismatch,causal_worlds=len(zz),causal_queries=32,
                causal_label_difference_fraction=causal_mismatch,physics_reset_verified=True,
                invalid_query_archive_semantics='Offline labels are -1 on invalid queries; runtime retains the last valid state. Equality checked on every valid query; before first valid query runtime also returns -1.',
                invalid_history_no_class_verified=True,checkpoint_binding_verified=True,
                stateless_ms_per8192worlds=ms,
                performance_note='Shared GPU6 with external jobs; this is metric routing time only, excluding encoder/simulation.',
                no_ground_truth_parameters_used_at_inference=True)
    dump(OUT/'runtime_verification.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
