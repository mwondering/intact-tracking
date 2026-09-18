"""Independent raw-history GPU replay and saved-center/statistics checks."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from intact_tracking.memory350_inference import load_memory350_checkpoint

ROOT = Path('runs/limb_context_20260916_dr16384')
OUT = ROOT/'analysis'


@torch.inference_mode()
def raw_audit():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    cp = load_memory350_checkpoint('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt', device='cuda:0')
    stats, replay_worlds, replay_fp32, replay_bf16 = [], [], [], []
    for shard in range(8):
        path = ROOT/f'shard_{shard:02d}'
        metadata = json.loads((path/'metadata.json').read_text())
        while not metadata['complete']:
            time.sleep(5)
            metadata = json.loads((path/'metadata.json').read_text())
        info = next(v for v in metadata['queries'] if v['step'] == 2400)
        with open(info['path'], 'rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == info['sha256']
        q = torch.load(info['path'], map_location='cpu', mmap=True, weights_only=False)
        cache = dict(np.load(path/'latents.npz'))
        eligible = np.flatnonzero(q['short_valid'].all(-1).numpy() & q['long_valid'].all(-1).numpy())
        chosen = np.random.default_rng(6931+shard).choice(eligible, 256, replace=False)
        batch = {key: q[key][chosen].to('cuda:0') for key in ('short', 'short_valid', 'long', 'long_valid')}
        for value in batch.values():
            assert torch.isfinite(value).all()
        assert torch.equal(batch['short'][:, :-1, 100:], batch['short'][:, 1:, :71])
        assert torch.equal(batch['long'][:, :, :-1, 100:], batch['long'][:, :, 1:, :71])
        lookup = {int(world): i for i, world in enumerate(cache['world']) if cache['step'][i] == 2400}
        rows = np.array([lookup[int(w)] for w in q['world'][chosen]])
        saved = cache['latent'][rows].astype(np.float64)
        saved /= np.linalg.norm(saved, axis=1, keepdims=True)
        def normalized(v):
            return torch.cat(((v[..., :71]-cp.state_mean)/cp.state_std,
                              (v[..., 71:100]-cp.action_mean)/cp.action_std,
                              (v[..., 100:]-cp.state_mean)/cp.state_std), dim=-1)
        short, long = normalized(batch['short']), normalized(batch['long'])
        record = dict(shard=shard, query_step=2400, windows=len(chosen), raw_sha_verified=True,
                      finite=True, exact_internal_transition_continuity=True)
        for precision in ('fp32', 'bf16'):
            with torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext():
                z = cp.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                               batch['short_valid'], long, batch['long_valid']).float()
            z = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
            diff = np.linalg.norm(z-saved, axis=1)
            if precision == 'bf16':
                assert diff.max() < 1e-5, (shard, precision, diff.max())
                replay_bf16.append(z)
            else:
                replay_fp32.append(z)
            record[precision+'_unit_latent_l2_mean'] = float(diff.mean())
            record[precision+'_unit_latent_l2_max'] = float(diff.max())
        stats.append(record)
        replay_worlds.append(q['world'][chosen].numpy())
        print(json.dumps(record), flush=True)
    (OUT/'raw_history_verification.json').write_text(json.dumps(stats, indent=2)+'\n')
    np.savez_compressed(OUT/'precision_replay.npz', world=np.concatenate(replay_worlds),
                        fp32=np.concatenate(replay_fp32), bf16=np.concatenate(replay_bf16), step=2400)


def models_audit():
    a = np.load(OUT/'selected_worlds.npz')
    p = a['physics'].astype(float)
    nominal = np.zeros(38); nominal[4]=.6; nominal[5:34]=1.
    residual = (p-nominal)/a['span']
    # Independently reconstruct ten equally weighted factors, including the
    # mean of 29 armature dimensions, instead of reusing coordinate_weights.
    factor_square = np.c_[residual[:, :5]**2, (residual[:, 5:34]**2).mean(1), residual[:, 34:]**2]
    expected = np.sqrt(factor_square.mean(1))
    np.testing.assert_allclose(expected, a['radius'], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(a['z'], axis=-1), 1, atol=2e-7)
    assert len(np.unique(a['world'])) == 18432
    results = json.loads((OUT/'results.json').read_text())
    assert len(results) == 18
    verified, precision_reports = [], []
    replay = np.load(OUT/'precision_replay.npz')
    world_lookup = {int(w): i for i, w in enumerate(a['world'])}
    replay_world = np.array([world_lookup[int(w)] for w in replay['world']])
    rng = np.random.default_rng(123419)
    for r in results:
        model = np.load(OUT/'models'/f"{r['key']}.npz")
        assert not set(model['train_dr']) & set(model['test_dr'])
        assert not set(model['train_nom']) & set(model['test_nom'])
        assert len(model['train_dr']) == r['fit_dr_worlds']
        value = {'true_dr': a['x'][:,None,:], 'world_mean_latent': a['zmean'][:,None,:], 'window_latent': a['z']}[r['representation']]
        ids = rng.choice(len(model['test_dr']), 256, replace=False)
        data = value[model['test_dr'][ids]]
        exact = ((data[:,:,None,:]-model['centers'][None,None,:,:])**2).sum(-1).argmin(-1)
        assert np.array_equal(exact, model['dr_labels'][ids])
        labels = model['dr_labels']
        nominal_class = r['nominal']['nominal_class']
        heavy = p[model['test_dr'],34:38].sum(1) >= 8
        rate = (labels[heavy] == nominal_class).mean()
        assert rate == r['nominal']['modal']['heavy_window_routed_fraction']
        assert r['cluster_pair_quality']['anchor_coverage'] > .999
        if r['representation'] == 'window_latent' and r['fit_dr_worlds'] == 8192:
            heldout = np.isin(replay_world, np.r_[model['test_dr'], model['test_nom']])
            centers = model['centers']
            lf = ((replay['fp32'][heldout,None,:]-centers[None,:,:])**2).sum(-1).argmin(-1)
            lb = ((replay['bf16'][heldout,None,:]-centers[None,:,:])**2).sum(-1).argmin(-1)
            precision_reports.append(dict(model=r['key'], windows=int(heldout.sum()),
                                          class_change_fraction=float((lf!=lb).mean()),
                                          nominal_membership_change_fraction=float(((lf==nominal_class)!=(lb==nominal_class)).mean())))
        verified.append(r['key'])
    report = dict(passed=True, independent_ten_factor_metric_matches=True,
                  saved_labels_match_explicit_nearest_center=True,
                  heldout_world_splits_disjoint=True, heavy_routing_rates_recomputed=True,
                  models=verified, precision_class_sensitivity=precision_reports)
    (OUT/'model_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    raw_audit() if args.raw else models_audit()
