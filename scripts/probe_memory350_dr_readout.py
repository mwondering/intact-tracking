"""Read DR coordinates from a frozen encoder on audited, cached raw histories."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from intact_tracking.memory350_inference import load_memory350_checkpoint
from probe_memory350_world_partitions import split


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unit(x):
    x = x.astype(np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@torch.inference_mode()
def encode(checkpoint, query, batch_size=128):
    output = []
    for start in range(0, len(query['world']), batch_size):
        def normalize(raw):
            raw = raw[start:start + batch_size]
            return torch.cat(((raw[..., :71] - checkpoint.state_mean) / checkpoint.state_std,
                              (raw[..., 71:100] - checkpoint.action_mean) / checkpoint.action_std,
                              (raw[..., 100:] - checkpoint.state_mean) / checkpoint.state_std), -1)

        short, long = normalize(query['short']), normalize(query['long'])
        output.append(checkpoint.encoder(
            short[..., :71], short[..., 71:100], short[..., 100:],
            query['short_valid'][start:start + batch_size], long,
            query['long_valid'][start:start + batch_size]).numpy())
    return np.concatenate(output)


def metrics(truth, prediction, names):
    error = prediction - truth
    r2 = 1 - np.square(error).sum(0) / np.square(truth - truth.mean(0)).sum(0)
    rows = [{'name': name, 'r2': float(r2[i]),
             'mae': float(np.abs(error[:, i]).mean()),
             'rmse': float(np.sqrt(np.square(error[:, i]).mean())),
             'absolute_error_p95': float(np.quantile(np.abs(error[:, i]), .95))}
            for i, name in enumerate(names)]
    groups = {}
    for family in sorted(set(name.split('/')[0] for name in names)):
        values = [r['r2'] for r in rows if r['name'].startswith(family + '/')]
        groups[family] = {'dimensions': len(values), 'mean_r2': float(np.mean(values)),
                          'min_r2': min(values), 'max_r2': max(values)}
    return {'per_parameter': rows, 'groups': groups}


def readout(z, data, train, test, ids, physics, names, fold):
    prototypes = np.stack([z[train & (data['world'] == w)].mean(0) for w in ids])
    lookup = np.searchsorted(ids, data['world'][test])
    truth = physics[data['world'][test]]
    prediction, shuffled, mean_prediction = [np.empty_like(truth) for _ in range(3)]
    for held in range(5):
        fitting = fold != held
        selected = fold[lookup] == held
        x, y = prototypes[fitting], physics[ids[fitting]]
        xm, xs, ym = x.mean(0), x.std(0).clip(1e-6), y.mean(0)
        x = (x - xm) / xs
        q = (z[test][selected] - xm) / xs
        gram = x.T @ x / len(x) + .1 * np.eye(x.shape[1])
        weight = np.linalg.solve(gram, x.T @ (y - ym) / len(x))
        prediction[selected] = q @ weight + ym
        shuffled_y = y[np.random.default_rng(20260916 + held).permutation(len(y))]
        shuffled[selected] = q @ np.linalg.solve(gram, x.T @ (shuffled_y - ym) / len(x)) + ym
        mean_prediction[selected] = ym
    # Query weighting matches the previous diagnostic; also report equal world weighting.
    squared = np.square(prediction - truth)
    world_mse = np.stack([squared[data['world'][test] == w].mean(0) for w in ids])
    world_r2 = 1 - world_mse.mean(0) / physics[ids].var(0)
    result = metrics(truth, prediction, names)
    result['world_balanced_r2'] = world_r2.tolist()
    result['train_mean_baseline'] = metrics(truth, mean_prediction, names)
    result['shuffled_world_label_control'] = metrics(truth, shuffled, names)
    return result, prediction, shuffled, prototypes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((source / 'metadata.json').read_text())
    with np.load(source / 'latents.npz', allow_pickle=False) as saved:
        all_data = {k: saved[k] for k in saved.files}
    physics_path = Path(meta['source_physics'])
    assert digest(physics_path) == meta['source_physics_sha256']
    with np.load(physics_path, allow_pickle=False) as saved:
        physics, names = saved['values'].astype(np.float64), saved['names'].tolist()
    checkpoint = load_memory350_checkpoint(args.checkpoint, device='cpu')
    latent = np.full((len(all_data['world']), 64), np.nan, np.float32)
    query_records = []
    for i, info in enumerate(meta['queries']):
        path = Path(info['path'])
        assert digest(path) == info['sha256']
        query = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        selected = all_data['step'] == query['step']
        assert np.array_equal(all_data['world'][selected], query['world'].numpy())
        assert np.array_equal(all_data['short_steps'][selected], query['short_valid'].sum(-1).numpy())
        assert np.array_equal(all_data['long_chunks'][selected], query['long_valid'].sum(-1).numpy())
        latent[selected] = encode(checkpoint, query)
        query_records.append(info)
        if (i + 1) % 8 == 0:
            print(json.dumps({'encoded_queries': i + 1, 'seconds': time.monotonic() - started}), flush=True)
    assert np.isfinite(latent).all()
    full = (~all_data['nominal']) & (all_data['short_steps'] == 50) & (all_data['long_chunks'] == 30)
    data = {k: v[full] for k, v in all_data.items()}
    train, test, ids = split(data, meta)
    fold = np.empty(len(ids), int)
    for index, worlds in enumerate(np.array_split(np.random.default_rng(20260916).permutation(ids), 5)):
        fold[np.isin(ids, worlds)] = index
    with np.load(source.parent / 'payload_group_linear_probe_predictions.npz', allow_pickle=False) as old_saved:
        assert np.array_equal(data['source_row'][test], old_saved['source_row'])
        assert np.array_equal(fold[np.searchsorted(ids, data['world'][test])], old_saved['world_fold'])
    z = unit(latent[full])
    result, predicted, shuffled, prototypes = readout(z, data, train, test, ids, physics, names, fold)
    old, old_predicted, _, _ = readout(unit(data[meta['models']['memory350']]), data, train, test,
                                     ids, physics, names, fold)
    old_report = json.loads((source.parent / 'payload_group_linear_probe.json').read_text())
    payload = [i for i, name in enumerate(names) if name.startswith('context_uniform_limb_payload/')]
    assert len(ids) == old_report['worlds'] and test.sum() == old_report['queries']
    assert np.allclose([old['per_parameter'][i]['r2'] for i in payload], old_report['per_limb_r2'], atol=1e-10, rtol=0)
    query_worlds = data['world'][test]
    truth = physics[query_worlds]
    report = {
        'checkpoint': {'path': checkpoint.path, 'sha256': checkpoint.sha256},
        'source': str(source), 'source_hashes': {name: digest(source / name) for name in ('metadata.json', 'latents.npz')},
        'physics': {'path': str(physics_path), 'sha256': digest(physics_path), 'names': names},
        'script_sha256': digest(Path(__file__)), 'raw_queries': query_records,
        'worlds': len(ids), 'queries': len(query_worlds), 'precision': 'CPU float32 encoder; float64 probe',
        'protocol': 'Five world folds, seed 20260916; array_split of permuted sorted world IDs, '
                    'verified against the previous saved folds. Fit only reference-motion world-mean unit latents '
                    'of other worlds; predict individual disjoint-history, other-family queries of held-out worlds. '
                    'Feature standardization on fit worlds only. Affine ridge, mean squared loss + 0.1 squared '
                    'weight penalty; intercept unpenalized, fixed alpha, no clipping or hyperparameter selection. '
                    'Shuffled-label control permutes complete physics vectors across fit worlds in each fold.',
        'scope': 'Current frozen encoder on saved LaFAN frozen-tracker trajectories, full tracker background '
                 'DR plus capped payload and complete Memory350 history. This does not test current load-only '
                 'PPO trajectories, cold memory, closed-loop routing, all unseen motions or encoder bias '
                 '(not present among these 38 saved physics coordinates). Poor linear readout does not '
                 'prove information is mathematically absent. CPU FP32 differs from runtime CUDA BF16.',
        'current': result, 'cached_u30750': old,
        'verification': {'cached_u30750_payload_scores_reproduced': True,
                         'all_raw_query_and_physics_hashes_verified': True},
        'seconds': time.monotonic() - started,
    }
    np.savez_compressed(output / 'predictions.npz', names=np.array(names), world_ids=ids, world_fold=fold,
                        query_worlds=query_worlds, query_source_rows=data['source_row'][test], truth=truth,
                        predicted=predicted, shuffled_prediction=shuffled, cached_prediction=old_predicted,
                        all_source_rows=all_data['source_row'], latent_current=latent,
                        current_world_prototypes=prototypes)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'groups': result['groups'],
                      'payload': [result['per_parameter'][i] for i in payload],
                      'seconds': report['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
