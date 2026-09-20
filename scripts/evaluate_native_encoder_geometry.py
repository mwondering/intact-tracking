"""Audit native-DR encoders on identical cached held-out histories, using CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import torch

from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_nominal_dr_rank import rank_view_valid

FIELDS = ('history_state', 'history_action', 'history_next_state', 'history_valid',
          'memory_interactions', 'memory_valid')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


@torch.inference_mode()
def encode(model, batch, prefix='', mask=None):
    values = [batch[prefix + k] if mask is None else batch[prefix + k][mask] for k in FIELDS]
    parts = [model.encoder(*(v[start:start + 128] for v in values)).float().numpy()
             for start in range(0, len(values[0]), 128)]
    return unit(np.concatenate(parts).astype(np.float64))


def geometry(data, anchor, mask, scale):
    nominal = mask & data['nominal']
    dr = mask & ~data['nominal'] & data['valid']
    radius = np.linalg.norm(data['z'] - anchor, axis=1)
    target = 2 * data['response'] / (data['response'] + scale)
    center = data['z'][nominal].mean(0)
    within = np.square(data['z'][nominal] - center).sum(1).mean()
    center_distance = np.linalg.norm(center - anchor)
    rms = np.sqrt(np.square(radius[nominal]).mean())
    assert np.isclose(rms ** 2, within + center_distance ** 2)
    error = np.abs(radius[dr] - target[dr])
    return {
        'nominal_samples': int(nominal.sum()), 'nominal_worlds': len(np.unique(data['world'][nominal])),
        'dr_samples': int(dr.sum()), 'dr_worlds': len(np.unique(data['world'][dr])),
        'nominal_anchor_rms': float(rms), 'nominal_center_anchor_distance': float(center_distance),
        'nominal_within_rms': float(np.sqrt(within)),
        'nominal_radius_p50': float(np.quantile(radius[nominal], .5)),
        'nominal_radius_p95': float(np.quantile(radius[nominal], .95)),
        'nominal_fraction_over_0_5': float((radius[nominal] > .5).mean()),
        'ab_target_mae': float(error.mean()), 'ab_target_p95_error': float(np.quantile(error, .95)),
        'ab_response_radius_pearson': float(np.corrcoef(radius[dr], data['response'][dr])[0, 1]),
        'ab_response_radius_spearman': float(spearmanr(radius[dr], data['response'][dr]).statistic),
        'ab_target_radius_pearson': float(np.corrcoef(radius[dr], target[dr])[0, 1]),
        'dr_radius_mean': float(radius[dr].mean()), 'ab_target_mean': float(target[dr].mean()),
    }


def pair_geometry(data):
    identities = np.column_stack((data['world'], data['session']))
    ids, first, inverse = np.unique(identities, axis=0, return_index=True, return_inverse=True)
    counts = np.bincount(inverse)

    def pool(x):
        out = np.zeros((len(ids), x.shape[1]))
        np.add.at(out, inverse, x)
        return out / counts[:, None]

    centers = pool((data['z'] + data['other']) / 2)
    theta = pool(data['parameters'])
    # This is descriptive geometry; pairwise distances are not independent observations.
    physical, latent = pdist(theta), pdist(centers)
    residual = (np.square(data['z'] - centers[inverse]).sum(1)
                + np.square(data['other'] - centers[inverse]).sum(1)) / 2
    within_by_world = np.bincount(inverse, weights=residual) / counts
    within = np.sqrt(within_by_world.mean())
    same = np.linalg.norm(data['z'] - data['other'], axis=1)
    # One deterministic pair per world prevents mixing archived/query windows when
    # constructing galleries. Each saved pair was validated as history-disjoint.
    query, gallery = data['z'][first], data['other'][first]
    scores = query @ gallery.T
    ranks = 1 + (scores > np.diag(scores)[:, None]).sum(1)
    centered = centers - centers.mean(0)
    eig = np.linalg.eigvalsh(centered.T @ centered / len(centers)).clip(0)
    result = {
        'pairs': len(data['z']), 'world_centers': len(ids),
        'parameter_center_distance_spearman': float(spearmanr(physical, latent).statistic),
        'parameter_center_distance_pearson': float(np.corrcoef(physical, latent)[0, 1]),
        'between_center_distance_mean': float(latent.mean()),
        'between_center_distance_rms': float(np.sqrt(np.square(latent).mean())),
        'within_center_rms_equal_world_weight': float(within),
        'within_over_between_rms': float(within / np.sqrt(np.square(latent).mean())),
        'same_world_cross_motion_distance_rms': float(np.sqrt(np.square(same).mean())),
        'same_world_cross_motion_cosine': float((data['z'] * data['other']).sum(1).mean()),
        'center_effective_rank': float(eig.sum() ** 2 / np.square(eig).sum()),
        'retrieval_worlds': len(first), 'retrieval_top1': float((ranks <= 1).mean()),
        'retrieval_top5': float((ranks <= 5).mean()), 'retrieval_median_rank': float(np.median(ranks)),
        'retrieval_chance_top1': 1 / len(first), 'retrieval_chance_top5': 5 / len(first),
        'minimum_archive_age_steps': int(data['age'].min()),
    }
    return result, first, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--updates', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = json.loads((args.run / 'run_config.json').read_text())
    ranks = range(config['distributed']['world_size'])
    paths = {str(u): args.run / f'update_{u:06d}.pt' for u in args.updates}
    states = {u: torch.load(p, map_location='cpu', weights_only=False, mmap=True) for u, p in paths.items()}
    reference = states[str(args.updates[0])]
    for s in states.values():
        for k in ('tracker', 'normalization', 'nominal_direction_anchor', 'model_config', 'dr_metric_schema', 'loss_config'):
            assert s[k] == reference[k], k
        assert s.get('context_input_contract') == reference.get('context_input_contract')
        assert s['native_dr_version'] == 1
        assert s['distributed_parameter_agreement']['passed']
        assert len(set(s['distributed_parameter_agreement']['sha256_by_rank'])) == 1
    models = {u: load_memory350_checkpoint(p, device='cpu') for u, p in paths.items()}
    anchor = np.array(reference['nominal_direction_anchor']['direction'])
    broad, pairs = {u: [] for u in paths}, {u: [] for u in paths}
    report = {'complete': False, 'precision': 'CPU float32', 'same_queries': True,
              'architecture_version': reference['architecture_version'],
              'context_input_contract': reference.get('context_input_contract'),
              'checkpoints': {u: {'path': str(p.resolve()), 'sha256': models[u].sha256} for u, p in paths.items()},
              'validation_sha256': {}, 'models': {}, 'anchor': anchor.tolist(),
              'sampling': config['native_dr_contract']['sampling'],
              'retrieval_contract': 'One archived gallery and one current query per held-out DR world; same-world histories full, nonoverlapping, different motion IDs. Known gallery identities, not new-world classification. Different motion families are not required.',
              'broad_contract': 'Fixed early held-out histories; samples within a world can overlap. Report full350 and all usable separately.',
              'history_overlap_audit': {'selected_pairs': 0, 'shared_exact_interaction_rows': 0}}
    save(args.output / 'summary.json', report)
    for rank in ranks:
        for kind, filename in [('broad', f'validation_broad_rank_{rank}.pt'), ('pairs', f'validation_rank_{rank}.pt')]:
            p = args.run / filename
            b = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
            report['validation_sha256'][filename] = sha(p)
            for k in ('state_mean', 'state_std', 'action_mean', 'action_std'):
                # These saved fields belong to the privileged predictor/AB-label
                # branch. Proprio122 encoders have separate context_* statistics.
                expected = torch.as_tensor(reference['normalization'][k], dtype=b[k].dtype)
                torch.testing.assert_close(b[k], expected, atol=0, rtol=0)
            state_width = models[str(args.updates[0])].state_mean.numel()
            assert b['history_state'].shape[-1] == b['history_next_state'].shape[-1] == state_width
            assert b['history_action'].shape[-1] == 29
            assert b['memory_interactions'].shape[-1] == 2 * state_width + 29
            if kind == 'broad':
                common = {'world': b['world_id'].numpy(), 'nominal': b['is_nominal'].numpy(),
                          'full': (b['history_valid'].all(1) & b['memory_valid'].all(1)).numpy(),
                          'usable': (b['history_valid'].any(1) | b['memory_valid'].any(1)).numpy(),
                          'valid': b['label_response_valid'].numpy(), 'motion': b['motion_id'].numpy(),
                          'response': b['label_response'].square().mean((1, 2)).sqrt().numpy(),
                          'rank': np.full(len(b['world_id']), rank)}
                for u, model in models.items():
                    broad[u].append({**common, 'z': encode(model, b)})
            else:
                valid = rank_view_valid(b)
                common = {'world': b['world_id'][valid].numpy(), 'session': b['physics_session'][valid].numpy(),
                          'parameters': b['dr_metric'][valid].numpy(), 'motion': b['motion_id'][valid].numpy(),
                          'other_motion': b['weak_motion_id'][valid].numpy(), 'age': b['weak_age_steps'][valid].numpy()}
                assert (common['motion'] != common['other_motion']).all()
                _, first = np.unique(common['world'], return_index=True)
                source_rows = torch.where(valid)[0][torch.tensor(first)]
                histories = []
                for prefix in ('', 'weak_'):
                    short = torch.cat([b[prefix + k][source_rows] for k in FIELDS[:3]], -1)
                    long = b[prefix + 'memory_interactions'][source_rows].flatten(1, 2)
                    histories.append(torch.cat((short, long), 1).contiguous().numpy())
                for current, old in zip(*histories, strict=True):
                    shared = len({r.tobytes() for r in current} & {r.tobytes() for r in old})
                    report['history_overlap_audit']['selected_pairs'] += 1
                    report['history_overlap_audit']['shared_exact_interaction_rows'] += shared
                assert report['history_overlap_audit']['shared_exact_interaction_rows'] == 0
                for u, model in models.items():
                    pairs[u].append({**common, 'z': encode(model, b, mask=valid),
                                    'other': encode(model, b, 'weak_', valid)})
        print(json.dumps({'rank_encoded': rank, 'seconds': time.monotonic() - started}), flush=True)
    for u in paths:
        b = {k: np.concatenate([x[k] for x in broad[u]]) for k in broad[u][0]}
        p = {k: np.concatenate([x[k] for x in pairs[u]]) for k in pairs[u][0]}
        assert np.isfinite(b['z']).all() and np.isfinite(p['z']).all() and np.isfinite(p['other']).all()
        result = {key: geometry(b, anchor, b[mask], states[u]['loss_config']['response_distance_scale'])
                  for key, mask in [('full350', 'full'), ('all_usable', 'usable')]}
        result['dr_geometry'], first, retrieval_rank = pair_geometry(p)
        np.savez_compressed(args.output / f'u{u}_broad.npz', **b)
        np.savez_compressed(args.output / f'u{u}_pairs.npz', **p, retrieval_rows=first, retrieval_rank=retrieval_rank)
        report['models'][u] = result
        print(json.dumps({'update': u, 'result': result}), flush=True)
    report.update(complete=True, elapsed_seconds=time.monotonic() - started)
    save(args.output / 'summary.json', report)


if __name__ == '__main__':
    main()
