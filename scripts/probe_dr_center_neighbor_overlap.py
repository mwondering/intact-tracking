"""Diagnose known-world readout margins and DR-distance targets on cached histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_forward_context_clusters import normalized
from evaluate_memory350_weak_pairs import write_json


def stats(x):
    x = np.asarray(x)
    return {'n': int(x.size), 'mean': float(x.mean()),
            'rms': float(np.sqrt(np.mean(x ** 2))),
            'p10': float(np.quantile(x, .1)), 'p50': float(np.median(x)),
            'p90': float(np.quantile(x, .9))}


def split(data, metadata):
    families = np.array([Path(p).name.split('_subject')[0] for p in metadata['motion_files']])
    family = families[data['motion']]
    order = np.random.default_rng(93011).permutation(np.unique(family))
    train_family = np.isin(family, order[:len(order) // 2])
    train = train_family & (data['step'] <= 1600)
    test = ~train_family & (data['step'] > 1600)
    world = data['world']
    eligible = np.zeros(len(world), bool)
    for key in np.unique(world[train]):
        refs = train & (world == key)
        end = data['total_chunks'][refs] + (data['short_steps'][refs] + data['pending_steps'][refs] + 9) // 10
        sessions = np.unique(data['memory_session'][refs])
        assert len(sessions) == 1
        eligible |= ((world == key) & (data['total_chunks'] - data['long_chunks'] >= end.max())
                     & (data['step'] >= data['step'][refs].max() + 50)
                     & (data['memory_session'] == sessions[0]))
    test &= eligible
    ids = np.array(sorted(set(world[train]).intersection(world[test])))
    test &= np.isin(world, ids)
    return train, test, ids, np.searchsorted(ids, world[test])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check_dir', type=Path)
    args = parser.parse_args()
    folder = args.check_dir.resolve()
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['complete']
    state = torch.load(summary['checkpoints']['memory350']['path'], map_location='cpu', weights_only=False, mmap=True)
    schema = state['dr_metric_schema']
    result = {'update': summary['candidate_update'], 'profiles': {},
              'mapping_note': 'Alternative targets only; no retraining or predicted accuracy gain.',
              'radius_note': 'Query radius versus nearest-center separation is descriptive, not a proof of overlap; exact decision margins are also reported.'}
    for profile in ('capped_loads', 'common'):
        source = folder / profile
        meta = json.loads((source / 'metadata.json').read_text())
        with np.load(source / 'latents.npz') as saved:
            mask = (~saved['nominal']) & (saved['short_steps'] == 50) & (saved['long_chunks'] == 30)
            data = {key: saved[key][mask] for key in saved.files}
        train, test, ids, truth = split(data, meta)
        expected = summary['profiles'][profile]['readout']
        assert len(ids) == expected['worlds'] and int(test.sum()) == expected['test_samples']
        with np.load(meta['source_physics']) as physics:
            names = physics['names'].tolist()
            values = physics['values']
        columns = []
        for name in schema['names']:
            if name in names:
                columns.append(values[ids, names.index(name)])
            else:
                assert '/added_mass_kg/' in name
                columns.append(np.zeros(len(ids)))
        raw = np.column_stack(columns)
        unit = (raw - schema['lower']) / (np.array(schema['upper']) - schema['lower'])
        assert unit.min() >= -1e-5 and unit.max() <= 1 + 1e-5
        dr = unit * np.sqrt(schema['coordinate_weights'])
        dr_distance = np.linalg.norm(dr[:, None] - dr[None], axis=-1)
        a, b = np.triu_indices(len(ids), 1)
        dr_pairs = dr_distance[a, b]
        dr_nn = dr_distance.copy()
        np.fill_diagonal(dr_nn, np.inf)
        nearest_dr = dr_nn.min(1)
        physical_rank = np.argsort(np.argsort(dr_nn, axis=1), axis=1) + 1
        models = {}
        for name in ('baseline', 'memory350'):
            z = normalized(data[meta['models'][name]].astype(np.float64))
            centers = np.stack([z[train & (data['world'] == key)].mean(0) for key in ids])
            q = z[test]
            distance = np.sqrt(np.maximum(0, np.sum(q*q, 1)[:, None] + np.sum(centers*centers, 1)[None] - 2*q@centers.T))
            pred = distance.argmin(1)
            correct = pred == truth
            assert float(correct.mean()) == expected['models'][name]['top1_accuracy']
            row = np.arange(len(truth))
            own = distance[row, truth].copy()
            distance[row, truth] = np.inf
            competing = distance.min(1)
            separation = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
            pair_distance = separation[a, b]
            np.fill_diagonal(separation, np.inf)
            erroneous_dr = dr_distance[truth[~correct], pred[~correct]]
            erroneous_rank = physical_rank[truth[~correct], pred[~correct]]
            models[name] = {
                'update': summary['checkpoints'][name]['update'], 'top1': float(correct.mean()),
                'query_to_correct_center': stats(own), 'nearest_center_separation': stats(separation.min(1)),
                'query_margin_wrong_minus_correct': stats(competing-own),
                'errors': int((~correct).sum()), 'wrong_prediction_dr_distance': stats(erroneous_dr),
                'wrong_prediction_dr_neighbor_rank': stats(erroneous_rank),
                'errors_to_nearest_5_dr_fraction': float((erroneous_rank <= 5).mean()),
                'errors_to_nearest_10_dr_fraction': float((erroneous_rank <= 10).mean()),
                'errors_to_nearest_20_dr_fraction': float((erroneous_rank <= 20).mean()),
                'center_pair_distance': stats(pair_distance),
                'center_pair_target_rmse': float(np.sqrt(np.mean((pair_distance-2*dr_pairs/(dr_pairs+.3))**2))),
            }
        maps = {str(scale): {'all_pair_targets': stats(2*dr_pairs/(dr_pairs+scale)),
                             'nearest_physical_neighbor_targets': stats(2*nearest_dr/(nearest_dr+scale))}
                for scale in (.3, .25, .2, .1)}
        result['profiles'][profile] = {'worlds': len(ids), 'queries': len(truth),
                                        'dr_pair_distance': stats(dr_pairs), 'nearest_dr_distance': stats(nearest_dr),
                                        'mapping': maps, 'models': models}
    write_json(folder / 'neighbor_overlap.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
