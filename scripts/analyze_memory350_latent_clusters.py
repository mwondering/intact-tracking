"""Paired latent geometry on common trajectories, with memory-overlap controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_forward_context_clusters import (
    describe, grouping, normalized, pair_stats, physics_prototype_probe,
    select_pairs, crossvalidated_group_r2,
)


MODELS = {'memory350': 'latent', 'v12': 'latent_v12', 'memory350_without_long': 'latent_no_long'}
PAIR_NAMES = (
    'nominal_cross_motion', 'nominal_same_motion_far_phase',
    'nominal_dr_same_motion_near_phase', 'dr_same_world_cross_motion',
    'dr_different_world_same_motion_near_phase', 'dr_different_world_cross_motion',
)


def disjoint_pairs(data, seed=4911):
    """Exclude even partially reused raw interactions in the positive pair.

    At query i, pending+short can eventually occupy at most ceil((pending+short)/10)
    new chunks. At the later query, requiring its oldest chunk to start beyond
    those sequence IDs excludes every interaction visible at the earlier query.
    Physics sessions are fixed and chunk counters increase through episode resets.
    """
    by_world = grouping(data['world'])
    rng = np.random.default_rng(seed)
    left, right = [], []
    for i in rng.permutation(np.flatnonzero(~data['nominal'])):
        ids = by_world[data['world'][i]]
        ids = ids[data['motion'][ids] != data['motion'][i]]
        early = np.where(data['step'][ids] < data['step'][i], ids, i)
        late = np.where(data['step'][ids] < data['step'][i], i, ids)
        reserved = (data['short_steps'][early] + data['pending_steps'][early] + 9) // 10
        keep = ((data['total_chunks'][late] - data['long_chunks'][late] >=
                 data['total_chunks'][early] + reserved) &
                (data['step'][late] - data['step'][early] >= 50) &
                (data['memory_session'][early] == data['memory_session'][late]))
        if keep.any():
            left.append(i)
            right.append(rng.choice(ids[keep]))
    return np.array(left, dtype=int), np.array(right, dtype=int)


def exact_pairs(data):
    groups = defaultdict(list)
    for i, key in enumerate(zip(data['motion'], data['motion_step'])):
        groups[key].append(i)
    left, right = [], []
    for values in groups.values():
        if len(values) < 2:
            continue
        ids = np.asarray(values)
        a, b = np.triu_indices(len(ids), 1)
        a, b = ids[a], ids[b]
        keep = data['world'][a] != data['world'][b]
        left.extend(a[keep]); right.extend(b[keep])
    a, b = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    return {
        'dr_different_world_exact_motion_frame': (a[~data['nominal'][a] & ~data['nominal'][b]],
                                                 b[~data['nominal'][a] & ~data['nominal'][b]]),
        'nominal_dr_exact_motion_frame': (a[data['nominal'][a] != data['nominal'][b]],
                                         b[data['nominal'][a] != data['nominal'][b]]),
        'nominal_nominal_exact_motion_frame': (a[data['nominal'][a] & data['nominal'][b]],
                                              b[data['nominal'][a] & data['nominal'][b]]),
    }


def analyze_subset(data, metadata, output, subset):
    pairs = {name: select_pairs(data, name, np.random.default_rng(192608 + i))
             for i, name in enumerate(PAIR_NAMES)
             if not name.startswith('nominal') or data['nominal'].any()}
    pairs['dr_same_world_cross_motion_disjoint'] = disjoint_pairs(data)
    pairs.update(exact_pairs(data))
    family_names = np.array([Path(p).name.split('_subject')[0] for p in metadata['motion_files']])
    families = family_names[data['motion']]
    result = {
        'n': len(data['world']), 'nominal_samples': int(data['nominal'].sum()),
        'dr_samples': int((~data['nominal']).sum()),
        'nominal_worlds': len(np.unique(data['world'][data['nominal']])),
        'dr_worlds': len(np.unique(data['world'][~data['nominal']])),
        'motion_files': len(np.unique(data['motion'])), 'models': {},
    }
    np.savez_compressed(output / f'{subset}_pairs.npz', **{
        f'{name}_{side}': ids for name, pair in pairs.items() for side, ids in zip(('left', 'right'), pair)
    }, source_row=data['source_row'])
    for model_name, field in metadata.get('models', MODELS).items():
        z = data[field].astype(np.float64)
        model_result = {'dr': describe(z[~data['nominal']]), 'pairs': {}}
        if data['nominal'].any():
            model_result['nominal'] = describe(z[data['nominal']])
            model_result['nominal_motion_dependence'] = crossvalidated_group_r2(
                normalized(z[data['nominal']]), data['motion'][data['nominal']],
                data['world'][data['nominal']], np.random.default_rng(321))
        for i, (name, (left, right)) in enumerate(pairs.items()):
            metrics, distances = pair_stats(z, left, right, data['world'], np.random.default_rng(7300 + i))
            if len(left):
                metrics.update(unit_distance_mean=float(distances.mean()),
                               unit_distance_p05=float(np.quantile(distances, .05)),
                               raw_distance_mean=float(np.linalg.norm(z[left] - z[right], axis=1).mean()),
                               phase_gap_mean=float(np.abs(data['phase'][left] - data['phase'][right]).mean()),
                               phase_gap_max=float(np.abs(data['phase'][left] - data['phase'][right]).max()))
            model_result['pairs'][name] = metrics
        for key in ('dr_same_world_cross_motion', 'dr_same_world_cross_motion_disjoint'):
            within = model_result['pairs'][key]
            between = model_result['pairs']['dr_different_world_same_motion_near_phase']
            model_result[key + '_over_between'] = within.get('unit_distance_rms', float('nan')) / between['unit_distance_rms']
        dr = ~data['nominal']
        model_result['dr_world_center_transfer'] = physics_prototype_probe(
            normalized(z[dr]), data['world'][dr], families[dr], np.random.default_rng(93011))
        result['models'][model_name] = model_result
        print(json.dumps({'subset': subset, 'model': model_name,
                          'within': model_result['pairs']['dr_same_world_cross_motion'].get('unit_distance_rms'),
                          'within_disjoint': model_result['pairs']['dr_same_world_cross_motion_disjoint'].get('unit_distance_rms'),
                          'between_matched': model_result['pairs']['dr_different_world_same_motion_near_phase']['unit_distance_rms'],
                          'within_over_between': model_result['dr_same_world_cross_motion_over_between'],
                          'center_transfer_top1': model_result['dr_world_center_transfer']['top1_accuracy']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.input / 'metadata.json').read_text())
    if not metadata.get('complete'):
        raise ValueError('Collection has not finished')
    with np.load(args.input / 'latents.npz') as f:
        all_data = {k: f[k] for k in f.files}
    all_data['source_row'] = np.arange(len(all_data['world']))
    common = (all_data['short_steps'] == 50) & (all_data['long_chunks'] == 30) & (all_data['v12_steps'] == 100)
    usable = (all_data['long_chunks'] == 30)
    memory_full = usable & (all_data['short_steps'] == 50)
    output = args.input / 'analysis'
    output.mkdir(exist_ok=True)
    result = {
        'metadata': metadata, 'main_queries': len(common),
        'common_full_queries': int(common.sum()), 'memory_full_queries': int(memory_full.sum()),
        'long_full_queries': int(usable.sum()),
        'primary_subset': 'common_full',
        'definitions': {'common_full': 'Identical query rows: v12 full 100 steps, Memory350 full short50 and long30x10.',
                        'memory_full': 'Full short50 and long30x10; v12 can have fewer than 100 short steps, so use common_full for the paired architecture comparison.',
                        'long_full': 'All queries with 30 long chunks, including short-history reset/warmup; v12 values are included descriptively and may have padded histories.',
                        'unit_distance_rms': 'sqrt(mean(||z_i/||z_i|| - z_j/||z_j||||^2))',
                        'pair_sampling': 'At most one random eligible partner per anchor for tolerance/cross-motion categories; all unordered eligible sample pairs for exact-frame categories.',
                        'phase': 'Reference motion file progress; actual state/action histories are not forced equal.',
                        'disjoint': 'Later oldest chunk sequence >= earlier total_chunks + ceil((short_steps+pending_steps)/10), with >=50 collector steps between queries.'},
        'subsets': {},
    }
    for name, mask in [('common_full', common), ('memory_full', memory_full), ('long_full', usable)]:
        data = {k: v[mask] for k, v in all_data.items()}
        result['subsets'][name] = analyze_subset(data, metadata, output, name)
        if name == 'common_full':
            np.savez_compressed(output / 'main_data.npz', **data)
    result['limitations'] = [
        '42 shared LAFAN/Qingtong motion files are a controlled subset of the 129827-file Memory350 training directory, not a full-dataset or unseen-motion benchmark.',
        'Clean nominal and no-payload common physics are diagnostic distributions; the memory_training profile uses all DR plus four-limb payloads. Training mixtures are recorded for each checkpoint separately.',
        'Both frozen encoders observe identical transitions but use different trained normalization and history architecture.',
        'Comparison to v12 changes training data, DR, losses, updates and history. Same-update Memory350 comparisons control architecture and training age but have separately collected normalization and trajectories.',
        'Bootstrap resamples query worlds; repeated partner worlds remain dependent. Intervals are descriptive.',
        'Geometry does not prove downstream policy benefit. Removing memory at inference is an ablation, not a separately trained short-history control.',
    ]
    (output / 'cluster_metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
