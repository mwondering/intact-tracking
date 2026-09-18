"""Offline world-level latent partition stability; no policy or simulator runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distances(x, centers):
    return np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None] - 2 * x @ centers.T, 0)


def kmeans(x, count, seed, starts=10):
    """Choose initialization using fit inertia only, never query agreement."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        centers = [x[rng.integers(len(x))].copy()]
        while len(centers) < count:
            cost = distances(x, np.array(centers)).min(1)
            centers.append(x[rng.choice(len(x), p=cost / cost.sum())].copy())
        centers = np.array(centers)
        for _ in range(200):
            cost = distances(x, centers)
            labels = cost.argmin(1)
            updated = centers.copy()
            empty = []
            for group in range(count):
                if np.any(labels == group):
                    updated[group] = x[labels == group].mean(0)
                else:
                    empty.append(group)
            if empty:
                candidates = np.argsort(cost.min(1))[::-1]
                for group, row in zip(empty, candidates):
                    updated[group] = x[row]
            if np.allclose(updated, centers, rtol=0, atol=1e-10):
                centers = updated
                break
            centers = updated
        inertia = distances(x, centers).min(1).sum()
        if best is None or inertia < best[0]:
            best = (float(inertia), centers.copy())
    return best[1]


def split(data, metadata):
    """Reproduce the audited LaFAN cross-family, disjoint-history split."""
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
    return train, test, ids


def score(reference_labels, query_labels, query_worlds):
    correct = reference_labels == query_labels
    ids = np.unique(query_worlds)
    return {
        'query_agreement': float(correct.mean()),
        'world_balanced_agreement': float(np.mean([correct[query_worlds == w].mean() for w in ids])),
        'worlds_single_query_route_fraction': float(np.mean([
            len(np.unique(query_labels[query_worlds == w])) == 1 for w in ids])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Audited u30750 check directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    folder = source / 'capped_loads'
    metadata = json.loads((folder / 'metadata.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    with np.load(folder / 'latents.npz', allow_pickle=False) as saved:
        mask = (~saved['nominal']) & (saved['short_steps'] == 50) & (saved['long_chunks'] == 30)
        data = {k: saved[k][mask] for k in saved.files}
    train, test, ids = split(data, metadata)
    expected = summary['profiles']['capped_loads']['readout']
    assert len(ids) == expected['worlds'] and test.sum() == expected['test_samples']
    z = data[metadata['models']['memory350']].astype(np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    prototypes = np.stack([z[train & (data['world'] == w)].mean(0) for w in ids])
    queries, query_worlds = z[test], data['world'][test]
    lookup = np.searchsorted(ids, query_worlds)
    known_world_accuracy = float((distances(queries, prototypes).argmin(1) == lookup).mean())
    assert abs(known_world_accuracy - expected['models']['memory350']['top1_accuracy']) < 1e-12
    fold = np.empty(len(ids), dtype=int)
    fold[np.random.default_rng(20260916).permutation(len(ids))] = np.arange(len(ids)) % 5
    results, artifacts = [], {}
    for count in (4, 8, 16):
        for seed in (731, 1731, 2731):
            centers = kmeans(prototypes, count, seed)
            assignments = distances(prototypes, centers).argmin(1)
            predicted = distances(queries, centers).argmin(1)
            sizes = np.bincount(assignments, minlength=count)
            seen = score(assignments[lookup], predicted, query_worlds)
            cross_reference, cross_predicted = np.empty(len(queries), int), np.empty(len(queries), int)
            minimum_fit_size = len(ids)
            for held_fold in range(5):
                fitting = fold != held_fold
                centers_fold = kmeans(prototypes[fitting], count, seed + 10000 * (held_fold + 1))
                fit_labels = distances(prototypes[fitting], centers_fold).argmin(1)
                minimum_fit_size = min(minimum_fit_size, int(np.bincount(fit_labels, minlength=count).min()))
                selected = fold[lookup] == held_fold
                # Reference uses separate motion families of the held-out world;
                # neither those prototypes nor its query data fit the centers.
                cross_reference[selected] = distances(prototypes[lookup[selected]], centers_fold).argmin(1)
                cross_predicted[selected] = distances(queries[selected], centers_fold).argmin(1)
                artifacts[f'k{count}_seed{seed}_fold{held_fold}_centers'] = centers_fold
            held = score(cross_reference, cross_predicted, query_worlds)
            results.append({'k': count, 'seed': seed, 'prototype_counts': sizes.tolist(),
                            'minimum_fold_fit_cluster_size': minimum_fit_size,
                            'seen_world_cross_motion': seen, 'heldout_world_cross_motion': held})
            artifacts[f'k{count}_seed{seed}_centers'] = centers
            artifacts[f'k{count}_seed{seed}_world_labels'] = assignments
            artifacts[f'k{count}_seed{seed}_query_labels'] = predicted
            artifacts[f'k{count}_seed{seed}_heldout_reference'] = cross_reference
            artifacts[f'k{count}_seed{seed}_heldout_query_labels'] = cross_predicted
    artifacts.update(world_ids=ids, world_fold=fold, world_prototypes=prototypes,
                     query_worlds=query_worlds, query_source_rows=data['source_row'][test])
    np.savez_compressed(output / 'partitions.npz', **artifacts)
    result = {
        'source': str(source), 'checkpoint': summary['checkpoints']['memory350'],
        'source_hashes': {name: digest(folder / name) for name in ('metadata.json', 'latents.npz')},
        'script_sha256': digest(Path(__file__)), 'worlds': len(ids), 'queries': len(queries),
        'known_world_identity_accuracy_reproduced': known_world_accuracy,
        'method': 'Unit-normalize individual latent; average per world over audited reference motion families; '
                  'Euclidean KMeans of world means, ten fit-only starts. No DR labels or query data fit clusters. '
                  'Five folds exclude query worlds from center fitting. Held-out reference class comes from '
                  'that world separate reference-motion prototype. Fold labels are local to each fold.',
        'scope': 'Exploratory cached u30750, LaFAN, full tracker DR plus capped payload, full memory, frozen-tracker '
                 'trajectories. Agreement with a separate world prototype is not physical-class accuracy, '
                 'temporal switch rate, new-policy online performance, or proof of useful experts. '
                 'Three clustering seeds are not independent encoder or policy training seeds.',
        'results': results,
    }
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for count in (4, 8, 16):
        rows = [r for r in results if r['k'] == count]
        print(json.dumps({'k': count, 'heldout_world_query_agreement': [r['heldout_world_cross_motion']['query_agreement'] for r in rows],
                          'worlds_single_route_fraction': [r['heldout_world_cross_motion']['worlds_single_query_route_fraction'] for r in rows],
                          'fit_cluster_min_max': [[min(r['prototype_counts']), max(r['prototype_counts'])] for r in rows]}))


if __name__ == '__main__':
    main()
