"""Group calibrated latent prototypes and retain explicit DR rows per class."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from probe_memory350_world_partitions import distances, kmeans


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def route_metrics(reference, predicted, ids, worlds, classes):
    correct = reference == predicted
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (reference, predicted), 1)
    return {
        'queries': len(ids), 'worlds': len(np.unique(worlds)),
        'query_agreement': float(correct.mean()),
        'anchor_balanced_agreement': float(np.mean([correct[ids == i].mean() for i in np.unique(ids)])),
        'world_balanced_agreement': float(np.mean([correct[worlds == w].mean() for w in np.unique(worlds)])),
        'class_recall': [float(correct[reference == c].mean()) for c in range(classes)],
        'confusion_reference_rows_predicted_columns': confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--seed', type=int, default=731)
    args = parser.parse_args()
    if len(set(args.classes)) != len(args.classes) or any(k < 2 or k > 32 for k in args.classes):
        raise ValueError('Choose distinct class counts in [2,32]')
    torch.set_num_threads(4)
    source, output = args.calibration.resolve(), args.output.resolve()
    saved = torch.load(source / 'prototypes.pt', map_location='cpu', weights_only=False, mmap=True)
    samples = torch.load(source / 'calibration_samples.pt', map_location='cpu', weights_only=False, mmap=True)
    report = json.loads((source / 'report.json').read_text())
    if saved['version'] != 'payload256_cross_world_prototypes_v1':
        raise ValueError('Expected audited payload-only 256-anchor calibration')
    if report['physics']['background'] != 'nominal; no pushes or observation corruption':
        raise ValueError('Explicit four-load rows require the same nominal background')
    assert saved['metadata']['context_sha256'] == report['context_sha256']
    z = F.normalize(samples['latent'].reshape(-1, 64).float(), dim=-1)
    ids, worlds = samples['ids'], samples['worlds']
    assert torch.equal(ids, worlds % 256)
    fold = (worlds // 256) % 4
    full = samples['full'].flatten()
    fit = (fold < 2) & full
    centers_check = torch.stack([z[fit & (ids == i)].mean(0) for i in range(256)])
    torch.testing.assert_close(centers_check, saved['centers'], atol=1e-7, rtol=0)
    old_distance = (z.square().sum(-1, keepdim=True) + saved['centers'].square().sum(-1)[None]
                    - 2 * z @ saved['centers'].T).clamp_min(0)
    # Reproduce the original topk tie handling, which can differ from argmin.
    old_top1 = old_distance.topk(5, largest=False).indices[:, 0]
    original_top1 = {}
    for split, number in (('validation', 2), ('test', 3)):
        selected = (fold == number) & full
        accuracy = float((old_top1[selected] == ids[selected]).float().mean())
        assert abs(accuracy - report['metrics'][split]['top1']) < 1e-7
        original_top1[split] = accuracy
    prototypes = saved['centers'].numpy().astype(np.float64)
    loads = saved['loads_kg'].numpy()
    assert loads.shape == (256, 4) and np.unique(loads, axis=0).shape[0] == 256
    limits = np.array([2.5, 2.5, 4., 4.])
    assert np.isfinite(loads).all() and ((loads >= 0) & (loads <= limits)).all()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        'source': str(source),
        'source_hashes': {name: digest(source / name) for name in ('prototypes.pt', 'calibration_samples.pt', 'report.json')},
        'encoder_checkpoint': report['context_checkpoint'], 'encoder_sha256': report['context_sha256'],
        'tracker_sha256': report['tracker_sha256'], 'script_sha256': digest(Path(__file__)),
    }
    summary = {
        **provenance,
        'method': 'KMeans of the 256 saved fit-world mean unit-latent prototypes; equally weighted anchors, '
                  'ten starts per K, fixed seed, selection by fit inertia only. No validation/test queries or '
                  'DR coordinates in clustering. Class IDs fixed in each exported bank.',
        'scope': 'Nominal background; four explicit limb loads. Test queries are independent worlds/histories '
                 'at the same 256 load combinations, not held-out physical parameters or disjoint motion families. '
                 'Frozen-tracker full-memory calibration; no expert PPO training or online routing changes.',
        'verification': {'fit_prototypes_reproduced': True, 'original_top1_reproduced': original_top1},
        'results': [],
    }
    z, ids, worlds = z.numpy().astype(np.float64), ids.numpy(), worlds.numpy()
    fold, full = fold.numpy(), full.numpy()
    for count in args.classes:
        centers = kmeans(prototypes, count, args.seed)
        labels = distances(prototypes, centers).argmin(1)
        assignments = distances(z, centers).argmin(1)
        groups = []
        for group in range(count):
            members = np.flatnonzero(labels == group)
            if not len(members):
                raise RuntimeError('Empty latent class')
            group_loads = loads[members]
            groups.append({
                'class_id': group, 'parameter_rows': len(members),
                'load_mean_kg': group_loads.mean(0).tolist(),
                'load_min_kg': group_loads.min(0).tolist(), 'load_max_kg': group_loads.max(0).tolist(),
                'rows': [{'prototype_id': int(i), 'masses_kg': loads[i].tolist()} for i in members],
            })
        assert sorted(row['prototype_id'] for g in groups for row in g['rows']) == list(range(256))
        bank = {
            'version': 'latent_class_explicit_payload_bank_v1', **provenance,
            'dr_profile': 'load_only', 'background': report['physics']['background'],
            'physics_contract': report['physics'],
            'limb_order': ['left_hand', 'right_hand', 'left_shin', 'right_shin'],
            'max_masses_kg': limits.tolist(), 'class_count': count, 'clustering_seed': args.seed,
            'membership': 'Fixed by nearest cluster center of each fit-world latent prototype',
            'sampling': 'Select an explicit row within the requested class and set its four masses at world '
                        'startup using the saved nominal-background physics contract; keep fixed across episodes. '
                        'The min/max summaries are not rectangular conditional sampling ranges.',
            'training_status': 'Export only. Not wired into existing --dr-bank (legacy eight fixed full-DR profiles). '
                               'No simulation, PPO training or runtime router has been started by this exporter.',
            'latent_cluster_centers': centers.tolist(), 'classes': groups,
        }
        (output / f'bank_k{count:02d}.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2) + '\n')
        metrics = {}
        for split, number in (('validation', 2), ('test', 3)):
            selected = (fold == number) & full
            metrics[split] = route_metrics(labels[ids[selected]], assignments[selected], ids[selected], worlds[selected], count)
        np.savez_compressed(output / f'partition_k{count:02d}.npz', centers=centers, prototype_latents=prototypes,
                            loads_kg=loads, prototype_class=labels, query_class=assignments,
                            query_worlds=worlds, query_prototype_ids=ids, replica_fold=fold, full_history=full)
        row = {'classes': count, 'parameter_rows_per_class': [g['parameter_rows'] for g in groups], 'metrics': metrics}
        summary['results'].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
