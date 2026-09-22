"""Compare checkpoint/latent/history interventions on matched query trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_native_tracking import audit_query_forces


def compare(reference_path, candidate_path):
    rows = [json.loads(path.read_text()) for path in (reference_path, candidate_path)]
    reference, candidate = rows
    keys = ('protocol', 'seed', 'motion_files', 'motion_ids', 'start_frames', 'horizons',
            'max_steps', 'metric_names', 'physics_world_fingerprints', 'is_nominal',
            'query_initial_state_sha256', 'context_sha256', 'reward_contract',
            'global_metric_contract', 'added_limb_payload_kg', 'payload_com_offsets_m')
    checks = {key: reference[key] == candidate[key] for key in keys}
    if not all(checks.values()):
        raise ValueError(f'Unmatched inputs: {checks}')
    lengths = [np.asarray(row['episode_lengths']) for row in rows]
    common = np.minimum(*lengths)
    if common.min() < 1:
        raise ValueError('Empty query')
    forces = audit_query_forces(reference['evaluation_diagnostics'],
                               candidate['evaluation_diagnostics'], int(common.max()))
    values = []
    for path, row, count in zip((reference_path, candidate_path), rows, lengths, strict=True):
        with np.load(path.with_suffix('.traces.npz')) as data:
            np.testing.assert_array_equal(data['lengths'], count)
            np.testing.assert_array_equal(data['metric_names'], row['metric_names'])
            trace = data['all_metrics'].astype(np.float64)
        mask = np.arange(trace.shape[1])[None, :] < count[:, None]
        own = np.where(mask[..., None], trace, 0).sum(1) / count[:, None]
        np.testing.assert_allclose(own, row['per_episode_metrics'], rtol=1e-10, atol=1e-12)
        mask = np.arange(trace.shape[1])[None, :] < common[:, None]
        values.append(np.where(mask[..., None], trace, 0).sum(1)/common[:, None])

    # Resample whole motions rather than treating their two replicas as independent.
    ids = np.asarray(reference['motion_ids'])
    clusters = np.unique(ids)
    counts = np.asarray([(ids == i).sum() for i in clusters])
    sums = [np.asarray([value[ids == i].sum(0) for i in clusters]) for value in values]
    rng = np.random.default_rng(20260921)
    changes = []
    for _ in range(40):
        draw = rng.integers(len(clusters), size=(250, len(clusters)))
        r, c = [v[draw].sum(1)/counts[draw].sum(1)[:, None] for v in sums]
        changes.append(100 * (c-r)/r)
    ci = np.quantile(np.concatenate(changes), [.025, .975], axis=0)
    r, c = [v.mean(0) for v in values]
    return {
        'reference': str(reference_path), 'candidate': str(candidate_path),
        'checkpoints_sha256': [row['checkpoint_sha256'] for row in rows],
        'completed_updates': [row['completed_training_updates'] for row in rows],
        'pairing_checks': checks, 'force_audit': {'common_steps': forces[0], 'full_hash_equal': forces[1]},
        'warmup_policies': [row['warmup']['policy'] for row in rows],
        'warmup_histories_bitwise_equal': reference['warmup']['trajectory_sample_sha256'] == candidate['warmup']['trajectory_sample_sha256'],
        'successes': [len(row['failed'])-sum(row['failed']) for row in rows],
        'episodes': len(common),
        'full_query_mean_episode_return': [row['mean_episode_return'] for row in rows],
        'latent_interventions': [row['latent_intervention'] for row in rows],
        'swapped_valid_step_fraction': [sum(row['swapped_steps'])/sum(row['episode_lengths']) for row in rows],
        'common_prefix_coverage': float((common/np.asarray(reference['horizons'])).mean()),
        'metrics': {name: {'reference': float(r[i]), 'candidate': float(c[i]),
            'candidate_error_increase_percent': float(100*(c[i]-r[i])/r[i]),
            'increase_motion_bootstrap_95pct': ci[:, i].tolist()}
            for i, name in enumerate(reference['metric_names'])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    protocol = json.loads((directory/'protocol.json').read_text())
    previous = Path(protocol['reference'])
    pairs = {
        'zero_hdr': (previous/'latent_hdr.json', directory/'latent4200_zero_hdr.json'),
        'swap_hdr': (previous/'latent_hdr.json', directory/'latent4200_swap_hdr.json'),
        'selfwarm_hdr': (previous/'latent_hdr.json', directory/'latent4200_selfwarm_hdr.json'),
        'zero_nominal': (previous/'latent_nominal.json', directory/'latent4200_zero_nominal.json'),
        'latent_2000_to_4200_hdr': (directory/'latent2000_hdr.json', previous/'latent_hdr.json'),
        'baseline_2000_to_4200_hdr': (directory/'baseline2000_hdr.json', previous/'baseline_matched_hdr.json'),
        'latent_vs_baseline_2000_hdr': (directory/'baseline2000_hdr.json', directory/'latent2000_hdr.json'),
    }
    comparisons = {name: compare(*paths) for name, paths in pairs.items()}
    files = {path for pair in pairs.values() for path in pair}
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result = {'comparisons': comparisons, 'source_json_sha256': sources,
        'gradient_probe': json.loads((directory/'latent4200_gradients.json').read_text()),
        'training_and_strata': json.loads((directory/'training_and_strata.json').read_text()),
        'aggregation': 'Equal-world average of errors over each pair common valid prefix; success counts use full query.',
        'bootstrap': '10,000 motion-cluster resamples; single physics seed and checkpoint pair.',
        'limitations': ['Independent warmups are not bitwise identical.',
            'Zero latent is out of distribution for the trained conditional policy, not equivalent to the separately trained baseline.',
            'Paired swap changes history/latent only, between active equal-phase worlds with full histories.',
            'The 2000-to-4200 difference also includes sampler reset and continued PPO; it does not isolate the auxiliary-loss change.',
            'Gradient probe uses one independent 512-world on-policy batch, not all training updates.']}
    (directory/'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, comparison in comparisons.items():
        print(name, 'successes', comparison['successes'])
        for key in ('error_joint_pos', 'error_body_pos', 'error_body_rot', 'error_anchor_pos_global', 'error_body_pos_global'):
            print(key, comparison['metrics'][key])


if __name__ == '__main__':
    main()
