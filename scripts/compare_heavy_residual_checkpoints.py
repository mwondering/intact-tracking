"""Audit and compare paired learned/zero-latent heavy residual evaluations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_native_tracking import audit_query_forces


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(directory, physics, baseline, protocol, *, resamples, seed):
    roles = ('latent', baseline)
    rows = {role: json.loads((directory / f'{role}_{physics}.json').read_text()) for role in roles}
    latent, reference = (rows[role] for role in roles)
    paired = (
        'protocol', 'seed', 'motion_files', 'motion_ids', 'start_frames', 'horizons',
        'metric_names', 'max_steps', 'physics_world_fingerprints', 'is_nominal',
        'reward_contract', 'query_initial_state_sha256', 'global_metric_contract',
        'context_sha256', 'policy_precision', 'physics', 'added_limb_payload_kg',
        'payload_com_offsets_m', 'failure_term_names',
    )
    checks = {key: latent[key] == reference[key] for key in paired}
    if not all(checks.values()):
        raise ValueError(f'{physics}/{baseline}: unmatched fields {checks}')
    for key in ('steps', 'policy', 'policy_sha256', 'seed'):
        if latent['warmup'][key] != reference['warmup'][key]:
            raise ValueError(f'Unmatched tracker warmup: {key}')
    manifest = Path(protocol['manifest']).read_text().splitlines()
    for role, row in rows.items():
        selected = protocol['checkpoints'][role]
        if (row['checkpoint_sha256'] != selected['sha256']
                or row['completed_training_updates'] != selected['completed_updates']
                or row['motion_files'] != manifest
                or row['seed'] != protocol['physics_seed']
                or row['max_steps'] != protocol['query_steps']
                or row['warmup']['steps'] != protocol['warmup_steps']
                or row['episodes'] != protocol['worlds_per_arm']
                or row['repeats_per_motion'] != protocol['repeats']
                or row['arguments']['physics'] != physics
                or row['memory_start'] != 'warm'
                or row['policy_precision'] != 'fp32'
                or not row['arguments']['paired_starts']
                or row['latent_intervention'] != 'correct'
                or row['arguments']['frozen_tracker_only']):
            raise ValueError(f'{role}/{physics}: evaluation differs from selected protocol')
        if any(bool(nominal) != (physics == 'nominal') for nominal in row['is_nominal']):
            raise ValueError(f'{physics} must evaluate the entire requested physics population')
        for key in ('reference_timeline_audited', 'partial_reset_survivor_state_audited',
                    'partial_reset_survivor_history_audited'):
            if not row[key]:
                raise ValueError(f'Missing runtime audit: {key}')
        if row['initial_context']['long_full_fraction'] != 1.:
            raise ValueError('Tracker warmup did not fill every long history')
        if bool(row['initial_context']['latent_input_is_zero']) != (role != 'latent'):
            raise ValueError(f'{role} uses the wrong latent input mode')

    lengths = {role: np.asarray(row['episode_lengths']) for role, row in rows.items()}
    common = np.minimum(lengths['latent'], lengths[baseline])
    if (common < 1).any():
        raise ValueError('Empty paired episode')
    force_steps, full_forces_equal = audit_query_forces(
        latent['evaluation_diagnostics'], reference['evaluation_diagnostics'], int(common.max()))
    checks['common_query_force_sequence'] = True
    means = {}
    for role, row in rows.items():
        with np.load(directory / f'{role}_{physics}.traces.npz') as archive:
            np.testing.assert_array_equal(archive['metric_names'], row['metric_names'])
            np.testing.assert_array_equal(archive['lengths'], lengths[role])
            values = archive['all_metrics'].astype(np.float64)
        own_mask = np.arange(values.shape[1])[None, :] < lengths[role][:, None]
        own_means = np.where(own_mask[..., None], values, 0).sum(1) / lengths[role][:, None]
        np.testing.assert_allclose(own_means, row['per_episode_metrics'], rtol=1e-10, atol=1e-12)
        mask = np.arange(values.shape[1])[None, :] < common[:, None]
        means[role] = np.where(mask[..., None], values, 0).sum(1) / common[:, None]
        if not np.isfinite(means[role]).all():
            raise ValueError('Non-finite scored metrics')

    motion_ids = np.asarray(latent['motion_ids'])
    ids = np.unique(motion_ids)
    counts = np.asarray([(motion_ids == i).sum() for i in ids])
    sums = {role: np.asarray([means[role][motion_ids == i].sum(0) for i in ids]) for role in roles}
    successes = {role: 1 - np.asarray(row['failed'], dtype=np.float64) for role, row in rows.items()}
    success_sums = {role: np.asarray([successes[role][motion_ids == i].sum() for i in ids]) for role in roles}
    rng = np.random.default_rng(seed)
    improvements, success_differences = [], []
    for offset in range(0, resamples, 256):
        draws = rng.integers(0, len(ids), size=(min(256, resamples-offset), len(ids)))
        denominator = counts[draws].sum(1)
        samples = {role: sums[role][draws].sum(1)/denominator[:, None] for role in roles}
        improvements.append(100 * (samples[baseline]-samples['latent']) / samples[baseline])
        success_differences.append(100 * (success_sums['latent'][draws].sum(1)
            - success_sums[baseline][draws].sum(1)) / denominator)
    ci = np.quantile(np.concatenate(improvements), [.025, .975], axis=0)
    metrics = {}
    for i, name in enumerate(latent['metric_names']):
        left, right = means['latent'][:, i].mean(), means[baseline][:, i].mean()
        metrics[name] = {'latent': float(left), 'baseline': float(right),
            'reduction_percent': float(100 * (right-left)/right),
            'reduction_motion_cluster_bootstrap_95pct': ci[:, i].tolist()}
    horizons = np.asarray(latent['horizons'])
    arms = {role: {'checkpoint': protocol['checkpoints'][role],
                   'failures': int(np.asarray(row['failed']).sum()),
                   'successes': int(successes[role].sum()),
                   'success_rate': float(successes[role].mean()),
                   'coverage_fraction': float((lengths[role]/horizons).mean()),
                   'mean_episode_return': row['mean_episode_return']}
            for role, row in rows.items()}
    return {'physics': physics, 'baseline_arm': baseline, 'episodes': len(common),
        'motions': len(ids), 'pairing_checks': checks, 'metrics': metrics, 'arms': arms,
        'common_prefix_coverage': float((common/horizons).mean()),
        'force_audit': {'common_steps': force_steps, 'equal_length_full_hash': full_forces_equal},
        'success_rate_difference_percentage_points': float(100 * (successes['latent'].mean()-successes[baseline].mean())),
        'success_difference_motion_cluster_bootstrap_95pct': np.quantile(
            np.concatenate(success_differences), [.025, .975]).tolist(),
        'warmup': {
            'trajectory_bitwise_equal': latent['warmup']['trajectory_sample_sha256'] == reference['warmup']['trajectory_sample_sha256'],
            'different_reset_count_worlds': sum(x != y for x, y in zip(
                latent['warmup']['resets_per_world'], reference['warmup']['resets_per_world'], strict=True)),
            'long_history_full_in_both': True,
            'note': 'Independent simulator warmups need not be bitwise identical. Query initial physics, state, and common force sequence are checked. The zero-latent arm ignores context.'},
        'original_failure_truncated_means': {role: row['mean'] for role, row in rows.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260921)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('Bootstrap samples must be positive')
    directory = args.directory.resolve()
    protocol = json.loads((directory/'protocol.json').read_text())
    if digest(protocol['manifest']) != protocol['manifest_sha256']:
        raise ValueError('Selected motion manifest changed')
    comparisons = {}
    for name, baseline in (('latest', 'baseline'), ('matched_updates', 'baseline_matched')):
        comparisons[name] = {physics: compare(directory, physics, baseline, protocol,
            resamples=args.bootstrap_samples, seed=args.seed) for physics in protocol['physics']}
    sources = [directory/'protocol.json', directory/'motions_256.txt', Path(__file__)]
    sources.extend(directory/f'{role}_{physics}{suffix}' for role in protocol['checkpoints']
                   for physics in protocol['physics'] for suffix in ('.json', '.traces.npz'))
    result = {
        'protocol': protocol,
        'aggregation': 'Mean per world over the common valid trajectory prefix, then equal world weights. Full-query success and coverage reported separately. Failure step included; no padding contributes.',
        'bootstrap': {'seed': args.seed, 'resamples': args.bootstrap_samples,
                      'unit': 'motion cluster, keeping its two repeats together',
                      'scope': 'One physics seed and checkpoint pair; not training-seed uncertainty'},
        'comparisons': comparisons,
        'source_sha256': {str(path): digest(path) for path in sources},
        'limitations': ['256 randomly selected training-catalog motions; not held-out generalization or a full-dataset evaluation.',
                        'Latest checkpoints have unequal update counts; matched-update comparison is separate.',
                        'Learned and zero-latent training also differ in auxiliary supervision.',
                        'Warm histories use 500 frozen-tracker steps; cold-start performance is not measured.'],
    }
    output = directory/'summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(output)
    for comparison, scenes in comparisons.items():
        for physics, row in scenes.items():
            print(comparison, physics, 'success', {role: arm['successes'] for role, arm in row['arms'].items()})
            for key in ('error_joint_pos', 'error_body_pos', 'error_body_rot', 'error_anchor_pos_global', 'error_body_pos_global'):
                print(key, row['metrics'][key])


if __name__ == '__main__':
    main()
