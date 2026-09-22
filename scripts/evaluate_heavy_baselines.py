"""Evaluate an entire ordered manifest in bounded batches, without truncation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aggregate(results):
    total = sum(row['episodes'] for row in results)
    if not total:
        raise ValueError('No evaluated episodes')
    metrics = tuple(results[0]['metric_names'])
    checkpoint = results[0]['checkpoint_sha256']
    if any(tuple(row['metric_names']) != metrics or row['checkpoint_sha256'] != checkpoint for row in results):
        raise ValueError('Cannot aggregate different metrics or model weights')
    return {'episodes': total, 'motions': sum(row['motions'] for row in results),
            'checkpoint_sha256': checkpoint, 'completed_updates': results[0]['completed_training_updates'],
            'mean': {name: sum(row['mean'][name]*row['episodes'] for row in results)/total for name in metrics},
            'failure_rate': sum(sum(row['failed']) for row in results)/total,
            'coverage_fraction': sum(row['coverage_fraction']*row['episodes'] for row in results)/total,
            'step_weighted_coverage_fraction': sum(sum(row['episode_lengths']) for row in results)/sum(sum(row['horizons']) for row in results),
            'mean_episode_return': sum(sum(row['episode_returns']) for row in results)/total,
            'metric_convention': 'episode-averaged, failure-truncated errors; compare with failure rate and coverage',
            'common_prefix_comparison': 'per-batch .traces.npz retains all per-step metrics for paired comparisons'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--motion-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--physics', choices=('nominal', 'hdr', 'both'), default='both')
    parser.add_argument('--motions-per-batch', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20001)
    parser.add_argument('--frozen-tracker-only', action='store_true')
    args = parser.parse_args()
    if (args.repeats != 2 or not 0 < args.motions_per_batch*args.repeats <= 4096 or args.steps <= 0):
        raise ValueError('Use two paired starts and batches of at most 4096 worlds')
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep evaluation artifacts in the repository')
    files = [str(Path(line).resolve()) for line in Path(args.motion_manifest).read_text().splitlines() if line.strip()]
    if not files or len(set(files)) != len(files) or any(not Path(p).is_file() for p in files):
        raise ValueError('Manifest must contain unique existing motions')
    contract = {'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(args.motion_manifest),
                'motion_count': len(files), 'motions_per_batch': args.motions_per_batch, 'repeats': args.repeats,
                'steps': args.steps, 'seed': args.seed, 'physics': args.physics, 'frozen_tracker_only': args.frozen_tracker_only,
                'seed_rule': 'seed + 1000003 * batch_index', 'history_start': 'cold'}
    output.mkdir(parents=True, exist_ok=True)
    protocol = output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != contract:
        raise ValueError('Existing evaluation protocol differs; use a fresh directory')
    protocol.write_text(json.dumps(contract, indent=2)+'\n')
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    summaries = {}
    for physics in (('nominal', 'hdr') if args.physics == 'both' else (args.physics,)):
        rows = []
        for index, offset in enumerate(range(0, len(files), args.motions_per_batch)):
            batch = files[offset:offset+args.motions_per_batch]
            manifest = output/f'motions_{index:05d}.txt'
            manifest.write_text('\n'.join(batch)+'\n')
            result = output/f'{physics}_{index:05d}.json'
            if not result.exists():
                command = [str(ROOT/'.venv/bin/python'), '-B', '-u', '-m', 'intact_tracking.cli.heavy_baseline_eval',
                           '--checkpoint', str(Path(args.checkpoint).resolve()), '--motion-manifest', str(manifest),
                           '--physics', physics, '--repeats', str(args.repeats), '--paired-starts', '--steps', str(args.steps),
                           '--seed', str(args.seed+1000003*index), '--output', str(result)]
                if args.frozen_tracker_only:
                    command.append('--frozen-tracker-only')
                with result.with_suffix('.log').open('ab') as stream:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            row = json.loads(result.read_text())
            # Frozen tracker rows identify the tracker hash; their selected
            # residual checkpoint is recorded in arguments for setup pairing.
            if (row['motion_files'] != batch or row['seed'] != args.seed+1000003*index
                    or row['episodes'] != len(batch)*args.repeats
                    or row['arguments']['physics'] != physics
                    or row['max_steps'] != args.steps or not row['arguments']['paired_starts']
                    or row['arguments']['frozen_tracker_only'] != args.frozen_tracker_only
                    or (not args.frozen_tracker_only and row['checkpoint_sha256'] != contract['checkpoint_sha256'])):
                raise ValueError(f'Saved batch does not match evaluation protocol: {result}')
            rows.append(row)
            print(json.dumps({'physics': physics, 'completed_motions': offset+len(batch), 'total_motions': len(files)}), flush=True)
        summaries[physics] = aggregate(rows)
    temporary = output/'summary.json.tmp'
    temporary.write_text(json.dumps({'protocol': contract, 'results': summaries}, indent=2)+'\n')
    temporary.replace(output/'summary.json')


if __name__ == '__main__':
    main()
