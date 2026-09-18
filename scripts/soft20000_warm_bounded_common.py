"""Shared paths and paired-result writer for the corrected warm evaluation."""
from pathlib import Path
import json
import numpy as np
from run_soft20000_history5_ppo import read, write, digest
from intact_tracking.memory350_policy_checkpoint_eval import load_protocol, validate_result
from intact_tracking.memory350_policy_results import compare


def output_directory(root, arm):
    return Path(root) / 'ppo' / f'{arm}_121_warm_bounded'


def protocol_path(root):
    return Path(root) / 'protocols/periodic_warm_bounded_v2.json'


def evaluation_history(root, arm):
    path = output_directory(root, arm) / 'endpoint_eval_metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().split('\n')[:-1] if line.strip()] if path.exists() else []
    by_update = {row['completed_updates']: row for row in rows}
    if len(by_update) != len(rows):
        raise ValueError('Duplicate evaluation update')
    return by_update


def paired_report(root, update):
    target = Path(root) / 'paired_results_warm_bounded_v2' / f'update_{update:06d}.json'
    if target.exists():
        return target
    protocol, files = load_protocol(protocol_path(root))
    protocol_sha = digest(protocol_path(root))
    histories = {arm: evaluation_history(root, arm)[update] for arm in ('baseline', 'concat')}
    if any(row['protocol_sha256'] != protocol_sha for row in histories.values()):
        raise ValueError('Paired protocol revision mismatch')
    if histories['baseline']['policy_precision'] != histories['concat']['policy_precision']:
        raise ValueError('Paired policy precision mismatch')
    result = {'completed_updates': update, 'training_seed_count': 1, 'training_seed': 121,
        'evaluation_protocol_revision': protocol['version'], 'protocol_sha256': protocol_sha,
        'primary_cases': ['mixture_warm'], 'upper_masses_kg': [2.5, 2.5, 4, 4],
        'comparisons': {}, 'inputs': {}, 'maximum_updates': None,
        'uncertainty': 'Paired motion bootstrap; not variation across training seeds',
        'scope': 'Training-catalog motions with held-out physics seeds and starts'}
    for case in protocol['cases']:
        rows, traces = {}, {}
        result['inputs'][case] = {}
        for arm in ('baseline', 'concat'):
            path = output_directory(root, arm) / 'endpoint_eval' / f'update_{update:06d}' / f'{case}.json'
            row = read(path)
            validate_result(row, protocol, files, histories[arm]['checkpoint_sha256'], update, case)
            rows[arm] = row
            with np.load(path.with_suffix('.traces.npz')) as saved:
                if saved['lengths'].tolist() != row['episode_lengths']:
                    raise ValueError('Evaluation trace lengths differ')
                traces[arm] = saved['all_metrics'].copy()
            result['inputs'][case][arm] = {'path': str(path), 'sha256': digest(path),
                'checkpoint_sha256': row['checkpoint_sha256']}
        result['comparisons'][case] = compare(rows['baseline'], rows['concat'], traces['baseline'], traces['concat'])
    write(target, result)
    return target
