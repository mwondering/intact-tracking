"""Backfill corrected warm tests on immutable saved policy checkpoints."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path

from run_limb_context_experiment import process_environment
from soft20000_warm_bounded_common import output_directory, protocol_path, paired_report, read, write, digest
from intact_tracking.memory350_policy_checkpoint_eval import load_protocol, validate_result, evaluate_checkpoint


def evaluate_arm(root, arm, update, slot):
    directory = output_directory(root, arm)
    directory.mkdir(parents=True, exist_ok=True)
    source = root / 'ppo' / f'{arm}_121'
    checkpoint = directory / f'checkpoint_update_{update:06d}.pt'
    if not checkpoint.exists():
        os.link(source / checkpoint.name, checkpoint)
    checkpoint_sha = digest(checkpoint)
    protocol, files = load_protocol(protocol_path(root))
    reuse = {}
    for case in ('mixture_warm', 'all_0_warm'):
        previous = source / 'endpoint_eval' / f'update_{update:06d}' / f'{case}.json'
        if not previous.exists():
            continue
        row = read(previous)
        if row['checkpoint_sha256'] != checkpoint_sha:
            continue
        validate_result(row, protocol, files, checkpoint_sha, update, case)
        target = directory / 'endpoint_eval' / f'update_{update:06d}' / previous.name
        target.parent.mkdir(parents=True, exist_ok=True)
        for old, new in ((previous, target), (previous.with_suffix('.traces.npz'), target.with_suffix('.traces.npz'))):
            if not new.exists():
                os.link(old, new)
        reuse[case] = {'original': str(previous), 'sha256': digest(previous), 'conditions_validated': True}
    first = (0 if arm == 'concat' else 4) + slot
    result = evaluate_checkpoint(checkpoint, directory, protocol_path(root), update, [first, first ^ 1])
    if digest(checkpoint) != checkpoint_sha:
        raise ValueError('Offline evaluation modified its checkpoint')
    result.update(training_state_preserved=True, offline_backfill=True,
        state_preservation_basis='Separate evaluation processes on an immutable checkpoint; no live trainer state accessed',
        reused_evaluations=reuse)
    with (directory / 'offline_evaluation_history.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history = directory / 'endpoint_eval_metrics.jsonl'
        rows = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        previous = [row for row in rows if row['completed_updates'] == update]
        if previous:
            if any(row['checkpoint_sha256'] != checkpoint_sha or row['protocol_sha256'] != result['protocol_sha256'] for row in previous):
                raise ValueError('Conflicting offline evaluation history')
        else:
            with history.open('a') as handle:
                handle.write(json.dumps(result, allow_nan=False) + '\n')
    return {'arm': arm, 'update': update, 'reused_cases': list(reuse), 'fresh_case': 'upper_train_warm'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--updates', nargs='+', type=int, required=True)
    args = parser.parse_args()
    if len(set(args.updates)) != len(args.updates) or not 1 <= len(args.updates) <= 2:
        raise ValueError('Evaluate one or two distinct saved updates per invocation')
    root = args.run_root.resolve()
    os.environ.update(process_environment())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(evaluate_arm, root, arm, update, slot)
                for slot, update in enumerate(args.updates) for arm in ('concat', 'baseline')]
        results = [job.result() for job in jobs]
    for update in sorted(args.updates):
        paired_report(root, update)
    write(root / 'evaluation_migration' / 'corrected_backfill.json', {'passed': True, 'evaluations': results})
    print(json.dumps({'corrected_backfill_complete': results}), flush=True)


if __name__ == '__main__':
    main()
