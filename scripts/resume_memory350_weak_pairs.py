"""Continue u5000 to u10000 unchanged; evaluate every 1000 updates on GPUs 4-7."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, process_environment
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json


MILESTONES = tuple(range(6000, 10001, 1000))
BASELINE_SHA256 = 'ec34427d1e3e38c2c7711bd2120f0fc7e6abfb994035232f279acb18818997ea'
JOURNAL = 'continuation_005000_010000'


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def continuation_command(root):
    command = list(read_json(root / 'launch_contract.json')['formal_command'])
    command[command.index('--stop-after-updates') + 1] = '10000'
    command += ['--resume', str(root / 'stage1_8192/update_005000.pt')]
    return command


def preflight(root, journal):
    import torch
    from intact_tracking.cli.forward_memory_weak_pairs_train import build_parser, reference_contract
    from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_arguments
    output = root / 'stage1_8192'
    baseline = output / 'update_005000.pt'
    assert sha256(baseline) == BASELINE_SHA256, 'The frozen u5000 checkpoint changed'
    original = read_json(output / 'run_config.json')
    command = continuation_command(root)
    index = command.index('intact_tracking.cli.forward_memory_weak_pairs_train')
    args = build_parser().parse_args(command[index + 1:])
    _validate_arguments(args)
    actual = deepcopy(original)
    actual['arguments'] = vars(args)
    # Reproduce the base trainer's metadata before the weak-pair hook runs.
    for key in ('weak_positive_weight', 'weak_negative_weight', 'weak_negative_margin'):
        actual['objective_weights'].pop(key, None)
    contract = reference_contract(args.comparison_reference_dir, actual)
    state = torch.load(baseline, map_location='cpu', weights_only=False, mmap=True)
    assert state['update'] == 5000 and state['optimizer_steps'] == 20000
    assert state['scheduler']['T_max'] == 32000
    assert state['scheduler']['last_epoch'] == 20000
    assert state['scheduler']['continuation_min_learning_rate'] == args.continuation_min_learning_rate == 1e-5
    assert args.updates == 8000 and args.stop_after_updates == 10000
    assert args.checkpoint_interval == 250 and all(u % args.checkpoint_interval == 0 for u in MILESTONES)
    assert all(int(value['step'].item()) == 20000 for value in state['optimizer']['state'].values())
    assert original['model'] == actual['model'] and original['loss'] == actual['loss']
    source_expected = read_json(root / 'source_sha256.json')
    protected = [
        'src/intact_tracking/cli/forward_memory_scale_nominal_train.py',
        'src/intact_tracking/memory350_weak_pairs.py',
        'src/intact_tracking/memory350_replay.py', 'src/intact_tracking/memory350_objective.py',
        'src/intact_tracking/memory350_model.py', 'src/intact_tracking/memory350_bank.py',
        'src/intact_tracking/memory350_rollout.py', 'src/intact_tracking/memory350_nominal_rollout.py',
        'src/intact_tracking/forward_predictor_objective.py', 'src/intact_tracking/forward_predictor_schedule.py',
        'src/intact_tracking/rollout/online.py', 'src/intact_tracking/rollout/nominal.py',
    ]
    changed = [name for name in protected if sha256(ROOT / name) != source_expected[name]]
    if changed:
        raise RuntimeError(f'Training or sampling implementation changed since u5000: {changed}')
    validation = {path.name: sha256(path) for path in sorted(output.glob('validation*rank_*.pt'))}
    assert len(validation) == 8
    result = {'passed': True, 'checkpoint': str(baseline), 'checkpoint_sha256': BASELINE_SHA256,
              'from_update': 5000, 'stop_after_updates': 10000, 'additional_updates': 5000,
              'optimizer_steps_restored': 20000, 'target_optimizer_steps': 40000,
              'restored_learning_rate': state['optimizer']['param_groups'][0]['lr'],
              'cosine_horizon_updates': 8000, 'continuation_min_learning_rate': 1e-5,
              'schedule_extension': None, 'training_implementation_unchanged': protected,
              'matched_control': contract, 'validation_sha256': validation,
              'simulator_replay_and_weak_archive_restarted': True, 'command': command,
              'checks_at_total_updates': list(MILESTONES), 'physical_gpus': [4, 5, 6, 7],
              'evaluation_gpu': 7, 'maximum_concurrent_evaluations': 1, 'unix_time': time.time()}
    write_json(journal / 'preflight.json', result)
    return result


def child_start(command, env, log_path):
    handle = log_path.open('x')
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        handle.close()
    record = {'pid': child.pid, 'command': command, 'physical_gpus': env['CUDA_VISIBLE_DEVICES'],
              'process_start_ticks': int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),
              'log': str(log_path), 'started_at': time.time()}
    return child, record


def refresh_report(root, env):
    journal = root / JOURNAL
    report_env = dict(env, PYTHONPATH='/tmp/intact-tsne-deps', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (journal / 'report.log').open('a') as log:
        result = subprocess.run([PYTHON, '-B', str(ROOT / 'scripts/report_memory350_weak_pairs_continuation.py'),
                                 '--run-root', str(root)], cwd=ROOT, env=report_env,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Could not refresh the convergence report; see report.log')


def poll_evaluation(root, env, active):
    journal = root / JOURNAL
    if active is not None:
        child, record, update = active
        if child.poll() is None:
            return active
        record.update(exit_code=child.returncode, finished_at=time.time())
        summary = read_json(journal / f'comparison_{update:06d}/summary.json')
        record['complete'] = child.returncode == 0 and bool(summary and summary.get('complete'))
        write_json(journal / f'evaluation_{update:06d}_result.json', record)
        refresh_report(root, env)
        print(json.dumps({'event': 'milestone_evaluation_finished', 'update': update,
                          'complete': record['complete'], 'assessment': summary.get('assessment') if summary else None}), flush=True)
    for update in MILESTONES:
        if (journal / f'evaluation_{update:06d}_result.json').exists():
            continue
        checkpoint = root / f'stage1_8192/update_{update:06d}.pt'
        if not checkpoint.exists():
            continue
        if gpu_status([7])[0]['free_mib'] < 12000:
            return None
        output = journal / f'comparison_{update:06d}'
        command = [PYTHON, '-B', '-u', str(ROOT / 'scripts/evaluate_memory350_weak_pairs.py'),
                   '--candidate', str(checkpoint), '--reference', str(root / 'stage1_8192/update_005000.pt'),
                   '--cache', read_json(root / 'launch_contract.json')['cached_trajectories'],
                   '--output', str(output), '--comparison-kind', 'continuation']
        eval_env = dict(env, CUDA_VISIBLE_DEVICES='7', PYTHONPATH='/tmp/intact-tsne-deps',
                        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        child, record = child_start(command, eval_env, journal / f'evaluation_{update:06d}.log')
        record['update'] = update
        write_json(journal / f'evaluation_{update:06d}_process.json', record)
        print(json.dumps({'event': 'milestone_evaluation_started', **record}), flush=True)
        return child, record, update
    return None


def verify_completion(root, plan):
    import torch
    output = root / 'stage1_8192'
    completion = read_json(output / 'completion.json')
    if not completion or completion['completed_updates'] != 10000 or completion['optimizer_steps'] != 40000 or not completion['hit_cap']:
        raise RuntimeError('Continuation did not complete exactly 5000 additional updates')
    config = read_json(output / 'run_config.json')
    resume = config['resume_history'][-1]
    assert resume['completed_update'] == 5000 and resume['checkpoint_sha256'] == BASELINE_SHA256
    assert resume['schedule_extension'] is None
    for name, digest in plan['validation_sha256'].items():
        assert sha256(output / name) == digest
    state = torch.load(output / 'update_010000.pt', map_location='cpu', weights_only=False, mmap=True)
    assert state['scheduler']['T_max'] == 32000 and state['scheduler']['last_epoch'] == 40000
    assert state['optimizer']['param_groups'][0]['lr'] == 1e-5
    assert all(int(value['step'].item()) == 40000 for value in state['optimizer']['state'].values())
    assert state['loss_config'] == read_json(root / JOURNAL / 'before/run_config.json')['loss']
    result = {'passed': True, 'completion': completion, 'learning_rate': 1e-5,
              'all_optimizer_state_steps': 40000, 'fixed_validation_unchanged': True,
              'checkpoint_sha256': sha256(output / 'update_010000.pt'), 'unix_time': time.time()}
    write_json(root / JOURNAL / 'completion_verification.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    journal = root / JOURNAL
    journal.mkdir(exist_ok=True)
    with (root / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (journal / 'training_process.json').exists():
            raise RuntimeError('This continuation has already been launched; inspect its recorded process')
        if (root / 'STOP').exists() or (journal / 'STOP').exists():
            raise RuntimeError('The run has a STOP marker')
        plan = preflight(root, journal)
        if args.prepare_only:
            print(json.dumps(plan), flush=True)
            return
        cards = gpu_status([4, 5, 6, 7])
        if any(card['free_mib'] < 60000 for card in cards):
            raise RuntimeError('Continuation needs at least 60000 MiB free on each GPU 4-7')
        write_json(journal / 'gpu_status_before.json', cards)
        before = journal / 'before'
        before.mkdir(exist_ok=True)
        for name in ('run_config.json', 'completion.json', 'history.json', 'metrics.jsonl',
                     'training_metrics.jsonl', 'progress.json', 'normalization.json', 'wandb_run.json'):
            if not (before / name).exists():
                shutil.copy2(root / 'stage1_8192' / name, before / name)
        if not (before / 'root_state.json').exists():
            shutil.copy2(root / 'state.json', before / 'root_state.json')
        sources = [ROOT / 'src/intact_tracking/cli/forward_memory_weak_pairs_train.py',
                   ROOT / 'scripts/evaluate_memory350_weak_pairs.py', Path(__file__).resolve(),
                   ROOT / 'scripts/report_memory350_weak_pairs_continuation.py']
        for source in sources:
            target = journal / 'source_snapshot' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write_json(journal / 'source_sha256.json', {str(p.relative_to(ROOT)): sha256(p) for p in sources})
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES='4,5,6,7', PYTHONUNBUFFERED='1',
                   WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
                   WANDB_RESUME='must', WANDB_RUN_ID=read_json(before / 'wandb_run.json')['id'])
        refresh_report(root, env)
        child, record = child_start(plan['command'], env, journal / 'training.log')
        write_json(journal / 'training_process.json', record)
        print(json.dumps({'event': 'continuation_launched', **record}), flush=True)
        active, sent_stop, training_checked = None, False, False
        try:
            while True:
                stopping = (root / 'STOP').exists() or (journal / 'STOP').exists()
                if stopping and not sent_stop:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                    if active is not None and active[0].poll() is None:
                        os.killpg(active[0].pid, signal.SIGTERM)
                    sent_stop = True
                if not stopping:
                    active = poll_evaluation(root, env, active)
                progress = read_json(root / 'stage1_8192/progress.json')
                results = {str(u): read_json(journal / f'evaluation_{u:06d}_result.json') for u in MILESTONES}
                state = {'status': 'continuation_training' if child.poll() is None else 'continuation_evaluating',
                         'launcher_pid': os.getpid(), 'job': record, 'progress': progress,
                         'active_evaluation': active[2] if active else None,
                         'completed_evaluations': [int(u) for u, r in results.items() if r and r['complete']],
                         'failed_evaluations': [int(u) for u, r in results.items() if r and not r['complete']],
                         'report': str(journal / 'README.md'), 'heartbeat': time.time()}
                write_json(journal / 'state.json', state)
                write_json(root / 'state.json', state)
                if child.poll() is not None:
                    if not training_checked:
                        record.update(exit_code=child.returncode, finished_at=time.time())
                        write_json(journal / 'training_process_result.json', record)
                        if not stopping:
                            if child.returncode:
                                raise RuntimeError(f'Continuation training failed with {child.returncode}')
                            verify_completion(root, plan)
                        training_checked = True
                    if stopping or (active is None and all(results.values())):
                        state['status'] = 'stopped' if stopping else ('complete_with_evaluation_failures' if state['failed_evaluations'] else 'complete')
                        state['heartbeat'] = time.time()
                        write_json(journal / 'state.json', state)
                        write_json(root / 'state.json', state)
                        break
                time.sleep(10)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            if active is not None and active[0].poll() is None:
                os.killpg(active[0].pid, signal.SIGTERM)
            raise


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import sys
        if '--run-root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--run-root') + 1])
            journal = root / JOURNAL
            if journal.exists():
                write_json(journal / 'launcher_error.json', {'error': repr(error), 'unix_time': time.time()})
                if (journal / 'training_process.json').exists():
                    state = read_json(journal / 'state.json') or {}
                    state.update(status='failed', error=repr(error), heartbeat=time.time())
                    write_json(journal / 'state.json', state)
                    write_json(root / 'state.json', state)
        raise
