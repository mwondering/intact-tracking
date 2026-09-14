"""Supervise u10000->u12000 weak weights 0.008 and margin 1.1 on GPUs 4-7."""

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
from resume_memory350_weak_pairs import child_start, sha256
from intact_tracking.memory350_weak_margin_tuning import MILESTONES, PARENT_UPDATE, TUNED_LOSS, TARGET_TOP1


def tuning_command(root, parent):
    command = list(read_json(parent / 'launch_contract.json')['formal_command'])
    old_module = 'intact_tracking.cli.forward_memory_weak_tune_train'
    command[command.index(old_module)] = 'intact_tracking.cli.forward_memory_weak_margin_train'
    options = {'output-dir': str(root / 'stage1_8192'), 'stop-after-updates': str(MILESTONES[-1]),
               'resume': str(parent / f'stage1_8192/update_{PARENT_UPDATE:06d}.pt'),
               'comparison-reference-dir': str(parent / 'stage1_8192'),
               'wandb-group': root.name, 'wandb-name': root.name + '-stage1_8192',
               **{key.replace('_', '-'): str(value) for key, value in TUNED_LOSS.items()}}
    for key, value in options.items():
        flag = '--' + key
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command += [flag, value]
    return command


def prepare(root, parent):
    import torch
    from intact_tracking.cli.forward_memory_weak_margin_train import build_parser, reference_contract
    from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_arguments
    parent_stage = parent / 'stage1_8192'
    checkpoint = parent_stage / f'update_{PARENT_UPDATE:06d}.pt'
    evaluation_path = parent / f'comparison_{PARENT_UPDATE:06d}/summary.json'
    evaluation = read_json(evaluation_path)
    if not evaluation or not evaluation.get('complete'):
        raise RuntimeError('The parent u10000 evaluation must finish before margin tuning is prepared')
    digest = sha256(checkpoint)
    if evaluation['checkpoints']['memory350']['sha256'] != digest:
        raise RuntimeError('The evaluated parent checkpoint changed')
    review = read_json(evaluation_path.with_name('review_verification.json'))
    if not review or not review.get('passed') or review['checkpoint_sha256'] != digest:
        raise RuntimeError('The parent u10000 independent review must pass before margin tuning')
    previous = read_json(parent_stage / 'run_config.json')
    command = tuning_command(root, parent)
    index = command.index('intact_tracking.cli.forward_memory_weak_margin_train')
    args = build_parser().parse_args(command[index + 1:])
    _validate_arguments(args)
    actual = deepcopy(previous)
    actual['arguments'] = vars(args)
    actual['loss'].update(TUNED_LOSS)
    actual['objective_weights'].update(TUNED_LOSS)
    contract = reference_contract(args.comparison_reference_dir, actual)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert state['update'] == PARENT_UPDATE and state['optimizer_steps'] == 4 * PARENT_UPDATE
    assert state['scheduler']['T_max'] == 32000 and state['scheduler']['last_epoch'] == 4 * PARENT_UPDATE
    assert state['optimizer']['param_groups'][0]['lr'] == args.continuation_min_learning_rate == 1e-5
    assert all(int(value['step'].item()) == 4 * PARENT_UPDATE for value in state['optimizer']['state'].values())
    output = root / 'stage1_8192'
    output.mkdir(exist_ok=True)
    if (output / 'progress.json').exists() or (root / 'training_process.json').exists():
        raise RuntimeError('This tuning branch has already started')
    files = sorted(parent_stage.glob('validation*rank_*.pt'))
    assert len(files) == 8
    validation = {}
    for source in files:
        target = output / source.name
        if not target.exists():
            shutil.copy2(source, target)
        validation[source.name] = sha256(source)
        assert sha256(target) == validation[source.name]
    shutil.copy2(parent_stage / 'normalization.json', output / 'normalization.json')
    write_json(output / 'run_config.json', previous)
    history = [row for row in read_json(parent_stage / 'history.json') if row['update'] <= PARENT_UPDATE]
    write_json(output / 'history.json', history)
    plan = {'parent_checkpoint': str(checkpoint), 'parent_sha256': digest,
            'parent_evaluation_summary': str(evaluation_path), 'parent_evaluation_sha256': sha256(evaluation_path),
            'from_update': PARENT_UPDATE, 'stop_after_updates': MILESTONES[-1], 'additional_updates': 2000,
            'evaluation_updates': list(MILESTONES), 'loss_changes': actual['tuning_stage']['loss_changes'],
            'primary_metric': actual['tuning_stage']['primary_metric'], 'profiles': ['common', 'memory_training'],
            'primary_profile': 'memory_training', 'target_top1': TARGET_TOP1,
            'optimizer_state_restored': True, 'learning_rate_schedule_unchanged': True, 'learning_rate': 1e-5,
            'validation_sha256': validation, 'matched_control': contract,
            'cached_trajectories': read_json(parent / 'launch_contract.json')['cached_trajectories'],
            'physical_gpus': [4, 5, 6, 7], 'evaluation_gpu': 7, 'maximum_concurrent_evaluations': 1,
            'formal_command': command, 'prepared_at': time.time()}
    write_json(root / 'launch_contract.json', plan)
    write_json(root / 'preflight.json', dict(plan, passed=True))
    sources = sorted((ROOT / 'src/intact_tracking').rglob('*.py')) + [
        Path(__file__).resolve(), ROOT / 'scripts/evaluate_memory350_weak_pairs.py',
        ROOT / 'scripts/evaluate_memory350_weak_margin.py',
        ROOT / 'scripts/verify_memory350_weak_margin_startup.py',
        ROOT / 'scripts/report_memory350_weak_margin_tuning.py']
    write_json(root / 'source_sha256.json', {str(p.relative_to(ROOT)): sha256(p) for p in sources})
    for source in sources:
        target = root / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return plan


def refresh_report(root, env):
    with (root / 'report.log').open('a') as log:
        subprocess.run([PYTHON, '-B', str(ROOT / 'scripts/report_memory350_weak_margin_tuning.py'),
                        '--run-root', str(root)], cwd=ROOT,
                       env=dict(env, PYTHONPATH='/tmp/intact-tsne-deps', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def poll_evaluation(root, env, plan, active):
    if active is not None:
        child, record, update = active
        if child.poll() is None:
            return active
        summary = read_json(root / f'comparison_{update:06d}/summary.json')
        record.update(exit_code=child.returncode, finished_at=time.time(),
                      complete=child.returncode == 0 and bool(summary and summary.get('complete')))
        write_json(root / f'evaluation_{update:06d}_result.json', record)
        refresh_report(root, env)
    for update in MILESTONES:
        if (root / f'evaluation_{update:06d}_result.json').exists():
            continue
        checkpoint = root / f'stage1_8192/update_{update:06d}.pt'
        if not checkpoint.exists():
            continue
        if gpu_status([7])[0]['free_mib'] < 12000:
            return None
        command = [PYTHON, '-B', '-u', str(ROOT / 'scripts/evaluate_memory350_weak_margin.py'),
                   '--candidate', str(checkpoint), '--reference', plan['parent_checkpoint'],
                   '--cache', plan['cached_trajectories'], '--output', str(root / f'comparison_{update:06d}'),
                   '--comparison-kind', 'tuning']
        eval_env = dict(env, CUDA_VISIBLE_DEVICES='7', PYTHONPATH='/tmp/intact-tsne-deps',
                        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        child, record = child_start(command, eval_env, root / f'evaluation_{update:06d}.log')
        record['update'] = update
        write_json(root / f'evaluation_{update:06d}_process.json', record)
        return child, record, update
    return None


def verify_completion(root, plan):
    import torch
    from evaluate_memory350_weak_margin import validate_checkpoints
    output = root / 'stage1_8192'
    completion = read_json(output / 'completion.json')
    assert completion['completed_updates'] == MILESTONES[-1] and completion['optimizer_steps'] == 4 * MILESTONES[-1] and completion['hit_cap']
    paths = {'baseline': Path(plan['parent_checkpoint']), 'memory350': output / f'update_{MILESTONES[-1]:06d}.pt'}
    states = {key: torch.load(path, map_location='cpu', weights_only=False, mmap=True) for key, path in paths.items()}
    validate_checkpoints(states, 'tuning')
    assert sha256(paths['baseline']) == plan['parent_sha256']
    assert all(int(value['step'].item()) == 4 * MILESTONES[-1] for value in states['memory350']['optimizer']['state'].values())
    for name, digest in plan['validation_sha256'].items():
        assert sha256(output / name) == digest
    write_json(root / 'completion_verification.json', {'passed': True, 'completion': completion,
        'checkpoint_sha256': sha256(paths['memory350']), 'fixed_validation_unchanged': True, 'unix_time': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root, parent = args.run_root.resolve(), args.parent_root.resolve()
    root.mkdir(exist_ok=True)
    with (root / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'STOP').exists():
            raise RuntimeError('The tuning branch has a STOP marker')
        plan = prepare(root, parent)
        env = process_environment()
        if args.prepare_only:
            refresh_report(root, env)
            print(json.dumps({'prepared': True, 'from_update': PARENT_UPDATE, 'checks': list(MILESTONES)}), flush=True)
            return
        cards = gpu_status([4, 5, 6, 7])
        if any(card['free_mib'] < 60000 for card in cards):
            raise RuntimeError('Margin tuning needs at least 60000 MiB free on each of GPUs 4-7')
        write_json(root / 'gpu_status_before.json', cards)
        env.update(CUDA_VISIBLE_DEVICES='4,5,6,7', PYTHONUNBUFFERED='1',
                   WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
                   WANDB_RESUME='never', WANDB_RUN_ID='m350margin-' + hashlib.sha256(str(root).encode()).hexdigest()[:12])
        refresh_report(root, env)
        child, record = child_start(plan['formal_command'], env, root / 'training.log')
        write_json(root / 'training_process.json', record)
        print(json.dumps({'event': 'tuning_launched', **record}), flush=True)
        active, sent_stop, checked = None, False, False
        try:
            while True:
                stopping = (root / 'STOP').exists()
                if stopping and not sent_stop:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                    if active is not None and active[0].poll() is None:
                        os.killpg(active[0].pid, signal.SIGTERM)
                    sent_stop = True
                if not stopping:
                    active = poll_evaluation(root, env, plan, active)
                results = {str(u): read_json(root / f'evaluation_{u:06d}_result.json') for u in MILESTONES}
                state = {'status': 'tuning_training' if child.poll() is None else 'tuning_evaluating',
                         'launcher_pid': os.getpid(), 'job': record,
                         'progress': read_json(root / 'stage1_8192/progress.json'),
                         'active_evaluation': active[2] if active else None,
                         'completed_evaluations': [int(u) for u, r in results.items() if r and r['complete']],
                         'failed_evaluations': [int(u) for u, r in results.items() if r and not r['complete']],
                         'report': str(root / 'README.md'), 'heartbeat': time.time()}
                write_json(root / 'state.json', state)
                if child.poll() is not None:
                    if not checked:
                        record.update(exit_code=child.returncode, finished_at=time.time())
                        write_json(root / 'training_process_result.json', record)
                        if not stopping:
                            if child.returncode:
                                raise RuntimeError(f'Tuning training failed with {child.returncode}')
                            verify_completion(root, plan)
                        checked = True
                    if stopping or (active is None and all(results.values())):
                        state['status'] = 'stopped' if stopping else ('complete_with_evaluation_failures' if state['failed_evaluations'] else 'complete')
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
            if root.exists():
                write_json(root / 'launcher_error.json', {'error': repr(error), 'unix_time': time.time()})
        raise
