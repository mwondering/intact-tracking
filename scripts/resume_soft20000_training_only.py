"""Resume the existing eight-GPU PPO pair without any scheduled evaluations."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from monitor_memory350_nominal_direction import process_identity
from run_limb_context_experiment import ROOT, process_environment
from run_memory350_scale_nominal_stage1 import gpu_status
from soft20000_warm_bounded_common import read, write, digest


ARMS = {'concat': [0, 1, 2, 3], 'baseline': [4, 5, 6, 7]}


def output_directory(root, arm):
    return root / 'ppo' / f'{arm}_121_train_only'


def verify_started(root, sources):
    target = root / 'training_only_migration' / 'resume_verification.json'
    if target.exists():
        return True
    if not all((output_directory(root, arm) / 'run_config.json').exists() for arm in ARMS):
        return False
    checks, evidence = {}, {}
    for arm in ARMS:
        config = read(output_directory(root, arm) / 'run_config.json')
        audit = config['input_audit']
        checks[arm + '_restored'] = bool(config.get('resume_state_audit', {}).get('passed'))
        checks[arm + '_resume_update'] = config['resume_state_audit']['checkpoint_update'] == sources[arm]['completed_updates']
        checks[arm + '_no_evaluations'] = not config.get('periodic_evaluation') and config['arguments']['endpoint_eval_protocol'] is None
        checks[arm + '_scale'] = config['distributed']['world_size'] == 4 and config['distributed']['num_envs_per_rank'] == 8192
        checks[arm + '_history5'] = audit['latent_history_frames'] == 5 and audit['actor_fusion_input_dim'] == audit['critic_fusion_input_dim'] == 448
        checks[arm + '_uniform'] = config['motion_sampling']['active_mode'] == 'uniform'
        checks[arm + '_termination'] = config['arguments']['training_terminations'] == 'original'
        checks[arm + '_unbounded'] = config['arguments']['until_user_stop'] and config['arguments']['iterations'] is None
        checks[arm + '_precision'] = config['policy_precision'] == 'fp32'
        checks[arm + '_nominal'] = all(x['physics']['nominal_population']['count'] == 820
            for x in config['physics']['runtime_audits_by_rank'])
        checks[arm + '_mixture'] = config['physics']['independent_nominal_mixture']['probability'] == .5
        evidence[arm] = {'resume_state_audit': config['resume_state_audit'],
                         'periodic_evaluation': config.get('periodic_evaluation')}
    result = {'passed': all(checks.values()), 'checks': checks, 'evidence': evidence}
    write(target, result)
    if not result['passed']:
        raise RuntimeError(f'Training-only startup mismatch: {[k for k, v in checks.items() if not v]}')
    return True


def read_progress(path, previous):
    """Retain the last observation if a concurrently replaced file is unavailable."""
    for attempt in range(3):
        try:
            return read(path), None
        except (FileNotFoundError, json.JSONDecodeError) as error:
            if attempt == 2:
                return dict(previous), {'path': str(path), 'error': repr(error), 'stale': True}
            time.sleep(.1)


def monitor(root, sources, processes=None, *, arms=None):
    assignments = {arm: ARMS[arm] for arm in (arms or ARMS)}
    records = {arm: read(root / f'ppo_{arm}_train_only_process.json') for arm in assignments}
    progress = {arm: {'completed_updates': sources[arm]['completed_updates']} for arm in assignments}
    write(root / 'coordinator_launch.json', process_identity(os.getpid()))
    while True:
        for arm, record in records.items():
            if processes is not None and processes[arm].poll() is not None:
                raise RuntimeError(f'{arm} PPO exited: {processes[arm].returncode}; no automatic restart')
            if not process_identity(record['pid'], record['start_ticks'])['live']:
                raise RuntimeError(f'{arm} PPO leader exited or changed identity; no automatic restart')
        verified = verify_started(root, sources)
        read_errors = {}
        for arm in assignments:
            progress[arm], error = read_progress(output_directory(root, arm) / 'progress.json', progress[arm])
            if error is not None:
                read_errors[arm] = error
        status = ('paired_ppo_training_only' if len(assignments) == 2 else 'ppo_training_only')
        write(root / 'state.json', {'status': status if verified else 'resuming_training_only',
            'unix_time': time.time(), 'progress': progress, 'maximum_updates': None,
            'assignments': assignments, 'inactive_arms': [arm for arm in ARMS if arm not in assignments],
            'evaluations_enabled': False, 'progress_read_errors': read_errors})
        time.sleep(30)


def run(root):
    migration = root / 'training_only_migration'
    sources = read(migration / 'sources.json')
    commands = copy.deepcopy(read(root / 'formal_commands.json')['commands'])
    for arm in ARMS:
        previous = read(root / f'ppo_{arm}_warm_bounded_process.json')
        if process_identity(previous['pid'], previous['start_ticks'])['live']:
            raise RuntimeError('Previous PPO leader is still running')
        if (root / f'ppo_{arm}_train_only_process.json').exists():
            raise RuntimeError('A training-only launch record already exists; inspect instead of restarting')
        if digest(sources[arm]['checkpoint']) != sources[arm]['sha256']:
            raise RuntimeError('Resume checkpoint changed')
        command = commands[arm]
        index = command.index('--endpoint-eval-protocol')
        del command[index:index + 2]
        command[command.index('--output-dir') + 1] = str(output_directory(root, arm))
        command += ['--resume', sources[arm]['checkpoint']]
        sources[arm]['wandb'] = read(root / 'ppo' / f'{arm}_121_warm_bounded' / 'wandb_run.json')
    cards = gpu_status(list(range(8)))
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise RuntimeError('GPUs still have processes; inspect without restarting jobs')
    before = migration / 'before'
    before.mkdir(exist_ok=True)
    for name in ['state.json', 'active_evaluation.json', 'coordinator_launch.json', 'latest_paired_result.json']:
        source = root / name
        if source.exists() and not (before / name).exists():
            shutil.copy2(source, before / name)
    write(migration / 'launch.json', {'commands': commands, 'sources': sources, 'assignments': ARMS,
                                    'maximum_updates': None, 'evaluations_enabled': False})
    write(migration / 'gpus_before_launch.json', cards)
    processes = {}
    for arm, command in commands.items():
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, ARMS[arm])), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
            WANDB_RUN_ID=sources[arm]['wandb']['id'], WANDB_RESUME='must')
        log = root / f'ppo_{arm}_train_only_stdout.log'
        with log.open('ab') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        identity = process_identity(process.pid)
        identity.update(output_dir=str(output_directory(root, arm)), log=str(log), resumed_from=sources[arm])
        write(root / f'ppo_{arm}_train_only_process.json', identity)
        processes[arm] = process
    reason = 'User requested all evaluations disabled and maximum training speed'
    write(root / 'evaluations_disabled.json', {'enabled': False, 'reason': reason, 'unix_time': time.time(),
        'periodic_evaluations': False, 'independent_diagnostics': False, 'planned_u2000_evaluation': 'paused by user'})
    write(root / 'active_evaluation.json', {'enabled': False, 'reason': reason,
        'outputs': {arm: str(output_directory(root, arm)) for arm in ARMS}})
    monitor(root, sources, processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--monitor-only', action='store_true',
                        help='Attach to the recorded live trainers without launching or restarting training')
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS),
                        help='Monitor only these existing PPO arms; requires --monitor-only')
    args = parser.parse_args()
    if args.arms and not args.monitor_only:
        parser.error('--arms requires --monitor-only; it never changes training launch assignments')
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('Run root must stay within the repository')
    with (root / 'coordinator_train_only.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'coordinator_train_only_process.json', process_identity(os.getpid()))
        try:
            if args.monitor_only:
                monitor(root, read(root / 'training_only_migration/sources.json'), arms=args.arms)
            else:
                run(root)
        except BaseException as error:
            write(root / 'training_only_migration/coordinator_error.json', {'error': repr(error), 'unix_time': time.time()})
            raise


if __name__ == '__main__':
    main()
