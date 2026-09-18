"""Resume the saved PPO pair with corrected warm tests and keep monitoring it."""
import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time

from run_limb_context_experiment import ROOT, process_environment
from monitor_memory350_nominal_direction import process_identity
from run_memory350_scale_nominal_stage1 import gpu_status
from soft20000_warm_bounded_common import (
    output_directory, protocol_path, evaluation_history, paired_report, read, write, digest)
from intact_tracking.memory350_policy_checkpoint_eval import load_protocol, resumed_evaluation_metadata


ARMS = {'concat': [0, 1, 2, 3], 'baseline': [4, 5, 6, 7]}


def prepare(root):
    import torch
    protocol, _ = load_protocol(protocol_path(root))
    selected = read(root / 'selected_context.json')
    if digest(selected['checkpoint']) != selected['sha256']:
        raise ValueError('Selected encoder changed')
    commands = copy.deepcopy(read(root / 'formal_commands.json')['commands'])
    evidence = {}
    for arm in ARMS:
        prior = read(root / f'ppo_{arm}_process.json')
        if process_identity(prior['pid'], prior['start_ticks'])['live']:
            raise RuntimeError('Old PPO is still running; do not duplicate it')
        if (root / f'ppo_{arm}_warm_bounded_process.json').exists():
            raise RuntimeError('A corrected PPO launch record already exists; inspect rather than restart')
        source = root / 'ppo' / f'{arm}_121'
        completion = read(source / 'completion.json')
        checkpoint = source / 'checkpoint_final.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if (not completion['stopped'] or not completion['distributed_parameter_agreement']['passed']
                or saved['completed_updates'] != completion['completed_updates']):
            raise ValueError('Missing complete, synchronized final save')
        old = saved['residual_policy']
        if arm == 'concat' and old['context_sha256'] != selected['sha256']:
            raise ValueError('Saved PPO uses a different encoder')
        current = {'protocol': protocol, 'protocol_file': str(protocol_path(root)),
                   'protocol_sha256': digest(protocol_path(root))}
        revision = resumed_evaluation_metadata(old['periodic_evaluation'], current,
                                               saved['completed_updates'], allow_change=True)
        command = commands[arm]
        command[command.index('--output-dir') + 1] = str(output_directory(root, arm))
        command[command.index('--endpoint-eval-protocol') + 1] = str(protocol_path(root))
        command += ['--resume', str(checkpoint), '--allow-evaluation-protocol-change']
        evidence[arm] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
            'completed_updates': saved['completed_updates'], 'revision': revision,
            'wandb': read(source / 'wandb_run.json')}
        del saved
    result = {'assignments': ARMS, 'commands': commands, 'sources': evidence,
              'maximum_updates': None, 'protocol_sha256': digest(protocol_path(root))}
    write(root / 'evaluation_migration' / 'resume_commands.json', result)
    return result


def verify_started(root, launch):
    target = root / 'evaluation_migration' / 'resume_verification.json'
    if target.exists():
        return True
    if not all((output_directory(root, arm) / 'run_config.json').exists() for arm in ARMS):
        return False
    checks, evidence = {}, {}
    for arm in ARMS:
        config = read(output_directory(root, arm) / 'run_config.json')
        source = launch['sources'][arm]
        checks[arm + '_restored'] = bool(config.get('resume_state_audit', {}).get('passed'))
        checks[arm + '_update'] = config['resume_state_audit']['checkpoint_update'] == source['completed_updates']
        checks[arm + '_scale'] = config['distributed']['world_size'] == 4 and config['distributed']['num_envs_per_rank'] == 8192
        checks[arm + '_history'] = (config['input_audit']['actor_fusion_input_dim'] == 448
            and config['input_audit']['critic_fusion_input_dim'] == 448
            and config['input_audit']['latent_history_frames'] == 5)
        checks[arm + '_uniform'] = config['motion_sampling']['active_mode'] == 'uniform'
        checks[arm + '_termination'] = config['arguments']['training_terminations'] == 'original'
        checks[arm + '_unbounded'] = config['arguments']['until_user_stop'] and config['arguments']['iterations'] is None
        checks[arm + '_protocol'] = config['periodic_evaluation']['protocol_sha256'] == launch['protocol_sha256']
        evidence[arm] = {'resume_state_audit': config['resume_state_audit'],
                        'periodic_evaluation': config['periodic_evaluation']}
    result = {'passed': all(checks.values()), 'checks': checks, 'evidence': evidence}
    write(target, result)
    if not result['passed']:
        raise ValueError('Corrected PPO startup audit failed')
    return True


def run(root):
    launch = prepare(root)
    if not read(root / 'evaluation_migration/corrected_backfill.json')['passed']:
        raise ValueError('Corrected full-scale warm evaluation has not passed')
    cards = gpu_status(list(range(8)))
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise RuntimeError('Training GPUs are not free; inspect rather than kill other jobs')
    write(root / 'evaluation_migration/gpu_before_resume.json', cards)
    processes = {}
    for arm, command in launch['commands'].items():
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, ARMS[arm])), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
            WANDB_RUN_ID=launch['sources'][arm]['wandb']['id'], WANDB_RESUME='must')
        log = root / f'ppo_{arm}_warm_bounded_stdout.log'
        with log.open('ab') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        identity = process_identity(process.pid)
        identity.update(output_dir=str(output_directory(root, arm)), log=str(log), resumed_from=launch['sources'][arm])
        write(root / f'ppo_{arm}_warm_bounded_process.json', identity)
        processes[arm] = process
    write(root / 'active_evaluation.json', {'version': load_protocol(protocol_path(root))[0]['version'],
        'protocol': str(protocol_path(root)), 'outputs': {arm: str(output_directory(root, arm)) for arm in ARMS},
        'process_records': {arm: str(root / f'ppo_{arm}_warm_bounded_process.json') for arm in ARMS},
        'paired_results': str(root / 'paired_results_warm_bounded_v2')})
    while True:
        for arm, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f'{arm} PPO exited: {process.returncode}; no automatic restart')
        verified = verify_started(root, launch)
        histories = {arm: evaluation_history(root, arm) for arm in ARMS}
        matched = sorted(histories['concat'].keys() & histories['baseline'].keys())
        for update in matched:
            report = paired_report(root, update)
        if matched:
            write(root / 'latest_paired_result.json', {'completed_updates': matched[-1], 'path': str(report),
                'sha256': digest(report), 'evaluation_protocol_revision': 'memory350_periodic_warm_bounded_v2',
                'stop_at_comparison': False, 'unix_time': time.time()})
        progress = {}
        for arm in ARMS:
            path = output_directory(root, arm) / 'progress.json'
            progress[arm] = read(path) if path.exists() else {'completed_updates': launch['sources'][arm]['completed_updates']}
        write(root / 'state.json', {'status': 'paired_ppo_training_warm_bounded' if verified else 'resuming_warm_bounded_ppo',
            'unix_time': time.time(), 'progress': progress, 'maximum_updates': None,
            'assignments': ARMS, 'protocol': str(protocol_path(root)), 'matched_updates': matched})
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('Run root must be inside the repository')
    if args.check:
        prepare(root)
        print('Saved checkpoints and evaluation-only revision validated; no launch', flush=True)
        return
    with (root / 'coordinator_warm_bounded.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = process_identity(os.getpid())
        write(root / 'coordinator_warm_bounded_process.json', identity)
        write(root / 'coordinator_launch.json', identity)
        try:
            run(root)
        except BaseException as error:
            write(root / 'coordinator_error.json', {'phase': 'warm_bounded_resume', 'error': repr(error), 'unix_time': time.time()})
            raise


if __name__ == '__main__':
    main()
