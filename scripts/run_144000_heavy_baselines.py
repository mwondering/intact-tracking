"""Four-GPU baseline controller: continuous uniform PPO, without a stage boundary.

Default prints a reviewable command plan. --run executes it; --smoke performs
two updates at the SAME 4x8192 scale with a short motion and disabled W&B.
Existing trainers are never stopped or oversubscribed by this launcher.
Uniform sampling is the default; the historical adaptive protocol remains explicit.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from monitor_memory350_nominal_direction import process_identity
from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT/'.venv/bin/python')
BOUNDARY = 2001


def command_for(method, stage, directory, *, resume=None, reset=False, smoke=False, motion_sampling='uniform'):
    result = [PYTHON, '-B', '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
              '--max-restarts=0', '-m', 'intact_tracking.cli.heavy_baseline_train',
              '--method', method, '--stage', stage, '--output-dir', str(directory),
              '--training-ranks', '4', '--num-envs', '8192', '--save-interval', '100',
              '--motion-sampling', motion_sampling,
              '--wandb-group', '144000-exp-heavy', '--wandb-name', '144000-exp-heavy-'+method.replace('_', '-')
              + ('-uniform' if motion_sampling == 'uniform' else '')]
    if smoke:
        result += ['--bounded-smoke', '--iterations', '2', '--save-interval', '1', '--motion-file',
                   str(ROOT/'runs/144000_exp/smoke_motions/walk1_subject1.motion.npz')]
    else:
        result += ['--motion-path', '/data_zcy/wxy/motion_data_correct']
    if resume:
        result += ['--resume', str(resume)]
    if reset:
        if motion_sampling != 'adaptive':
            raise ValueError('Uniform training has no planned sampler reset')
        result += ['--reset-adaptive-sampling']
    return result


def latest_checkpoint(directory):
    candidates = sorted(directory.glob('checkpoint_*.pt'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return candidates[0] if candidates else None


def preflight_valid(root, method, motion_sampling='uniform'):
    record = root/'preflight.json'
    if not record.exists():
        return False
    audit = json.loads(record.read_text())
    if audit.get('method') != method or audit.get('motion_sampling', 'adaptive') != motion_sampling:
        return False
    for name, expected in audit['source_sha256'].items():
        with (ROOT/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                return False
    return audit['passed']


def check_preflight(directory, root):
    completion = json.loads((directory/'completion.json').read_text())
    config = json.loads((directory/'run_config.json').read_text())
    if (completion['completed_updates'] != 2 or not completion['complete'] or completion['stopped']
            or not completion['distributed_parameter_agreement']['passed']
            or completion['distributed']['world_size'] != 4
            or completion['distributed']['num_envs_per_rank'] != 8192):
        raise RuntimeError('Formal 4x8192 preflight did not pass')
    (root/'preflight.json').write_text(json.dumps({
        'passed': True, 'method': config['method'], 'motion_sampling': config['motion_sampling']['active_mode'],
        'directory': str(directory), 'source_sha256': config['research_source_sha256'],
        'agreement': completion['distributed_parameter_agreement'], 'checked_at': time.time()}, indent=2)+'\n')


def select_stage(root, motion_sampling='adaptive'):
    """Once stage B has any checkpoint, NEVER repeat its sampler reset."""
    import torch
    if motion_sampling == 'uniform':
        checkpoint = latest_checkpoint(root/'continuous_uniform')
        if checkpoint:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            meta = saved['residual_policy']
            if (meta.get('baseline_stage') != 'continuous_uniform' or meta.get('sampling_reset_generation', 0) != 0
                    or meta['motion_sampling']['active_mode'] != 'uniform'):
                raise ValueError('Continuous uniform training must restore its own checkpoint')
        return 'continuous_uniform', checkpoint, False
    second = latest_checkpoint(root/'resume_sampling_reset')
    if second:
        saved = torch.load(second, map_location='cpu', weights_only=False, mmap=True)
        meta = saved['residual_policy']
        if meta['sampling_reset_generation'] != 1:
            raise ValueError('Stage B checkpoint has no completed sampling refresh')
        return 'resume_sampling_reset', second, False
    first = latest_checkpoint(root/'stage1')
    if first:
        saved = torch.load(first, map_location='cpu', weights_only=False, mmap=True)
        stored_mode = saved.get('residual_policy', {}).get('motion_sampling', {}).get('active_mode')
        if stored_mode is not None and stored_mode != motion_sampling:
            raise ValueError('An existing baseline cannot change its motion sampling protocol')
        if saved['completed_updates'] > BOUNDARY:
            raise ValueError('Stage A passed its required checkpoint boundary')
        if saved['completed_updates'] == BOUNDARY:
            return 'resume_sampling_reset', first, True
    return 'stage1', first, False


def launch(method, stage, directory, gpus, *, resume, reset, smoke, motion_sampling='uniform'):
    cards = gpu_status(gpus)
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise RuntimeError(f'GPUs {gpus} must be free for a 4x8192 baseline; existing jobs were left running')
    command = command_for(method, stage, directory, resume=resume, reset=reset, smoke=smoke, motion_sampling=motion_sampling)
    directory.parent.mkdir(parents=True, exist_ok=True)
    attempt = str(time.time_ns())
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               WANDB_MODE='disabled' if smoke else 'online', WANDB_RESUME='allow',
               WANDB_RUN_ID='heavy-'+method+'-'+hashlib.sha256(str(directory).encode()).hexdigest()[:10])
    key = ROOT/'.runtime/limb_context/wandb_api_key'
    if key.exists():
        env['WANDB_API_KEY'] = key.read_text().strip()
    log = directory.parent/f'{stage}_{attempt}.log'
    with log.open('ab') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(child.pid)
    record = {**identity, 'method': method, 'stage': stage, 'command': command, 'gpus': gpus,
              'log': str(log), 'output': str(directory), 'sampling_reset_requested': reset and motion_sampling == 'adaptive',
              'motion_sampling': motion_sampling,
              'parent_checkpoint': str(resume) if resume else None, 'wandb_id': env['WANDB_RUN_ID'],
              'started_at': time.time(), 'maximum_completed_updates': 2 if smoke else BOUNDARY if stage == 'stage1' else None}
    record_path = directory.parent/'active_process.json'
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        # Signal workers, not torchrun first: let every rank finish its complete
        # update and checkpoint before the launcher tears down the process group.
        children = Path(f'/proc/{child.pid}/task/{child.pid}/children')
        if children.exists():
            for worker in children.read_text().split():
                try:
                    os.kill(int(worker), signal.SIGTERM)
                except ProcessLookupError:
                    pass
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = child.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    record.update(returncode=code, ended_at=time.time(), controller_stop_requested=stopping)
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    if code:
        raise RuntimeError(f'{method}/{stage} exited with {code}; inspect {log}')
    return not stopping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('rma_teacher', 'any2track'), required=True)
    parser.add_argument('--gpus', help='Four comma-separated physical GPUs; default 0-3 for RMA, 4-7 for Any2Track')
    parser.add_argument('--root')
    parser.add_argument('--motion-sampling', choices=('uniform', 'adaptive'), default='uniform')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    gpus = list(map(int, (args.gpus or ('0,1,2,3' if args.method == 'rma_teacher' else '4,5,6,7')).split(',')))
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
        raise ValueError('Exactly four distinct physical GPUs are required')
    root = Path(args.root).resolve() if args.root else ROOT/'runs/144000-exp-heavy/baselines'/(
        args.method + ('_uniform' if args.motion_sampling == 'uniform' else ''))
    if not root.is_relative_to(ROOT):
        raise ValueError('Baseline artifacts must remain in the repository')
    if not args.run:
        plan = {'method': args.method, 'gpus': gpus, 'motion_sampling': args.motion_sampling}
        if args.motion_sampling == 'uniform':
            plan.update(maximum_completed_updates=None, planned_restart=False,
                        command=command_for(args.method, 'continuous_uniform', root/'continuous_uniform'))
        else:
            plan.update(stage1=command_for(args.method, 'stage1', root/'stage1', motion_sampling='adaptive'),
                        stage2=command_for(args.method, 'resume_sampling_reset', root/'resume_sampling_reset',
                                           resume=root/'stage1/checkpoint_2000.pt', reset=True, motion_sampling='adaptive'))
        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_stage = 'continuous_uniform' if args.motion_sampling == 'uniform' else 'stage1'
        if args.smoke:
            directory = root/'smoke_4gpu8192'
            launch(args.method, initial_stage, directory, gpus, resume=None, reset=False, smoke=True,
                   motion_sampling=args.motion_sampling)
            check_preflight(directory, root)
            return
        if not preflight_valid(root, args.method, args.motion_sampling):
            directory = root/f'smoke_4gpu8192_{time.time_ns()}'
            if not launch(args.method, initial_stage, directory, gpus, resume=None, reset=False, smoke=True,
                          motion_sampling=args.motion_sampling):
                return
            check_preflight(directory, root)
        while True:
            stage, resume, reset = select_stage(root, args.motion_sampling)
            keep_running = launch(args.method, stage, root/stage, gpus, resume=resume, reset=reset, smoke=False,
                                  motion_sampling=args.motion_sampling)
            completion = json.loads((root/stage/'completion.json').read_text())
            if not keep_running or completion['stopped'] or stage != 'stage1':
                return
            if completion['completed_updates'] != BOUNDARY or not completion['complete']:
                raise RuntimeError('Stage A did not reach the exact sampling-refresh boundary')


if __name__ == '__main__':
    main()
