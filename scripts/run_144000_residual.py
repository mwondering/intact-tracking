"""Launch noisy-proprio stage two: five latents plus tracker output in both heads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from monitor_memory350_nominal_direction import process_identity
from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/144000_exp'
CONTEXT = RUN / 'stage1_proprio122_8192/update_007179.pt'
CONTEXT_SHA256 = 'a799a059934acd27e1bea6924897087dec433787873c1a47e0638999c2e48e89'
STAGE2 = RUN / 'stage2_proprio122_history5_tracker_action'


def command_for(phase, output, *, resume=None):
    smoke = phase == 'smoke'
    source = json.loads((RUN/'source_identity.json').read_text())
    command = [str(ROOT/'.venv/bin/python'), '-B', '-u', '-m', 'torch.distributed.run',
               '--standalone', '--nproc-per-node=8', '--max-restarts=0', '-m',
               'intact_tracking.cli.memory350_proprio_native_policy_train',
               '--fusion', 'concat', '--training-ranks', '8', '--num-envs', '256' if smoke else '8192',
               '--tracker-checkpoint', source['tracker_checkpoint'],
               '--context-checkpoint', str(CONTEXT),
               '--output-dir', str(output), '--seed', '121', '--episode-steps', '500',
               '--motion-sampling', 'adaptive', '--adaptive-after-update', '0',
               '--training-terminations', 'original', '--dr-profile', 'checkpoint_native_flat_v1',
               '--policy-precision', 'fp32', '--rollout-steps', '24', '--actor-lr', '0.0001',
               '--critic-lr', '0.0005', '--entropy-coef', '0.005', '--initial-action-std', '1.0',
               '--dr-aux-coef', '0.05', '--dr-aux-motor-weight', '0.0',
               '--epochs', '5', '--mini-batches', '4',
               '--residual-output-mode', 'unbounded', '--residual-scale', '1.0',
               '--save-interval', '24' if smoke else '100',
               '--wandb-group', '144000_exp', '--wandb-name',
               '144000_exp-residual-proprio122-u7179-history5-tracker-action-auxdr' + ('-smoke' if smoke else '')]
    if smoke:
        command += ['--motion-file', str(RUN/'smoke_motions/walk1_subject1.motion.npz'),
                    '--bounded-smoke', '--iterations', '25' if resume else '24']
    else:
        command += ['--motion-path', source['root'], '--until-user-stop']
    if resume:
        command += ['--resume', str(Path(resume).resolve())]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--attempt', default='initial')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--run-root', type=Path, default=STAGE2,
                        help='Experiment directory containing smoke, preflight and fresh PPO outputs')
    args = parser.parse_args()
    if not args.attempt.replace('_', '').isalnum():
        raise ValueError('Use an alphanumeric attempt name')
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    smoke = args.phase == 'smoke'
    output = root/('smoke_8gpu256' if smoke else 'ppo_8gpu8192')
    record = root/f'{args.phase}_{args.attempt}_process.json'
    if record.exists() or (output.exists() and not args.resume):
        raise FileExistsError('Existing launch/output must be inspected before starting another attempt')
    with CONTEXT.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != CONTEXT_SHA256:
            raise ValueError('The selected noisy-proprio u7179 encoder changed')
    if not smoke:
        verified = json.loads((root/'preflight_verification.json').read_text())
        if not verified['passed']:
            raise ValueError('Native PPO preflight has not passed')
        for name, expected in verified['source_sha256'].items():
            with (ROOT/name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Source changed after preflight: {name}')
    cards = gpu_status(list(range(8)))
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise RuntimeError('All eight GPUs must be free before this stage starts')
    command = command_for(args.phase, output, resume=args.resume)
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cache = root/'runtime'
    cache.mkdir(exist_ok=True)
    env['SP_TRACKING_MULTIMOTION_MANIFEST_DIR'] = str(cache)
    env['WANDB_RUN_ID'] = '144000proprio-residual-' + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    env['WANDB_RESUME'] = 'allow'
    env['WANDB_MODE'] = 'offline' if smoke else 'online'
    key = ROOT/'.runtime/limb_context/wandb_api_key'
    if key.exists():
        env['WANDB_API_KEY'] = key.read_text().strip()
    log = root/f'{args.phase}_{args.attempt}.log'
    with log.open('ab') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(child.pid)
    payload = {**identity, 'process_start_ticks': identity['start_ticks'], 'command': command,
               'physical_gpus': list(range(8)), 'num_envs_per_rank': 256 if smoke else 8192,
               'started_at': time.time(), 'output': str(output), 'log': str(log),
               'initialization': 'resume' if args.resume else 'scratch', 'maximum_updates': 25 if args.resume and smoke else 24 if smoke else None,
               'wandb_id': env['WANDB_RUN_ID']}
    record.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({key: payload[key] for key in ('pid', 'physical_gpus', 'num_envs_per_rank', 'output', 'log')}))


if __name__ == '__main__':
    main()
