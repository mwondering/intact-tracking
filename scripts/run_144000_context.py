"""Launch the user-approved, scratch 144000_exp context stage after preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status
from monitor_memory350_nominal_direction import process_identity

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/144000_exp')
    args = parser.parse_args()
    run = args.run_root.resolve()
    source = json.loads((run/'source_identity.json').read_text())
    assert json.loads((run/'old_training_stop_result.json').read_text())['complete']
    with Path(source['tracker_checkpoint']).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == source['tracker_sha256']
    smoke = args.phase == 'smoke'
    if not smoke:
        assert json.loads((run/'preflight_verification.json').read_text())['passed']
    gpus = [0] if smoke else list(range(8))
    cards = gpu_status(gpus)
    assert all(not c['processes'] and c['free_mib'] > 60000 for c in cards), cards
    output = run/('smoke' if smoke else 'stage1_8192')
    record_path = run/(args.phase+'_process.json')
    assert not record_path.exists(), 'Inspect and archive the old attempt before another launch'
    assert not (output/'run_config.json').exists(), 'Refusing to overwrite completed initialization'
    command = [str(ROOT/'.venv/bin/python'), '-B', '-u', '-m', 'torch.distributed.run', '--standalone',
               f'--nproc-per-node={len(gpus)}', '--max-restarts=0', '-m',
               'intact_tracking.cli.forward_memory_native_dr_train',
               '--checkpoint-file', source['tracker_checkpoint'], '--motion-path',
               str(run/'smoke_motions') if smoke else source['root'], '--output-dir', str(output),
               '--num-envs', '256' if smoke else '8192', '--validation-worlds', '32' if smoke else '128',
               '--seed', '717', '--nominal-fraction', '0.1', '--no-payload',
               '--dr-nominal-probability', '0', '--warmup-steps', '1000', '--max-warmup-steps', '10000',
               '--rollout-steps-per-update', '5', '--gradient-steps-per-update', '4',
               '--batch-size', '128' if smoke else '1024', '--micro-batch-size', '128' if smoke else '256',
               '--fixed-probe-batch-size', '128' if smoke else '512',
               '--replay-capacity', '16384' if smoke else '262144', '--replay-sampling', 'motion_balanced',
               '--amp-dtype', 'bfloat16', '--model-learning-rate', '0.0003', '--weight-decay', '0.001',
               '--continuation-min-learning-rate', '0.00001', '--context-history-steps', '50',
               '--chunk-depth', '2', '--memory-depth', '4', '--context-depth', '4', '--dynamics-latent-dim', '64',
               '--recursive-weight', '0.5', '--representation-weight', '0.01',
               '--representation-relation-weight', '10', '--nominal-anchor-weight', '0.08',
               '--weak-positive-weight', '0.008', '--dr-soft-weight', '0.02',
               '--dr-soft-h', '0.15', '--dr-soft-temperature', '0.1', '--response-distance-scale', '0.6',
               '--weak-archive-slots', '4', '--weak-archive-interval', '200',
               '--anchor-calibration-batches', '8', '--anchor-calibration-batch-size', '256',
               '--checkpoint-interval', '1' if smoke else '250',
               '--validation-interval', '1' if smoke else '100', '--log-interval', '10',
               '--warmup-log-interval', '100']
    if smoke:
        command += ['--bounded-smoke', '--updates', '2', '--no-wandb']
    else:
        command += ['--until-user-stop', '--updates', '8000', '--wandb',
                    '--wandb-project', 'intact-forward-predictor',
                    '--wandb-entity', '2486344338-zhejiang-university',
                    '--wandb-group', '144000_exp', '--wandb-name', '144000_exp',
                    '--wandb-tag', 'scratch', '--wandb-tag', 'native-dr-flat', '--wandb-tag', 'memory350']
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Keep the resolved motion manifest scoped to this experiment.
    cache = run/'runtime'; cache.mkdir(exist_ok=True)
    env['SP_TRACKING_MULTIMOTION_MANIFEST_DIR'] = str(cache)
    key = ROOT/'.runtime/limb_context/wandb_api_key'
    if key.exists():
        env['WANDB_API_KEY'] = key.read_text().strip()
    env['WANDB_RUN_ID'] = '144000context-' + hashlib.sha256(str(run).encode()).hexdigest()[:12]
    env['WANDB_RESUME'] = 'allow'
    log = run/(args.phase+'.log')
    with log.open('ab') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(child.pid)
    record = {**identity, 'process_start_ticks': identity['start_ticks'], 'command': command,
              'physical_gpus': gpus, 'started_at': time.time(), 'output': str(output),
              'log': str(log), 'initialization': 'scratch', 'wandb_id': env['WANDB_RUN_ID']}
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'phase': args.phase, 'pid': child.pid, 'gpus': gpus, 'output': str(output)}))


if __name__ == '__main__':
    main()
