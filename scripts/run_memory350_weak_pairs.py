"""Run smoke, exactly 5000 matched training updates, then paired DR evaluation."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, TRACKER, DATASET, SMOKE_MOTION, process_environment
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json


REFERENCE = ROOT / 'runs/limb_context_20260911_memory350_encoder2x_nominal50/stage1_8192'
CACHE = ROOT / 'runs/latent_cluster_probe_memory350_nominal50_u5000_20260911'


def command_for(root, smoke, reference):
    output = root / ('smoke_8192' if smoke else 'stage1_8192')
    command = [PYTHON, '-B', '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=4', '--max-restarts=0', '-m',
               'intact_tracking.cli.forward_memory_weak_pairs_train',
               '--checkpoint-file', TRACKER, '--output-dir', str(output),
               '--nominal-fraction', '.5', '--num-envs', '8192',
               '--weak-positive-weight', '.002', '--weak-negative-weight', '.002',
               '--weak-negative-margin', '1.0', '--response-distance-scale', '.5',
               '--wandb', '--wandb-entity', '2486344338-zhejiang-university',
               '--wandb-project', 'intact-forward-predictor',
               '--wandb-group', root.name, '--wandb-name', output.parent.name + '-' + output.name]
    if smoke:
        command += ['--motion-file', SMOKE_MOTION, '--bounded-smoke', '--updates', '2']
    else:
        command += ['--motion-path', DATASET, '--until-user-stop', '--updates', '8000',
                    '--stop-after-updates', '5000', '--comparison-reference-dir', str(reference)]
    return command


def run_child(root, command, env, *, phase, output=None):
    log_path = root / (phase + '.log')
    with log_path.open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': child.pid, 'process_start_ticks': int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),
              'command': command, 'physical_gpus': env['CUDA_VISIBLE_DEVICES'],
              'log': str(log_path), 'started_at': time.time()}
    write_json(root / (phase + '_process.json'), record)
    sent_stop = False
    last_update = -1
    try:
        while child.poll() is None:
            progress = read_json(output / 'progress.json') if output else None
            state = {'status': phase + '_running', 'launcher_pid': os.getpid(), 'job': record,
                     'progress': progress, 'heartbeat': time.time()}
            write_json(root / 'state.json', state)
            update = progress.get('completed_updates', 0) if progress else 0
            if update // 100 != last_update // 100:
                print(json.dumps({'phase': phase, 'progress': progress, 'time': time.time()}), flush=True)
                last_update = update
            if (root / 'STOP').exists() and not sent_stop:
                os.killpg(child.pid, signal.SIGTERM)
                sent_stop = True
            time.sleep(10)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise
    record.update(exit_code=child.returncode, finished_at=time.time())
    write_json(root / (phase + '_process_result.json'), record)
    if sent_stop:
        write_json(root / 'state.json', {'status': 'stopped', 'phase': phase, 'heartbeat': time.time()})
        return False
    if child.returncode:
        raise RuntimeError(f'{phase} failed with {child.returncode}; inspect {log_path}')
    return True


def verify_training(output, target):
    completion = read_json(output / 'completion.json')
    config = read_json(output / 'run_config.json')
    if not completion or completion['completed_updates'] != target or not completion['hit_cap']:
        raise RuntimeError(f'Training did not finish the exact {target} update budget')
    if config['parameter_counts'] != {'total': 21086486, 'context_encoder': 2026688, 'predictor': 19059798}:
        raise RuntimeError('Expanded model parameter counts changed')
    expected = {'weak_positive_weight': .002, 'weak_negative_weight': .002,
                'weak_negative_margin': 1., 'response_distance_scale': .5,
                'representation_weight': .01, 'representation_relation_weight': 2.}
    if any(config['loss'][key] != value for key, value in expected.items()):
        raise RuntimeError('Actual loss configuration differs from the experiment')
    if config['arguments']['nominal_fraction'] != .5:
        raise RuntimeError('Training lost the nominal50 mixture')
    ranks = config['dataset']['runtime_audits_by_rank']
    if len(ranks) != 4 or any(rank['num_envs'] != 8192 for rank in ranks):
        raise RuntimeError('Training world count changed')
    if target == 5000:
        rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        if not any(row['optimization_train']['weak_positive_pairs'] > 0 for row in rows):
            raise RuntimeError('No cross-motion positives contributed during training')
        if not any(row['optimization_train']['weak_negative_pairs'] > 0 for row in rows):
            raise RuntimeError('No DR negatives contributed during training')
        if completion['optimizer_steps'] != 20000:
            raise RuntimeError('Optimizer-step count differs from the baseline')
    result = {'passed': True, 'completion': completion, 'parameter_counts': config['parameter_counts'],
              'loss': config['loss'], 'nominal_fraction': .5}
    write_json(output.parent / (output.name + '_verification.json'), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--cache', type=Path, default=CACHE)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_relative_to(ROOT):
        raise ValueError('Run must stay inside the shared project')
    with (root / '.launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'stage1_8192/run_config.json').exists():
            raise RuntimeError('Refusing to restart an existing formal training run')
        cards = gpu_status([4, 5, 6, 7])
        if any(card['free_mib'] < 60000 for card in cards):
            raise RuntimeError('GPUs 4-7 need at least 60000 MiB free each')
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES='4,5,6,7', PYTHONUNBUFFERED='1',
                   WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
                   WANDB_RESUME='allow')
        contract = {'maximum_updates': 5000, 'cosine_horizon_updates': 8000,
                    'gpus': [4, 5, 6, 7], 'reference_checkpoint': str(args.reference / 'update_005000.pt'),
                    'cached_trajectories': str(args.cache), 'formal_command': command_for(root, False, args.reference),
                    'initialization': 'from scratch, seed717', 'stage2_jobs': [],
                    'primary_outcome': 'DR cross-motion compactness, separation and held-out-motion environment readout',
                    'created_at': time.time()}
        write_json(root / 'launch_contract.json', contract)
        source_files = sorted((ROOT / 'src/intact_tracking').rglob('*.py')) + [Path(__file__).resolve()]
        write_json(root / 'source_sha256.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in source_files})
        for smoke in (True, False):
            name = 'smoke_8192' if smoke else 'stage1_8192'
            output = root / name
            env['WANDB_RUN_ID'] = 'm350weak-' + hashlib.sha256(str(output).encode()).hexdigest()[:12]
            if not run_child(root, command_for(root, smoke, args.reference), env, phase=name, output=output):
                return
            verify_training(output, 2 if smoke else 5000)
        evaluation = [PYTHON, '-B', '-u', str(ROOT / 'scripts/evaluate_memory350_weak_pairs.py'),
                      '--candidate', str(root / 'stage1_8192/update_005000.pt'),
                      '--reference', str(args.reference / 'update_005000.pt'),
                      '--cache', str(args.cache), '--output', str(root / 'comparison_005000')]
        evaluation_env = dict(env, CUDA_VISIBLE_DEVICES='7',
                              PYTHONPATH='/tmp/intact-tsne-deps', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        if run_child(root, evaluation, evaluation_env, phase='comparison_005000'):
            summary = read_json(root / 'comparison_005000/summary.json')
            if not summary or not summary.get('complete'):
                raise RuntimeError('Comparison returned without a complete audited result')
            write_json(root / 'state.json', {'status': 'complete', 'summary': str(root / 'comparison_005000/README.md'),
                                            'heartbeat': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import sys
        if '--run-root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--run-root') + 1])
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / 'state.json', {'status': 'failed', 'error': repr(error), 'heartbeat': time.time()})
        raise
