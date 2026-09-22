"""Launch the 144000-exp-heavy scratch context experiment after local checks."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

from monitor_memory350_nominal_direction import process_identity
from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/144000-exp-heavy'
MODULE = 'intact_tracking.cli.forward_memory_proprio_heavy_train'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command_for(output, *, smoke_motion_path=None):
    # Keep every architecture, loss, optimizer and collection setting from the
    # previous scratch proprio run. Only experiment/data/allocation change.
    record = json.loads((ROOT / 'runs/144000_exp/proprio122_train_process.json').read_text())
    previous = record['command']
    entry = previous.index('intact_tracking.cli.forward_memory_proprio_native_train')
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', '-m']
    if smoke_motion_path is None:
        command += ['torch.distributed.run', '--standalone', '--nproc-per-node=8', '--max-restarts=0', '-m']
    command += [MODULE, *previous[entry + 1:]]
    plan = json.loads((RUN / 'plan.json').read_text())
    changes = {'--checkpoint-file': plan['tracker_checkpoint'], '--motion-path': plan['dataset']['root'],
               '--output-dir': str(output), '--num-envs': '16384',
               '--wandb-group': '144000-exp-heavy', '--wandb-name': '144000-exp-heavy'}
    if smoke_motion_path is not None:
        changes.update({'--motion-path': str(smoke_motion_path), '--updates': '2',
                        '--checkpoint-interval': '1', '--validation-interval': '1', '--log-interval': '1'})
        command.remove('--wandb')
        command.remove('--until-user-stop')
        command += ['--bounded-smoke', '--no-wandb']
    for key, value in changes.items():
        command[command.index(key) + 1] = value
    command += ['--heavy-max-masses-kg', '2.5', '2.5', '4', '4',
                '--heavy-com-half-width-m', '0.05', '--weak-archive-device', 'cpu',
                '--wandb-tag', 'stratified-payload-256', '--wandb-tag', 'payload-com-cube5cm']
    return command


def launch(smoke_motion_path=None):
    smoke = smoke_motion_path is not None
    plan = json.loads((RUN / 'plan.json').read_text())
    assert sha256(plan['tracker_checkpoint']) == plan['tracker_checkpoint_sha256']
    assert sha256(plan['baseline_config']) == plan['baseline_config_sha256']
    dataset = plan['dataset']
    assert sha256(dataset['manifest_file']) == dataset['manifest_file_sha256']
    for source in dataset['exclusion_files']:
        assert sha256(source['source']) == sha256(source['snapshot']) == source['sha256']
    gpus = [0] if smoke else list(range(8))
    cards = gpu_status(gpus)
    assert all(not card['processes'] and card['free_mib'] > 60000 for card in cards), cards
    output = RUN / ('smoke_16384' if smoke else 'stage1_proprio122_16384')
    record_path = RUN / ('smoke_process.json' if smoke else 'train_process.json')
    assert not output.exists() and not record_path.exists(), 'Archive a failed attempt before relaunching'
    if not smoke:
        verification = json.loads((RUN / 'launch_verification.json').read_text())
        assert verification['passed']
        for name, digest in verification['source_sha256'].items():
            assert sha256(ROOT / name) == digest, f'Verified source changed: {name}'
        with tarfile.open(RUN / 'source_snapshot.tar.gz', 'w:gz') as archive:
            for path in [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'),
                         *ROOT.glob('tests/*.py'), ROOT / 'pyproject.toml']:
                archive.add(path, arcname=str(path.relative_to(ROOT)))
    command = command_for(output, smoke_motion_path=smoke_motion_path)
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    runtime = RUN / 'runtime'
    runtime.mkdir(exist_ok=True)
    env['SP_TRACKING_MULTIMOTION_MANIFEST_DIR'] = str(runtime)
    if not smoke:
        key = ROOT / '.runtime/limb_context/wandb_api_key'
        if key.exists():
            env['WANDB_API_KEY'] = key.read_text().strip()
        env['WANDB_RUN_ID'] = '144000-heavy-' + hashlib.sha256(str(output).encode()).hexdigest()[:12]
        env['WANDB_RESUME'] = 'never'
    log = RUN / ('smoke_16384.log' if smoke else 'train.log')
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(child.pid)
    record = {**identity, 'process_start_ticks': identity['start_ticks'], 'command': command,
              'physical_gpus': gpus, 'started_at': time.time(), 'output': str(output), 'log': str(log),
              'initialization': 'scratch', 'maximum_updates': 2 if smoke else None,
              'wandb_id': None if smoke else env['WANDB_RUN_ID'],
              'tracker_sha256': plan['tracker_checkpoint_sha256'],
              'num_envs_per_rank': 16384, 'dataset_manifest_sha256': dataset['manifest_sha256'],
              'user_authorization': 'Stop the old training and start new training on all eight GPUs; payload xyz independent uniform as tracker COM'}
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'pid': child.pid, 'gpus': gpus, 'output': str(output), 'log': str(log)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-motion-path', type=Path)
    args = parser.parse_args()
    launch(args.smoke_motion_path)
