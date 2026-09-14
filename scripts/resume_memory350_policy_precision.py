"""Resume the existing paired policies with checkpointed FP32 numerical settings."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import time

from run_limb_context_experiment import ROOT, process_environment
from resume_memory350_weak_pairs import child_start, sha256
from run_memory350_scale_nominal_stage1 import read_json, write_json
from run_memory350_compressed_ppo import ASSIGNMENTS, update_comparison


def signal_workers(parent):
    """Let rank workers complete the current update and write final checkpoints."""
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (path / 'cmdline').read_bytes().replace(b'\0', b' ')
            if int(stat[1]) == parent and b'intact_tracking.cli.memory350_action_policy_train' in command:
                os.kill(int(path.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('Run must remain inside the project')
    phase = root / 'precision_fp32_resume'
    ready = read_json(phase / 'FP32_READY.json')
    if not ready or not ready['passed']:
        raise ValueError('FP32 resume preflight is incomplete')
    for name, expected in ready['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Validated source changed: {name}')
    for record in ready['arms'].values():
        if sha256(Path(record['checkpoint'])) != record['checkpoint_sha256']:
            raise ValueError('Resume checkpoint changed')
    lock = (root / '.ppo_launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children = {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        for child, _ in children.values():
            if child.poll() is None:
                signal_workers(child.pid)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for fusion, gpus in ASSIGNMENTS.items():
            output = root / 'ppo' / f'{fusion}_121'
            env = process_environment()
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                       WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
                       WANDB_RESUME='allow',
                       WANDB_RUN_ID='m350compressed-' + hashlib.sha256(str(output).encode()).hexdigest()[:12])
            child, record = child_start(ready['arms'][fusion]['command'], env,
                                        phase / f'{fusion}_resume.log')
            record.update(fusion=fusion, phase='fp32_resume', maximum_updates=None,
                          resumed_from_update=ready['arms'][fusion]['completed_updates'])
            children[fusion] = child, record
            write_json(phase / f'{fusion}_process.json', record)
        while True:
            if not stopping and ((root / 'STOP').exists() or (root / 'STOP_PPO').exists()):
                stop(None, None)
            statuses = {fusion: child.poll() for fusion, (child, _) in children.items()}
            if not stopping and any(code is not None for code in statuses.values()):
                raise RuntimeError(f'FP32 PPO stopped unexpectedly: {statuses}')
            update_comparison(root)
            write_json(root / 'state.json', {'status':'ppo_stop_requested' if stopping else 'paired_ppo_training',
                'maximum_updates':None, 'launcher_pid':os.getpid(), 'heartbeat':time.time(),
                'training_terminations':'original', 'adaptive_after_update':0, 'initialization':'resume',
                'policy_precision':'fp32', 'precision_phase':str(phase),
                'selected_context':ready['selected_context'],
                'arms':{fusion:{'job':record,'progress':read_json(root / 'ppo' / f'{fusion}_121/progress.json')}
                        for fusion, (_,record) in children.items()}})
            if stopping and all(code is not None for code in statuses.values()):
                return
            time.sleep(10)
    except BaseException as error:
        for child, _ in children.values():
            if child.poll() is None:
                signal_workers(child.pid)
        write_json(phase / 'launcher_error.json', {'error':repr(error),'unix_time':time.time()})
        raise


if __name__ == '__main__':
    main()
