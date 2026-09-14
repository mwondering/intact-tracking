"""Run the online-KMeans16 MoE and latent-free MLP on four GPUs each."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import time

from run_limb_context_experiment import ROOT, process_environment
from resume_memory350_weak_pairs import child_start, sha256
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json
from run_memory350_compressed_ppo import ASSIGNMENTS, command_for, update_comparison


def moe_command(root, fusion, context, **kwargs):
    command = command_for(root, fusion, context, **kwargs)
    index = command.index("intact_tracking.cli.memory350_compressed_policy_train")
    command[index] = "intact_tracking.cli.memory350_moe_policy_train"
    command[command.index("--training-terminations") + 1] = "original"
    command[command.index("--adaptive-after-update") + 1] = "0"
    command[command.index("--motion-sampling") + 1] = "adaptive"
    if fusion == "concat":
        index = command.index("--wandb-name") + 1
        command[index] = command[index].replace("-concat-", "-moe16-")
    command += ["--dr-sampling", "independent_uniform", "--policy-precision", "fp32",
                "--num-experts", "16", "--router-center-rate", "0.01",
                "--router-max-switch-fraction", "0.02", "--router-bootstrap-steps", "500"]
    return command


def stop_rank_workers(parent):
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (path / 'cmdline').read_bytes()
            if int(stat[1]) == parent and b'intact_tracking.cli.memory350_moe_policy_train' in command:
                os.kill(int(path.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('Run must remain in the project')
    ready, plan = read_json(root / 'PPO_READY.json'), read_json(root / 'ppo_launch_contract.json')
    if not ready or not ready['passed']:
        raise ValueError('Complete preflight before launching formal PPO')
    for name, expected in {**ready['source_sha256'], **ready['artifact_sha256']}.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Validated file changed: {name}')
    if (root / 'ppo').exists():
        raise FileExistsError('Formal training already exists; do not restart from zero')
    for fusion in ASSIGNMENTS:
        if plan['formal_commands'][fusion] != moe_command(root, fusion, plan['selected_context']['checkpoint']):
            raise ValueError('Formal launch command differs from the reviewed contract')
    lock = (root / '.ppo_launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cards = gpu_status(list(range(8)))
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise ValueError('All eight GPUs must be free before the paired experiment')
    write_json(root / 'gpu_before_launch.json', cards)
    (root / 'ppo').mkdir()
    children, stopping = {}, False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        for child, _ in children.values():
            if child.poll() is None:
                stop_rank_workers(child.pid)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for fusion, gpus in ASSIGNMENTS.items():
            output = root / 'ppo' / f'{fusion}_121'
            env = process_environment()
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                       WANDB_API_KEY=(ROOT / '.runtime/limb_context/wandb_api_key').read_text().strip(),
                       WANDB_RESUME='allow',
                       WANDB_RUN_ID='m350moe-' + hashlib.sha256(str(output).encode()).hexdigest()[:12])
            child, record = child_start(plan['formal_commands'][fusion], env, root / 'ppo' / f'{fusion}_initial.log')
            record.update(fusion=fusion, display_name='MLP baseline' if fusion == 'baseline' else 'online KMeans16 MoE',
                          phase='scratch_online_kmeans', maximum_updates=None)
            children[fusion] = child, record
            write_json(root / 'ppo' / f'{fusion}_initial_process.json', record)
        while True:
            if not stopping and ((root / 'STOP').exists() or (root / 'STOP_PPO').exists()):
                stop(None, None)
            statuses = {fusion: child.poll() for fusion, (child, _) in children.items()}
            if not stopping and any(code is not None for code in statuses.values()):
                raise RuntimeError(f'PPO stopped unexpectedly: {statuses}')
            update_comparison(root)
            write_json(root / 'state.json', {'status':'ppo_stop_requested' if stopping else 'paired_ppo_training',
                'maximum_updates':None, 'launcher_pid':os.getpid(), 'heartbeat':time.time(),
                'training_terminations':'original', 'adaptive_after_update':0,
                'dr_sampling':'independent_uniform', 'initialization':'scratch', 'policy_precision':'fp32',
                'router':'online KMeans16; hard top1; update only after each complete PPO update',
                'selected_context':plan['selected_context'],
                'arms':{fusion:{'job':record,'progress':read_json(root / 'ppo' / f'{fusion}_121/progress.json')}
                        for fusion, (_, record) in children.items()}})
            if stopping and all(code is not None for code in statuses.values()):
                return
            time.sleep(10)
    except BaseException as error:
        stop(None, None)
        write_json(root / 'ppo_launcher_error.json', {'error':repr(error), 'unix_time':time.time()})
        raise


if __name__ == '__main__':
    main()
