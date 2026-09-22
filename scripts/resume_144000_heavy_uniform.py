"""Resume both heavy arms from their original parents with uniform motion draws."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from run_144000_heavy_residual import ROOT, ARMS, CONTEXT, CONTEXT_SHA256, sha256, command_for as original_command
from monitor_memory350_nominal_direction import process_identity
from run_limb_context_experiment import process_environment
from run_memory350_scale_nominal_stage1 import gpu_status

REVISION = 'resume2000_uniform_mass5x'
COMPARISON = ROOT/'runs/144000-exp-heavy-residual-comparison'/REVISION
OUTPUTS = {'smoke': 'smoke_resume_uniform_mass5x', 'train': 'ppo_4gpu8192_resume2000_uniform_mass5x'}


def command_for(arm, phase):
    root = ROOT/'runs'/('144000-exp-heavy-residual-'+arm)
    parent = root/('smoke_4gpu8192_aux108/checkpoint_final.pt' if phase == 'smoke' else 'ppo_4gpu8192_aux108/checkpoint_2000.pt')
    command, _ = original_command(arm, phase, resume=parent)
    output = root/OUTPUTS[phase]
    values = {'--output-dir':str(output), '--motion-sampling':'uniform',
              '--dr-aux-coef':'.5' if arm == 'latent' else '0.', '--dr-aux-payload-targets':'mass',
              '--wandb-name':f'144000-exp-heavy-residual-{arm}-uniform'}
    if phase == 'smoke':
        values.update({'--iterations':'28', '--save-interval':'1'})
    for key, value in values.items():
        command[command.index(key)+1] = value
    command += ['--resume-uniform-sampling', '--allow-dr-aux-change']
    return command, output, parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--arm', choices=(*ARMS, 'both'), default='both')
    args = parser.parse_args()
    if sha256(CONTEXT) != CONTEXT_SHA256:
        raise ValueError('Frozen context checkpoint changed')
    COMPARISON.mkdir(parents=True, exist_ok=True)
    if args.phase == 'train':
        preflight = json.loads((COMPARISON/'preflight_verification.json').read_text())
        if not preflight['passed'] or any(sha256(ROOT/p) != h for p,h in preflight['source_sha256'].items()):
            raise ValueError('Uniform-resume preflight is missing or source changed')
    arms = list(ARMS) if args.arm == 'both' else [args.arm]
    cards = gpu_status([gpu for arm in arms for gpu in ARMS[arm][1]])
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise RuntimeError('Wait for previous training/diagnostics to release the allocated GPUs')
    for arm in arms:
        _, output, parent = command_for(arm, args.phase)
        if output.exists() or (output.parent/f'{args.phase}_{REVISION}_process.json').exists():
            raise FileExistsError(output)
        if not parent.is_file():
            raise FileNotFoundError(parent)
        if args.phase == 'train' and (output.parent/'monitor').exists():
            raise FileExistsError('Archive the previous stopped monitor first')
    for arm in arms:
        mode, gpus = ARMS[arm]
        command, output, parent = command_for(arm, args.phase)
        root = output.parent
        record, log = root/f'{args.phase}_{REVISION}_process.json', root/f'{args.phase}_{REVISION}.log'
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            WANDB_RUN_ID=f'heavy-residual-{arm}-uniform-'+hashlib.sha256(str(output).encode()).hexdigest()[:10],
            WANDB_RESUME='allow', WANDB_MODE='online' if args.phase == 'train' else 'disabled',
            WANDB_TAGS=REVISION)
        key = ROOT/'.runtime/limb_context/wandb_api_key'
        if key.exists():
            env['WANDB_API_KEY'] = key.read_text().strip()
        with log.open('xb') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        identity = process_identity(child.pid)
        data = {**identity, 'process_start_ticks':identity['start_ticks'], 'command':command,
            'started_at':time.time(), 'phase':args.phase, 'arm':arm, 'latent_input_mode':mode,
            'physical_gpus':list(gpus), 'output':str(output), 'log':str(log), 'revision':REVISION,
            'dr_aux_coef':.5 if arm == 'latent' else 0., 'dr_aux_payload_targets':'mass',
            'maximum_updates':None if args.phase == 'train' else 28, 'num_envs_per_rank':8192,
            'parent_checkpoint':str(parent), 'parent_checkpoint_sha256':sha256(parent),
            'motion_sampling':'uniform', 'failure_rewind_enabled':False,
            'resume_uniform_sampling':True, 'adaptive_statistics':'disabled_and_discarded',
            'wandb_id':env['WANDB_RUN_ID']}
        record.write_text(json.dumps(data,indent=2)+'\n')
        if args.phase == 'train':
            (root/'latest_training_process.json').write_text(json.dumps({'process_record':str(record)},indent=2)+'\n')
            monitor = root/'monitor'; monitor.mkdir()
            with (monitor/'observer.log').open('xb') as stream:
                observer = subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B','-u',
                    str(ROOT/'scripts/monitor_memory350_residual.py'),'--run-root',str(root)],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            (monitor/'observer_process.json').write_text(json.dumps(process_identity(observer.pid),indent=2)+'\n')
            from watch_memory350_onnx import launch_watcher
            exporter = launch_watcher(output,monitor/'onnx_export',child.pid,identity['start_ticks'])
            data.update(health_observer_pid=observer.pid, onnx_exporter_pid=exporter['pid'])
            record.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({k:data[k] for k in ('arm','pid','output','parent_checkpoint','log')}),flush=True)


if __name__ == '__main__':
    main()
