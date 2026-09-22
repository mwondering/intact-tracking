"""Launch the heavy learned/zero latent comparison on disjoint groups of four GPUs."""

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
CONTEXT = ROOT / 'runs/144000-exp-heavy/stage1_proprio122_16384/update_008816.pt'
CONTEXT_SHA256 = 'decd72ae598603755d0a7df93250b2925574bd71bae3bc9a6d53eefe3d057931'
TRACKER = '/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/2026-09-10_16-19-06_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct/checkpoint_144000.pt'
DATASET = '/data_zcy/wxy/motion_data_correct'
REVISION = 'aux108_8192'
COMPARISON = ROOT / 'runs/144000-exp-heavy-residual-comparison' / REVISION
ARMS = {'latent': ('learned', (0,1,2,3)), 'baseline': ('zero', (4,5,6,7))}
DIRECTORIES = {'smoke': 'smoke_4gpu8192_aux108', 'train': 'ppo_4gpu8192_aux108'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command_for(arm, phase, *, resume=None):
    mode, _ = ARMS[arm]
    name = '144000-exp-heavy-residual-' + arm
    output = ROOT / 'runs' / name / DIRECTORIES[phase]
    command = [str(ROOT/'.venv/bin/python'), '-B','-u','-m','torch.distributed.run',
        '--standalone','--nproc-per-node=4','--max-restarts=0','-m',
        'intact_tracking.cli.memory350_proprio_heavy_policy_train',
        '--fusion','concat','--latent-input-mode',mode,'--training-ranks','4',
        '--num-envs','8192',
        '--tracker-checkpoint',TRACKER,'--context-checkpoint',str(CONTEXT),
        '--output-dir',str(output),'--seed','121','--episode-steps','500',
        '--motion-sampling','adaptive','--adaptive-after-update','0',
        '--training-terminations','original','--dr-profile','checkpoint_native_flat_heavy_v1',
        '--policy-precision','fp32','--rollout-steps','24','--actor-lr','0.0001',
        '--critic-lr','0.0005','--entropy-coef','0.005','--initial-action-std','1.0',
        '--dr-aux-coef','0.1' if arm == 'latent' else '0.0','--dr-aux-payload-targets','all',
        '--dr-aux-motor-weight','0.0','--epochs','5','--mini-batches','4',
        '--residual-output-mode','unbounded','--residual-scale','1.0',
        '--save-interval','100' if phase == 'train' else '24',
        '--wandb-group','144000-exp-heavy','--wandb-name',name]
    if phase == 'smoke':
        command += ['--motion-file',str(ROOT/'runs/144000_exp/smoke_motions/walk1_subject1.motion.npz')]
    else:
        command += ['--motion-path',DATASET]
    if phase == 'train':
        command += ['--until-user-stop']
    else:
        command += ['--bounded-smoke','--iterations',str(25 if resume else 24)]
    if resume:
        command += ['--resume',str(Path(resume).resolve())]
    return command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=tuple(DIRECTORIES),required=True)
    parser.add_argument('--arm',choices=(*ARMS,'both'),default='both')
    parser.add_argument('--attempt',default=REVISION)
    parser.add_argument('--resume-final',action='store_true')
    args = parser.parse_args()
    if not args.attempt.replace('_','').isalnum():
        raise ValueError('Use an alphanumeric attempt name')
    assert sha256(CONTEXT) == CONTEXT_SHA256
    if args.phase == 'train':
        verified = json.loads((COMPARISON/'preflight_verification.json').read_text())
        if not verified['passed']:
            raise ValueError('Heavy comparison preflight has not passed')
        for name, digest in verified['source_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Source changed after preflight: {name}')
    arms = list(ARMS) if args.arm == 'both' else [args.arm]
    COMPARISON.mkdir(parents=True, exist_ok=True)
    cards = gpu_status([g for arm in arms for g in ARMS[arm][1]])
    if any(card['processes'] or card['free_mib'] < 60000 for card in cards):
        raise RuntimeError('Allocated GPUs must be free before starting a new process')
    # Validate both destinations before starting either trainer.
    for arm in arms:
        root = ROOT/'runs'/('144000-exp-heavy-residual-'+arm)
        output = root/DIRECTORIES[args.phase]
        record = root/f'{args.phase}_{args.attempt}_process.json'
        if record.exists() or (output.exists() and not args.resume_final):
            raise FileExistsError(f'Existing launch must be inspected: {record} / {output}')
        if args.resume_final and not (output/'checkpoint_final.pt').is_file():
            raise FileNotFoundError(output/'checkpoint_final.pt')
    for arm in arms:
        mode, gpus = ARMS[arm]
        name = '144000-exp-heavy-residual-' + arm
        root = ROOT/'runs'/name
        output = root/DIRECTORIES[args.phase]
        resume = output/'checkpoint_final.pt' if args.resume_final else None
        record = root/f'{args.phase}_{args.attempt}_process.json'
        if record.exists() or (output.exists() and not resume):
            raise FileExistsError(f'Existing launch must be inspected: {record} / {output}')
        if resume and not resume.is_file():
            raise FileNotFoundError(resume)
        root.mkdir(parents=True,exist_ok=True)
        command, output = command_for(arm,args.phase,resume=resume)
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   WANDB_RUN_ID='heavy-residual-' + arm + '-' + hashlib.sha256(str(output).encode()).hexdigest()[:10],
                   WANDB_RESUME='allow', WANDB_MODE='online' if args.phase == 'train' else 'disabled')
        key = ROOT/'.runtime/limb_context/wandb_api_key'
        if key.exists():
            env['WANDB_API_KEY'] = key.read_text().strip()
        log = root/f'{args.phase}_{args.attempt}.log'
        with log.open('ab') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        identity = process_identity(child.pid)
        data = {**identity,'process_start_ticks':identity['start_ticks'],'command':command,
                'started_at':time.time(),'phase':args.phase,'arm':arm,'latent_input_mode':mode,
                'revision':REVISION,'dr_aux_coef':.1 if arm == 'latent' else 0.,'num_envs_per_rank':8192,
                'physical_gpus':list(gpus),'output':str(output),'log':str(log),
                'wandb_id':env['WANDB_RUN_ID'],'maximum_updates':None if args.phase == 'train' else 25 if resume else 24}
        record.write_text(json.dumps(data,indent=2)+'\n')
        if args.phase == 'train':
            (root/'latest_training_process.json').write_text(json.dumps({'process_record':str(record)},indent=2)+'\n')
            monitor = root/'monitor'
            monitor.mkdir(exist_ok=True)
            with (monitor/'observer.log').open('ab') as stream:
                observer = subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B','-u',
                    str(ROOT/'scripts/monitor_memory350_residual.py'),'--run-root',str(root)],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
            monitor_identity = process_identity(observer.pid)
            (monitor/'observer_process.json').write_text(json.dumps(monitor_identity,indent=2)+'\n')
            data['health_observer_pid'] = observer.pid
            from watch_memory350_onnx import launch_watcher
            exporter = launch_watcher(output,root/'monitor/onnx_export',child.pid,identity['start_ticks'])
            data['onnx_exporter_pid'] = exporter['pid']
            record.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({key:data[key] for key in ('pid','arm','phase','physical_gpus','output','log')}),flush=True)


if __name__ == '__main__':
    main()
