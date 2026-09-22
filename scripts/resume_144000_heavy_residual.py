"""Resume paired heavy policies with mass-only auxiliary supervision and fresh sampling."""

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

REVISION = 'resume2000_mass5x_sampling_reset'
COMPARISON = ROOT/'runs/144000-exp-heavy-residual-comparison'/REVISION
OUTPUTS = {'smoke':'smoke_resume_mass5x_sampling_reset','train':'ppo_4gpu8192_resume2000_mass5x_sampling_reset'}


def command_for(arm,phase):
    root = ROOT/'runs'/('144000-exp-heavy-residual-'+arm)
    parent = root/('smoke_4gpu8192_aux108/checkpoint_final.pt' if phase == 'smoke' else 'ppo_4gpu8192_aux108/checkpoint_2000.pt')
    command,_ = original_command(arm,phase,resume=parent)
    output = root/OUTPUTS[phase]
    for key,value in {'--output-dir':str(output),'--dr-aux-coef':'.5' if arm == 'latent' else '0.',
                      '--dr-aux-payload-targets':'mass'}.items():
        command[command.index(key)+1]=value
    if phase == 'smoke':
        command[command.index('--iterations')+1]='28'
        command[command.index('--save-interval')+1]='1'
    command += ['--reset-adaptive-sampling','--allow-dr-aux-change']
    return command,output,parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','train'),required=True)
    parser.add_argument('--arm',choices=(*ARMS,'both'),default='both')
    args = parser.parse_args()
    assert sha256(CONTEXT)==CONTEXT_SHA256
    COMPARISON.mkdir(parents=True,exist_ok=True)
    if args.phase == 'train':
        preflight=json.loads((COMPARISON/'preflight_verification.json').read_text())
        assert preflight['passed']
        for name,digest in preflight['source_sha256'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Source changed after verification: '+name)
    arms=list(ARMS) if args.arm=='both' else [args.arm]
    cards=gpu_status([gpu for arm in arms for gpu in ARMS[arm][1]])
    if any(c['processes'] or c['free_mib']<60000 for c in cards):
        raise RuntimeError('Training GPUs must be free')
    for arm in arms:
        _,output,parent=command_for(arm,args.phase)
        record=output.parent/f'{args.phase}_{REVISION}_process.json'
        if output.exists() or record.exists():raise FileExistsError(str(output))
        if not parent.is_file():raise FileNotFoundError(parent)
        if args.phase=='train' and (output.parent/'monitor').exists():raise FileExistsError('Archive the stopped monitor before relaunch')
    for arm in arms:
        mode,gpus=ARMS[arm]
        command,output,parent=command_for(arm,args.phase)
        root=output.parent
        record=root/f'{args.phase}_{REVISION}_process.json'
        log=root/f'{args.phase}_{REVISION}.log'
        env=process_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            WANDB_RUN_ID='heavy-residual-'+arm+'-'+hashlib.sha256(str(output).encode()).hexdigest()[:10],
            WANDB_RESUME='allow',WANDB_MODE='online' if args.phase=='train' else 'disabled',WANDB_TAGS=REVISION)
        key=ROOT/'.runtime/limb_context/wandb_api_key'
        if key.exists():env['WANDB_API_KEY']=key.read_text().strip()
        with log.open('ab') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_identity(child.pid)
        data={**identity,'process_start_ticks':identity['start_ticks'],'command':command,'started_at':time.time(),
            'phase':args.phase,'arm':arm,'latent_input_mode':mode,'physical_gpus':list(gpus),'output':str(output),'log':str(log),
            'revision':REVISION,'dr_aux_coef':.5 if arm=='latent' else 0.,'dr_aux_payload_targets':'mass',
            'maximum_updates':None if args.phase=='train' else 28,'num_envs_per_rank':8192,
            'parent_checkpoint':str(parent),'parent_checkpoint_sha256':sha256(parent),'reset_adaptive_sampling':True,
            'wandb_id':env['WANDB_RUN_ID']}
        record.write_text(json.dumps(data,indent=2)+'\n')
        if args.phase=='train':
            (root/'latest_training_process.json').write_text(json.dumps({'process_record':str(record)},indent=2)+'\n')
            monitor=root/'monitor';monitor.mkdir()
            with (monitor/'observer.log').open('ab') as stream:
                observer=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B','-u',str(ROOT/'scripts/monitor_memory350_residual.py'),
                    '--run-root',str(root)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            (monitor/'observer_process.json').write_text(json.dumps(process_identity(observer.pid),indent=2)+'\n')
            from watch_memory350_onnx import launch_watcher
            exporter=launch_watcher(output,monitor/'onnx_export',child.pid,identity['start_ticks'])
            data.update(health_observer_pid=observer.pid,onnx_exporter_pid=exporter['pid'])
            record.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({k:data[k] for k in ('arm','pid','output','parent_checkpoint','log')}),flush=True)


if __name__=='__main__':main()
