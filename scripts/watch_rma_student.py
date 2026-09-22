"""CPU export and periodic paired tracking evaluations for the RMA student."""

import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from run_144000_rma_student import ROOT,PYTHON,RUN_ROOT,digest
from run_limb_context_experiment import process_environment
from monitor_memory350_nominal_direction import process_identity
from intact_tracking.rma_student_distillation import atomic_json
from intact_tracking.rma_student_export import export_policy
from watch_memory350_onnx import publish_latest

REFERENCE=ROOT/'runs/144000-exp-heavy/baselines/evaluation_latest_four_20260922'
LATENT=ROOT/'runs/144000-exp-heavy-residual-latent/ppo_4gpu8192_resume2000_uniform_mass5x'
VANILLA=ROOT/'runs/144000-exp-heavy-residual-baseline/ppo_4gpu8192_resume2000_uniform_mass5x/checkpoint_final.pt'


def numeric_checkpoints(directory):
    return sorted(directory.glob('checkpoint_[0-9]*.pt'),key=lambda p:int(p.stem.split('_')[-1]))


def record_evaluation_sources(directory,protocol):
    path=directory/'evaluation_source_sha256.json'
    names=set(json.loads((REFERENCE/'evaluation_source_sha256.json').read_text()))
    names.update(('src/intact_tracking/heavy_rma_student.py','src/intact_tracking/rma_student_env.py',
                  'src/intact_tracking/memory350_proprio_inputs.py'))
    hashes={name:digest(ROOT/name) for name in sorted(names)}
    if path.exists():
        if json.loads(path.read_text())!=hashes:
            raise RuntimeError('Evaluation source changed while resuming a paired evaluation')
        return
    existing={f'{role}_{physics}.json':digest(directory/f'{role}_{physics}.json')
              for physics in protocol['physics'] for role in protocol['roles']
              if (directory/f'{role}_{physics}.json').exists()}
    atomic_json(directory/'evaluation_source_capture.json',{
        'captured_at_utc':datetime.now(timezone.utc).isoformat(),
        'phase':'after_existing_results' if existing else 'before_evaluation',
        'existing_results_sha256':existing,
        'note':'Current source fingerprint backfilled after existing results; not a contemporaneous pre-evaluation snapshot.'
               if existing else 'Source fingerprints captured before evaluation and checked on resume.'})
    atomic_json(path,hashes)


def select_protocol(root,checkpoint,directory):
    if (directory/'protocol.json').exists():return json.loads((directory/'protocol.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    sources={'rma_teacher':root/'teacher_checkpoint_9000.pt','latent':numeric_checkpoints(LATENT)[-1],
             'vanilla':VANILLA,'rma_student':checkpoint}
    protocol=json.loads((REFERENCE/'protocol.json').read_text())
    manifest=directory/'motions_256.txt';shutil.copyfile(REFERENCE/'motions_256.txt',manifest)
    protocol.update(comparison_mode='distillation',roles=list(sources),manifest=str(manifest),
        selected_at_utc=datetime.now(timezone.utc).isoformat(),
        selection='Fixed source teacher9000; stopped vanilla; latest latent at snapshot; selected student.',
        history_contract='Common frozen-tracker warmup pairs controller state. Student clears history at query reset, retaining only the current proprio frame; its cold-to-full-history query is scored in full. Latent retains its trained long-memory protocol.',
        training_history_caveat='Student updates count supervised adaptation; teacher/vanilla/latent updates count PPO. The fixed teacher source continues to identify the distillation target.',
        training_history_caveat_zh='Student 轮数为监督蒸馏，其他组为 PPO；teacher 始终使用蒸馏源快照。Student 从空交互历史开始查询，latent 保留其长期物理记忆。')
    selected={}
    for role,path in sources.items():
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        snapshot=directory/(role+'_checkpoint_'+str(state['iter'])+'.pt')
        shutil.copyfile(path,snapshot)
        meta=state['residual_policy']
        selected[role]={'source':str(path),'snapshot':str(snapshot),'filename_iteration':state['iter'],
            'completed_updates':state['completed_updates'],'sha256':digest(snapshot),
            'context_sha256':meta.get('context_sha256'),'tracker_sha256':meta['tracker_sha256'],
            'method':meta.get('method'),'latent_input_mode':meta.get('latent_input_mode')}
    protocol['checkpoints']=selected
    atomic_json(directory/'protocol.json',protocol)
    return protocol


def evaluate(root,checkpoint,gpu):
    update=int(checkpoint.stem.split('_')[-1])
    directory=root/'evaluations'/f'update_{update:06d}'
    protocol=select_protocol(root,checkpoint,directory)
    record_evaluation_sources(directory,protocol)
    env=process_environment();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled')
    for physics in ('nominal','mixed'):
        for role in protocol['roles']:
            output=directory/f'{role}_{physics}.json'
            if output.exists():continue
            module=('memory350_proprio_heavy_policy_eval' if role in ('latent','vanilla') else 'heavy_baseline_eval')
            command=[PYTHON,'-m','intact_tracking.cli.'+module,'--checkpoint',protocol['checkpoints'][role]['snapshot'],
                '--motion-manifest',protocol['manifest'],'--output',str(output),'--physics',physics,
                '--repeats','2','--steps','500','--seed',str(protocol['physics_seed']),'--paired-starts',
                '--global-metrics','--memory-start','warm','--warmup-steps','500','--policy-precision','fp32']
            with output.with_suffix('.log').open('w') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    record_evaluation_sources(directory,protocol)
    subprocess.run([PYTHON,str(ROOT/'scripts/compare_heavy_teacher_tracking.py'),str(directory)],
                   cwd=ROOT,env=env,check=True,stdout=subprocess.DEVNULL)
    # Also show post-adaptation tracking within the same cold-start trajectories.
    import numpy as np
    segments={}
    for physics in ('nominal','mixed'):
        row=json.loads((directory/f'rma_student_{physics}.json').read_text())
        trace=np.load(directory/f'rma_student_{physics}.traces.npz')
        values=trace['all_metrics'];lengths=np.asarray(row['episode_lengths'])
        phase_metrics={}
        for phase,start,stop in (('cold',0,49),('full_history',49,values.shape[1])):
            mask=(np.arange(values.shape[1])[None]>=start)&(np.arange(values.shape[1])[None]<np.minimum(lengths,stop)[:,None])
            counts=mask.sum(1);valid=counts>0
            means=np.where(mask[...,None],values,0).sum(1)/counts[:,None].clip(1)
            phase_metrics[phase]={'contributing_worlds':int(valid.sum()),'steps_per_world':counts.tolist(),
                'mean':dict(zip(row['metric_names'],means[valid].mean(0).tolist())) if valid.any() else {},
                'scope':'Own student trajectory segment; excludes worlds with no valid steps in segment; use overall success alongside.'}
        segments[physics]={'history_full_steps_per_world':row['context_full_steps'],
            'cold_start_success_rate':1-row['failure_rate'],'phase_metrics':phase_metrics}
    atomic_json(directory/'history_diagnostics.json',segments)
    atomic_json(directory/'completion.json',{'complete':True,'student_updates':update,'unix_time':time.time()})
    return str(directory)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=RUN_ROOT);p.add_argument('--evaluation-gpu',type=int,default=7)
    p.add_argument('--once',action='store_true');p.add_argument('--evaluate-checkpoint',type=Path)
    args=p.parse_args();torch.set_num_threads(1);root=args.root.resolve()
    status=root/'monitor';status.mkdir(exist_ok=True)
    if args.evaluate_checkpoint:
        print(evaluate(root,args.evaluate_checkpoint,args.evaluation_gpu),flush=True);return
    with (status/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last_export=None
        while True:
            errors=[];live=True
            try:
                record=json.loads((root/'active_process.json').read_text())
                identity=process_identity(record['pid'],record['start_ticks']);live=identity['live']
                run=Path(record['output']);candidates=numeric_checkpoints(run)
                if not live and (run/'checkpoint_final.pt').exists():candidates.append(run/'checkpoint_final.pt')
                progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
                if candidates:
                    checkpoint=candidates[-1]
                    if str(checkpoint)!=last_export:
                        exported=run/'deploy'/checkpoint.stem
                        if not exported.exists():export_policy(checkpoint,exported)
                        metadata=json.loads((exported/'policy.json').read_text())
                        if metadata['checkpoint_sha256']!=digest(checkpoint) or not metadata['validation']['passed']:
                            raise RuntimeError('Export integrity check failed')
                        publish_latest(run,exported);last_export=str(checkpoint)
                    pending=[p for p in numeric_checkpoints(run) if int(p.stem.split('_')[-1])>0
                             and int(p.stem.split('_')[-1])%500==0
                             and not (root/'evaluations'/f"update_{int(p.stem.split('_')[-1]):06d}"/'completion.json').exists()]
                    if pending:
                        # Small 512-world evaluator shares GPU7 only while sufficient memory is free.
                        free=int(subprocess.check_output(['nvidia-smi','-i',str(args.evaluation_gpu),
                            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                        if free>=12000:evaluate(root,pending[0],args.evaluation_gpu)
                # A paired evaluation can take several minutes while training advances.
                if (run/'progress.json').exists():progress=json.loads((run/'progress.json').read_text())
                progressed = bool(progress) and progress.get('unix_time',0)>=record['started_at']
                age=time.time()-(progress['unix_time'] if progressed else record['started_at'])
                if live and age>(300 if progressed else 1200):errors.append('training_progress_stalled')
                metrics=run/'metrics.jsonl'
                if metrics.exists():
                    with metrics.open('rb') as f:
                        f.seek(max(0,metrics.stat().st_size-100000));lines=f.read().splitlines()
                    if lines:
                        recent=json.loads(lines[-1])
                        if any(isinstance(v,float) and not math.isfinite(v) for v in recent.values()):errors.append('nonfinite_training_metric')
                        if recent.get('Distill/gradient_accumulation_steps')!=1:errors.append('unexpected_gradient_accumulation')
                atomic_json(status/'health.json',{'checked_at':time.time(),'training_live':live,'progress':progress,
                    'last_export':last_export,'errors':errors,'status':'needs_attention' if errors else ('healthy' if progressed else 'initializing') if live else 'stopped'})
            except Exception as error:
                errors.append(repr(error));atomic_json(status/'health.json',{'checked_at':time.time(),'errors':errors,'status':'monitor_error'})
            print(json.dumps({'time':time.time(),'training_live':live,'errors':errors,'last_export':last_export}),flush=True)
            if args.once or (not live and not errors):return
            time.sleep(10)


if __name__=='__main__':main()
