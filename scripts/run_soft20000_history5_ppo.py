"""Wait for the authorized soft u20000, evaluate, then start the matched PPO pair.

No implicit restart: saved process identities and outputs are authoritative.
STOP in run-root stops this coordinator; it never kills unknown GPU processes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_limb_context_experiment import process_environment, ROOT, PYTHON
from monitor_memory350_nominal_direction import process_identity, worker_processes


def read(path):
    for retry in range(3):
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if retry == 2:
                raise
            time.sleep(.1)


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def validate_ready(root):
    ready=read(root/'PPO_READY.json')
    if not ready['passed']:raise RuntimeError('PPO preflight not passed')
    for path,sha in {**ready['source_sha256'],**ready['artifact_sha256']}.items():
        if digest(ROOT/path)!=sha:raise RuntimeError(f'Validated file changed: {path}')
    plan=read(root/'plan.json');launch=read(root/'formal_commands.json')
    expected={'concat':[0,1,2,3],'baseline':[4,5,6,7]}
    if plan['assignments']!=expected or launch['assignments']!=expected:
        raise RuntimeError('GPU assignment changed')
    from intact_tracking.cli.memory350_history5_policy_train import build_parser
    from intact_tracking.memory350_policy_checkpoint_eval import load_protocol
    for arm,command in launch['commands'].items():
        module='intact_tracking.cli.memory350_history5_policy_train'
        args=build_parser().parse_args(command[command.index(module)+1:])
        if (args.fusion!=arm or args.num_envs!=8192 or args.training_ranks!=4
                or args.motion_sampling!='uniform' or args.training_terminations!='original'
                or not args.until_user_stop or args.bounded_smoke or args.resume
                or args.policy_precision!='fp32' or args.adaptive_after_update!=0
                or args.seed!=121 or args.endpoint_eval_protocol!=plan['periodic_evaluation']
                or args.output_dir!=str(root/'ppo'/f'{arm}_121')):
            raise RuntimeError(f'PPO command violates the confirmed contract: {arm}')
        if arm=='concat' and args.context_checkpoint!=plan['encoder_checkpoint']:
            raise RuntimeError('PPO context is not the selected u20000 encoder')
    protocol,_=load_protocol(plan['periodic_evaluation'])
    if not protocol.get('global_metrics') or protocol['primary_cases']!=['mixture_cold','mixture_warm']:
        raise RuntimeError('Paired evaluation contract changed')
    return ready


def update_comparison(root):
    rows={}
    for arm in ('baseline','concat'):
        path=root/'ppo'/f'{arm}_121/endpoint_eval_metrics.jsonl'
        # A writer may be partway through its last append. Only read full rows.
        saved=[json.loads(line) for line in path.read_text().split('\n')[:-1] if line.strip()] if path.exists() else []
        rows[arm]={row['completed_updates']:row for row in saved}
        if len(saved)!=len(rows[arm]):raise ValueError('Duplicate completed PPO evaluation')
    result=[]
    for update in sorted(rows['baseline'].keys()&rows['concat'].keys()):
        a,b=rows['baseline'][update],rows['concat'][update]
        if a['protocol_sha256']!=b['protocol_sha256'] or a['policy_precision']!=b['policy_precision']:
            raise ValueError('PPO comparison protocol or precision differs')
        result.append({'completed_updates':update,'baseline':a['metrics'],'latent':b['metrics'],
                       'policy_precision':a['policy_precision'],'same_update_comparison':True})
    write(root/'ppo_comparison.json',{'matched_updates':[r['completed_updates'] for r in result],
          'rows':result,'maximum_updates':None,'updated_at':time.time()})
    return [r['completed_updates'] for r in result]


def paired_report(root, update):
    target=root/'paired_results'/f'update_{update:06d}.json'
    if target.exists():return
    import numpy as np
    from intact_tracking.memory350_policy_results import compare
    protocol=read(root/'protocols/periodic.json')
    result={'completed_updates':update,'training_seed_count':1,'training_seed':121,
            'primary_cases':protocol['primary_cases'],'comparisons':{},'inputs':{},
            'uncertainty':'Paired motion bootstrap; not variation across training seeds',
            'scope':'Training-catalog motions with held-out physics seeds and starts',
            'maximum_updates':None}
    for case in protocol['cases']:
        rows={};traces={};result['inputs'][case]={}
        for arm in ('baseline','concat'):
            path=root/'ppo'/f'{arm}_121/endpoint_eval'/f'update_{update:06d}'/f'{case}.json'
            row=read(path)
            if row['completed_training_updates']!=update:raise RuntimeError('Unmatched PPO update')
            rows[arm]=row
            with np.load(path.with_suffix('.traces.npz')) as saved:
                if saved['lengths'].tolist()!=row['episode_lengths']:
                    raise ValueError('Trace lengths do not match evaluation')
                traces[arm]=saved['all_metrics'].copy()
            result['inputs'][case][arm]={'path':str(path),'sha256':digest(path),
                                        'checkpoint_sha256':row['checkpoint_sha256']}
        result['comparisons'][case]=compare(rows['baseline'],rows['concat'],traces['baseline'],traces['concat'])
    write(target,result)
    write(root/'latest_paired_result.json',{'completed_updates':update,'path':str(target),
          'sha256':digest(target),'stop_at_comparison':False,'unix_time':time.time()})


def audit_started_ppo(root, selection):
    output=root/'ppo_start_verification.json'
    if output.exists():return
    directories={arm:root/'ppo'/f'{arm}_121' for arm in ('baseline','concat')}
    if not all((p/name).exists() for p in directories.values()
               for name in ('run_config.json','checkpoint_initial.pt','wandb_run.json')):return
    import torch
    configs={arm:read(p/'run_config.json') for arm,p in directories.items()}
    checks={}
    for arm,p in directories.items():
        config=configs[arm];audit=config['input_audit'];physics=config['physics']
        initial=torch.load(p/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
        checks[arm+'_scratch']=initial['completed_updates']==0 and not initial['optimizer_state_dict']['state']
        del initial
        checks[arm+'_shape']=audit['actor_fusion_input_dim']==audit['critic_fusion_input_dim']==448 and audit['latent_history_frames']==5
        checks[arm+'_scale']=config['distributed']['world_size']==4 and config['distributed']['num_envs_per_rank']==8192
        checks[arm+'_uniform']=config['motion_sampling']['active_mode']=='uniform' and config['arguments']['motion_sampling']=='uniform'
        checks[arm+'_termination']=config['arguments']['training_terminations']=='original'
        checks[arm+'_nominal_minimum']=len(physics['runtime_audits_by_rank'])==4 and all(
            row['physics']['nominal_population']['count']==820 and row['physics']['nominal_population']['fraction']>=.1
            for row in physics['runtime_audits_by_rank'])
        checks[arm+'_coordinate_mixture']=physics['independent_nominal_mixture']['probability']==.5
        checks[arm+'_unbounded']=config['arguments']['until_user_stop'] and not config['arguments']['bounded_smoke']
    a,b=configs['baseline'],configs['concat']
    checks['context_identity']=a['context_sha256'] is None and b['context_sha256']==selection['sha256']
    checks['frozen_context']=b['encoder_frozen'] and b['context_normalization_frozen']
    checks['paired_initial_weights']=all(a['input_audit'][k]==b['input_audit'][k]
        for k in ('actor_common_trunk_sha256','critic_common_trunk_sha256'))
    checks['paired_mass_samples']=all(x['physics']['sampled_mass_sha256']==y['physics']['sampled_mass_sha256']
        for x,y in zip(a['physics']['runtime_audits_by_rank'],b['physics']['runtime_audits_by_rank']))
    result={'passed':all(checks.values()),'checks':checks,'unix_time':time.time(),
            'wandb':{arm:read(p/'wandb_run.json') for arm,p in directories.items()}}
    write(output,result)
    if not result['passed']:raise RuntimeError(f'PPO startup mismatch: {[k for k,v in checks.items() if not v]}')


def verified_leader(root):
    rec=read(root/'train_process.json')
    return process_identity(rec['pid'],rec['process_start_ticks'])


def stop_encoder(root):
    leader=verified_leader(root)
    if not leader['live']:
        return {'already_exited':True,'leader':leader}
    workers=worker_processes(leader)
    if len(workers)!=4 or not all(w['live'] for w in workers):
        raise RuntimeError(f'Expected four live, known encoder workers at {root}')
    # One worker sets the trainer's synchronized finish-and-save flag.
    first=workers[0];again=process_identity(first['pid'],first['start_ticks'])
    if not again['live']:raise RuntimeError('Encoder worker identity changed')
    os.kill(first['pid'],signal.SIGTERM)
    return {'signal_worker':first,'leader':leader,'requested_at':time.time()}


class Coordinator:
    def __init__(self, root):
        self.root=root;self.plan=read(root/'plan.json');self.children={}
        self.phase='starting';self.stop=False

    def status(self, **values):
        write(self.root/'state.json',{'status':self.phase,'unix_time':time.time(),
            'coordinator_pid':os.getpid(),**values})

    def check_stop(self):
        if self.stop or (self.root/'STOP').exists():
            raise InterruptedError('Coordinator stopped by user request')

    def wait_processes(self, jobs):
        while True:
            self.check_stop()
            codes={k:p.poll() for k,p in jobs.items()}
            if any(v is not None and v!=0 for v in codes.values()):
                raise RuntimeError(f'{self.phase} child failed: {codes}')
            if all(v is not None for v in codes.values()):return
            self.status(children={k:{'pid':p.pid,'returncode':codes[k]} for k,p in jobs.items()})
            time.sleep(15)

    def launch(self, name, command, env):
        record=self.root/f'{name}_process.json'
        if record.exists():raise FileExistsError(f'Refuse duplicate launch: {record}')
        with (self.root/f'{name}.log').open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_identity(p.pid)
        write(record,{'pid':p.pid,'start_ticks':identity['start_ticks'],'command':command,'started_at':time.time()})
        self.children[name]=p
        return p

    def run(self):
        validate_ready(self.root)
        soft=Path(self.plan['encoder_source_root']);target=Path(self.plan['encoder_checkpoint'])
        control=ROOT/'runs/limb_context_20260917_memory350_nominal_dr_rank_scale06_resume_4x8192'
        self.phase='waiting_for_encoder_20000'
        while not target.exists():
            self.check_stop();leader=verified_leader(soft)
            if not leader['live']:raise RuntimeError('Encoder exited before u20000; no automatic restart')
            try:progress=read(soft/'stage1_8192/progress.json')
            except FileNotFoundError:progress=read(soft/'monitor/status.json')['progress']
            self.status(encoder=leader,progress=progress,target=str(target));time.sleep(30)
        self.check_stop()
        validate_ready(self.root)
        import torch
        state=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
        if state['update']!=20000 or state['loss_config'].get('dr_soft_h')!=.15:
            raise RuntimeError('Unexpected target encoder')
        if not state['distributed_parameter_agreement']['passed']:
            raise RuntimeError('Encoder checkpoint lacks rank agreement')
        selection={'checkpoint':str(target),'sha256':digest(target),'update':20000,
                   'source_loss':state['loss_config']}
        del state
        write(self.root/'selected_context.json',selection)
        self.phase='evaluating_encoder_20000'
        env=process_environment();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        out=self.root/'encoder_evaluation';out.mkdir(exist_ok=True)
        jobs={}
        jobs['encoder_cached']=self.launch('encoder_cached',[PYTHON,'-B','-u','scripts/evaluate_memory350_nominal_direction_cached.py',
            '--reference',str(soft/'stage1_8192/update_010750.pt'),'--candidate',str(target),
            '--cache',str(ROOT/'runs/latent_cluster_probe_memory350_nominal50_u5000_20260911'),
            '--output',str(out/'cached'),'--profiles','common','memory_training','--threads','4'],env)
        jobs['encoder_heldout']=self.launch('encoder_heldout',[PYTHON,'-B','-u','scripts/evaluate_soft_encoder_heldout.py',
            '--reference',str(soft/'stage1_8192/update_010750.pt'),'--candidate',str(target),
            '--validation',str(ROOT/'runs/limb_context_20260917_memory350_nominal_dr_rank_resume_4x8192/stage1_8192'),
            '--output',str(out)],env)
        self.wait_processes(jobs)
        cached=read(out/'cached/summary.json');heldout=read(out/'heldout.json')
        if (not cached['complete'] or not heldout['complete']
                or heldout['checkpoints']['soft']['sha256']!=selection['sha256']
                or cached['checkpoints']['memory350']['sha256']!=selection['sha256']):
            raise RuntimeError('Encoder evaluation identity mismatch')
        write(out/'verification.json',{'passed':True,'checkpoint':selection,'cached_complete':True,
              'same_query_heldout_completed':True})
        self.check_stop()
        validate_ready(self.root)
        self.phase='releasing_encoder_gpus'
        write(self.root/'encoder_stop_requests.json',{str(p):stop_encoder(p) for p in (soft,control)})
        deadline=time.time()+300
        while any(verified_leader(p)['live'] for p in (soft,control)):
            self.check_stop();self.status();time.sleep(10)
            if time.time()>deadline:raise RuntimeError('Graceful encoder stop is taking longer than five minutes; inspect, do not force-kill')
        for p in (soft,control):
            completion=read(p/'stage1_8192/completion.json')
            if not completion.get('stopped'):raise RuntimeError(f'Encoder did not confirm final save: {p}')
        from run_memory350_scale_nominal_stage1 import gpu_status
        cards=gpu_status(list(range(8)))
        if any(c['processes'] or c['free_mib']<60000 for c in cards):
            raise RuntimeError('GPUs not released or occupied by another job; never kill unknown jobs')
        write(self.root/'gpu_before_ppo.json',cards)
        validate_ready(self.root)
        launch=read(self.root/'formal_commands.json')
        if launch['assignments']!={'concat':[0,1,2,3],'baseline':[4,5,6,7]}:
            raise RuntimeError('GPU assignment changed')
        self.phase='starting_paired_ppo'
        jobs={}
        for arm,command in launch['commands'].items():
            env=process_environment();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,launch['assignments'][arm])),
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                WANDB_API_KEY=(ROOT/'.runtime/limb_context/wandb_api_key').read_text().strip(),
                WANDB_RESUME='allow',WANDB_RUN_ID='soft20k-h5-'+hashlib.sha256((str(self.root)+arm).encode()).hexdigest()[:12])
            if arm=='concat' and command[command.index('--context-checkpoint')+1]!=str(target):
                raise RuntimeError('PPO context is not the evaluated encoder')
            jobs[arm]=self.launch('ppo_'+arm,command,env)
        self.phase='paired_ppo_training'
        while True:
            self.check_stop()
            for arm,p in jobs.items():
                if p.poll() is not None:raise RuntimeError(f'{arm} PPO exited: {p.returncode}; no restart')
            audit_started_ppo(self.root,selection)
            for update in update_comparison(self.root):
                paired_report(self.root,update)
            progress={}
            for arm in jobs:
                p=self.root/'ppo'/f'{arm}_121/progress.json'
                if p.exists():progress[arm]=read(p)
            self.status(progress=progress,selected_context=selection,maximum_updates=None,
                        assignments=launch['assignments'])
            time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--check',action='store_true',help='Validate preparation without starting or stopping any job')
    args=p.parse_args();root=args.run_root.resolve()
    if not root.is_relative_to(ROOT):raise ValueError('Run must stay inside repository')
    if args.check:
        validate_ready(root);print('Ready; no jobs launched or stopped',flush=True);return
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=Coordinator(root)
        def stop(*_):c.stop=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        identity=process_identity(os.getpid());write(root/'coordinator_process.json',identity)
        try:c.run()
        except BaseException as e:
            # Never erase a checkpoint or restart a stopped job. Preserve live PPO
            # on observation failure; user STOP explicitly stops owned children.
            if isinstance(e,InterruptedError):
                for child in c.children.values():
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            write(root/'coordinator_error.json',{'phase':c.phase,'error':repr(e),'unix_time':time.time()})
            c.status(error=repr(e));raise


if __name__=='__main__':main()
