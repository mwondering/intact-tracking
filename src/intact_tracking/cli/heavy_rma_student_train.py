"""Four-rank, uniform HDR RMA student distillation with a fixed embedded teacher."""

import argparse
import copy
import faulthandler
import json
import os
from pathlib import Path
import signal
import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking import memory350_native_policy as native, memory350_heavy_policy as heavy
from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import capture_original_rewards, assert_rewards_unchanged
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.distributed import DistributedContext
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.heavy_baseline_env import physical_schema
from intact_tracking.heavy_rma_teacher import parameter_encoder, PHYSICS_GROUP
from intact_tracking.heavy_rma_student import (VERSION, INPUT_CONTRACT, RMAStudentActor, HistoryBank,
                                               EMBEDDING_GROUP, build_actor)
from intact_tracking.limb_context_distributed import main_process_call
from intact_tracking.limb_context_terminations import configure_training_terminations, audit_runtime_terminations
from intact_tracking.memory350_checkpoint import embedded_tracker
from intact_tracking.memory350_native_dr import native_dataset_identity
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.residual_dr_aux import capture_dr_aux_targets
from intact_tracking.rma_student_env import RMAStudentWrapper
from intact_tracking.rma_student_distillation import Distiller, atomic_json, accumulate_episode_logs
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.wandb_logger import WandbLogger

ROOT = Path(__file__).resolve().parents[3]
TEACHER = ROOT/'runs/144000-exp-heavy/baselines/rma_teacher_uniform/continuous_uniform/checkpoint_9000.pt'
TEACHER_SHA256 = 'fbd26367b3cc014f2d7c2fa181bb7801912fefe017f4dee68286d5e4f53ec9e2'
MANIFEST_SHA256 = '59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac'
METRIC_GROUPS = ('Distill','DistillLatent','DistillGroup','Residual','Metrics','Episode',
                 'Episode_Metrics','Episode_Reward','Episode_Termination','Train','Perf')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--teacher-checkpoint', default=str(TEACHER))
    p.add_argument('--resume')
    p.add_argument('--num-envs', type=int, default=8192)
    p.add_argument('--training-ranks', type=int, default=4)
    p.add_argument('--rollout-steps', type=int, default=24)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    p.add_argument('--mini-batches', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=49152, help='Default is one complete per-rank minibatch, without accumulation')
    p.add_argument('--seed', type=int, default=121)
    p.add_argument('--iterations', type=int, help='Optional total completed distillation updates; default no cap')
    p.add_argument('--save-interval', type=int, default=100)
    p.add_argument('--motion-path', default=native.FULL_DATASET)
    p.add_argument('--motion-file')
    p.add_argument('--bounded-smoke', action='store_true')
    p.add_argument('--wandb-name', default='144000-exp-heavy-rma-student-uniform')
    return p


@torch.no_grad()
def diagnostics(actor, obs, targets, wrapped, distributed):
    indices = torch.linspace(0,len(targets)-1,min(1024,len(targets)),device=targets.device).long()
    obs = obs[indices]
    z = obs[EMBEDDING_GROUP]
    target = targets[indices]
    predicted = actor(obs)
    expected = actor.action_from_embedding(obs, target)
    base, residual = actor.last_base_action, actor.last_residual_mean
    stats = distributed.mean_scalars({
        'Distill/action_mse': float((predicted-expected).square().mean()),
        'Distill/current_latent_mse': float((z-target).square().mean()),
        'Distill/student_embedding_ms': float(z.square().mean()),
        'Distill/teacher_embedding_ms': float(target.square().mean()),
        'Residual/mean_ms': float(residual.square().mean()),
        'Residual/tracker_action_ms': float(base.square().mean()),
        **wrapped.latent_metrics})
    for src,dst in [('Distill/action_mse','Distill/action_rmse'),
                    ('Distill/student_embedding_ms','Distill/student_embedding_rms'),
                    ('Distill/teacher_embedding_ms','Distill/teacher_embedding_rms'),
                    ('Residual/mean_ms','Residual/mean_rms'),('Residual/tracker_action_ms','Residual/tracker_action_rms')]:
        stats[dst] = stats.pop(src)**.5
    stats['Residual/mean_to_tracker_rms_ratio'] = stats['Residual/mean_rms']/max(stats['Residual/tracker_action_rms'],1e-8)
    maximum = residual.abs().max()
    if distributed.enabled:
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    stats['Residual/mean_abs_max'] = float(maximum)
    nominal = torch.zeros(len(targets),dtype=torch.bool,device=targets.device)
    nominal[wrapped.unwrapped.native_policy_nominal_ids] = True
    nominal = nominal[indices]
    full = wrapped.history.count[indices] == 50
    masks = {'nominal':nominal,'hdr':~nominal,'cold':~full,'full_history':full}
    mass = wrapped.unwrapped.heavy_policy_payload.mass[indices]
    for limb,limit in enumerate((2.5,2.5,4.,4.)):
        bins = (mass[:,limb]/(limit/4)).long().clamp(0,3)
        for b in range(4):masks[f'limb{limb}_mass_bin{b}'] = (~nominal)&(bins==b)
    latent_error,action_error=(z-target).square().mean(-1),(predicted-expected).square().mean(-1)
    packed=torch.stack([torch.stack((latent_error[m].sum(),action_error[m].sum(),m.sum())) for m in masks.values()])
    distributed.all_reduce_sum(packed)
    for name,(latent_sum,action_sum,count) in zip(masks,packed):
        stats[f'DistillGroup/{name}_count']=float(count)
        if count>0:
            stats[f'DistillGroup/{name}_latent_mse']=float(latent_sum/count)
            stats[f'DistillGroup/{name}_action_rmse']=float((action_sum/count).sqrt())
    return stats


def run(args, distributed):
    if min(args.num_envs,args.rollout_steps,args.mini_batches,args.microbatch,args.save_interval) <= 0:
        raise ValueError('Sizes and intervals must be positive')
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError('iterations must be positive')
    if args.training_ranks != distributed.world_size or args.learning_rate <= 0:
        raise ValueError('Invalid rank count or learning rate')
    if not args.bounded_smoke and (args.num_envs,args.training_ranks,args.rollout_steps,args.mini_batches,args.microbatch,args.seed,args.learning_rate) != (8192,4,24,4,49152,121,5e-4):
        raise ValueError('Formal student scale/hyperparameters must match the approved plan')
    if not args.bounded_smoke and (args.motion_file or Path(args.motion_path).resolve()!=Path(native.FULL_DATASET)):
        raise ValueError('Formal student requires the complete filtered catalog')
    if args.bounded_smoke and args.iterations is None:
        raise ValueError('Smoke must have a finite update budget')
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Artifacts must remain in repository')
    def make_output():
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
    main_process_call(distributed, make_output)
    configure_policy_precision('fp32')
    source_path = Path(args.resume or args.teacher_checkpoint).resolve()
    state = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    if args.resume:
        if state['residual_policy']['version'] != VERSION:
            raise ValueError('Expected RMA student resume checkpoint')
        source = state['teacher_source']
    else:
        source = {'checkpoint': str(source_path), 'sha256': _sha256(source_path),
                  'completed_updates': state['completed_updates']}
        if source['sha256'] != TEACHER_SHA256 or state['residual_policy']['method'] != 'rma_teacher':
            raise ValueError('The approved fixed RMA teacher differs')
    tracker = embedded_tracker(state)
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    rank_seed = args.seed+1000003*distributed.rank
    files, dataset = native_dataset_identity(source_path,args.motion_file,None if args.motion_file else args.motion_path)
    if not args.bounded_smoke and (len(files)!=220480 or dataset['manifest_sha256']!=MANIFEST_SHA256):
        raise ValueError('Full filtered dataset identity changed')
    _seed_everything(rank_seed)
    prepared = prepare_rollout(checkpoint_file=tracker['source_path'],checkpoint_config=tracker['cfg'],
                               num_envs=args.num_envs,motion_file=args.motion_file,
                               motion_path=None if args.motion_file else args.motion_path)
    cfg = prepared.env
    cfg.seed = rank_seed
    cfg.episode_length_s = 500*cfg.decimation*cfg.sim.mujoco.timestep
    rewards = capture_original_rewards(cfg)
    physics = heavy.configure_physics(cfg,rank_seed,profile=heavy.PROFILE,rank=distributed.rank)
    terms = configure_training_terminations(cfg,OmegaConf.create(tracker['cfg']),'original')
    sampling = native.configure_sampling(cfg,'uniform',0,0)
    starts = configure_training_starts(cfg,'original')
    assert_rewards_unchanged(rewards,cfg)
    if prepared.clip_actions is not None or cfg.actions['joint_pos'].clip is not None or cfg.actions['joint_pos'].raw_action_clip is not None:
        raise ValueError('Student must use unclipped total residual commands')
    env = heavy.environment_factory(ManagerBasedRlEnv,cfg=cfg,device=str(distributed.device))
    logger, success, handlers = None, False, {}
    try:
        command = env.command_manager.get_term('motion')
        expected = files[distributed.rank::distributed.world_size] if len(files)>1 else files
        if not distributed.all_true(tuple(map(str,expected))==command.motion_files):
            raise ValueError('Actual motion shard differs from requested catalog')
        audits = distributed.all_gather_object({'rank':distributed.rank,'motion_count':len(command.motion_files),
            'physics':heavy.audit_physics(env,physics),'sampling':native.audit_sampling(command,sampling),
            'terminations':audit_runtime_terminations(env.termination_manager,terms)})
        wrapped = RMAStudentWrapper(env,prepared.clip_actions)
        schema = physical_schema(env)
        if schema != state['residual_policy']['physics_input_schema']:
            raise ValueError('Teacher physical target schema changed')
        actor = build_actor(state,RMAStudentActor).to(distributed.device)
        label_encoder = parameter_encoder().to(distributed.device).eval().requires_grad_(False)
        if args.resume:
            actor.load_state_dict(state['actor_state_dict'],strict=True)
            label_encoder.load_state_dict(state['teacher_dr_encoder_state_dict'],strict=True)
        else:
            teacher = build_actor(state).to(distributed.device).eval().requires_grad_(False)
            teacher.load_state_dict(state['actor_state_dict'],strict=True)
            actor.initialize_from_teacher(teacher)
            label_encoder.load_state_dict(teacher.dr_encoder.state_dict(),strict=True)
        with torch.no_grad():
            theta = capture_dr_aux_targets(env,schema)*2-1
            targets = label_encoder(theta).detach()
        wrapped.bind_policy(actor)
        obs = wrapped.get_observations()
        with torch.no_grad():
            checked = obs[:128].clone()
            if not args.resume:
                checked.set(PHYSICS_GROUP,theta[:128])
                expected_action = teacher(checked)
                torch.testing.assert_close(actor.action_from_embedding(checked,targets[:128]),expected_action,atol=2e-5,rtol=2e-5)
                del teacher
            clean = obs[:128].select(*actor.tracker.obs_groups,EMBEDDING_GROUP)
            expected_action = actor(clean)
            poisoned = clean.clone(); poisoned.set(PHYSICS_GROUP,torch.full((len(clean),108),float('nan'),device=env.device))
            torch.testing.assert_close(actor(poisoned),expected_action,atol=0,rtol=0)
        trainer = Distiller(actor,distributed,learning_rate=args.learning_rate,
                            mini_batches=args.mini_batches,microbatch=args.microbatch,seed=args.seed)
        checkpoint_cfg = copy.deepcopy(state['cfg'])
        OmegaConf.set_struct(checkpoint_cfg,False)
        checkpoint_cfg.agent.actor.class_name = 'intact_tracking.heavy_rma_student:RMAStudentActor'
        metadata = copy.deepcopy(state['residual_policy'])
        metadata.update(version=VERSION,method='rma_student',fusion='concat',arguments=vars(args),
            teacher_source=source,teacher_frozen=True,context_checkpoint=None,context_sha256=None,
            maximum_updates=args.iterations,initialization_protocol='RMA phase2: frozen teacher control; fresh adaptation',
            encoder_frozen=False,encoder_freezing_scope='only adaptation trainable; frozen tracker and RMA controller',
            predictor_executed_in_ppo=False,critic_normalization_initialization=None,
            actor_initialization='copy frozen teacher controller; new proprio50 adaptation',
            critic_initialization=None,extra_current_state_physics_privilege=False,
            frozen_inference='SPV5-2A tracker and RMA residual controller',input_contract=INPUT_CONTRACT,
            baseline_contract={'encoder_objective':'actor embedding MSE','action_loss_coefficient':0.,'PPO':False},
            dataset=dataset,physics=physics,motion_sampling=sampling,training_starts=starts,
            training_terminations=terms,reward_contract=rewards,physics_input_schema=schema,
            execution_protocol='rma_student_online_distillation_4gpu8192_v1',
            distributed={'world_size':distributed.world_size,'num_envs_per_rank':args.num_envs,
                         'global_num_envs':args.num_envs*distributed.world_size},
            runtime_audits=audits,student_parameters=sum(p.numel() for p in actor.adaptation.parameters()),
            startup_audit={'teacher_action_alignment':True,'no_privileged_actor_inputs':True},
            resume_contract='restore model optimizer moments rank RNG; restart simulator and clear history',
            research_source_sha256={str(p.relative_to(ROOT)):_sha256(p) for p in (
                Path(__file__),*(ROOT/'src/intact_tracking').glob('*rma_student*.py'),
                ROOT/'src/intact_tracking/rma_student_distillation.py',ROOT/'src/intact_tracking/rma_student_env.py')})
        metadata['wandb_logging'].update(metric_groups=list(METRIC_GROUPS),step_metric='completed_updates')
        if args.resume:
            old = state['residual_policy']
            for key in ('num_envs','training_ranks','rollout_steps','mini_batches','microbatch','seed','learning_rate'):
                if old['arguments'][key]!=vars(args)[key]:
                    raise ValueError('Resume changed '+key)
            if old['dataset']['manifest_sha256']!=dataset['manifest_sha256'] or state['teacher_source']['sha256']!=TEACHER_SHA256:
                raise ValueError('Resume changed teacher or dataset')
            trainer.resume(state)
        else:
            _seed_everything(rank_seed+1)
        main_process_call(distributed,lambda:atomic_json(output/'run_config.json',metadata))
        logger = WandbLogger(enabled=not args.bounded_smoke,is_main=distributed.is_main,
            project='intact-preview-v2',entity='2486344338-zhejiang-university',group='144000-exp-heavy',
            name=args.wandb_name,output_dir=output,config=metadata,tags=('heavy','rma','student','uniform','distillation'))
        if logger.run is not None:
            logger.run.define_metric('completed_updates')
            for group in METRIC_GROUPS:
                logger.run.define_metric(group+'/*',step_metric='completed_updates')
            atomic_json(output/'wandb_run.json',{'id':logger.id,'url':logger.url})
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
            os.write(2,b'Saving RMA student after current complete distillation update.\n')
        handlers = {s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        def save(name):
            return trainer.save(output/name,source_state=state,cfg=checkpoint_cfg,metadata=metadata,teacher_encoder=label_encoder)
        save('checkpoint_resume.pt' if args.resume else 'checkpoint_0.pt')
        bank = HistoryBank(args.num_envs,args.rollout_steps,env.device)
        episode_returns = torch.zeros(args.num_envs,device=env.device)
        episode_lengths = torch.zeros_like(episode_returns)
        while args.iterations is None or trainer.completed_updates<args.iterations:
            start = time.perf_counter(); bank.start(wrapped.history)
            episode = torch.zeros(4,device=env.device)
            logs, log_counts = {}, {}
            with torch.no_grad():
                for step in range(args.rollout_steps):
                    action = actor(obs)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('Nonfinite student action')
                    env.action_manager.get_term('joint_pos').record_policy_mean(action)
                    obs,reward,dones,extras = wrapped.step(action)
                    bank.append(step,wrapped.history)
                    episode_returns.add_(reward); episode_lengths.add_(1)
                    done = dones.bool()
                    episode += torch.stack((episode_returns[done].sum(),episode_lengths[done].sum(),done.sum(),env.reset_terminated.sum()))
                    episode_returns[done]=0;episode_lengths[done]=0
                    accumulate_episode_logs(logs,log_counts,extras.get('log',extras.get('episode',{})))
            torch.cuda.synchronize(distributed.device)
            collected = time.perf_counter()-start; start=time.perf_counter()
            torch.cuda.reset_peak_memory_stats(distributed.device)
            metrics = trainer.update(bank,targets)
            torch.cuda.synchronize(distributed.device)
            learned = time.perf_counter()-start
            metrics['Perf/torch_learning_peak_gib'] = torch.cuda.max_memory_allocated(distributed.device)/2**30
            metrics['Distill/gradient_accumulation_steps'] = max(1,(args.num_envs*args.rollout_steps//args.mini_batches+args.microbatch-1)//args.microbatch)
            with torch.no_grad():obs=wrapped.get_observations()
            metrics.update(diagnostics(actor,obs,targets,wrapped,distributed))
            distributed.all_reduce_sum(episode)
            keys=sorted(set().union(*distributed.all_gather_object(list(logs))))
            if keys:
                values=torch.stack([torch.as_tensor(logs.get(k,0.),device=env.device,dtype=torch.float32) for k in keys])
                counts=torch.tensor([log_counts.get(k,0) for k in keys],device=env.device)
                distributed.all_reduce_sum(values);distributed.all_reduce_sum(counts)
                metrics.update({k if '/' in k else 'Episode/'+k:float(v/c.clamp_min(1)) for k,v,c in zip(keys,values,counts)})
            timing=distributed.mean_scalars({'Perf/collection_time':collected,'Perf/learning_time':learned})
            metrics.update(timing,completed_updates=trainer.completed_updates,unix_time=time.time(),
                **{'Perf/total_fps':args.num_envs*distributed.world_size*args.rollout_steps/(timing['Perf/collection_time']+timing['Perf/learning_time']),
                   'Train/completed_episodes':float(episode[2]),
                   'Train/failure_count':float(episode[3])})
            if episode[2]>0:
                metrics.update({'Train/mean_reward':float(episode[0]/episode[2]),
                                'Train/mean_episode_length':float(episode[1]/episode[2])})
            if distributed.is_main:
                with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                atomic_json(output/'progress.json',{'completed_updates':trainer.completed_updates,'unix_time':time.time()})
                logger.log(metrics,step=trainer.completed_updates)
                print(json.dumps({k:metrics[k] for k in ('completed_updates','Distill/latent_mse','Distill/action_rmse','Train/mean_episode_length','Perf/total_fps') if k in metrics}),flush=True)
            stopping = not distributed.all_true(not stopping)
            if trainer.completed_updates%args.save_interval==0:
                save(f'checkpoint_{trainer.completed_updates}.pt')
            if stopping:break
            torch.cuda.empty_cache()
        agreement=save('checkpoint_final.pt')
        main_process_call(distributed,lambda:atomic_json(output/'completion.json',{
            'completed_updates':trainer.completed_updates,'stopped':stopping,
            'complete':args.iterations is not None and trainer.completed_updates>=args.iterations,
            'distributed_parameter_agreement':agreement,'distributed':metadata['distributed'],'unix_time':time.time()}))
        success=True
    finally:
        for s,h in handlers.items():signal.signal(s,h)
        env.close()
        if logger is not None:logger.finish(exit_code=0 if success else 1)


def main():
    args=build_parser().parse_args()
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    distributed=DistributedContext.initialize()
    try:run(args,distributed)
    finally:distributed.close()


if __name__=='__main__':main()
