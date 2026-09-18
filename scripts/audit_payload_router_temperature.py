"""Actual early-PPO trajectories: cold/full memory, grid/continuous gate sharpness."""

import argparse
import copy
import importlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout,_load_saved_config
from intact_tracking.limb_context_protocol import TRACKER,TRACKER_SHA256,PAYLOAD_EVENT,FULL_DATASET
from intact_tracking.limb_context_terminations import configure_training_terminations
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.payload_prototype_moe import PayloadMoEActor,PayloadMoEWrapper
from intact_tracking.payload_prototype_physics import configure_physics,audit_physics
from report_payload_expert_mapping import routing_mapping


@torch.inference_mode()
def capture_expert_actions(actor, obs, full, action_term, step):
    """Counterfactual expert means on identical current inputs; no sampled actions."""
    worlds=full.nonzero(as_tuple=False).flatten()
    if not worlds.numel(): return None
    rng_cpu=torch.random.get_rng_state()
    rng_cuda=torch.cuda.get_rng_state(full.device)
    features,base=actor._base_features_and_action(obs)
    common=actor.obs_encoder(features[worlds])
    fixed_base=base[worlds]
    residuals=torch.stack([expert(common,fixed_base) for expert in actor.expert_bank.experts],dim=1)
    ids,weights=actor.last_route_ids[worlds],actor.last_route_weights[worlds]
    selected=residuals.gather(1,ids[...,None].expand(-1,-1,residuals.shape[-1]))
    mixture=(selected*weights[...,None]).sum(1)
    expected=actor.last_residual_mean[worlds]
    torch.testing.assert_close(fixed_base,actor.last_base_action[worlds],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(mixture,expected,atol=2e-5,rtol=2e-4)
    if not torch.equal(rng_cpu,torch.random.get_rng_state()) or not torch.equal(rng_cuda,torch.cuda.get_rng_state(full.device)):
        raise RuntimeError('Counterfactual expert evaluation consumed the rollout RNG')
    scale=torch.broadcast_to(torch.as_tensor(action_term._scale,device=base.device),base.shape)[worlds]
    return {'worlds':worlds.cpu(),'step':torch.full((len(worlds),),step,dtype=torch.long),
            'features':features[worlds].cpu(),'base_action':fixed_base.cpu(),
            'expert_residual_means':residuals.cpu(),'ids':ids.cpu(),'weights':weights.cpu(),
            'actual_residual_mean':expected.cpu(),'action_std':actor.output_std[worlds].cpu(),
            'joint_target_scale':scale.cpu(),
            'dense_sparse_max_absolute_error':float((mixture-expected).abs().max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--motion-manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--mapping',action='store_true',help='Also retain traffic and compare selected experts with actual load labels')
    parser.add_argument('--action-mode',choices=('sample','mean','tracker_mean'),default='sample')
    args=parser.parse_args()
    torch.set_num_threads(2)
    configure_policy_precision('fp32')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    meta=state['residual_policy']
    completed_updates=state['completed_updates']
    prepared=prepare_rollout(checkpoint_file=TRACKER,num_envs=1024,motion_path=FULL_DATASET,motion_file=None)
    cfg=prepared.env
    cfg.episode_length_s=1000*cfg.decimation*cfg.sim.mujoco.timestep
    cfg.seed=910616
    physics=configure_physics(cfg,cfg.seed,anchor_fraction=.5)
    cfg.commands['motion'].motion_manifest_file=args.motion_manifest
    configure_training_starts(cfg,'original')
    configure_training_terminations(cfg,_load_saved_config(Path(TRACKER)),'original')
    context=load_memory350_checkpoint(meta['context_checkpoint'],device='cuda:0',expected_tracker_sha256=TRACKER_SHA256)
    _seed_everything(cfg.seed)
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
        audit=audit_physics(env,physics)
        wrapped=PayloadMoEWrapper(env,prepared.clip_actions,context)
        obs=wrapped.get_observations()
        agent=OmegaConf.to_container(state['cfg'].agent,resolve=True)
        kwargs=copy.deepcopy(agent['actor']);kwargs.pop('class_name')
        actor=PayloadMoEActor(obs,agent['obs_groups'],'actor',29,**kwargs).cuda()
        actor.load_state_dict(state['actor_state_dict'],strict=True)
        actor.eval().requires_grad_(False)
        del state
        rows={'cold_grid':[],'cold_continuous':[],'full_grid':[],'full_continuous':[]}
        traffic={name:[] for name in rows}
        payload=env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
        actual_masses=payload.observe().detach().cpu()
        anchor_ids=payload.anchor_ids.detach().cpu()
        grid=torch.tensor(physics['load_grid_kg'])
        full_ids=[]
        expert_action_snapshots=[]
        action_term=env.action_manager.get_term('joint_pos')
        previous=None
        switches=[0,0]
        with torch.inference_mode():
            for step in range(args.steps):
                actions=actor(obs,stochastic_output=True)
                if args.action_mode=='mean': actions=actor.output_mean
                elif args.action_mode=='tracker_mean': actions=actor.last_base_action
                ids,w=actor.last_route_ids,actor.last_route_weights
                bank=wrapped.context.memory
                full=(bank.short_count==50)&(bank.total_chunks-bank.session_start>=30)
                if args.mapping and step in (500,600,700):
                    snapshot=capture_expert_actions(actor,obs,full,action_term,step)
                    if snapshot is not None: expert_action_snapshots.append(snapshot)
                if step>=500 and previous is not None:
                    valid=full&previous[1]
                    switches[0]+=int(((ids[:,0]!=previous[0])&valid).sum())
                    switches[1]+=int(valid.sum())
                previous=(ids[:,0].clone(),full.clone())
                if step<50 or step>=500 and step%10==0:
                    for group,mask in [('grid',torch.arange(1024,device='cuda')<512),('continuous',torch.arange(1024,device='cuda')>=512)]:
                        if step<50: rows['cold_'+group].append(w[mask].cpu())
                        elif (full&mask).any(): rows['full_'+group].append(w[full&mask].cpu())
                        if args.mapping:
                            selected=mask if step<50 else full&mask
                            world_ids=selected.nonzero(as_tuple=False).flatten()
                            if world_ids.numel():
                                traffic[('cold_' if step<50 else 'full_')+group].append({
                                    'worlds':world_ids.cpu(),'ids':ids[selected].cpu(),'weights':w[selected].cpu(),
                                    'step':torch.full((len(world_ids),),step,dtype=torch.long)})
                    if step>=500: full_ids.append(ids[full].cpu())
                policy_mean=actor.last_base_action if args.action_mode=='tracker_mean' else actor.output_mean
                env.action_manager.get_term('joint_pos').record_policy_mean(policy_mean)
                obs,_,_,_=wrapped.step(actions)
                if (step+1)%100==0: print(json.dumps({'step':step+1,'full_history_fraction':float(full.float().mean())}),flush=True)
        result={'checkpoint':args.checkpoint,'prototype_sha256':meta['prototype_sha256'],
                'completed_updates':completed_updates,
                'action_mode':args.action_mode,
                'temperature':float(actor.gate.temperature),'physics_audit':audit,'seed':cfg.seed,
                'rollout':'1024 worlds with 50/50 grid/continuous loads; Gaussian draws consumed in every mode for matched RNG before trajectory divergence',
                'full_history_top1_switch_fraction':switches[0]/max(switches[1],1),'groups':{}}
        for name,batches in rows.items():
            w=torch.cat(batches);p=w.max(-1).values;eff=1/w.square().sum(-1)
            result['groups'][name]={'samples':len(w),'mean_max_weight':float(p.mean()),
                'max_weight_quantiles_05_25_50_75_95':p.quantile(torch.tensor([.05,.25,.5,.75,.95])).tolist(),
                'fraction_max_le_0p30':float((p<=.3).float().mean()),'fraction_max_ge_0p7':float((p>=.7).float().mean()),
                'fraction_max_ge_0p9':float((p>=.9).float().mean()),'mean_effective_experts':float(eff.mean()),
                'fraction_effective_ge_4p5':float((eff>=4.5).float().mean())}
        result['full_history_used_expert_ids']=int(torch.cat(full_ids).unique().numel())
        if args.mapping:
            traces={name:{key:torch.cat([batch[key] for batch in batches]) for key in batches[0]} for name,batches in traffic.items()}
            result['mapping']={name:routing_mapping(trace['ids'],trace['weights'],actual_masses[trace['worlds']],grid,
                anchor_ids=anchor_ids[trace['worlds']] if name.endswith('_grid') else None) for name,trace in traces.items()}
            trace_path=Path(args.output).with_suffix('.traffic.pt')
            torch.save({'traces':traces,'actual_masses_kg':actual_masses,'anchor_ids':anchor_ids,'grid_kg':grid,
                        'checkpoint':args.checkpoint,'completed_updates':completed_updates},trace_path)
            result['traffic_file']=str(trace_path)
            if expert_action_snapshots:
                action_path=Path(args.output).with_suffix('.expert_actions.pt')
                torch.save({'snapshots':expert_action_snapshots,'joint_names':list(action_term._target_names),
                            'actual_masses_kg':actual_masses,'anchor_ids':anchor_ids,'grid_kg':grid,
                            'checkpoint':args.checkpoint,'completed_updates':completed_updates,
                            'action_mode':args.action_mode,
                            'contract':'Same FP32 obs features and frozen tracker action for every expert; compare deterministic residual means. Adding the same tracker action leaves pairwise differences unchanged.'},action_path)
                result['expert_action_file']=str(action_path)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('mapping','physics_audit')}),flush=True)
    finally: env.close()


if __name__=='__main__':main()
