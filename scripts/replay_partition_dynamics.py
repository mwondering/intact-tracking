"""Matched-action mjwarp fingerprints for independent latent partition evaluation."""
import hashlib
import json
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from intact_tracking.rollout.nominal import NominalPairRollout, NominalPairRolloutConfig
from intact_tracking.limb_context_protocol import TRACKER
from intact_tracking.preview_protocol import LIMBS
from intact_tracking.environment.mdp.randomizations import rigid_body_payload, _reconstruct_pseudo_inertia_J, _decompose_pseudo_inertia_J
from mjlab.sim.sim import RecomputeLevel
from probe_motion_response_metrics import snapshots

ROOT=Path('runs/limb_context_20260916_latent_environment_partitions')
SOURCE=Path('runs/limb_context_20260916_dr16384/analysis')
ANCHORS=Path('runs/limb_context_20260916_motion_metric_probe')


def fresh_anchors(out, query_step=2400):
    """New states from the 16384 collection, never used to fit response metric."""
    root=Path('runs/limb_context_20260916_dr16384/shard_07')
    meta=json.loads((root/'metadata.json').read_text())
    cache=dict(np.load(root/'latents.npz'))
    info=next(x for x in meta['queries'] if x['step']==query_step)
    query=torch.load(info['path'],map_location='cpu',weights_only=False,mmap=True)
    files=meta['motion_files'];family=np.array([Path(files[int(i)]).name.split('_subject')[0] for i in cache['motion']])
    selected_families=json.loads((ROOT/'dynamics_evaluation/protocol.json').read_text())['test_families']
    valid=(cache['step']==query_step)&cache['nominal']&(cache['short_steps']==50)&(cache['long_chunks']==30)
    rng=np.random.default_rng(29172026 if query_step==2400 else 29172026+query_step);records=[];states=[];actions=[]
    lookup={int(w):i for i,w in enumerate(query['world'])}
    for name in selected_families:
        pool=np.flatnonzero(valid&(family==name))
        assert len(pool)>=8,(name,len(pool))
        for row in rng.choice(pool,8,replace=False):
            local=lookup[int(cache['world'][row])]
            h=query['short'][local,-10:]
            assert query['short_valid'][local,-10:].all()
            torch.testing.assert_close(h[:-1,100:],h[1:,:71],atol=1e-6,rtol=0)
            states.append(h[0,:71].numpy());actions.append(h[:,71:100].numpy())
            motion=int(cache['motion'][row])
            records.append(dict(source_row=int(row),world=int(cache['world'][row]),query_step=query_step,
                                motion_id=motion,motion_file=files[motion],family=name))
    states=np.stack(states);actions=np.stack(actions)
    np.savez_compressed(out/'fresh_anchors.npz',state=states,targets=actions)
    return states,actions,records


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fresh',action='store_true');parser.add_argument('--confirmation',action='store_true');args=parser.parse_args()
    torch.set_num_threads(3)
    start=time.monotonic()
    out=ROOT/('dynamics_confirmation' if args.confirmation else ('dynamics_fresh' if args.fresh else 'dynamics'));out.mkdir(exist_ok=False)
    a=dict(np.load(SOURCE/'selected_worlds.npz'))
    rng=np.random.default_rng(16092026)
    excluded=np.load(ROOT/'dynamics/worlds.npz')['indices'] if args.confirmation else np.array([],int)
    chosen=np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero((a['fold']==f)&~a['nominal']),excluded),1024,replace=False) for f in (0,1)])
    if args.confirmation:
        assert not set(chosen)&set(excluded)
        assert json.loads((ROOT/'export/runtime_verification.json').read_text())['passed']
        (out/'frozen_model.json').write_text(json.dumps(dict(artifact_sha256=hashlib.sha256((ROOT/'export/partition_k64.npz').read_bytes()).hexdigest(),
                 protocol='Final confirmation: DR worlds disjoint from previous response probes, source query2800 states disjoint from query2400 and original probes. No further tuning.'),indent=2)+'\n')
    # Repeat identical DR worlds and nominal worlds to quantify simulator noise.
    repeat=np.arange(16)
    physics=np.r_[np.zeros((2,38),np.float32),a['physics'][chosen],a['physics'][chosen[repeat]]]
    physics[:2,4]=.6;physics[:2,5:34]=1.
    n=len(physics)
    if args.fresh or args.confirmation:
        initial,targets,records=fresh_anchors(out,2800 if args.confirmation else 2400)
        anchor_file=out/'fresh_anchors.npz'
    else:
        source=np.load(ANCHORS/'anchors.npz');records=json.loads((ANCHORS/'anchors.json').read_text())
        initial=source['state'];targets=source['targets'];anchor_file=ANCHORS/'anchors.npz'
    na=len(initial)
    assert na==len(records)
    np.savez_compressed(out/'worlds.npz',indices=chosen,world=a['world'][chosen],physics=physics,repeat=repeat)
    (out/'anchors.json').write_text(json.dumps(records,indent=2)+'\n')
    r=NominalPairRollout(NominalPairRolloutConfig(checkpoint_file=TRACKER,motion_file=records[0]['motion_file'],
                                               num_envs=n,device='cuda:0',horizon=10,seed=91761))
    try:
        env=r.env;env.scene.env_origins.zero_()
        fields=tuple(dict.fromkeys(rigid_body_payload.model_fields+('geom_friction','dof_armature')))
        env.sim.expand_model_fields(tuple(f for f in fields if f not in env.sim.expanded_fields))
        names=('body_mass','body_ipos','body_inertia','body_iquat','geom_friction','dof_armature')
        base={f:getattr(env.sim.model,f).clone() for f in names}
        local,_=r.robot.find_bodies([x[0] for x in LIMBS.values()],preserve_order=True)
        limbs=torch.tensor(local,device=r.device);bids=r.robot.indexing.body_ids[limbs].long()
        feetlocal,_=r.robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)
        feet=torch.tensor(feetlocal,device=r.device)
        mj=env.sim.mj_model
        torso=next(i for i in range(mj.nbody) if mj.body(i).name.split('/')[-1]=='torso_link')
        geoms=[i for i in range(mj.ngeom) if '_foot' in mj.geom(i).name and mj.geom(i).name.endswith('_collision')]
        joint_lookup={mj.joint(i).name.split('/')[-1]:int(mj.jnt_dofadr[i]) for i in range(mj.njnt)}
        dofs=torch.tensor([joint_lookup[name.split('/')[-1]] for name in a['names'][5:34]],device=r.device)
        p=torch.from_numpy(physics).to(r.device)
        loads=p[:,34:38]
        positions=torch.tensor([x[1] for x in LIMBS.values()],device=r.device).expand(n,-1,-1)
        sizes=torch.tensor([x[2] for x in LIMBS.values()],device=r.device)
        inertia=loads[...,None]*torch.stack([sizes[:,1]**2+sizes[:,2]**2,sizes[:,0]**2+sizes[:,2]**2,sizes[:,0]**2+sizes[:,1]**2],-1)/12
        quat=torch.zeros((n,4,4),device=r.device);quat[...,0]=1
        original=[base[f][:,bids] for f in ('body_mass','body_ipos','body_inertia','body_iquat')]
        combined=_decompose_pseudo_inertia_J(_reconstruct_pseudo_inertia_J(*original)+_reconstruct_pseudo_inertia_J(loads,positions,inertia,quat))
        for f,new,old in zip(('body_mass','body_ipos','body_inertia','body_iquat'),combined,original):
            mask=loads==0
            if new.ndim==3:mask=mask[...,None]
            getattr(env.sim.model,f)[:,bids]=torch.where(mask,old,new)
        env.sim.model.body_mass[:,torso]=base['body_mass'][:,torso]*(1+p[:,3])
        env.sim.model.body_ipos[:,torso]=base['body_ipos'][:,torso]+p[:,:3]
        env.sim.model.geom_friction[:,geoms,0]=p[:,4:5]
        env.sim.model.dof_armature[:,dofs]=base['dof_armature'][:,dofs]*p[:,5:34]
        env.sim.recompute_constants(RecomputeLevel.set_const)
        measured=torch.cat([env.sim.model.body_ipos[:,torso]-base['body_ipos'][:,torso],
                            (env.sim.model.body_mass[:,torso]/base['body_mass'][:,torso]-1)[:,None],
                            env.sim.model.geom_friction[:,geoms[0],0,None],
                            env.sim.model.dof_armature[:,dofs]/base['dof_armature'][:,dofs],
                            env.sim.model.body_mass[:,bids]-base['body_mass'][:,bids]],dim=1)
        torch.testing.assert_close(measured,p,atol=1e-6,rtol=1e-6)
        np.save(out/'measured_physics.npy',measured.cpu().numpy())
        states=np.lib.format.open_memmap(out/'state.npy',mode='w+',dtype=np.float32,shape=(na,n,11,71))
        features=np.lib.format.open_memmap(out/'feature.npy',mode='w+',dtype=np.float32,shape=(na,n,11,32))
        restores=[];noise=[]
        for index in range(na):
            x=torch.from_numpy(initial[index]).to(r.device).expand(n,-1)
            u=torch.from_numpy(targets[index]).to(r.device)
            restores.append(r._restore(x,torch.zeros((n,29),device=r.device)))
            ss=[];ff=[]
            s,f=snapshots(r,limbs,feet);ss.append(s);ff.append(f)
            for h in range(10):
                for _ in range(env.cfg.decimation):
                    r.robot.set_joint_position_target(u[h].expand(n,-1),env_ids=r._env_ids)
                    env.scene.write_data_to_sim();env.sim.step();env.scene.update(dt=env.physics_dt)
                env.sim.forward();s,f=snapshots(r,limbs,feet);ss.append(s);ff.append(f)
            ss=torch.stack(ss,1);ff=torch.stack(ff,1)
            assert torch.isfinite(ss).all() and torch.isfinite(ff).all()
            states[index]=ss.cpu().numpy();features[index]=ff.cpu().numpy()
            noise.append(dict(anchor=index,nominal_state_max=float((ss[0]-ss[1]).abs().max()),
                              repeated_dr_state_max=float((ss[2:18]-ss[-16:]).abs().max())))
            if (index+1)%8==0:
                states.flush();features.flush()
                print(json.dumps(dict(anchors=index+1,total=na,worlds=2048,seconds=time.monotonic()-start)),flush=True)
        torch.testing.assert_close(measured,p,atol=1e-6,rtol=1e-6)
        states.flush();features.flush()
        audit=dict(complete=True,dr_worlds=2048,nominal_repeats=2,dr_repeats=16,anchors=na,
                   motion_files=len(set(x['motion_file'] for x in records)),simulator='mjwarp',device=str(r.device),
                   initial_state_restore_max=max(restores),physics_schema_matches=True,noise=noise,
                   seconds=time.monotonic()-start,control_dt=env.step_dt,horizon=10,
                   scope='Same physical initial states and exact PD target sequences for all DR; all 38 measured physics coordinates reconstructed. Encoder bias excluded from physical target replay. No contact gating, no policy or encoder optimization.',
                   anchors_sha256=hashlib.sha256(anchor_file.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({k:v for k,v in audit.items() if k!='noise'}),flush=True)
    finally:r.close()


if __name__=='__main__':main()
