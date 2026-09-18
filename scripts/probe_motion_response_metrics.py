"""GPU mjwarp paired motion replay: compare old state response and grouped body response."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
from intact_tracking.rollout.nominal import NominalPairRollout,NominalPairRolloutConfig
from intact_tracking.limb_context_protocol import TRACKER
from intact_tracking.preview_protocol import LIMBS
from intact_tracking.environment.mdp.randomizations import rigid_body_payload,_reconstruct_pseudo_inertia_J,_decompose_pseudo_inertia_J
from intact_tracking.rollout.mjlab_adapter import _robot_raw_state
from intact_tracking.forward_predictor import physical_state_delta
from mjlab.sim.sim import RecomputeLevel

SOURCE=Path('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/cluster_check_030750/common')
OUT=Path('runs/limb_context_20260916_motion_metric_probe')

def variants():
 out=[]
 def add(name,family,branch,level,load=None,mass=0,com=None,friction=.6):
  out.append(dict(name=name,family=family,branch=branch,level=level,load=load or [0]*4,mass=mass,com=com or [0]*3,friction=friction))
 add('nominal','nominal','nominal',0);add('nominal_repeat','nominal','nominal',0)
 for limb,maxmass in enumerate([2.5,2.5,4,4]):
  for f in [.2,.4,.6,.8,1]:
   load=[0]*4;load[limb]=maxmass*f;add(f'load{limb}_{f}','load',f'load{limb}',f,load=load)
 for f in [.2,.4,.6,.8,1]:add(f'all_load_{f}','load','all_load',f,load=[v*f for v in [2.5,2.5,4,4]])
 for sign in [-1,1]:
  for f in [.25,.5,1]:add(f'mass_{sign*f}','mass',f'mass{sign}',f,mass=sign*f)
 for axis in range(3):
  for sign in [-1,1]:
   for f in [1/3,2/3,1]:
    c=[0]*3;c[axis]=sign*.075*f;add(f'com{axis}_{sign*f}','com',f'com{axis}_{sign}',f,com=c)
 for v in [.3,.4,.5,.85,1.2,1.6,2.]:add(f'friction_{v}','friction','friction_down' if v<.6 else 'friction_up',abs(v-.6)/(.3 if v<.6 else 1.4),friction=v)
 rng=np.random.default_rng(9751)
 for group in ('random_full','random_load'):
  for i in range(64):
   load=(rng.random(4)*np.array([2.5,2.5,4,4])).tolist()
   mass=float(rng.uniform(-1,1)) if group=='random_full' else 0.
   com=rng.uniform(-.075,.075,3).tolist() if group=='random_full' else [0]*3
   friction=float(rng.uniform(.3,2)) if group=='random_full' else .6
   add(f'{group}_{i}',group,group,1.,load=load,mass=mass,com=com,friction=friction)
 return out

def invrotate(q,v):
 xyz=-q[...,1:];t=2*torch.cross(xyz,v,dim=-1);return v+q[...,:1]*t+torch.cross(xyz,t,dim=-1)

def snapshots(r,limbs,feet):
 d=r.robot.data;state=_robot_raw_state(r.env).clone();p=d.body_link_pos_w[:,limbs]-d.root_link_pos_w[:,None];q=d.root_link_quat_w[:,None].expand(-1,4,-1)
 v=d.body_link_lin_vel_w[:,limbs]-d.root_link_lin_vel_w[:,None]-torch.cross(d.root_link_ang_vel_w[:,None].expand_as(p),p,dim=-1)
 relp=invrotate(q,p);relv=invrotate(q,v)
 fp=d.body_link_pos_w[:,feet,:2]-r.env.scene.env_origins[:,None,:2];fv=d.body_link_lin_vel_w[:,feet,:2]
 return state,torch.cat([relp.flatten(1),relv.flatten(1),fp.flatten(1),fv.flatten(1)],-1)

def prepare_anchors(maxmotions=24,per_motion=4):
 OUT.mkdir(exist_ok=True);meta=json.loads((SOURCE/'metadata.json').read_text());d=np.load(SOURCE/'latents.npz')
 valid=d['nominal']&(d['short_steps']==50)&(d['long_chunks']==30)&(d['step']>=800)
 available=np.unique(d['motion'][valid]);rng=np.random.default_rng(8251)
 # Cover all named action families first, then additional recordings.
 byfamily={}
 for m in available:
  family=Path(meta['motion_files'][m]).name.split('_subject')[0];byfamily.setdefault(family,[]).append(int(m))
 chosen=[]
 for values in byfamily.values():chosen.append(int(rng.choice(values)))
 rest=[int(m) for m in available if m not in chosen];rng.shuffle(rest);chosen=(chosen+rest)[:maxmotions]
 selected=[]
 for m in chosen:
  pool=np.flatnonzero(valid&(d['motion']==m));rng.shuffle(pool);picked=[];seen=set()
  for row in pool:
   identity=(int(d['world'][row]),int(d['step'][row]))
   if identity not in seen:picked.append(int(row));seen.add(identity)
   if len(picked)==per_motion:break
  assert len(picked)==per_motion;selected.extend(picked)
 state=np.empty((len(selected),71),np.float32);targets=np.empty((len(selected),10,29),np.float32);records=[None]*len(selected)
 for info in meta['queries']:
  loc=[i for i,row in enumerate(selected) if d['step'][row]==info['step']]
  if not loc:continue
  q=torch.load(info['path'],map_location='cpu',weights_only=False,mmap=True);lookup={int(v):i for i,v in enumerate(q['world'])}
  for i in loc:
   row=selected[i];qi=lookup[int(d['world'][row])];h=q['short'][qi,-10:];assert q['short_valid'][qi,-10:].all()
   torch.testing.assert_close(h[:-1,100:],h[1:,:71],atol=1e-5,rtol=1e-5)
   state[i]=h[0,:71].numpy();targets[i]=h[:,71:100].numpy();motion=int(d['motion'][row]);records[i]={'source_row':int(d['source_row'][row]),'world':int(d['world'][row]),'query_step':int(info['step']),'motion_id':motion,'motion_file':meta['motion_files'][motion],'family':Path(meta['motion_files'][motion]).name.split('_subject')[0]}
 np.savez_compressed(OUT/'anchors.npz',state=state,targets=targets)
 (OUT/'anchors.json').write_text(json.dumps(records,indent=2)+'\n');return state,targets,records

@torch.inference_mode()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();torch.set_num_threads(4);start=time.monotonic()
 state,targets,records=prepare_anchors(4 if args.smoke else 24,1 if args.smoke else 4);vs=variants();nv=len(vs);batch=4 if args.smoke else 8;n=batch*nv
 r=NominalPairRollout(NominalPairRolloutConfig(checkpoint_file=TRACKER,motion_file=records[0]['motion_file'],num_envs=n,device='cuda:0',horizon=10))
 env=r.env;env.scene.env_origins.zero_();fields=tuple(dict.fromkeys(rigid_body_payload.model_fields+('geom_friction',)))
 env.sim.expand_model_fields(tuple(f for f in fields if f not in env.sim.expanded_fields))
 base={f:getattr(env.sim.model,f).clone() for f in ('body_mass','body_ipos','body_inertia','body_iquat','geom_friction')}
 local,_=r.robot.find_bodies([x[0] for x in LIMBS.values()],preserve_order=True);limbs=torch.tensor(local,device=r.device);bids=r.robot.indexing.body_ids[limbs].long()
 feetlocal,_=r.robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True);feet=torch.tensor(feetlocal,device=r.device)
 model=env.sim.mj_model;torso=next(i for i in range(model.nbody) if model.body(i).name.split('/')[-1]=='torso_link');geoms=[i for i in range(model.ngeom) if '_foot' in model.geom(i).name and model.geom(i).name.endswith('_collision')];assert len(geoms)==14
 loads=torch.tensor([v['load'] for v in vs],device=r.device).repeat(batch,1)
 positions=torch.tensor([x[1] for x in LIMBS.values()],device=r.device).expand(n,-1,-1);sizes=torch.tensor([x[2] for x in LIMBS.values()],device=r.device)
 inertia=loads[...,None]*torch.stack([sizes[:,1]**2+sizes[:,2]**2,sizes[:,0]**2+sizes[:,2]**2,sizes[:,0]**2+sizes[:,1]**2],-1)/12
 quat=torch.zeros((n,4,4),device=r.device);quat[...,0]=1
 original=[base[f][:,bids] for f in ('body_mass','body_ipos','body_inertia','body_iquat')]
 combined=_decompose_pseudo_inertia_J(_reconstruct_pseudo_inertia_J(*original)+_reconstruct_pseudo_inertia_J(loads,positions,inertia,quat))
 for f,new,orig in zip(('body_mass','body_ipos','body_inertia','body_iquat'),combined,original):
  mask=loads==0
  if new.ndim==3:mask=mask[...,None]
  getattr(env.sim.model,f)[:,bids]=torch.where(mask,orig,new)
 mass=torch.tensor([v['mass'] for v in vs],device=r.device).repeat(batch);com=torch.tensor([v['com'] for v in vs],device=r.device).repeat(batch,1);fr=torch.tensor([v['friction'] for v in vs],device=r.device).repeat(batch)
 env.sim.model.body_mass[:,torso]=base['body_mass'][:,torso]+mass;env.sim.model.body_ipos[:,torso]=base['body_ipos'][:,torso]+com;env.sim.model.geom_friction[:,geoms,0]=fr[:,None]
 torch.testing.assert_close(env.sim.model.body_mass[:,bids]-base['body_mass'][:,bids],loads,atol=1e-5,rtol=0)
 env.sim.recompute_constants(RecomputeLevel.set_const)
 torch.testing.assert_close(env.sim.model.body_mass[:,torso]-base['body_mass'][:,torso],mass,atol=1e-6,rtol=0)
 torch.testing.assert_close(env.sim.model.body_ipos[:,torso]-base['body_ipos'][:,torso],com,atol=1e-7,rtol=0)
 torch.testing.assert_close(env.sim.model.geom_friction[:,geoms,0],fr[:,None].expand(-1,14),atol=0,rtol=0)
 assert torch.allclose(base['geom_friction'][:,geoms,0],torch.full_like(fr[:,None].expand(-1,14),.6))
 np.savez_compressed(OUT/'physics_audit.npz',load_measured=(env.sim.model.body_mass[:,bids]-base['body_mass'][:,bids]).cpu().numpy(),mass_measured=(env.sim.model.body_mass[:,torso]-base['body_mass'][:,torso]).cpu().numpy(),com_measured=(env.sim.model.body_ipos[:,torso]-base['body_ipos'][:,torso]).cpu().numpy(),friction_measured=env.sim.model.geom_friction[:,geoms,0].cpu().numpy())
 states=[];features=[];restore=[]
 for begin in range(0,len(state),batch):
  ix=np.minimum(np.arange(begin,begin+batch),len(state)-1);initial=torch.from_numpy(state[ix]).to(r.device).repeat_interleave(nv,0);actions=torch.from_numpy(targets[ix]).to(r.device).repeat_interleave(nv,0)
  restore.append(r._restore(initial,torch.zeros((n,29),device=r.device)));ss=[];ff=[]
  s,f=snapshots(r,limbs,feet);ss.append(s);ff.append(f)
  for h in range(10):
   for _ in range(env.cfg.decimation):
    r.robot.set_joint_position_target(actions[:,h],env_ids=r._env_ids);env.scene.write_data_to_sim();env.sim.step();env.scene.update(dt=env.physics_dt)
   env.sim.forward();s,f=snapshots(r,limbs,feet);ss.append(s);ff.append(f)
  count=min(batch,len(state)-begin);states.append(torch.stack(ss,1).reshape(batch,nv,11,71)[:count].cpu().numpy());features.append(torch.stack(ff,1).reshape(batch,nv,11,32)[:count].cpu().numpy())
  print(json.dumps({'anchors_done':min(begin+batch,len(state)),'anchors_total':len(state),'variants':nv,'seconds':time.monotonic()-start}),flush=True)
 states=np.concatenate(states);features=np.concatenate(features);assert np.isfinite(states).all() and np.isfinite(features).all()
 # Duplicate nominal worlds provide numerical floor, including near-contact sensitivities.
 audit={'state_restore_max':max(restore),'nominal_repeat_state_max':float(np.max(np.abs(states[:,0]-states[:,1]))),'nominal_repeat_feature_max':float(np.max(np.abs(features[:,0]-features[:,1]))),'device':str(r.device),'physics_dt':env.physics_dt,'control_dt':env.step_dt,'simulator':'mjwarp','num_envs':n,'motion_count':len(set(x['motion_id'] for x in records)),'anchors':len(records),'variants':len(vs),'seconds':time.monotonic()-start,'feet_contact_gate':False,'model_fields_audited':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Frozen tracker cached nominal motion histories. All variants replay the same physical targets and initial state; no policy or encoder training. Torso mass add only (matches existing DR), limb payload updates composite mass/COM/inertia. Bias and armature remain nominal.'}
 prefix='smoke' if args.smoke else 'simulation';np.savez_compressed(OUT/(prefix+'.npz'),state=states,feature=features);(OUT/(prefix+'.json')).write_text(json.dumps({'audit':audit,'variants':vs,'anchors':records},indent=2)+'\n');r.close();print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
