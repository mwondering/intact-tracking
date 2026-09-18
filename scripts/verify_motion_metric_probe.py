"""Independent integrity checks of saved GPU motion-response experiment artifacts."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
P=Path('runs/limb_context_20260916_motion_metric_probe')

def main():
 meta=json.load(open(P/'simulation.json'));sim=np.load(P/'simulation.npz');resp=np.load(P/'responses.npz');phys=np.load(P/'physics_audit.npz');anchors=np.load(P/'anchors.npz');s=sim['state'];vs=meta['variants'];na,nv,nh,width=s.shape
 assert (na,nv,nh,width)==(96,186,11,71);assert np.isfinite(s).all() and np.isfinite(sim['feature']).all()
 np.testing.assert_allclose(s[:,:,0],np.broadcast_to(anchors['state'][:,None],s[:,:,0].shape),atol=1e-6,rtol=1e-6)
 repeats=len(phys['mass_measured'])//nv
 for key,expected in [('load_measured',[v['load'] for v in vs]),('mass_measured',[v['mass'] for v in vs]),('com_measured',[v['com'] for v in vs])]:
  values=np.array(expected);tiled=np.tile(values,(repeats,1)) if values.ndim==2 else np.tile(values,repeats);np.testing.assert_allclose(phys[key],tiled,atol=1e-6,rtol=0)
 np.testing.assert_allclose(phys['friction_measured'],np.broadcast_to(np.tile([v['friction'] for v in vs],repeats)[:,None],phys['friction_measured'].shape),atol=1e-7,rtol=0)
 # Independent scipy quaternion composition, not the training delta helper.
 i,j,h=np.indices((na,nv,10));i=i.ravel()[::41];j=j.ravel()[::41];h=h.ravel()[::41]+1
 actual=s[i,j,h].astype(float);nominal=s[i,0,h].astype(float)
 rot=(Rotation.from_quat(actual[:,3:7],scalar_first=True)*Rotation.from_quat(nominal[:,3:7],scalar_first=True).inv()).as_rotvec()
 delta=np.c_[actual[:,:3]-nominal[:,:3],rot,actual[:,7:]-nominal[:,7:]]
 measured=resp['old_checkpoint_scale'][i,j,h-1]*resp['old_std']*np.sqrt(70)
 np.testing.assert_allclose(measured,delta,atol=2e-6,rtol=2e-5)
 train=resp['train_anchors'];families=np.array([x['family'] for x in meta['anchors']]);assert not set(families[train])&set(families[~train])
 assert not meta['audit']['feet_contact_gate'];assert meta['audit']['simulator']=='mjwarp' and meta['audit']['device']=='cuda:0'
 checks=[]
 for suffix in ('','_reverse'):
  report=json.load(open(P/('diagnostics'+suffix+'.json')));saved=np.load(P/('diagnostic_clusters'+suffix+'.npz'))
  for r in report['results']:
   if r['scenario'].startswith('random'):
    assert set(r['train_variants'])&set(r['test_variants'])=={0}
   key=f'{r["scenario"]}_{r["method"]}_{r["k"]}';labels=saved[key+'_labels'];ids=np.tile(r['test_variants'],r['test_motion_anchors']);nom=int(np.square(saved[key+'_centers']).sum(1).argmin())
   heavy=np.array([resp['physical_parameters'][idx,:4].sum()>=8 if r['scenario'].startswith('random') else vs[idx]['family']!='nominal' and vs[idx]['level']>=.8 for idx in ids]);np.testing.assert_allclose(np.mean(labels[heavy]==nom),r['heavy_nominal_fraction'],atol=1e-12)
   rng=np.random.default_rng(614);ix=rng.integers(len(ids),size=30000);pools=[np.flatnonzero(labels==c) for c in range(r['k'])];partners=np.array([rng.choice(pools[labels[i]]) for i in ix]);random=rng.integers(len(ids),size=len(ix))
   scale=np.array([2.5,2.5,4,4,2,.15,.15,.15,1.7]);groups=[[0,1,2,3],[4],[5,6,7],[8]];params=resp['physical_parameters']
   def mse(a,b):
    dif=(params[ids[a]]-params[ids[b]])/scale;return np.mean([np.mean(dif[:,g]**2) for g in groups])
   ratio=1-np.sqrt(mse(ix,partners)/mse(ix,random));np.testing.assert_allclose(ratio,r['overall_distance_reduction'],atol=1e-12)
   checks.append(key+suffix)
 result={'passed':True,'gpu_trajectories':na*nv,'independent_rotation_delta_samples':len(i),'physical_parameters_verified':['four_limb_mass','torso_mass','torso_COM','14_foot_geom_friction'],'all_initial_states_match':True,'motion_family_split_disjoint':True,'random_DR_fit_test_disjoint_except_nominal':True,'recomputed_cluster_metrics':len(checks),'new_no_contact_gate':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (P/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
