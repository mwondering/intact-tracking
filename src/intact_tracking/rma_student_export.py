"""Portable student export, validated against the actual actor and standalone runtime."""

import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from intact_tracking.heavy_rma_student import VERSION, INPUT_CONTRACT, EMBEDDING_GROUP, build_actor
from intact_tracking.memory350_deploy import TRACKER_WIDTH, TRACKER_FIELDS
from intact_tracking.memory350_onnx_export import tracker_observations, exportable_attention, robot_metadata
from intact_tracking.rollout.mjlab_adapter import _sha256


class StudentExport(torch.nn.Module):
    def __init__(self,actor):
        super().__init__()
        self.preprocessing=actor.tracker.as_onnx()
        self.preprocessing.mlp=torch.nn.Identity()
        self.preprocessing.deterministic_output=torch.nn.Identity()
        self.base_mlp=copy.deepcopy(actor.tracker.mlp)
        self.base_output=actor.tracker.distribution.as_deterministic_output_module()
        self.residual=copy.deepcopy(actor.residual_mlp)
        self.adaptation=copy.deepcopy(actor.adaptation)

    def forward(self,tracker_observation,proprio_history,history_count):
        features=self.preprocessing(tracker_observation)
        base=self.base_output(self.base_mlp(features))
        z=self.adaptation(proprio_history,history_count)
        return base+self.residual(torch.cat((features,z,base),-1)),z


def export_policy(checkpoint,output):
    import onnx
    import onnxruntime as ort
    from intact_tracking import rma_student_deploy
    checkpoint,output=Path(checkpoint).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['residual_policy']['version']!=VERSION or state['input_contract']!=INPUT_CONTRACT:
        raise ValueError('Unsupported student checkpoint')
    actor=build_actor(state);actor.load_state_dict(state['actor_state_dict'],strict=True)
    actor.eval().requires_grad_(False)
    module=StudentExport(actor).eval().requires_grad_(False)
    generator=torch.Generator().manual_seed(20260922)
    cases=[]
    for count in (0,1,23,50):
        observation=torch.randn(1,TRACKER_WIDTH,generator=generator)*.05
        observation[:,:4]/=observation[:,:4].norm(dim=-1,keepdim=True)
        history=torch.randn(1,50,122,generator=generator)*.05
        history[:,:50-count]=0
        cases.append((observation,history,torch.tensor([count])))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
    try:
        path=temporary/'policy.onnx'
        names=['tracker_observation','proprio_history','history_count']
        with exportable_attention(),torch.inference_mode():
            torch.onnx.export(module,cases[-1],str(path),opset_version=18,input_names=names,
                output_names=['action','embedding'],dynamo=False)
        onnx.checker.check_model(str(path),full_check=True)
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        errors=[]
        with torch.inference_mode():
            for observation,history,count in cases:
                z=actor.adaptation(history,count)
                obs=tracker_observations(observation);obs.set(EMBEDDING_GROUP,z)
                wanted=(actor(obs),z)
                eager=module(observation,history,count)
                actual=session.run(None,dict(zip(names,[observation.numpy(),history.numpy(),count.numpy()])))
                for a,b,c in zip(wanted,eager,actual):
                    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
                    np.testing.assert_allclose(a.numpy(),c,atol=2e-4,rtol=2e-4)
                errors.append([float(np.abs(a.numpy()-b).max()) for a,b in zip(wanted,actual)])
        metadata={'format':'motion_tracking_sim2real_policy','deployment_contract':'heavy_rma_student_onnx_v1',
            'method':'rma_student','completed_updates':state['completed_updates'],
            'checkpoint_sha256':_sha256(checkpoint),'onnx_sha256':_sha256(path),
            'tracker_sha256':state['residual_policy']['tracker_sha256'],'teacher_source':state['teacher_source'],
            'in_keys':names,'out_keys':['action','embedding'],
            'in_shapes':[[[1,8199]],[[1,50,122]],[[1]]],'out_shapes':[[[1,29]],[[1,64]]],
            'input_dtypes':['float32','float32','int64'],'dtype':'float32','opset':18,
            'num_actions':29,'physical_input_schema':None,'history':INPUT_CONTRACT,
            'tracker_observation_layout':[{'name':n,'width':w} for n,w in TRACKER_FIELDS],
            'action_semantics':{'output':'total deterministic raw command','clipping':None,
                'pd_target':'default_joint_pos + action_scale * action','controller':'source SP delay/smoothing/PD'},
            'standalone_runtime':'policy_runtime.py:RMAStudentPolicy; numpy + onnxruntime',
            'stock_tracker_client_compatible':False,
            'validation':{'passed':True,'atol':2e-4,'rtol':2e-4,'max_absolute_errors_by_case':errors},
            **robot_metadata(state)}
        (temporary/'policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
        runtime=rma_student_deploy.RMAStudentPolicy(temporary)
        frames=[]
        with torch.inference_mode():
            for step in range(54):
                observation=cases[step%4][0].numpy()[0].copy()
                boundary=step in (0,52)
                if boundary:frames=[]
                frames.append(rma_student_deploy.current_proprio(observation));frames=frames[-50:]
                history=torch.zeros(1,50,122);history[0,-len(frames):]=torch.from_numpy(np.stack(frames))
                expected=module(torch.from_numpy(observation)[None],history,torch.tensor([len(frames)]))[0]
                actual=runtime.step(observation,reset_boundary=boundary)
                np.testing.assert_allclose(actual,expected[0].numpy(),atol=2e-4,rtol=2e-4)
        metadata['validation']['runtime_cold_partial_full_reset_steps']=54
        for name in ('policy.json','deploy_metadata.json'):
            (temporary/name).write_text(json.dumps(metadata,indent=2)+'\n')
        shutil.copyfile(rma_student_deploy.__file__,temporary/'policy_runtime.py')
        temporary.rename(output)
        return {'directory':str(output),'completed_updates':state['completed_updates'],'validation':metadata['validation']}
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
