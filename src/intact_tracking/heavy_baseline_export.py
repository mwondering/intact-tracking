"""Self-contained SPV5-2A + RMA/AnyAdapter ONNX and tracker-style JSON."""

import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from intact_tracking.heavy_rma_teacher import VERSION, RMATeacherActor, PHYSICS_GROUP
from intact_tracking.heavy_anyadapter import AnyAdapterActor
from intact_tracking.memory350_checkpoint import embedded_tracker
from intact_tracking.memory350_deploy import TRACKER_WIDTH, TRACKER_FIELDS
from intact_tracking.memory350_onnx_export import tracker_observations, exportable_attention, robot_metadata
from intact_tracking.rollout.mjlab_adapter import _sha256


def make_actor_observation(value, kwargs):
    from intact_tracking.environment.mdp.spv5 import SPV5_REFERENCE_TARGET_DIM
    obs = tracker_observations(value)
    config = kwargs['tracker_actor_kwargs']
    for name, width in ((config.get('reference_encoder_target_group', 'reference_encoder_target'), SPV5_REFERENCE_TARGET_DIM),
                         (config.get('estimator_target_group', 'estimator_target'), 1),
                         (config.get('foot_contact_target_group', 'foot_contact_target'), 2)):
        obs.set(name, value.new_zeros(value.shape[0], width))
    obs.set(PHYSICS_GROUP, value.new_zeros(value.shape[0], 108))
    obs.set('dynamics_latent', value.new_zeros(value.shape[0], 128))
    return obs


class BaselineExport(torch.nn.Module):
    def __init__(self, actor, schema, method):
        super().__init__()
        if method not in ('rma_teacher', 'any2track') or actor.residual_output_mode != 'unbounded' or actor.residual_scale != 1.:
            raise ValueError('Deployment requires the matched unbounded heavy baseline action contract')
        self.method = method
        self.preprocessing = actor.tracker.as_onnx()
        self.preprocessing.mlp = torch.nn.Identity()
        self.preprocessing.deterministic_output = torch.nn.Identity()
        self.base_mlp = copy.deepcopy(actor.tracker.mlp)
        self.base_output = actor.tracker.distribution.as_deterministic_output_module()
        if method == 'rma_teacher':
            self.dr_encoder, self.residual = copy.deepcopy(actor.dr_encoder), copy.deepcopy(actor.residual_mlp)
            self.register_buffer('lower', torch.tensor(schema['lower']))
            self.register_buffer('ranges', torch.tensor(schema['upper'])-self.lower)
        else:
            self.encoder, self.adapters = copy.deepcopy(actor.history_encoder), copy.deepcopy(actor.adapters)

    def forward(self, tracker_observation, condition):
        features = self.preprocessing(tracker_observation)
        if self.method == 'rma_teacher':
            embedding = self.dr_encoder(2*(condition-self.lower)/self.ranges-1)
            base = self.base_output(self.base_mlp(features))
            return base+self.residual(torch.cat((features, embedding, base), -1)), embedding
        embedding = self.encoder(condition.reshape(-1, 79, 93))
        return self.base_output(self.adapters(self.base_mlp, features, embedding)), embedding


def load_models(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['residual_policy']['version'] != VERSION:
        raise ValueError('Expected a heavy RMA/AnyAdapter checkpoint')
    tracker = embedded_tracker(state)
    if tracker is None:
        raise ValueError('Checkpoint must embed the frozen tracker')
    agent = OmegaConf.to_container(state['cfg'].agent, resolve=True)
    kwargs = copy.deepcopy(agent['actor'])
    method = state['residual_policy']['method']
    if method not in ('rma_teacher', 'any2track'):
        raise ValueError('Unknown heavy baseline method')
    cls = RMATeacherActor if method == 'rma_teacher' else AnyAdapterActor
    if kwargs.pop('class_name') != f'{cls.__module__}:{cls.__name__}':
        raise ValueError('Actor class differs from method metadata')
    value = torch.zeros(1, TRACKER_WIDTH)
    value[:, 0] = 1
    obs = make_actor_observation(value, kwargs)
    actor = cls(obs, agent['obs_groups'], 'actor', 29, tracker_state_dict=tracker['actor_state_dict'], **kwargs)
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    actor.eval().requires_grad_(False)
    module = BaselineExport(actor, state['residual_policy']['physics_input_schema'], method).eval().requires_grad_(False)
    return state, actor, module


def export_policy(checkpoint, output):
    import onnx
    import onnxruntime as ort
    from intact_tracking import heavy_baseline_deploy

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state, actor, module = load_models(checkpoint)
    meta, method = state['residual_policy'], module.method
    schema = meta['physics_input_schema']
    names = ['tracker_observation', 'physics_parameters' if method == 'rma_teacher' else 'history_state_action']
    condition_width, latent_width = (108, 64) if method == 'rma_teacher' else (79*93, 128)
    generator = torch.Generator().manual_seed(20260921)
    cases = []
    for i in range(4):
        observation = torch.randn(1, TRACKER_WIDTH, generator=generator)*.05
        observation[:, :4] /= observation[:, :4].norm(dim=-1, keepdim=True)
        if method == 'rma_teacher':
            normalized = torch.full((1, 108), float(i)/3)
            condition = torch.tensor(schema['lower']) + normalized * (torch.tensor(schema['upper'])-torch.tensor(schema['lower']))
        else:
            condition = torch.randn(1, 79*93, generator=generator)*.05
            condition[:, :(79-(0, 1, 40, 79)[i])*93] = 0
        cases.append((observation, condition))
    temporary = Path(tempfile.mkdtemp(prefix='.'+output.name+'.', dir=output.parent))
    try:
        path = temporary/'policy.onnx'
        with exportable_attention(), torch.inference_mode():
            module(*cases[0])
            torch.onnx.export(module, cases[0], str(path), opset_version=18, input_names=names,
                              output_names=['action', 'embedding'], dynamic_axes={}, dynamo=False)
        onnx.checker.check_model(str(path), full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        audits = []
        with torch.inference_mode():
            for observation, condition in cases:
                obs = tracker_observations(observation)
                if method == 'rma_teacher':
                    theta = 2*(condition-module.lower)/module.ranges-1
                    obs.set(PHYSICS_GROUP, theta)
                    embedding = actor.dr_encoder(theta)
                else:
                    embedding = actor.history_encoder(condition.reshape(1, 79, 93))
                    obs.set('dynamics_latent', embedding)
                expected = (actor(obs), embedding)
                eager = module(observation, condition)
                onnx_values = session.run(['action', 'embedding'], dict(zip(names, [observation.numpy(), condition.numpy()])))
                errors = []
                for want, actual, exported in zip(expected, eager, onnx_values, strict=True):
                    torch.testing.assert_close(actual, want, atol=2e-4, rtol=2e-4)
                    np.testing.assert_allclose(exported, want.numpy(), atol=2e-4, rtol=2e-4)
                    errors.append(float(np.abs(exported-want.numpy()).max()))
                audits.append(errors)
        metadata = {
            'format': 'motion_tracking_sim2real_policy', 'deployment_contract': 'heavy_baselines_onnx_v1',
            'method': method, 'run_name': meta['arguments'].get('wandb_name'),
            'iteration': state['iter'], 'completed_updates': state['completed_updates'],
            'checkpoint_sha256': _sha256(checkpoint), 'onnx_sha256': _sha256(path),
            'in_keys': names, 'out_keys': ['action', 'embedding'],
            'in_shapes': [[[1, TRACKER_WIDTH]], [[1, condition_width]]],
            'out_shapes': [[[1, 29]], [[1, latent_width]]], 'dtype': 'float32', 'opset': 18,
            'num_actions': 29, 'tracker_sha256': meta['tracker_sha256'],
            'tracker_observation_layout': [{'name': n, 'width': w} for n, w in TRACKER_FIELDS],
            'physical_input_schema': schema if method == 'rma_teacher' else None,
            'physical_input_normalization': 'raw physical coordinates; fixed [-1,1] mapping embedded in ONNX' if method == 'rma_teacher' else None,
            'history': None if method == 'rma_teacher' else {
                'length': 79, 'frame_dim': 93, 'frame': ['gyro*0.05', 'gravity', 'q-default_q', '(qdot-default_qdot)*0.05', 'raw_total_action'],
                'order': 'oldest to newest, zero padding on the left',
                'timing': '(pre-state_t, actual raw command_t), appended AFTER a completed nonboundary transition',
                'boundary': 'clear on episode reset, motion switch or physical parameter change'},
            'action_semantics': {'output': 'total deterministic raw command', 'clipping': None,
                                 'pd_target': 'default_joint_pos + action_scale * action',
                                 'controller': 'source SP delay/smoothing/PD contract',
                                 'world_model_required_for_deployment': False},
            'standalone_runtime': 'policy_runtime.py; numpy + onnxruntime',
            'stock_tracker_client_compatible': False,
            'validation': {'passed': True, 'atol': 2e-4, 'rtol': 2e-4, 'max_absolute_errors_by_case': audits},
            **robot_metadata(state),
        }
        for filename in ('policy.json', 'deploy_metadata.json'):
            (temporary/filename).write_text(json.dumps(metadata, indent=2)+'\n')
        shutil.copyfile(heavy_baseline_deploy.__file__, temporary/'policy_runtime.py')
        temporary.rename(output)
        return {'directory': str(output), 'method': method, 'validation': metadata['validation']}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
