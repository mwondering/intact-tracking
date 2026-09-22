"""Standalone NumPy/ONNX Runtime client for RMA teacher or AnyAdapter."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np


def proprio_state(observation):
    offset, values = 4, []
    for width in (29, 29, 3, 3, 29, 29):
        offset += 50*width
        values.append(observation[offset-width:offset])
    q, qdot, gravity, gyro = values[:4]
    return np.concatenate((gyro*.05, gravity, q, qdot*.05))


class HeavyBaselinePolicy:
    def __init__(self, directory, *, threads=1, providers=None):
        import onnxruntime as ort
        directory = Path(directory)
        model = (directory/'policy.onnx').resolve()
        self.metadata = json.loads(model.with_suffix('.json').read_text())
        if self.metadata.get('deployment_contract') != 'heavy_baselines_onnx_v1':
            raise ValueError('Unsupported heavy baseline deployment format')
        with model.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != self.metadata['onnx_sha256']:
                raise ValueError('ONNX and JSON checksums differ')
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=providers or ['CPUExecutionProvider'])
        self.method = self.metadata['method']
        self.history = deque(maxlen=79)
        self.previous_state = self.previous_action = None

    def reset(self):
        self.history.clear()
        self.previous_state = self.previous_action = None

    def step(self, tracker_observation, *, physics_parameters=None, reset_boundary=False,
             parameters_changed=False, command_applied=None):
        observation = np.asarray(tracker_observation, dtype=np.float32).reshape(-1)
        if observation.shape != (8199,) or not np.isfinite(observation).all():
            raise ValueError('Expected a finite 8199D source tracker observation')
        if reset_boundary or parameters_changed:
            self.reset()
        if self.method == 'rma_teacher':
            if physics_parameters is None:
                raise ValueError('RMA teacher requires real physical parameters, in the JSON schema order')
            if isinstance(physics_parameters, dict):
                physics_parameters = [physics_parameters[name] for name in self.metadata['physical_input_schema']['names']]
            condition = np.asarray(physics_parameters, dtype=np.float32)
            if condition.shape != (108,) or not np.isfinite(condition).all():
                raise ValueError('Expected 108 finite real physical coordinates')
        else:
            if self.previous_state is not None:
                previous = self.previous_action if command_applied is None else np.asarray(command_applied, dtype=np.float32)
                if previous.shape != (29,) or not np.isfinite(previous).all():
                    raise ValueError('Expected the previous actual raw command29')
                self.history.append(np.concatenate((self.previous_state, previous)))
            condition = np.zeros((79, 93), dtype=np.float32)
            if self.history:
                condition[-len(self.history):] = np.stack(self.history)
            condition = condition.reshape(-1)
        action, embedding = self.session.run(self.metadata['out_keys'], {
            self.metadata['in_keys'][0]: observation[None], self.metadata['in_keys'][1]: condition[None]})
        if not np.isfinite(action).all() or not np.isfinite(embedding).all():
            raise FloatingPointError('Nonfinite deployed baseline output')
        if self.method == 'any2track':
            self.previous_state, self.previous_action = proprio_state(observation), action[0].copy()
        return action[0].copy()
