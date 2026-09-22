"""Standalone RMA student runtime: numpy + ONNX Runtime; no DR arguments."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np


def current_proprio(observation):
    offset, frames = 4, []
    for width in (29,29,3,3,29,29):
        offset += 50*width
        frames.append(observation[offset-width:offset])
    return np.concatenate(frames).copy()


class RMAStudentPolicy:
    def __init__(self,directory,*,threads=1,providers=None):
        import onnxruntime as ort
        directory=Path(directory)
        self.metadata=json.loads((directory/'policy.json').read_text())
        if self.metadata.get('deployment_contract')!='heavy_rma_student_onnx_v1':
            raise ValueError('Unsupported student deployment contract')
        model=directory/'policy.onnx'
        with model.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=self.metadata['onnx_sha256']:
                raise ValueError('ONNX/JSON checksum mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=providers or ['CPUExecutionProvider'])
        self.history=deque(maxlen=50)

    def reset(self):
        self.history.clear()

    def step(self,tracker_observation,*,reset_boundary=False,parameters_changed=False,command_applied=None):
        observation=np.asarray(tracker_observation,dtype=np.float32).reshape(-1)
        if observation.shape!=(8199,) or not np.isfinite(observation).all():
            raise ValueError('Expected finite 8199D tracker observation')
        if reset_boundary or parameters_changed:self.reset()
        proprio=current_proprio(observation)
        if command_applied is not None:
            command=np.asarray(command_applied,dtype=np.float32)
            if command.shape!=(29,) or not np.isfinite(command).all():
                raise ValueError('Expected previous executed raw command29')
            proprio[64:93]=command
        self.history.append(proprio)
        history=np.zeros((1,50,122),dtype=np.float32)
        history[0,-len(self.history):]=np.stack(self.history)
        action,embedding=self.session.run(['action','embedding'],{
            'tracker_observation':observation[None], 'proprio_history':history,
            'history_count':np.array([len(self.history)],dtype=np.int64)})
        if not np.isfinite(action).all() or not np.isfinite(embedding).all():
            raise FloatingPointError('Nonfinite student output')
        return action[0].copy()
