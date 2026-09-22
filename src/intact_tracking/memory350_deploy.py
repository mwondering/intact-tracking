"""Standalone NumPy/ONNX Runtime client; no PyTorch or simulator dependency."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

TRACKER_FIELDS = (("robot_root_quat", 4), ("estimator_history", 6100),
                  ("reference_encoder_input", 1900), ("robot_key_body", 195))
TRACKER_WIDTH = 8199
PROPRIO_WIDTHS = (29, 29, 3, 3, 29, 29)
INPUT_FIELDS = (("tracker_observation", (8199,)), ("short_interactions", (50, 273)),
                ("short_valid", (50,)), ("long_interactions", (30, 10, 273)),
                ("long_valid", (30,)), ("previous_latents", (4, 64)))
INPUT_WIDTH = sum(int(np.prod(shape)) for _, shape in INPUT_FIELDS)
INPUT_NAME = "memory350_residual_observation"
OUTPUT_NAMES = ("action", "context_latent")


def input_layout():
    offset, result = 0, []
    for name, shape in INPUT_FIELDS:
        width = int(np.prod(shape))
        result.append({"name": name, "shape": list(shape), "offset": offset, "width": width})
        offset += width
    return result


def current_proprio(observation):
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.shape != (TRACKER_WIDTH,) or not np.isfinite(observation).all():
        raise ValueError("Expected one finite 8199-D SPV5-2 tracker observation")
    offset, values = 4, []
    for width in PROPRIO_WIDTHS:
        offset += 50 * width
        values.append(observation[offset - width:offset])
    return np.concatenate(values).copy()


class InteractionHistory:
    """Chronological short50, eviction chunks of 10, last30 complete old chunks."""

    def __init__(self):
        self.short = deque()
        self.pending = []
        self.chunks = deque(maxlen=30)

    def reset(self, *, keep_long_memory=True):
        if keep_long_memory:
            sequence = list(self.pending) + list(self.short)
            for start in range(0, len(sequence) - 9, 10):
                self.chunks.append(np.stack(sequence[start:start + 10]))
        else:
            self.chunks.clear()
        self.short.clear()
        self.pending.clear()

    def append(self, before, command, after, *, reset_boundary=False, discontinuity=False):
        before, command, after = (np.asarray(v, dtype=np.float32) for v in (before, command, after))
        if (before.shape != (122,) or command.shape != (29,) or after.shape != (122,)
                or not all(np.isfinite(v).all() for v in (before, command, after))):
            raise ValueError("A completed interaction requires finite proprio122/command29/proprio122")
        if discontinuity:
            self.reset(keep_long_memory=True)
        if reset_boundary:
            self.reset(keep_long_memory=True)
            return  # Never admit the transition across a reset/motion boundary.
        if len(self.short) == 50:
            self.pending.append(self.short.popleft())
            if len(self.pending) == 10:
                self.chunks.append(np.stack(self.pending))
                self.pending.clear()
        self.short.append(np.concatenate((before, command, after)))

    def tensors(self):
        short, short_valid = np.zeros((50, 273), np.float32), np.zeros(50, np.float32)
        long, long_valid = np.zeros((30, 10, 273), np.float32), np.zeros(30, np.float32)
        if self.short:
            short[-len(self.short):] = np.stack(self.short)
            short_valid[-len(self.short):] = 1
        if self.chunks:
            long[-len(self.chunks):] = np.stack(self.chunks)
            long_valid[-len(self.chunks):] = 1
        return short, short_valid, long, long_valid


class Memory350Policy:
    """Call step once per 20ms control step with the original SP tracker observation.

    reset_boundary marks an episode/motion change since the previous call;
    parameters_changed starts a new physical session and discards all history.
    command_applied is the previous 29-D policy command actually sent to the
    controller (before PD scale/offset/delay), if it differs from our last output.
    """

    def __init__(self, directory, *, threads=1, providers=None):
        import onnxruntime as ort

        directory = Path(directory).resolve()
        # Resolve a versioned generation once, even through the live latest links.
        model = (directory / "policy.onnx").resolve()
        self.metadata = json.loads(model.with_suffix(".json").read_text())
        if self.metadata.get("deployment_contract") != "memory350_proprio122_history5_onnx_v1":
            raise ValueError("Not a compatible Memory350 deployment package")
        with model.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.metadata["onnx_sha256"]:
            raise ValueError("policy.onnx and policy.json do not match")
        if (self.metadata["in_keys"] != [INPUT_NAME]
                or self.metadata["out_keys"] != list(OUTPUT_NAMES)
                or self.metadata["in_shapes"] != [[[1, INPUT_WIDTH]]]):
            raise ValueError("ONNX input/output contract changed")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=providers or ["CPUExecutionProvider"])
        self.history = InteractionHistory()
        self.previous_latents = np.zeros((4, 64), np.float32)
        self.previous_proprio = None
        self.previous_action = None

    def reset(self, *, keep_long_memory=True):
        self.history.reset(keep_long_memory=keep_long_memory)
        self.previous_latents.fill(0)
        self.previous_proprio = self.previous_action = None

    def step(self, tracker_observation, *, reset_boundary=False, parameters_changed=False,
             command_applied=None):
        observation = np.asarray(tracker_observation, dtype=np.float32).reshape(-1)
        proprio = current_proprio(observation)
        if parameters_changed:
            self.reset(keep_long_memory=False)
        if self.previous_proprio is not None:
            command = self.previous_action if command_applied is None else command_applied
            self.history.append(self.previous_proprio, command, proprio,
                                reset_boundary=reset_boundary)
        elif reset_boundary:
            self.history.reset(keep_long_memory=True)
        if reset_boundary:
            self.previous_latents.fill(0)
        packed = np.concatenate((observation, *(v.reshape(-1) for v in self.history.tensors()),
                                 self.previous_latents.reshape(-1)))[None]
        action, latent = self.session.run(list(OUTPUT_NAMES), {INPUT_NAME: packed})
        if (action.shape != (1, 29) or latent.shape != (1, 64)
                or not np.isfinite(action).all() or not np.isfinite(latent).all()):
            raise RuntimeError("ONNX policy returned invalid actions or context")
        self.previous_latents = np.concatenate((self.previous_latents[1:], latent), axis=0)
        self.previous_proprio, self.previous_action = proprio, action[0].copy()
        return self.previous_action.copy()
