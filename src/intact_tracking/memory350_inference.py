"""Strict loading and causal online inference for the separate memory350 version."""

from contextlib import nullcontext
from pathlib import Path

import torch

from intact_tracking.memory350_bank import InteractionMemory
from intact_tracking.memory350_model import HierarchicalContextEncoder, Memory350Config
from intact_tracking.residual_context import FrozenContextCheckpoint, _as_vector
from intact_tracking.rollout.mjlab_adapter import _sha256


def load_memory350_checkpoint(path, *, device, expected_tracker_sha256=None):
    path = Path(path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = Memory350Config(**state["model_config"])
    if state["architecture_version"] != config.architecture_version or config.architecture_version != Memory350Config().architecture_version:
        raise ValueError("Not a compatible hierarchical-memory checkpoint")
    tracker_sha = state["tracker"]["checkpoint_sha256"]
    if expected_tracker_sha256 is not None and tracker_sha != expected_tracker_sha256:
        raise ValueError("Hierarchical encoder and frozen tracker checkpoints differ")
    encoder = HierarchicalContextEncoder(config)
    encoder.load_state_dict({name.removeprefix("context_encoder."): value for name, value in state["model"].items()
                             if name.startswith("context_encoder.")}, strict=True)
    encoder.to(device).requires_grad_(False).eval()
    statistics = state["normalization"]
    return FrozenContextCheckpoint(
        encoder=encoder, config=config,
        state_mean=_as_vector(statistics, "state_mean", 71, torch.device(device)),
        state_std=_as_vector(statistics, "state_std", 71, torch.device(device)),
        action_mean=_as_vector(statistics, "action_mean", 29, torch.device(device)),
        action_std=_as_vector(statistics, "action_std", 29, torch.device(device)),
        path=str(path), sha256=_sha256(path), tracker_sha256=tracker_sha)


class Memory350Inference:
    def __init__(self, checkpoint, num_worlds, *, batch_size=512, use_bfloat16=True):
        self.checkpoint = checkpoint
        self.memory = InteractionMemory(num_worlds, device=checkpoint.state_mean.device)
        self.batch_size = batch_size
        self.use_bfloat16 = use_bfloat16 and checkpoint.state_mean.device.type == "cuda"

    def append(self, batch):
        self.memory.begin_step(batch["episode_id"], batch["episode_step"], batch["motion_id"],
                               batch["motion_step"], batch.get("parameters_changed"))
        raw = torch.cat((batch["robot_state"], batch["joint_target"], batch["next_robot_state"]), dim=-1)
        self.memory.finish_step(raw, batch["reset_boundary"])

    @torch.inference_mode()
    def encode(self):
        checkpoint = self.checkpoint

        def normalize(raw):
            return torch.cat(((raw[..., :71] - checkpoint.state_mean) / checkpoint.state_std,
                              (raw[..., 71:100] - checkpoint.action_mean) / checkpoint.action_std,
                              (raw[..., 100:] - checkpoint.state_mean) / checkpoint.state_std), dim=-1)

        result = []
        for start in range(0, self.memory.num_worlds, self.batch_size):
            ids = self.memory._worlds[start:start + self.batch_size]
            short, valid = self.memory.ordered_short(ids)
            long, long_valid = self.memory.read_chunks(ids)
            short, long = normalize(short), normalize(long)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bfloat16 else nullcontext()
            with autocast:
                result.append(checkpoint.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                                                 valid, long, long_valid).float())
        return torch.cat(result)
