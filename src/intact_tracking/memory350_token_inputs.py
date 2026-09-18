"""Causal, parameter-independent snapshots for the 29-token residual policy."""

from __future__ import annotations

import torch
from torch import nn

HISTORY_STEPS = 5
FUTURE_STEPS = 4
FRAME_DIMS = (320, 269, 260, 64, 29)
FRAME_NAMES = ("proprio", "reference", "error", "latent", "tracker_action")
FRAME_DIM = sum(FRAME_DIMS)
LATENT_START = sum(FRAME_DIMS[:3])
LATENT_SLICE = slice(LATENT_START, LATENT_START + 64)
HISTORY = "token_frame_history"
VALID = "token_frame_valid"
FUTURE = "token_future_reference"
BASE_ACTION = "token_tracker_action"


class SPV52TokenSplitter(nn.Module):
    """Reindex frozen, normalized SPV5-2 features without changing their values.

    The source is term-major, not frame-major. The 577-D reference packet is
    partitioned exactly into a 269-D current target and four 77-D future targets.
    Future root offsets retain the source reference-root coordinate convention.
    No future actual states, estimates, latent codes, or actions are constructed.
    """

    def __init__(self):
        super().__init__()
        proprio, offset = [], 0
        for width in (29, 29, 3, 3, 29, 29):
            proprio.extend(range(offset + 4 * width, offset + 5 * width))
            offset += 5 * width
        proprio.extend(range(610, 806))  # Current height and 13 key bodies.
        proprio.extend(range(1643, 1645))  # Estimated contact probabilities.
        # Source standard reference: offsets, rotations, height, linear velocity,
        # joint position/velocity, gravity, angular velocity (each over 5 frames).
        fields = ((12, 6), (42, 1), (47, 3), (62, 29),
                  (207, 29), (352, 3), (367, 3))
        current = [806 + start + j for start, width in fields for j in range(width)]
        current.extend(range(1188, 1383))  # Current reference key-body state.
        future = []
        for step in range(1, 5):
            indices = list(range(806 + 3 * (step - 1), 806 + 3 * step))
            indices.extend(806 + start + step * width + j
                           for start, width in fields for j in range(width))
            future.append(indices)
        self.register_buffer("proprio_indices", torch.tensor(proprio), persistent=False)
        self.register_buffer("reference_indices", torch.tensor(current), persistent=False)
        self.register_buffer("future_indices", torch.tensor(future), persistent=False)

    def forward(self, features, tracker_action, latent):
        if features.ndim != 2 or features.shape[-1] != 1645:
            raise ValueError("Tokenization requires the audited 1645-D SPV5-2 feature vector")
        if tracker_action.shape != (len(features), 29) or latent.shape != (len(features), 64):
            raise ValueError("Tracker action and latent must align with the current observation")
        features, tracker_action, latent = (x.detach() for x in (features, tracker_action, latent))
        frame = torch.cat((features[:, self.proprio_indices],
                           features[:, self.reference_indices], features[:, 1383:1643],
                           latent.to(features), tracker_action.to(features)), -1)
        future = features[:, self.future_indices]
        return frame, future


class TokenHistory:
    """Immutable per-observation snapshots, updated only by environment collection.

    Re-observing the same decision replaces its last frame; it never advances the
    window. New episodes, changed motions and nonconsecutive motion frames clear
    the affected world's history. Old snapshots remain valid for shuffled PPO.
    This short policy memory is independent of the encoder's long memory.
    """

    def __init__(self, num_envs, device):
        self.frames = torch.zeros(num_envs, HISTORY_STEPS, FRAME_DIM, device=device)
        self.valid = torch.zeros(num_envs, HISTORY_STEPS, dtype=torch.bool, device=device)
        self.stamp = torch.full((num_envs, 3), -1, dtype=torch.long, device=device)

    def clear(self, env_ids=None):
        # Never mutate tensors already attached to a collected observation.
        self.frames, self.valid, self.stamp = (x.clone() for x in (self.frames, self.valid, self.stamp))
        ids = slice(None) if env_ids is None else env_ids
        self.frames[ids] = 0
        self.valid[ids] = False
        self.stamp[ids] = -1

    @torch.no_grad()
    def observe(self, frame, stamp):
        if frame.shape != (len(self.frames), FRAME_DIM) or stamp.shape != self.stamp.shape:
            raise ValueError("History frame/stamp batch differs from the environment batch")
        initialized = self.valid[:, -1]
        same = initialized & (stamp == self.stamp).all(-1)
        continuous = (initialized & (stamp[:, :2] == self.stamp[:, :2]).all(-1)
                      & (stamp[:, 2] == self.stamp[:, 2] + 1))
        prefix = torch.where(same[:, None, None], self.frames[:, :-1], self.frames[:, 1:])
        mask = torch.where(same[:, None], self.valid[:, :-1], self.valid[:, 1:])
        keep = same | continuous
        prefix = prefix.masked_fill(~keep[:, None, None], 0)
        mask = mask & keep[:, None]
        self.frames = torch.cat((prefix, frame.detach()[:, None]), 1)
        self.valid = torch.cat((mask, torch.ones_like(mask[:, :1])), 1)
        self.stamp = stamp.detach().clone()
        return self.frames, self.valid


def intervene_latent_history(obs, mode, *, donors=None, eligible=None):
    """Replace the complete latent stream, leaving motion/state/action tokens alone."""
    if mode not in ("zero", "paired-swap"):
        raise ValueError("Expected zero or paired-swap latent intervention")
    result = obs.clone(recurse=False)
    history = obs[HISTORY].clone()
    current = obs["dynamics_latent"]
    if mode == "zero":
        history[..., LATENT_SLICE] = 0
        current = torch.zeros_like(current)
    else:
        if donors is None or eligible is None:
            raise ValueError("Paired swaps require donor indices and an eligibility mask")
        # Only exchange complete, equally valid short windows.
        eligible = eligible & obs[VALID].all(-1) & obs[VALID][donors].all(-1)
        history[..., LATENT_SLICE] = torch.where(
            eligible[:, None, None], obs[HISTORY][donors, :, LATENT_SLICE],
            history[..., LATENT_SLICE])
        current = torch.where(eligible[:, None], current[donors], current)
    result.set(HISTORY, history)
    result.set("dynamics_latent", current)
    return result
