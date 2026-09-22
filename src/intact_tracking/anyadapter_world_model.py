"""AnyAdapter history representation and causal 20-step dynamics prediction.

Port of OpenTrack cb9b751993a2483e5d1805a2565ddbfe950c04c9, using SP raw
commands. Histories contain whole (pre-state64, command29) frames, never the
64-coordinate flat shift present in that reference's autoregressive loss.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

HISTORY_STEPS, FRAME_DIM, EMBEDDING_DIM, STATE_DIM = 79, 93, 128, 65
VELOCITY_SCALE, DT = .05, .02
STATE_GROUP, WORLD_GROUP, COUNT_GROUP = "adapter_proprio", "adapter_world_state", "adapter_history_count"
LOSS_NAMES = ("gyro", "gravity", "joint_position", "joint_velocity", "height")
LOSS_WEIGHTS = (500., 500., 1., .5, 500.)


def lecun_uniform(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        fan_in = module.weight[0].numel()
        nn.init.uniform_(module.weight, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        nn.init.zeros_(module.bias)


class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(FRAME_DIM, 64, 9, stride=5), nn.SiLU(),
                                  nn.Conv1d(64, 64, 6, stride=3), nn.SiLU())
        self.output = nn.Linear(256, EMBEDDING_DIM)
        self.apply(lecun_uniform)

    def forward(self, history):
        # Flatten time-major, as Flax ConvMLP does after its channels-last conv.
        hidden = self.conv(history.transpose(-1, -2)).transpose(-1, -2)
        return self.output(hidden.flatten(-2))


class WorldModel(nn.Module):
    def __init__(self, hidden_dims=(512, 512, 256, 256, 256, 128)):
        super().__init__()
        widths = (STATE_DIM + EMBEDDING_DIM + 29, *hidden_dims, 33)
        layers = []
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(widths) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.apply(lecun_uniform)

    def forward(self, state, embedding, action):
        delta = self.mlp(torch.cat((state, embedding, action), -1))
        gyro = state[..., :3] + delta[..., :3]
        velocity = state[..., 35:64] + delta[..., 3:32]
        position = state[..., 6:35] + velocity * (DT / VELOCITY_SCALE)
        gravity = state[..., 3:6] - torch.linalg.cross(gyro / VELOCITY_SCALE, state[..., 3:6], dim=-1) * DT
        # Exact unit direction (reference uses norm(g + 1e-4), not norm(g)).
        gravity = F.normalize(gravity, dim=-1, eps=1e-8)
        height = state[..., 64:] + delta[..., 32:]
        return torch.cat((gyro, gravity, position, velocity, height), -1)


def append_frame(history, state, action):
    """At t+1 the newest frame is (s_t, a_t), not (s_{t+1}, a_t)."""
    frame = torch.cat((state[..., :64], action), -1)
    return torch.cat((history[:, 1:], frame[:, None]), 1)


def component_losses(prediction, target, valid):
    errors = torch.stack(((prediction[..., :3] - target[..., :3]).abs().sum(-1),
                          1 - (prediction[..., 3:6] * target[..., 3:6]).sum(-1),
                          (prediction[..., 6:35] - target[..., 6:35]).abs().sum(-1),
                          (prediction[..., 35:64] - target[..., 35:64]).abs().sum(-1),
                          (prediction[..., 64:] - target[..., 64:]).abs().sum(-1)), -1)
    # Mean over all transitions, sum over coordinates, as the reference does.
    # Mask the ENTIRE gravity error; resets must not introduce a 500 constant.
    return (errors * valid[..., None]).reshape(-1, 5).mean(0) * errors.new_tensor(LOSS_WEIGHTS)


def autoregressive_loss(encoder, model, history, states, actions, boundaries, *, reanchor):
    """states [B,T+1,65], actions [B,T,29]; reanchor(i) returns real history at i."""
    current, predictions = states[:, 0], []
    for step in range(actions.shape[1]):
        predicted = model(current, encoder(history), actions[:, step])
        predictions.append(predicted)
        next_history = append_frame(history, current, actions[:, step])
        cut = boundaries[:, step]
        current = torch.where(cut[:, None], states[:, step + 1], predicted)
        history = torch.where(cut[:, None, None], reanchor(step + 1), next_history)
    components = component_losses(torch.stack(predictions, 1), states[:, 1:], ~boundaries)
    return components.sum(), components


class CausalHistory:
    """Live chronological history; zero padding does not count as observations."""
    def __init__(self, num_envs, device):
        self.frames = torch.zeros(num_envs, HISTORY_STEPS, FRAME_DIM, device=device)
        self.count = torch.zeros(num_envs, 1, dtype=torch.long, device=device)

    @torch.no_grad()
    def clear(self, ids=None):
        if ids is None:
            self.frames.zero_()
            self.count.zero_()
        else:
            self.frames[ids] = 0
            self.count[ids] = 0

    @torch.no_grad()
    def append(self, state, action, boundary):
        self.frames.copy_(append_frame(self.frames, state, action))
        self.count.add_(1).clamp_(max=HISTORY_STEPS)
        self.clear(boundary)


class RolloutHistoryBank:
    """N*(79+T)*93 storage instead of N*T*79*93 history duplication."""
    def __init__(self, num_envs, steps, device):
        self.frames = torch.zeros(num_envs, HISTORY_STEPS + steps, FRAME_DIM, device=device)
        self.counts = torch.zeros(num_envs, steps + 1, dtype=torch.long, device=device)

    @torch.no_grad()
    def start(self, history):
        self.frames[:, :HISTORY_STEPS].copy_(history.frames)
        self.counts[:, 0].copy_(history.count[:, 0])

    @torch.no_grad()
    def append(self, step, state, action, next_count):
        self.frames[:, HISTORY_STEPS + step].copy_(torch.cat((state, action), -1))
        self.counts[:, step + 1].copy_(next_count[:, 0])

    def history(self, env_ids, steps):
        steps = torch.as_tensor(steps, device=self.frames.device)
        positions = steps[..., None] + torch.arange(HISTORY_STEPS, device=self.frames.device)
        frames = self.frames[env_ids[:, None], positions]
        count = self.counts[env_ids, steps]
        mask = torch.arange(HISTORY_STEPS, device=self.frames.device) >= HISTORY_STEPS - count[:, None]
        return frames * mask[..., None]
