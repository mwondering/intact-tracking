"""RMA adaptation from causal noisy proprioception; no privileged actor inputs."""

import copy

import torch
from torch import nn
from rsl_rl.modules import MLP

from intact_tracking.heavy_rma_teacher import CachedTrackerActor, embedding_residual_input
from intact_tracking.memory350_proprio_inputs import PROPRIO_DIM, PROPRIO_TERMS, PROPRIO_WIDTHS

VERSION = 'heavy_rma_student_v1'
HISTORY_STEPS, EMBEDDING_DIM = 50, 64
EMBEDDING_GROUP = 'rma_student_embedding'
INPUT_CONTRACT = {
    'version': VERSION, 'history_steps': HISTORY_STEPS, 'frame_dim': PROPRIO_DIM,
    'terms': list(PROPRIO_TERMS), 'widths': list(PROPRIO_WIDTHS), 'frequency_hz': 50,
    'order': 'oldest to newest; current sensor frame includes previous executed raw command',
    'reset': 'clear before appending first post-reset or post-motion-teleport observation',
    'padding': 'zero invalid frames after normalization and point MLP; explicit valid count',
    'noise': 'reuse cached tracker sensor sample', 'privileged_inputs': False,
    'reference': 'used only by frozen tracker; adaptation sees no reference',
}


class ProprioNormalizer(nn.Module):
    """Globally pooled Welford moments, explicitly updated only by the trainer."""
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(PROPRIO_DIM))
        self.register_buffer('variance', torch.ones(PROPRIO_DIM))
        self.register_buffer('count', torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, frames, distributed=None):
        x = frames.reshape(-1, PROPRIO_DIM).double()
        moments = torch.cat((x.sum(0), x.square().sum(0), x.new_tensor([len(x)])))
        if distributed is not None:
            distributed.all_reduce_sum(moments)
        n = moments[-1]
        if n <= 0:
            return
        mean = moments[:PROPRIO_DIM] / n
        var = (moments[PROPRIO_DIM:2*PROPRIO_DIM] / n - mean.square()).clamp_min(0)
        total = self.count + n
        delta = mean - self.mean.double()
        pooled = (self.variance.double()*self.count + var*n + delta.square()*self.count*n/total)/total
        self.mean.add_((delta*n/total).float())
        self.variance.copy_(pooled.float())
        self.count.copy_(total)

    def forward(self, value):
        return (value-self.mean) / self.variance.clamp_min(1e-4).sqrt()


class RMAHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = ProprioNormalizer()
        self.point = nn.Sequential(nn.Linear(PROPRIO_DIM, 128), nn.ELU(), nn.Linear(128, 32), nn.ELU())
        self.temporal = nn.Sequential(nn.Conv1d(32, 32, 8, stride=4), nn.ELU(),
                                      nn.Conv1d(32, 32, 5), nn.ELU(), nn.Conv1d(32, 32, 5), nn.ELU())
        self.output = nn.Linear(96, EMBEDDING_DIM)
        self.register_buffer('positions', torch.arange(HISTORY_STEPS), persistent=False)

    def forward(self, history, count):
        mask = self.positions[None, :] >= HISTORY_STEPS - count.reshape(-1, 1)
        # Inactive data may contain arbitrary bytes; mask BEFORE all nonlinear operations.
        values = torch.where(mask[..., None], self.normalizer(history), 0.)
        features = self.point(values) * mask[..., None]
        return self.output(self.temporal(features.transpose(1, 2)).flatten(1))


class ProprioHistory:
    def __init__(self, num_envs, device):
        self.frames = torch.zeros(num_envs, HISTORY_STEPS, PROPRIO_DIM, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    @torch.no_grad()
    def clear(self, ids=None):
        if ids is None:
            self.frames.zero_(); self.count.zero_()
        else:
            self.frames[ids] = 0; self.count[ids] = 0

    @torch.no_grad()
    def append(self, frame, boundary=None):
        if boundary is not None:
            self.clear(boundary)
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = frame
        self.count.add_(1).clamp_(max=HISTORY_STEPS)


class HistoryBank:
    """N*(H+T)*D storage; reconstruct boundary-masked windows on demand."""
    def __init__(self, num_envs, steps, device):
        self.steps = steps
        self.frames = torch.empty(num_envs, HISTORY_STEPS+steps, PROPRIO_DIM, device=device)
        self.counts = torch.empty(num_envs, steps+1, dtype=torch.long, device=device)

    @torch.no_grad()
    def start(self, history):
        self.frames[:, :HISTORY_STEPS].copy_(history.frames)
        self.counts[:, 0].copy_(history.count)

    @torch.no_grad()
    def append(self, step, history):
        self.frames[:, HISTORY_STEPS+step].copy_(history.frames[:, -1])
        self.counts[:, step+1].copy_(history.count)

    def batch(self, flat_ids):
        env_ids, times = flat_ids // self.steps, flat_ids % self.steps
        positions = times[:, None] + torch.arange(HISTORY_STEPS, device=self.frames.device)
        count = self.counts[env_ids, times]
        return self.frames[env_ids[:, None], positions], count, env_ids


class RMAStudentActor(CachedTrackerActor):
    def __init__(self, obs, *args, initialization_seed=None, initial_action_std=1.,
                 residual_hidden_dims=(512, 256, 128), **kwargs):
        kwargs.pop('fusion_mode', None)
        kwargs.pop('dynamics_latent_dim', None)
        super().__init__(obs, *args, fusion_mode='baseline', initialization_seed=initialization_seed,
                         initial_action_std=initial_action_std, residual_hidden_dims=residual_hidden_dims, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(initialization_seed or 0))
            self.residual_input_dim = self.tracker.policy_input_dim + 29 + 64
            self.residual_mlp = MLP(self.residual_input_dim, 29, list(residual_hidden_dims), 'elu')
            self.adaptation = RMAHistoryEncoder()
        self.requires_grad_(False)
        self.adaptation.requires_grad_(True)
        self.eval()

    def train(self, mode=True):
        # Frozen controller and all of its normalization stay in eval mode.
        super().train(False)
        if hasattr(self, 'adaptation'):
            self.adaptation.train(mode)
        return self

    def _residual_input(self, obs, tracker_features, base_action, *, latent_override=None):
        embedding = obs[EMBEDDING_GROUP] if latent_override is None else latent_override
        return embedding_residual_input(tracker_features, embedding, base_action)

    def action_from_embedding(self, obs, embedding):
        features, base = self._base_features_and_action(obs)
        return base + self._residual(embedding_residual_input(features, embedding, base))

    @torch.no_grad()
    def initialize_from_teacher(self, teacher):
        own, source = self.state_dict(), teacher.state_dict()
        transferable = {k: v for k, v in source.items() if not k.startswith('dr_encoder.')}
        expected = {k for k in own if not k.startswith('adaptation.')}
        if set(transferable) != expected:
            raise ValueError('Teacher/student controller state keys differ')
        self.load_state_dict({**own, **transferable}, strict=True)


def build_actor(state, cls=None):
    """Construct from embedded tracker/config without opening external weight files."""
    from omegaconf import OmegaConf
    from intact_tracking.memory350_checkpoint import embedded_tracker
    from intact_tracking.heavy_baseline_export import make_actor_observation
    from intact_tracking.memory350_deploy import TRACKER_WIDTH
    from intact_tracking.heavy_rma_teacher import RMATeacherActor
    cls = cls or (RMAStudentActor if state['residual_policy']['method'] == 'rma_student' else RMATeacherActor)
    tracker = embedded_tracker(state)
    if tracker is None:
        raise ValueError('Self-contained tracker bundle required')
    agent = OmegaConf.to_container(state['cfg'].agent, resolve=True)
    kwargs = copy.deepcopy(agent['actor']); kwargs.pop('class_name')
    value = torch.zeros(1, TRACKER_WIDTH); value[:, 0] = 1
    obs = make_actor_observation(value, kwargs)
    actor = cls(obs, agent['obs_groups'], 'actor', 29,
                tracker_state_dict=tracker['actor_state_dict'], **kwargs)
    return actor
