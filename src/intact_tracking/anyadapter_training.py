"""Independent world-model optimization followed by ordinary SP residual PPO."""

import time

import torch

from intact_tracking.anyadapter_world_model import (
    WorldModel, RolloutHistoryBank, STATE_GROUP, WORLD_GROUP, COUNT_GROUP,
    LOSS_NAMES, autoregressive_loss,
)
from intact_tracking.memory350_tracker_action_policy import TrackerActionPPO
from intact_tracking.residual_policy import DYNAMICS_LATENT_GROUP


class AnyAdapterPPO(TrackerActionPPO):
    def __init__(self, *args, world_model_lr=1e-4, world_model_epochs=5,
                 world_model_mini_batches=4, prediction_steps=20, world_model_seed=40130,
                 world_model_hidden_dims=(512, 512, 256, 256, 256, 128), **kwargs):
        super().__init__(*args, **kwargs)
        n, steps = self.storage.num_envs, self.storage.num_transitions_per_env
        if not 1 <= prediction_steps <= steps or n % world_model_mini_batches:
            raise ValueError('World-model windows/minibatches must fit this rollout')
        if min(world_model_epochs, world_model_mini_batches, world_model_lr) <= 0:
            raise ValueError('World-model training budgets must be positive')
        self.prediction_steps = prediction_steps
        self.wm_epochs, self.wm_batches = world_model_epochs, world_model_mini_batches
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(world_model_seed)
            self.world_model = WorldModel(world_model_hidden_dims).to(self.device)
        self.wm_parameters = list(self.actor.history_encoder.parameters()) + list(self.world_model.parameters())
        self.world_optimizer = torch.optim.Adam(self.wm_parameters, lr=world_model_lr)
        self.world_optimizer_steps = 0
        # Independent generator; checkpoint its state. No environment, Gaussian
        # action or PPO shuffle draws are consumed by prediction-window sampling.
        self.window_rng = torch.Generator(device='cpu').manual_seed(world_model_seed)
        self.history_bank = RolloutHistoryBank(n, steps, self.device)
        self.world_states = torch.zeros(n, steps + 1, 65, device=self.device)
        self.boundaries = torch.zeros(n, steps, dtype=torch.bool, device=self.device)
        self.live_history = None
        self._latest_obs = None

    def bind_environment(self, wrapper):
        self.live_history = wrapper.history

    def act(self, obs):
        if self.live_history is None:
            raise RuntimeError('Bind the AnyAdapter history provider before collecting a rollout')
        if self.storage.step == 0:
            self.history_bank.start(self.live_history)
            self.world_states[:, 0].copy_(obs[WORLD_GROUP])
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        step = self.storage.step
        self.history_bank.append(step, self.transition.observations[STATE_GROUP], self.transition.actions, obs[COUNT_GROUP])
        self.world_states[:, step + 1].copy_(obs[WORLD_GROUP])
        self.boundaries[:, step].copy_(dones.bool() | extras['motion_resample_boundary'].bool())
        self._latest_obs = obs
        return super().process_env_step(obs, rewards, dones, extras)

    @torch.no_grad()
    def broadcast_parameters(self):
        super().broadcast_parameters()
        for value in self.wm_parameters:
            torch.distributed.broadcast(value, src=0)

    def _world_update(self):
        n, steps = self.storage.num_envs, self.storage.num_transitions_per_env
        starts = torch.randint(steps-self.prediction_steps+1, (n,), generator=self.window_rng).to(self.device)
        stats = torch.zeros(5, device=self.device)
        norms = torch.zeros((), device=self.device)
        self.actor.history_encoder.requires_grad_(True)
        try:
            for _ in range(self.wm_epochs):
                order = torch.randperm(n, generator=self.window_rng).to(self.device)
                for ids in order.chunk(self.wm_batches):
                    begin = starts[ids]
                    times = begin[:, None] + torch.arange(self.prediction_steps, device=self.device)
                    states = self.world_states[ids[:, None], torch.cat((times, times[:, -1:] + 1), -1)]
                    actions = self.storage.actions[times, ids[:, None]]
                    boundary = self.boundaries[ids[:, None], times]
                    def reanchor(offset):
                        return self.history_bank.history(ids, begin + offset)
                    self.world_optimizer.zero_grad(set_to_none=True)
                    loss, components = autoregressive_loss(
                        self.actor.history_encoder, self.world_model, reanchor(0), states, actions, boundary,
                        reanchor=reanchor)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite AnyAdapter world-model loss')
                    loss.backward()
                    if self.is_multi_gpu:
                        flat = torch.cat([p.grad.flatten() for p in self.wm_parameters])
                        torch.distributed.all_reduce(flat)
                        flat.div_(self.gpu_world_size)
                        offset = 0
                        for p in self.wm_parameters:
                            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
                            offset += p.numel()
                    norm = torch.nn.utils.clip_grad_norm_(self.wm_parameters, self.max_grad_norm, error_if_nonfinite=True)
                    self.world_optimizer.step()
                    self.world_optimizer_steps += 1
                    stats.add_(components.detach())
                    norms.add_(norm.detach())
        finally:
            self.world_optimizer.zero_grad(set_to_none=True)
            self.actor.history_encoder.requires_grad_(False)
        batches = self.wm_epochs * self.wm_batches
        result = {f'WorldModel/{name}_loss': float(value / batches) for name, value in zip(LOSS_NAMES, stats)}
        result.update({'WorldModel/loss': float(stats.sum() / batches),
                       'WorldModel/gradient_norm': float(norms / batches),
                       'WorldModel/optimizer_steps': self.world_optimizer_steps,
                       'WorldModel/valid_transition_fraction': float((~self.boundaries).float().mean())})
        return result

    @torch.no_grad()
    def _refresh_embeddings(self):
        storage = self.storage
        n, steps = storage.num_envs, storage.num_transitions_per_env
        drift_sq, dimensions = 0., 0
        result = {}
        for step in range(steps):
            for ids in torch.arange(n, device=self.device).split(1024):
                old = storage.observations[DYNAMICS_LATENT_GROUP][step, ids].clone()
                value = self.actor.history_encoder(self.history_bank.history(ids, step))
                storage.observations[DYNAMICS_LATENT_GROUP][step, ids] = value
                drift_sq += float((value-old).square().sum())
                dimensions += value.numel()
                if step == 0 and int(ids[0]) == 0:
                    obs = storage.observations[step, ids]
                    self.actor.distribution.update(self.actor(obs))
                    # Distribution params are stored in the RSL storage, not
                    # reconstructed from a mutable last-forward value.
                    old_params = tuple(v[step, ids] for v in storage.distribution_params)
                    result['WorldModel/pre_ppo_kl'] = float(self.actor.get_kl_divergence(
                        old_params, self.actor.output_distribution_params).mean())
                    result['WorldModel/action_mean_drift_rms'] = float((self.actor.output_mean-old_params[0]).square().mean().sqrt())
        if self._latest_obs is not None:
            self._latest_obs.set(DYNAMICS_LATENT_GROUP, self.actor.history_encoder(self.live_history.frames))
        result['WorldModel/embedding_drift_rms'] = (drift_sq / max(dimensions, 1)) ** .5
        return result

    def update(self):
        start = time.perf_counter()
        result = self._world_update()
        result.update(self._refresh_embeddings())
        result['WorldModel/update_seconds'] = time.perf_counter() - start
        result.update(super().update())
        return result

    def save(self):
        return {**super().save(), 'world_model_state_dict': self.world_model.state_dict(),
                'world_optimizer_state_dict': self.world_optimizer.state_dict(),
                'world_optimizer_steps': self.world_optimizer_steps, 'world_window_rng': self.window_rng.get_state()}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        self.world_model.load_state_dict(loaded_dict['world_model_state_dict'], strict=strict)
        # This algorithm's supported resume is full state, never a silent
        # actor-only load with a newly initialized world model optimizer.
        self.world_optimizer.load_state_dict(loaded_dict['world_optimizer_state_dict'])
        self.world_optimizer_steps = int(loaded_dict['world_optimizer_steps'])
        self.window_rng.set_state(loaded_dict['world_window_rng'].cpu())
        return result
