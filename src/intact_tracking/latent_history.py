"""Causal per-world history of frozen latent frames for actor and critic."""
import torch


class LatentHistory:
    def __init__(self, num_envs, *, frames=5, latent_dim=64, device='cpu'):
        if min(num_envs, frames, latent_dim) < 1:
            raise ValueError('History dimensions must be positive')
        self.values = torch.zeros(num_envs, frames, latent_dim, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    @torch.no_grad()
    def append(self, latent, *, reset=None):
        if latent.shape != self.values[:, -1].shape:
            raise ValueError('Incorrect per-world latent frame dimensions')
        if reset is not None:
            if reset.shape != self.count.shape or reset.dtype != torch.bool:
                raise ValueError('Reset mask must be one bool per world')
            self.values[reset] = 0
            self.count[reset] = 0
        self.values[:, :-1] = self.values[:, 1:].clone()
        self.values[:, -1] = latent.detach()
        self.count = (self.count + 1).clamp_max(self.values.shape[1])
        return self.snapshot()

    def snapshot(self):
        # Ownership matters: previous PPO observations must not change on append.
        return self.values.flatten(1).clone()

    @torch.no_grad()
    def clear(self):
        self.values.zero_()
        self.count.zero_()
