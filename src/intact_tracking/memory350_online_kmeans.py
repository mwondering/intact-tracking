"""Checkpointed online K-means routing, independent of policy gradients."""

import torch
from torch import nn
from torch.nn import functional as F


def normalized_latent(z):
    return F.normalize(z.detach().float(), dim=-1, eps=1e-8)


def squared_distances(x, centers):
    return (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None]
            - 2 * x @ centers.T).clamp_min(0)


@torch.no_grad()
def fit_kmeans(x, k=16, *, seed=731, restarts=3, iterations=60):
    """Euclidean K-means on unit-normalized latent; center indices stay explicit."""
    if x.ndim != 2 or len(x) < k or not torch.isfinite(x).all():
        raise ValueError("K-means requires enough finite latent vectors")
    x = normalized_latent(x)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    best = None
    for _ in range(restarts):
        centers = [x[torch.randint(len(x), (), device=x.device, generator=generator)]]
        minimum = squared_distances(x, torch.stack(centers)).squeeze(-1)
        for _ in range(1, k):
            if float(minimum.sum()) < 1e-8:
                raise ValueError("Warm history has fewer distinct latent groups than experts")
            index = torch.multinomial(minimum, 1, generator=generator)[0]
            centers.append(x[index])
            minimum = torch.minimum(minimum, squared_distances(x, x[index:index + 1]).squeeze(-1))
        centers = torch.stack(centers)
        for _ in range(iterations):
            distances = squared_distances(x, centers)
            ids = distances.argmin(-1)
            counts = torch.bincount(ids, minlength=k)
            sums = torch.zeros_like(centers).index_add_(0, ids, x)
            candidate = sums / counts.clamp_min(1)[:, None]
            empty = (counts == 0).nonzero().flatten()
            if len(empty):
                far = distances.min(-1).values.topk(len(empty)).indices
                candidate[empty] = x[far]
            shift = (candidate - centers).square().sum(-1).max()
            centers = candidate
            if float(shift) < 1e-10:
                break
        distances = squared_distances(x, centers)
        inertia = float(distances.min(-1).values.mean())
        if best is None or inertia < best[0]:
            best = inertia, centers.clone()
    return best[1], best[0]


class OnlineKMeansRouter(nn.Module):
    """No parameters. Update only AFTER a rollout's complete PPO update.

    Sufficient statistics are pooled across ranks. Empty centers keep their
    identity and position. Neither forward nor evaluation mutates the router.
    """

    def __init__(self, k=16, latent_dim=64, center_rate=0.01, max_switch_fraction=0.02):
        super().__init__()
        if k < 2 or not 0 < center_rate <= 1 or not 0 < max_switch_fraction <= 1:
            raise ValueError("Invalid online K-means settings")
        self.k, self.latent_dim = int(k), int(latent_dim)
        self.center_rate = float(center_rate)
        self.max_switch_fraction = float(max_switch_fraction)
        self.register_buffer("centers", torch.zeros(k, latent_dim))
        self.register_buffer("assignment_count", torch.zeros(k, dtype=torch.float64))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @staticmethod
    def distributed():
        return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

    @torch.no_grad()
    def initialize(self, latent, seed=731):
        x = normalized_latent(latent.reshape(-1, self.latent_dim))
        if self.distributed():
            gathered = [torch.empty_like(x) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, x)
            x = torch.cat(gathered)
        # All ranks execute the same small fit, then rank zero is authoritative.
        # This also makes a fit failure collective, before any broadcast wait.
        centers, inertia = fit_kmeans(x, self.k, seed=seed)
        self.centers.copy_(centers)
        if self.distributed():
            torch.distributed.broadcast(self.centers, src=0)
        counts = torch.bincount(self(x), minlength=self.k)
        if not bool((counts > 0).all()):
            raise ValueError("Initialization left an unused expert")
        self.assignment_count.copy_(counts)
        self.initialized.fill_(True)
        return {"bootstrap_inertia": inertia, "bootstrap_samples": len(x),
                "bootstrap_counts": counts.cpu().tolist()}

    def forward(self, latent):
        shape = latent.shape[:-1]
        x = normalized_latent(latent.reshape(-1, self.latent_dim))
        return squared_distances(x, self.centers).argmin(-1).reshape(shape)

    @torch.no_grad()
    def update_centers(self, latent):
        if not bool(self.initialized):
            raise RuntimeError("Initialize routing from warm interaction history first")
        x = normalized_latent(latent.reshape(-1, self.latent_dim))
        old = self.centers.clone()
        ids = squared_distances(x, old).argmin(-1)
        counts = torch.bincount(ids, minlength=self.k)
        sums = torch.zeros_like(old).index_add_(0, ids, x)
        packed = torch.cat((sums.flatten(), counts.to(sums)))
        if self.distributed():
            torch.distributed.all_reduce(packed)
        counts = packed[-self.k:]
        means = packed[:-self.k].reshape_as(old) / counts.clamp_min(1)[:, None]
        delta = self.center_rate * (means - old) * (counts > 0)[:, None]
        # Unsupervised routing is an additional policy change. Limit its effect
        # on this rollout instead of silently moving many samples to new heads.
        scale = 1.0
        for _ in range(9):
            candidate = old + scale * delta
            moved = (squared_distances(x, candidate).argmin(-1) != ids).sum()
            stats = torch.stack((moved, moved.new_tensor(len(x)))).double()
            if self.distributed():
                torch.distributed.all_reduce(stats)
            switched = float(stats[0] / stats[1])
            if switched <= self.max_switch_fraction:
                break
            scale *= 0.5
        else:
            candidate, switched, scale = old, 0.0, 0.0
        self.centers.copy_(candidate)
        self.assignment_count.add_(counts.double())
        self.update_count.add_(1)
        fractions = counts / counts.sum().clamp_min(1)
        return {"router_center_shift_rms": float((candidate - old).square().mean().sqrt()),
                "router_center_update_switch_fraction": switched,
                "router_center_effective_rate": self.center_rate * scale,
                "router_center_updates": float(self.update_count),
                "router_rollout_active_experts": float((counts > 0).sum()),
                "router_rollout_max_fraction": float(fractions.max()),
                **{f"router_rollout_fraction_{i:02d}": float(v) for i, v in enumerate(fractions)}}
