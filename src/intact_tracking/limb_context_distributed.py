"""Collective statistics and checkpoint audits for distributed residual PPO."""

import hashlib

import torch

from intact_tracking.residual_policy import DecayVecNorm, ResidualPPO


class DistributedResidualPPO(ResidualPPO):
    """Plain residual PPO with rollout advantage scaling shared across ranks."""

    def compute_returns(self, obs):
        super().compute_returns(obs)
        if not self.is_multi_gpu or self.normalize_advantage_per_mini_batch:
            return
        # Keep the existing GAE/termination semantics; pool the raw advantages
        # before normalization, as in SP_Tracking's cross-rank PPO path.
        value = self.storage.returns - self.storage.values
        flat = value.double()
        moments = torch.stack((flat.sum(), flat.square().sum(), flat.new_tensor(flat.numel())))
        torch.distributed.all_reduce(moments)
        total, squares, count = moments
        mean = total / count
        variance = ((squares - total * mean) / (count - 1).clamp_min(1)).clamp_min(0)
        self.storage.advantages = (value - mean.to(value.dtype)) / (variance.sqrt().to(value.dtype) + 1e-8)


class SynchronizedDecayVecNorm(DecayVecNorm):
    """The source critic normalization, updated from the union of all ranks."""

    @torch.no_grad()
    def update(self, value):
        if not self.training:
            return
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return super().update(value)
        flat = value.reshape(-1, value.shape[-1])
        packed = torch.cat((flat.sum(0), flat.square().sum(0),
                            flat.new_tensor([flat.shape[0]])))
        torch.distributed.all_reduce(packed)
        size = self.sum.numel()
        self.sum.mul_(self.decay).add_(packed[:size])
        self.ssq.mul_(self.decay).add_(packed[size:2 * size])
        self.count.mul_(self.decay).add_(packed[-1:])


def synchronize_critic_normalization(critic):
    original = critic.obs_normalizer
    replacement = SynchronizedDecayVecNorm(critic.obs_dim, original.decay, original.eps).to(original.sum.device)
    replacement.load_state_dict(original.state_dict(), strict=True)
    replacement.train(original.training)
    critic.obs_normalizer = replacement


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(f"{name}/{value.dtype}/{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_rank_agreement(runner, distributed):
    actor, critic = runner.alg.actor, runner.alg.critic
    actor_parameters = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
    critic_parameters = [(n, p) for n, p in critic.named_parameters() if p.requires_grad]
    local = {"rank": distributed.rank, "completed_updates": runner.completed_learning_updates,
             "actor_trainable_sha256": tensor_digest(actor_parameters),
             "critic_trainable_sha256": tensor_digest(critic_parameters),
             "critic_normalizer_sha256": tensor_digest(critic.obs_normalizer.state_dict().items()),
             "finite": all(bool(torch.isfinite(p).all()) for _, p in actor_parameters + critic_parameters)}
    rows = distributed.all_gather_object(local)
    for key in ("completed_updates", "actor_trainable_sha256", "critic_trainable_sha256", "critic_normalizer_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise RuntimeError(f"Distributed workers diverged in {key}")
    if not all(row["finite"] for row in rows):
        raise RuntimeError("Nonfinite trainable parameters")
    return {"passed": True, "world_size": distributed.world_size, "ranks": rows}


def main_process_call(distributed, function, *, process_group=None):
    packet = None
    if distributed.is_main:
        try:
            packet = {"result": function(), "error": None}
        except Exception as error:
            packet = {"result": None, "error": f"{type(error).__name__}: {error}"}
    if process_group is None:
        packet = distributed.broadcast_object(packet)
    else:
        objects = [packet]
        torch.distributed.broadcast_object_list(objects, src=0, group=process_group)
        packet = objects[0]
    if packet["error"]:
        raise RuntimeError(packet["error"])
    return packet["result"]
