"""ABC load-only fine-tuning: original policy head, no residual or extra input."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from mjlab.envs.mdp.dr.body import _decompose_pseudo_inertia_J, _reconstruct_pseudo_inertia_J
from mjlab.managers.event_manager import EventTermCfg

from intact_tracking.environment.mdp.randomizations import rigid_body_payload
from intact_tracking.environment.policy import SPV52HeightContactEstimatorActor
from intact_tracking.preview_protocol import LIMBS
from intact_tracking.residual_policy import WarmStartedHeftCritic
from intact_tracking.simulator_preview_experiment import ScratchHeftCritic

VERSION = "lafan_abc_tracker_head_finetune_warmcritic_v2"
LAFAN = "/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong/lafan_qingtong"


def lafan_files(root=LAFAN):
    # Do not double-weight the additional cropped fight/singlejump derivatives.
    files = sorted(path.resolve() for path in Path(root).glob("*.motion.npz")
                   if re.fullmatch(r"[A-Za-z]+\d+_subject\d+\.motion\.npz", path.name))
    if not files:
        raise ValueError(f"No original-named LaFAN motions in {root}")
    return files


def payload_samples(condition, n, seed):
    if condition not in ("A", "B", "C"):
        raise ValueError("Expected A/B/C load condition")
    if condition == "A":
        return torch.zeros(n, 4)
    if condition == "B":
        return torch.full((n, 4), 4.0)
    if n % 4:
        raise ValueError("C requires a multiple of 4 worlds for exact endpoint proportions")
    generator = torch.Generator().manual_seed(seed)
    masses = 4 * torch.rand(n, 4, generator=generator)
    masses[:n // 4] = 0
    masses[n // 4:n // 2] = 4
    return masses[torch.randperm(n, generator=generator)]


class FourLimbExperimentPayload:
    model_fields = rigid_body_payload.model_fields
    recompute = rigid_body_payload.recompute

    def __init__(self, cfg, env):
        self.env = env
        missing = tuple(name for name in self.model_fields if name not in env.sim.expanded_fields)
        if missing:
            env.sim.expand_model_fields(missing)
        asset = env.scene["robot"]
        local, _ = asset.find_bodies([row[0] for row in LIMBS.values()], preserve_order=True)
        if len(local) != 4:
            raise ValueError("Expected both wrist-yaw and knee bodies")
        self.body_ids = asset.indexing.body_ids[torch.tensor(local, device=env.device)].long()
        self.base = {name: getattr(env.sim.model, name)[:, self.body_ids].clone()
                     for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat")}
        self.mass = payload_samples(cfg.params["condition"], env.num_envs, cfg.params["seed"]).to(env.device)

    def __call__(self, env, env_ids, **kwargs):
        del kwargs
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        original = [self.base[name][ids] for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat")]
        mass = self.mass[ids]
        position = torch.tensor([row[1] for row in LIMBS.values()], device=env.device).expand(len(ids), -1, -1)
        size = torch.tensor([row[2] for row in LIMBS.values()], device=env.device)
        inertia = mass[..., None] * torch.stack((size[:, 1]**2 + size[:, 2]**2,
                                                size[:, 0]**2 + size[:, 2]**2,
                                                size[:, 0]**2 + size[:, 1]**2), -1) / 12
        quat = torch.zeros((*mass.shape, 4), device=env.device)
        quat[..., 0] = 1
        result = _decompose_pseudo_inertia_J(
            _reconstruct_pseudo_inertia_J(*original)
            + _reconstruct_pseudo_inertia_J(mass, position, inertia, quat))
        for name, value, base in zip(self.base, result, original, strict=True):
            mask = mass == 0
            if value.ndim == 3:
                mask = mask[..., None]
            # Nominal slots retain exactly the compiled inertial fields; all
            # calls recompute from the original body, never accumulate payloads.
            getattr(env.sim.model, name)[ids[:, None], self.body_ids] = torch.where(mask, base, value)

    def audit(self):
        actual = self.env.sim.model.body_mass[:, self.body_ids] - self.base["body_mass"]
        torch.testing.assert_close(actual, self.mass, atol=1e-5, rtol=0)
        values = actual.cpu()
        return {"per_limb_mean_kg": values.mean(0).tolist(), "min_kg": float(values.min()),
                "max_kg": float(values.max()), "nominal_worlds": int((values.abs() < 1e-5).all(-1).sum()),
                "hardest_worlds": int(((values - 4).abs() < 1e-5).all(-1).sum()),
                "actual_mass_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest()}


def configure_abc_physics(cfg, condition, seed):
    from intact_tracking.cli.adaptation_eval import configure_physics

    nominal = configure_physics(cfg, "nominal")
    if condition not in ("A", "B", "C"):
        raise ValueError("Expected A/B/C")
    if condition != "A":
        cfg.events["abc_payload"] = EventTermCfg(
            mode="startup", func=FourLimbExperimentPayload,
            params={"condition": condition, "seed": seed + 91283})
    return {"physics": {"A": "nominal", "B": "hardest", "C": "load_dr"}[condition],
            "dr_profile": "abc-load-only", "condition": condition,
            "nominal_configuration": nominal,
            "details": {"limbs": list(LIMBS), "positions": [list(row[1]) for row in LIMBS.values()],
                        "sizes": [list(row[2]) for row in LIMBS.values()],
                        "A": "0 kg", "B": "4 kg per limb",
                        "C": "25% all-zero + 25% all-four + 50% independent U(0,4) per limb",
                        "sampling": "startup, independently seeded per rank, fixed per world"}}


class TrackerFinetuneActor(SPV52HeightContactEstimatorActor):
    """Same original actor, directly optimize its existing MLP and action std."""

    def __init__(self, *args, tracker_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        checkpoint = torch.load(tracker_checkpoint, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.requires_grad_(False)
        self.mlp.requires_grad_(True)
        self.distribution.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for name, child in self.named_children():
            if name not in ("mlp", "distribution"):
                child.eval()
        return self

    def update_normalization(self, obs):
        # Preserve pretrained frontends AND their statistics. No privileged
        # targets are used to update the estimator/reference encoder.
        pass


class SynchronizedScratchCritic(ScratchHeftCritic):
    @torch.no_grad()
    def update_normalization(self, obs):
        if not torch.distributed.is_initialized():
            return super().update_normalization(obs)
        x = torch.cat([obs[name] for name in self.obs_groups], -1).double()
        packed = torch.cat((x.sum(0), x.square().sum(0), x.new_tensor([len(x)])))
        torch.distributed.all_reduce(packed)
        n = packed[-1]
        dim = x.shape[-1]
        mean = packed[:dim] / n
        var = (packed[dim:2 * dim] / n - mean.square()).clamp_min(0)
        norm = self.obs_normalizer
        norm.count.add_(n.long())
        rate = n / norm.count
        delta = mean - norm._mean.double()
        new_mean = norm._mean.double() + rate * delta
        new_var = norm._var.double() + rate * (var - norm._var.double() + delta * (mean - new_mean))
        norm._mean.copy_(new_mean)
        norm._var.copy_(new_var.clamp_min(0))
        norm._std.copy_(norm._var.sqrt())


class SynchronizedWarmStartedCritic(WarmStartedHeftCritic):
    """Original HEFT weights and DecayVecNorm; pool stats across all ranks."""

    @torch.no_grad()
    def update_normalization(self, obs):
        if not self.obs_normalization or not self.obs_normalizer.training:
            return
        if not torch.distributed.is_initialized():
            return super().update_normalization(obs)
        x = self._flat_obs(obs).double()
        packed = torch.cat((x.sum(0), x.square().sum(0), x.new_tensor([len(x)])))
        torch.distributed.all_reduce(packed)
        norm, dim = self.obs_normalizer, self.obs_dim
        norm.sum.mul_(norm.decay).add_(packed[:dim].to(norm.sum.dtype))
        norm.ssq.mul_(norm.decay).add_(packed[dim:2 * dim].to(norm.ssq.dtype))
        norm.count.mul_(norm.decay).add_(packed[-1].to(norm.count.dtype))
