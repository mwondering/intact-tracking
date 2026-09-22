"""Native tracker DR plus 256 stratified four-limb payload mass intervals."""

from dataclasses import dataclass
import hashlib
import math

import torch
from mjlab.envs.mdp.dr.body import _decompose_pseudo_inertia_J, _reconstruct_pseudo_inertia_J
from mjlab.managers.event_manager import EventTermCfg

from intact_tracking.environment.mdp.randomizations import rigid_body_payload
from intact_tracking.preview_protocol import LIMBS
from intact_tracking.rollout.online import FixedDRRolloutConfig


PROFILE = "checkpoint_native_flat_heavy_v1"
EVENT = "stratified_limb_payload"
MASS_LIMITS = (2.5, 2.5, 4., 4.)
INERTIAL_FIELDS = ("body_mass", "body_ipos", "body_inertia", "body_iquat")


@dataclass(frozen=True)
class HeavyDRRolloutConfig(FixedDRRolloutConfig):
    heavy_payload: bool = True
    heavy_max_masses_kg: tuple[float, float, float, float] = MASS_LIMITS
    heavy_com_half_width_m: float = .05

    def __post_init__(self):
        super().__post_init__()
        if not self.checkpoint_native_dr or not self.heavy_payload:
            raise ValueError("Heavy collection requires checkpoint-native DR and payloads")
        if len(self.heavy_max_masses_kg) != 4 or any(
                not math.isfinite(x) or x <= 0 for x in self.heavy_max_masses_kg):
            raise ValueError("Specify four finite positive payload mass limits")
        if not math.isfinite(self.heavy_com_half_width_m) or self.heavy_com_half_width_m <= 0:
            raise ValueError("Payload COM half width must be finite and positive")
        if self.world_id_offset % self.num_envs:
            raise ValueError("Heavy worlds require contiguous equal-sized rank blocks")


def stratified_payload_samples(num_envs, nominal_fraction, rank, seed,
                               max_masses_kg=MASS_LIMITS, com_half_width_m=.05):
    """256 mass-bin combinations; independent continuous samples within bins.

    Remainders rotate across ranks before shuffling world assignments. Other
    DR does not share this grouping, and no worlds share a physics prototype.
    """
    from intact_tracking.memory350_native_dr import native_nominal_ids
    nominal_ids = native_nominal_ids(num_envs, nominal_fraction, allow_endpoints=True)
    is_dr = torch.ones(num_envs, dtype=torch.bool)
    is_dr[nominal_ids] = False
    dr_ids = is_dr.nonzero().flatten()
    generator = torch.Generator().manual_seed(seed)
    count = len(dr_ids)
    groups = (torch.arange(count) + rank * (count % 256)) % 256
    groups = groups[torch.randperm(count, generator=generator)]
    digits = (groups[:, None] // torch.tensor([64, 16, 4, 1])) % 4
    mass = torch.zeros(num_envs, 4)
    mass[dr_ids] = (digits + torch.rand(count, 4, generator=generator)) * torch.tensor(max_masses_kg) / 4
    offsets = torch.zeros(num_envs, 4, 3)
    offsets[dr_ids] = (2 * torch.rand(count, 4, 3, generator=generator) - 1) * com_half_width_m
    group_ids = torch.full((num_envs,), -1, dtype=torch.long)
    group_ids[dr_ids] = groups
    return mass, offsets, group_ids


def composite_inertial_fields(base, mass, offsets):
    """Sum rigid-body pseudo-inertias about a common link-local origin."""
    # This one-time eigendecomposition uses double precision; simulation fields
    # keep their original dtype. Nominal worlds retain bit-identical values.
    original = [base[name].double() for name in INERTIAL_FIELDS]
    added_mass = mass.double()
    position = torch.tensor([row[1] for row in LIMBS.values()], device=mass.device,
                            dtype=torch.float64) + offsets.double()
    size = torch.tensor([row[2] for row in LIMBS.values()], device=mass.device, dtype=torch.float64)
    inertia = added_mass[..., None] * torch.stack((size[:, 1]**2 + size[:, 2]**2,
        size[:, 0]**2 + size[:, 2]**2, size[:, 0]**2 + size[:, 1]**2), -1) / 12
    quat = torch.zeros((*mass.shape, 4), device=mass.device, dtype=torch.float64)
    quat[..., 0] = 1
    result = _decompose_pseudo_inertia_J(
        _reconstruct_pseudo_inertia_J(*original)
        + _reconstruct_pseudo_inertia_J(added_mass, position, inertia, quat))
    combined = {}
    for name, value in zip(INERTIAL_FIELDS, result, strict=True):
        mask = mass.eq(0)
        if value.ndim == 3:
            mask = mask[..., None]
        combined[name] = torch.where(mask, base[name], value.to(base[name].dtype))
    return combined


class StratifiedLimbPayload:
    model_fields = rigid_body_payload.model_fields
    recompute = rigid_body_payload.recompute

    def __init__(self, cfg, env):
        self.env, self.params = env, cfg.params
        missing = tuple(name for name in self.model_fields if name not in env.sim.expanded_fields)
        if missing:
            env.sim.expand_model_fields(missing)
        asset = env.scene["robot"]
        local, _ = asset.find_bodies([row[0] for row in LIMBS.values()], preserve_order=True)
        if len(local) != 4:
            raise ValueError("Expected both wrist-yaw and knee bodies")
        self.body_ids = asset.indexing.body_ids[torch.tensor(local, device=env.device)].long()
        values = stratified_payload_samples(env.num_envs, cfg.params["nominal_fraction"],
            cfg.params["rank"], cfg.params["seed"], cfg.params["max_masses_kg"],
            cfg.params["com_half_width_m"])
        self.mass, self.offsets, self.group_ids = (v.to(env.device) for v in values)
        self.base = None
        self.combined = None

    def __call__(self, env, env_ids, **kwargs):
        if self.base is None:
            # The event is appended after native startup DR. Snapshot once and
            # always reconstruct from it, so later calls cannot add mass twice.
            self.base = {name: getattr(env.sim.model, name)[:, self.body_ids].clone()
                         for name in INERTIAL_FIELDS}
            self.combined = composite_inertial_fields(self.base, self.mass, self.offsets)
        ids = torch.arange(env.num_envs, device=env.device)
        if env_ids is not None:
            ids = ids[env_ids] if isinstance(env_ids, slice) else env_ids
        for name, value in self.combined.items():
            getattr(env.sim.model, name)[ids[:, None], self.body_ids] = value[ids]

    def privileged_dynamics_targets(self):
        labels = ([f"added_mass_kg/{limb}" for limb in LIMBS]
                  + [f"payload_com_offset/{limb}/{axis}" for limb in LIMBS for axis in "xyz"])
        # These parameters are checked against all four actual inertial fields
        # by audit(), before the rollout captures its immutable physics snapshot.
        return labels, torch.cat((self.mass, self.offsets.flatten(1)), dim=1)

    def audit(self):
        for name, expected in self.combined.items():
            actual = getattr(self.env.sim.model, name)[:, self.body_ids]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(f"Nonfinite payload physics: {name}")
        actual_mass = self.env.sim.model.body_mass[:, self.body_ids] - self.base["body_mass"]
        torch.testing.assert_close(actual_mass, self.mass, atol=1e-6, rtol=1e-6)
        if not bool((self.env.sim.model.body_inertia[:, self.body_ids] > 0).all()):
            raise ValueError("Nonpositive composite principal inertia")
        nominal = self.group_ids.lt(0)
        for name in INERTIAL_FIELDS:
            if not torch.equal(getattr(self.env.sim.model, name)[nominal][:, self.body_ids],
                               self.base[name][nominal]):
                raise ValueError(f"Payload changed nominal {name}")
        dr = ~nominal
        mass, offsets = self.mass[dr].cpu(), self.offsets[dr].cpu()
        counts = torch.bincount(self.group_ids[dr].cpu(), minlength=256)
        return {"profile": PROFILE, "rank": self.params["rank"],
                "nominal_count": int(nominal.sum()), "dr_count": int(dr.sum()),
                "mass_group_counts": counts.tolist(), "all_256_groups_present": bool(counts.gt(0).all()),
                "per_limb_mass_min_kg": mass.amin(0).tolist() if len(mass) else None,
                "per_limb_mass_max_kg": mass.amax(0).tolist() if len(mass) else None,
                "per_limb_mass_mean_kg": mass.mean(0).tolist() if len(mass) else None,
                "com_offset_min_m": offsets.amin((0, 1)).tolist() if len(mass) else None,
                "com_offset_max_m": offsets.amax((0, 1)).tolist() if len(mass) else None,
                "com_offset_norm_max_m": float(offsets.norm(dim=-1).max()) if len(mass) else None,
                "sample_sha256": hashlib.sha256(torch.cat((self.mass.cpu(),
                    self.offsets.cpu().flatten(1)), 1).numpy().tobytes()).hexdigest(),
                "nominal_inertial_fields_exact": True,
                "actual_composite_inertial_fields_verified": True}


def configure_heavy_payload(env_cfg, config):
    rank = config.world_id_offset // config.num_envs
    params = {"nominal_fraction": config.nominal_fraction, "rank": rank,
              "seed": (config.seed if config.dynamics_seed is None else config.dynamics_seed) + 91283,
              "max_masses_kg": tuple(config.heavy_max_masses_kg),
              "com_half_width_m": config.heavy_com_half_width_m}
    env_cfg.events[EVENT] = EventTermCfg(mode="startup", func=StratifiedLimbPayload, params=params)
    return {**params, "limbs": list(LIMBS), "mass_bins_per_limb": 4, "mass_groups": 256,
            "mount_positions_body_m": [list(row[1]) for row in LIMBS.values()],
            "sizes_m": [list(row[2]) for row in LIMBS.values()],
            "mass_sampling": "equal Cartesian mass-bin allocation; independent uniform within each interval",
            "com_sampling": "independent xyz uniform in a cube centered on the payload mount, link-local frame",
            "com_max_norm_m": math.sqrt(3) * config.heavy_com_half_width_m,
            "inertia": "sum link and cuboid payload pseudo-inertias; includes both parallel-axis shifts",
            "lifetime": "sample once per world, fixed across resets; no shared physics prototypes"}
