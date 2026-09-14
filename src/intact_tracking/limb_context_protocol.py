"""Shared, load-only physics contract for the context/residual experiment."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from mjlab.managers.event_manager import EventTermCfg

from intact_tracking.preview_protocol import FULL_DATASET, LIMBS
from intact_tracking.tracker_finetune import FourLimbExperimentPayload

VERSION = "limb_uniform_context_residual_v1"
PAYLOAD_EVENT = "context_uniform_limb_payload"
TRACKER = (
    "/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/"
    "2026-09-02_04-48-42_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_"
    "8gpu_12288env_motion_data_correct/checkpoint_72000.pt"
)
TRACKER_SHA256 = "fd7bd90d5552e573bbbce1417e9b415c64bb487a76b683ba3c20503b5ec77635"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sample_limb_masses(num_envs: int, seed: int, fixed_masses=None) -> torch.Tensor:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if fixed_masses is not None:
        masses = torch.as_tensor(fixed_masses, dtype=torch.float32)
        if masses.shape != (4,) or not torch.isfinite(masses).all() or not ((masses >= 0) & (masses <= 4)).all():
            raise ValueError("Fixed evaluation masses must be four finite values in [0,4]")
        return masses.expand(num_envs, -1).clone()
    return 4 * torch.rand(num_envs, 4, generator=torch.Generator().manual_seed(seed))


class UniformLimbPayload(FourLimbExperimentPayload):
    """Reuse audited composite inertias; replace only the ABC mixture sampler."""

    def __init__(self, cfg, env):
        base_cfg = copy.deepcopy(cfg)
        base_cfg.params["condition"] = "A"
        super().__init__(base_cfg, env)
        self.mass = sample_limb_masses(
            env.num_envs, cfg.params["seed"], cfg.params.get("fixed_masses")
        ).to(env.device)

    def observe(self):
        return self.env.sim.model.body_mass[:, self.body_ids] - self.base["body_mass"]


def configure_load_only(env_cfg, seed: int, *, fixed_masses=None, observation_noise=False):
    from intact_tracking.cli.residual_policy_train import _configure_nominal_physics
    from intact_tracking.rollout.mjlab_adapter import _filter_disturbance_events

    removed_disturbances = _filter_disturbance_events(env_cfg)
    nominal = _configure_nominal_physics(env_cfg)
    for group in env_cfg.observations.values():
        group.enable_corruption = bool(observation_noise)
    env_cfg.events[PAYLOAD_EVENT] = EventTermCfg(
        mode="startup", func=UniformLimbPayload,
        params={"seed": int(seed) + 91283, "fixed_masses": fixed_masses},
    )
    command = env_cfg.commands["motion"]
    command.motion_manifest_file = ""
    command.excluded_motion_files = ()
    command.motion_exclude_files = ()
    command.motion_exclude_file = ""
    command.adaptive_bin_snapshot_interval_iterations = 0
    command.sampling_mode = "uniform"
    command.rewind.enabled = False
    return {
        "profile": "hands-shins-independent-uniform-0-4kg",
        "nominal_configuration": nominal, "removed_disturbances": removed_disturbances,
        "observation_noise": bool(observation_noise),
        "sampling": "independent U(0,4) per limb and world, fixed at startup",
        "fixed_masses": list(fixed_masses) if fixed_masses is not None else None,
        "limbs": {name: {"body": row[0], "position": list(row[1]), "size": list(row[2])}
                  for name, row in LIMBS.items()},
        "total_added_mass_range_kg": [0, 16],
    }


def audit_load_only(env):
    from intact_tracking.cli.residual_policy_train import _is_persistent_domain_randomization

    for names in env.event_manager.active_terms.values():
        for name in names:
            term = env.event_manager.get_term_cfg(name)
            if name != PAYLOAD_EVENT and _is_persistent_domain_randomization(term):
                raise RuntimeError(f"Unexpected remaining DR: {name}")
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    result = payload.audit()
    actual = payload.observe()
    result["sampled_mass_sha256"] = hashlib.sha256(payload.mass.cpu().numpy().tobytes()).hexdigest()
    result["per_limb_min_kg"] = actual.amin(0).tolist()
    result["per_limb_max_kg"] = actual.amax(0).tolist()
    if env.num_envs > 4 and (actual.std(0) > 1e-5).all():
        result["correlation"] = torch.corrcoef(actual.T).cpu().tolist()
    action = env.action_manager.get_term("joint_pos")
    if int(getattr(action, "max_delay", 0)) != 0:
        raise RuntimeError("Action delay DR survived")
    for name, desired in (("alpha", 1), ("joint_offset", 0)):
        value = getattr(action, name, None)
        if isinstance(value, torch.Tensor) and not torch.equal(value, torch.full_like(value, desired)):
            raise RuntimeError(f"Action {name} DR survived")
    # Verify that the nominal construction has no hidden random model changes.
    nominal_errors = {}
    for field in ("geom_friction", "dof_armature", "dof_damping", "dof_frictionloss",
                  "actuator_gainprm", "actuator_biasprm", "jnt_pos"):
        try:
            current = getattr(env.sim.model, field)
            default = env.sim.get_default_field(field).to(current.device)
        except (AttributeError, KeyError):
            continue
        if current.shape == default.shape:
            error = (current - default).abs().max() if current.numel() else current.new_zeros(())
        elif current.shape[1:] == default.shape:
            error = (current - default.unsqueeze(0)).abs().max()
        else:
            continue
        nominal_errors[field] = float(error)
        if float(error) > 1e-5:
            raise RuntimeError(f"Unexpected non-nominal {field}: {float(error)}")
    result["nominal_field_max_errors"] = nominal_errors
    return result


def local_process_environment():
    """Keep experiment caches and temporary files inside the authorized project."""
    root = PROJECT_ROOT / ".runtime" / "limb_context"
    paths = {
        "TMPDIR": root / "tmp", "MPLCONFIGDIR": root / "matplotlib",
        "XDG_CACHE_HOME": root / "cache", "WARP_CACHE_PATH": root / "warp",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor", "WANDB_CACHE_DIR": root / "wandb_cache",
        "WANDB_DIR": root / "wandb", "CUDA_CACHE_PATH": root / "cuda",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {**{name: str(path) for name, path in paths.items()},
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
