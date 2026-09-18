"""256 load anchors and continuous interpolation, with nominal background physics."""

import torch

from intact_tracking.limb_context_dr import LOAD_ONLY, configure_limb_dr
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT, UniformLimbPayload
from intact_tracking.residual_uniform_protocol import (
    MAX_MASSES, wrist_articulation, physics_contract, audit_physics as audit_base,
)

VERSION = "payload_grid256_half_continuous_nominal_background_v1"


def load_grid():
    levels = torch.linspace(0, 1, 4)
    return torch.cartesian_prod(levels, levels, levels, levels) * torch.tensor(MAX_MASSES)


def load_bank(num_envs, seed, anchor_fraction=0.5):
    anchors = round(num_envs * anchor_fraction)
    if not 0 <= anchor_fraction <= 1 or anchors % 256:
        raise ValueError("The anchor world count must be a multiple of 256")
    mass = torch.rand(num_envs, 4, generator=torch.Generator().manual_seed(seed)) * torch.tensor(MAX_MASSES)
    ids = torch.full((num_envs,), -1, dtype=torch.long)
    ids[:anchors] = torch.arange(anchors) % 256
    mass[:anchors] = load_grid()[ids[:anchors]]
    return mass, ids


class PrototypePayload(UniformLimbPayload):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        masses, ids = load_bank(env.num_envs, cfg.params["seed"], cfg.params["anchor_fraction"])
        self.mass, self.anchor_ids = masses.to(env.device), ids.to(env.device)

    def __call__(self, env, env_ids, **kwargs):
        # anchor_fraction configures construction, not the physical update.
        kwargs.pop("anchor_fraction", None)
        return super().__call__(env, env_ids, **kwargs)


def configure_physics(env_cfg, seed, *, profile=LOAD_ONLY, anchor_fraction=0.5, fixed_masses=None):
    if profile != LOAD_ONLY:
        raise ValueError("This experiment changes only the four limb loads")
    robot = env_cfg.scene.entities["robot"]
    robot.articulation = wrist_articulation(robot.articulation)
    metadata = configure_limb_dr(env_cfg, seed, profile=LOAD_ONLY, max_masses_kg=MAX_MASSES,
                                fixed_masses=fixed_masses)
    payload = env_cfg.events[PAYLOAD_EVENT]
    if fixed_masses is None:
        payload.func = PrototypePayload
        payload.params["anchor_fraction"] = float(anchor_fraction)
    else:
        anchor_fraction = None
    sampling = ("Independent continuous uniform hand [0,2.5] and shin [0,4] kg in every world; fixed per world"
                if anchor_fraction == 0 else
                "Balanced 256 load anchors plus independent continuous uniform loads; fixed per world")
    if fixed_masses is not None:
        sampling = "Specified fixed limb loads in every world"
    metadata.update(profile=VERSION, residual_physics_contract=physics_contract(),
                    anchor_fraction=anchor_fraction, load_grid_kg=load_grid().tolist(),
                    sampling=sampling,
                    background="nominal; no pushes or observation corruption")
    return metadata


def audit_physics(env, metadata):
    result = audit_base(env, metadata)
    if metadata["anchor_fraction"] is None:
        return result
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    expected, ids = load_bank(env.num_envs, env.event_manager.get_term_cfg(PAYLOAD_EVENT).params["seed"],
                              metadata["anchor_fraction"])
    torch.testing.assert_close(payload.observe(), expected.to(env.device), atol=1e-5, rtol=0)
    result.update(anchor_worlds=int((ids >= 0).sum()), continuous_worlds=int((ids < 0).sum()),
                  anchor_counts=torch.bincount(ids[ids >= 0], minlength=256).tolist())
    return result
