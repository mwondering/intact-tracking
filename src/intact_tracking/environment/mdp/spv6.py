"""SPV6 normalized physical context and causal external-force targets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

from . import sp as sp_mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


SPV6_GLOBAL_CONTEXT_DIM = 5
SPV6_ARMATURE_DIM = 29
SPV6_STATIC_CONTEXT_DIM = SPV6_GLOBAL_CONTEXT_DIM + SPV6_ARMATURE_DIM
SPV6_FORCE_FRAME_DIM = 3
SPV6_FORCE_HISTORY_LENGTH = 50
SPV6_FORCE_HISTORY_DIM = SPV6_FORCE_FRAME_DIM * SPV6_FORCE_HISTORY_LENGTH


class normalized_physical_context:
    """Return normalized torso CoM/mass deltas and shared foot friction.

    The fixed transform is part of the task contract:
    ``[delta_com / com_scale, delta_mass / mass_scale, log_friction]``.
    Every output component is clipped to ``[-1, 1]``.  Raw physical units are
    deliberately excluded from actor, critic, and auxiliary-loss inputs.
    """

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        body_name = str(params.get("body_name", "torso_link"))
        geom_names = str(params.get("geom_names", r"^(left|right)_foot.*collision$"))
        body_ids, matched_bodies = self.asset.find_bodies(body_name)
        geom_ids, matched_geoms = self.asset.find_geoms(geom_names)
        if len(body_ids) != 1:
            raise ValueError(
                "SPV6 normalized physical context requires one body for "
                f"{body_name!r}, got {matched_bodies}"
            )
        if not geom_ids:
            raise ValueError(
                "SPV6 normalized physical context matched no geoms for "
                f"{geom_names!r}: {matched_geoms}"
            )
        self.body_id = int(self.asset.indexing.body_ids[body_ids[0]].item())
        self.geom_ids = self.asset.indexing.geom_ids[
            torch.as_tensor(geom_ids, device=env.device, dtype=torch.long)
        ]
        self.com_scale_m = float(params.get("com_scale_m", 0.075))
        self.mass_scale_kg = float(params.get("mass_scale_kg", 1.0))
        friction_range = tuple(float(value) for value in params.get("friction_range", (0.3, 2.0)))
        if self.com_scale_m <= 0.0 or self.mass_scale_kg <= 0.0:
            raise ValueError("SPV6 physical normalization scales must be positive")
        if (
            len(friction_range) != 2
            or friction_range[0] <= 0.0
            or friction_range[1] <= friction_range[0]
        ):
            raise ValueError("SPV6 friction_range must contain two increasing positive values")
        self.log_friction_low = math.log(friction_range[0])
        self.log_friction_span = math.log(friction_range[1]) - self.log_friction_low

        default_com = env.sim.get_default_field("body_ipos")[self.body_id]
        default_mass = env.sim.get_default_field("body_mass")[self.body_id]
        self.default_com = default_com.to(device=env.device).clone()
        self.default_mass = default_mass.to(device=env.device).clone()

    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        model = env.sim.model
        actual_com = model.body_ipos[:, self.body_id]
        actual_mass = model.body_mass[:, self.body_id : self.body_id + 1]
        friction = model.geom_friction[:, self.geom_ids, 0].mean(dim=1, keepdim=True)
        com = (actual_com - self.default_com) / self.com_scale_m
        mass = (actual_mass - self.default_mass.reshape(1, 1)) / self.mass_scale_kg
        log_friction = (
            2.0
            * (friction.clamp_min(1.0e-8).log() - self.log_friction_low)
            / self.log_friction_span
            - 1.0
        )
        result = torch.cat((com, mass, log_friction), dim=-1).clamp(-1.0, 1.0)
        if result.shape != (env.num_envs, SPV6_GLOBAL_CONTEXT_DIM):
            raise RuntimeError(
                f"SPV6 physical context has shape {tuple(result.shape)}, "
                f"expected {(env.num_envs, SPV6_GLOBAL_CONTEXT_DIM)}"
            )
        return result


class normalized_armature_context:
    """Return per-joint armature scales mapped from a fixed range to [-1, 1]."""

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        params = cfg.params
        self.event_name = str(params.get("event_name", "motor_params_implicit"))
        scale_range = tuple(float(value) for value in params.get("scale_range", (0.8, 1.2)))
        if len(scale_range) != 2 or scale_range[1] <= scale_range[0]:
            raise ValueError("SPV6 armature scale_range must be increasing")
        self.low, self.high = scale_range

    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        event_manager = getattr(env, "event_manager", None)
        if event_manager is None:
            value = torch.ones((env.num_envs, SPV6_ARMATURE_DIM), device=env.device)
        else:
            try:
                event = event_manager.get_term_cfg(self.event_name).func
            except (KeyError, ValueError) as error:
                raise RuntimeError(
                    f"SPV6 requires event {self.event_name!r} for armature targets"
                ) from error
            observe = getattr(event, "observe", None)
            if not callable(observe):
                raise RuntimeError(
                    f"SPV6 event {self.event_name!r} does not expose armature values"
                )
            value = observe()
        if value.shape != (env.num_envs, SPV6_ARMATURE_DIM):
            raise RuntimeError(
                f"SPV6 armature context has shape {tuple(value.shape)}, "
                f"expected {(env.num_envs, SPV6_ARMATURE_DIM)}"
            )
        return (2.0 * (value - self.low) / (self.high - self.low) - 1.0).clamp(-1.0, 1.0)


class normalized_past_force:
    """Return the previous-step torso-frame force normalized by a fixed scale."""

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        self.event_name = str(params.get("event_name", "push_robot"))
        body_name = str(params.get("body_name", "torso_link"))
        body_ids, body_names = self.asset.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"SPV6 normalized past force requires one body for {body_name!r}, got {body_names}"
            )
        self.body_id = int(body_ids[0])
        self.force_scale_n = float(params.get("force_scale_n", 10.0))
        if self.force_scale_n <= 0.0:
            raise ValueError("SPV6 force_scale_n must be positive")
        self.pending_force_b = torch.zeros((env.num_envs, SPV6_FORCE_FRAME_DIM), device=env.device)
        self.output_force_b = torch.zeros_like(self.pending_force_b)
        self.last_step = -1

    def _current_force_w(self) -> torch.Tensor:
        event_manager = getattr(self.env, "event_manager", None)
        if event_manager is None:
            return torch.zeros_like(self.pending_force_b)
        try:
            event = event_manager.get_term_cfg(self.event_name).func
        except (KeyError, ValueError):
            return torch.zeros_like(self.pending_force_b)
        observe = getattr(event, "observe", None)
        if not callable(observe):
            return torch.zeros_like(self.pending_force_b)
        value = observe()
        if value.shape != (self.env.num_envs, SPV6_FORCE_FRAME_DIM):
            raise RuntimeError(
                f"SPV6 force event observation has shape {tuple(value.shape)}, "
                f"expected {(self.env.num_envs, SPV6_FORCE_FRAME_DIM)}"
            )
        return value

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.pending_force_b[ids] = 0.0
        self.output_force_b[ids] = 0.0

    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        step = int(getattr(env, "common_step_counter", self.last_step + 1))
        if step != self.last_step:
            current_force_w = self._current_force_w()
            body_quat_w = self.asset.data.body_link_quat_w[:, self.body_id]
            current_force_b = sp_mdp._quat_apply_inverse(body_quat_w, current_force_w)
            self.output_force_b.copy_(self.pending_force_b)
            self.pending_force_b.copy_((current_force_b / self.force_scale_n).clamp(-1.0, 1.0))
            self.last_step = step
        return self.output_force_b
