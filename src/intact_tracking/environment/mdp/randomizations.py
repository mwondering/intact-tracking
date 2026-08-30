from __future__ import annotations

import math
from typing import Any

import torch

try:
    import warp as wp
except Exception:  # pragma: no cover - warp is present in training envs.
    wp = None

from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel
from mjlab.utils.lab_api.string import resolve_matching_names_values

if False:
    from mjlab.envs import ManagerBasedRlEnv


def _as_env_ids(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _uniform(shape: tuple[int, ...], low: float, high: float, device: str) -> torch.Tensor:
    return torch.rand(shape, device=device) * (float(high) - float(low)) + float(low)


def _uniform_range(low: torch.Tensor, high: torch.Tensor, n_envs: int) -> torch.Tensor:
    low = low.unsqueeze(0).expand(n_envs, -1)
    high = high.unsqueeze(0).expand(n_envs, -1)
    return torch.rand_like(low) * (high - low) + low


def _log_uniform_range(low: torch.Tensor, high: torch.Tensor, n_envs: int) -> torch.Tensor:
    return torch.exp(_uniform_range(torch.log(low), torch.log(high), n_envs))


def _rand_unit_vectors(shape: tuple[int, ...], device: str) -> torch.Tensor:
    vec = torch.randn(shape, device=device)
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _add_spherical_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Match the reference task's bounded isotropic 3-D noise distribution."""
    if noise_std <= 0.0:
        return x
    if x.shape[-1] != 3:
        raise ValueError(f"_add_spherical_noise expects last dim 3, got shape {tuple(x.shape)}")
    direction = torch.randn_like(x)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    radius = torch.rand((*x.shape[:-1], 1), device=x.device, dtype=x.dtype) * float(noise_std)
    return x + direction * radius


def _expand_model_fields(env: "ManagerBasedRlEnv", *fields: str) -> None:
    missing = tuple(field for field in fields if field not in env.sim.expanded_fields)
    if missing:
        env.sim.expand_model_fields(missing)


class perturb_body_com:
    model_fields = (
        "body_ipos",
        "body_subtreemass",
        "dof_invweight0",
        "body_invweight0",
        "tendon_length0",
        "tendon_invweight0",
    )
    recompute = RecomputeLevel.set_const

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        _expand_model_fields(env, *self.model_fields)
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        body_names = params.get("body_names", ".*")
        self.body_ids, _ = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError("perturb_body_com did not match any body.")
        self.global_body_ids = self.asset.indexing.body_ids[
            torch.as_tensor(self.body_ids, device=env.device, dtype=torch.long)
        ]
        self.com_range = tuple(float(v) for v in params.get("com_range", (-0.05, 0.05)))
        self.default_body_ipos = env.sim.model.body_ipos[:, self.global_body_ids].clone()

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        ids = _as_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        low, high = self.com_range
        offsets = _uniform((ids.numel(), self.global_body_ids.numel(), 3), low, high, env.device)
        env.sim.model.body_ipos[ids.unsqueeze(1), self.global_body_ids] = (
            self.default_body_ipos[ids] + offsets
        )


class perturb_body_materials:
    model_fields = ("geom_friction", "geom_solref")
    recompute = RecomputeLevel.none

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        _expand_model_fields(env, *self.model_fields)
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        body_names = params.get("body_names", ".*")
        body_ids, _ = self.asset.find_bodies(body_names)
        if len(body_ids) == 0:
            raise ValueError("perturb_body_materials did not match any body.")
        global_body_ids = self.asset.indexing.body_ids[
            torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
        ]
        selected_body_set = set(global_body_ids.cpu().tolist())
        geom_local_ids: list[int] = []
        geom_global_ids: list[int] = []
        geom_names = self.asset.geom_names
        for local_idx, global_idx in enumerate(self.asset.indexing.geom_ids.cpu().tolist()):
            body_id = int(env.sim.mj_model.geom_bodyid[global_idx])
            if body_id in selected_body_set:
                geom_local_ids.append(local_idx)
                geom_global_ids.append(global_idx)
        if not geom_global_ids:
            raise ValueError("perturb_body_materials did not match any geom.")
        self.geom_names = [geom_names[idx] for idx in geom_local_ids]
        self.geom_global_ids = torch.as_tensor(geom_global_ids, device=env.device, dtype=torch.long)
        self.static_friction_range = tuple(
            float(v) for v in params.get("static_friction_range", (0.6, 1.0))
        )
        self.solref_time_constant_range = tuple(
            float(v) for v in params.get("solref_time_constant_range", (0.02, 0.02))
        )
        self.solref_dampratio_range = tuple(
            float(v) for v in params.get("solref_dampratio_range", (1.0, 1.0))
        )
        self.homogeneous = bool(params.get("homogeneous", False))
        self._obs_static_friction = torch.ones((env.num_envs, 1), device=env.device)
        self._obs_solref_time_constant = torch.ones((env.num_envs, 1), device=env.device)
        self._obs_solref_dampratio = torch.ones((env.num_envs, 1), device=env.device)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        ids = _as_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        num_geoms = self.geom_global_ids.numel()
        sample_cols = 1 if self.homogeneous else num_geoms
        shape = (ids.numel(), sample_cols)
        sf = _uniform(shape, *self.static_friction_range, env.device)
        tc = _uniform(shape, *self.solref_time_constant_range, env.device)
        dr_low, dr_high = self.solref_dampratio_range
        dr = torch.exp(_uniform(shape, math.log(dr_low), math.log(dr_high), env.device))
        if sample_cols == 1:
            sf = sf.expand(-1, num_geoms)
            tc = tc.expand(-1, num_geoms)
            dr = dr.expand(-1, num_geoms)
        env.sim.model.geom_friction[ids.unsqueeze(1), self.geom_global_ids, 0] = sf
        env.sim.model.geom_solref[ids.unsqueeze(1), self.geom_global_ids, 0] = tc
        env.sim.model.geom_solref[ids.unsqueeze(1), self.geom_global_ids, 1] = dr
        self._obs_static_friction[ids] = sf[:, :1]
        self._obs_solref_time_constant[ids] = tc[:, :1]
        self._obs_solref_dampratio[ids] = dr[:, :1]

    def observe(self, **_: Any) -> torch.Tensor:
        return torch.cat(
            (
                self._obs_static_friction,
                self._obs_solref_time_constant,
                self._obs_solref_dampratio,
            ),
            dim=-1,
        )


class motor_params_implicit:
    model_fields = (
        "actuator_gainprm",
        "actuator_biasprm",
        "dof_armature",
        "dof_frictionloss",
        "dof_invweight0",
        "body_invweight0",
        "tendon_length0",
        "tendon_invweight0",
    )
    recompute = RecomputeLevel.set_const_0

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        _expand_model_fields(env, *self.model_fields)
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        self.mode = str(params.get("mode", "log_uniform")).strip().lower()
        if self.mode not in {"uniform", "log_uniform"}:
            raise ValueError(f"Unsupported motor_params_implicit mode: {self.mode}")

        kp_ids, self.kp_names, kp_ranges = resolve_matching_names_values(
            dict(params.get("stiffness_range", {})), self.asset.actuator_names
        )
        kd_ids, self.kd_names, kd_ranges = resolve_matching_names_values(
            dict(params.get("damping_range", {})), self.asset.actuator_names
        )
        arm_ids, self.arm_names, arm_ranges = resolve_matching_names_values(
            dict(params.get("armature_range", {})), self.asset.joint_names
        )
        fric_ids, self.fric_names, fric_ranges = resolve_matching_names_values(
            dict(params.get("frictionloss_range", {})), self.asset.joint_names
        )

        self.kp_ctrl_ids = self.asset.indexing.ctrl_ids[
            torch.as_tensor(kp_ids, device=env.device, dtype=torch.long)
        ]
        self.kd_ctrl_ids = self.asset.indexing.ctrl_ids[
            torch.as_tensor(kd_ids, device=env.device, dtype=torch.long)
        ]
        self.arm_dof_ids = self.asset.indexing.joint_v_adr[
            torch.as_tensor(arm_ids, device=env.device, dtype=torch.long)
        ]
        self.fric_dof_ids = self.asset.indexing.joint_v_adr[
            torch.as_tensor(fric_ids, device=env.device, dtype=torch.long)
        ]

        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        default_armature = env.sim.get_default_field("dof_armature")
        self.kp_gain_def = default_gainprm[self.kp_ctrl_ids, 0]
        self.kp_bias_def = default_biasprm[self.kp_ctrl_ids, 1]
        self.kd_bias_def = default_biasprm[self.kd_ctrl_ids, 2]
        self.arm_def = default_armature[self.arm_dof_ids]

        self.kp_low, self.kp_high = self._ranges(kp_ranges, env.device)
        self.kd_low, self.kd_high = self._ranges(kd_ranges, env.device)
        self.arm_low, self.arm_high = self._ranges(arm_ranges, env.device)
        self.fric_low, self.fric_high = self._ranges(fric_ranges, env.device)

        self._obs_kp_scale = torch.ones((env.num_envs, len(kp_ids)), device=env.device)
        self._obs_kd_scale = torch.ones((env.num_envs, len(kd_ids)), device=env.device)
        self._obs_arm_scale = torch.ones((env.num_envs, len(arm_ids)), device=env.device)
        self._obs_frictionloss = torch.zeros((env.num_envs, len(fric_ids)), device=env.device)

    @staticmethod
    def _ranges(ranges: list[Any], device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if len(ranges) == 0:
            empty = torch.empty(0, device=device)
            return empty, empty
        values = torch.as_tensor(ranges, dtype=torch.float32, device=device)
        low, high = values.unbind(1)
        if torch.any(high < low):
            raise ValueError("motor_params_implicit ranges must satisfy low <= high.")
        return low, high

    def _sample_scale(self, n_envs: int, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        if low.numel() == 0:
            return torch.empty((n_envs, 0), device=self.env.device)
        if self.mode == "uniform":
            return _uniform_range(low, high, n_envs)
        if torch.any(low <= 0.0) or torch.any(high <= 0.0):
            raise ValueError("log_uniform motor params require positive ranges.")
        return _log_uniform_range(low, high, n_envs)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        ids = _as_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        if self.arm_dof_ids.numel() > 0:
            arm_scale = self._sample_scale(ids.numel(), self.arm_low, self.arm_high)
            env.sim.model.dof_armature[ids.unsqueeze(1), self.arm_dof_ids] = (
                self.arm_def.unsqueeze(0) * arm_scale
            )
            self._obs_arm_scale[ids] = arm_scale
        if self.fric_dof_ids.numel() > 0:
            frictionloss = _uniform_range(self.fric_low, self.fric_high, ids.numel())
            env.sim.model.dof_frictionloss[ids.unsqueeze(1), self.fric_dof_ids] = frictionloss
            self._obs_frictionloss[ids] = frictionloss
        self.reset(env_ids=ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = _as_env_ids(self.env, env_ids)
        if ids.numel() == 0:
            return
        if self.kp_ctrl_ids.numel() > 0:
            kp_samples = self._sample_scale(ids.numel(), self.kp_low, self.kp_high)
            self.env.sim.model.actuator_gainprm[ids.unsqueeze(1), self.kp_ctrl_ids, 0] = (
                self.kp_gain_def.unsqueeze(0) * kp_samples
            )
            self.env.sim.model.actuator_biasprm[ids.unsqueeze(1), self.kp_ctrl_ids, 1] = (
                self.kp_bias_def.unsqueeze(0) * kp_samples
            )
            self._obs_kp_scale[ids] = kp_samples
        if self.kd_ctrl_ids.numel() > 0:
            kd_samples = self._sample_scale(ids.numel(), self.kd_low, self.kd_high)
            self.env.sim.model.actuator_biasprm[ids.unsqueeze(1), self.kd_ctrl_ids, 2] = (
                self.kd_bias_def.unsqueeze(0) * kd_samples
            )
            self._obs_kd_scale[ids] = kd_samples

    def observe(self, **_: Any) -> torch.Tensor:
        return torch.cat(
            (
                self._obs_kp_scale,
                self._obs_kd_scale,
                self._obs_arm_scale,
                self._obs_frictionloss,
            ),
            dim=-1,
        )


class random_joint_offset:
    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        ranges = dict(cfg.params.get("ranges", cfg.params))
        ranges.pop("enabled", None)
        self.joint_ids, self.joint_names, values = resolve_matching_names_values(
            ranges, self.asset.joint_names
        )
        self.joint_ids_t = torch.as_tensor(self.joint_ids, device=env.device, dtype=torch.long)
        self.offset_range = torch.as_tensor(values, device=env.device, dtype=torch.float32)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        del env, env_ids

    def _action_term(self):
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is None:
            return None
        return action_manager.get_term("joint_pos")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        term = self._action_term()
        if term is None:
            return
        ids = _as_env_ids(self.env, env_ids)
        if ids.numel() == 0:
            return
        low = self.offset_range[:, 0].unsqueeze(0).expand(ids.numel(), -1)
        high = self.offset_range[:, 1].unsqueeze(0).expand(ids.numel(), -1)
        sampled = torch.rand_like(low) * (high - low) + low
        full_offset = torch.zeros((ids.numel(), term.action_dim), device=self.env.device)
        target_ids = [int(v) for v in term.target_ids.detach().cpu().tolist()]
        for src_col, joint_id in enumerate(self.joint_ids):
            if int(joint_id) in target_ids:
                full_offset[:, target_ids.index(int(joint_id))] = sampled[:, src_col]
        if hasattr(term, "set_joint_offset"):
            term.set_joint_offset(ids, full_offset)
        elif hasattr(term, "joint_offset"):
            term.joint_offset[ids] = full_offset

    def observe(self, **_: Any) -> torch.Tensor:
        term = self._action_term()
        if term is None or not hasattr(term, "joint_offset"):
            return torch.zeros((self.env.num_envs, len(self.joint_ids)), device=self.env.device)
        offset = term.joint_offset
        target_ids = [int(v) for v in term.target_ids.detach().cpu().tolist()]
        cols = [
            target_ids.index(int(joint_id))
            for joint_id in self.joint_ids
            if int(joint_id) in target_ids
        ]
        if not cols:
            return torch.zeros((self.env.num_envs, len(self.joint_ids)), device=self.env.device)
        return offset[:, torch.as_tensor(cols, device=self.env.device, dtype=torch.long)]


class perturb_root_vel:
    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        self.min_s = float(params.get("min_s", 3.0))
        self.max_s = float(params.get("max_s", 6.0))
        if self.max_s < self.min_s:
            raise ValueError("perturb_root_vel max_s must be >= min_s.")
        ranges = [params.get(name, (0.0, 0.0)) for name in ("x", "y", "z", "roll", "pitch", "yaw")]
        self.low = torch.as_tensor([float(v[0]) for v in ranges], device=env.device)
        self.high = torch.as_tensor([float(v[1]) for v in ranges], device=env.device)
        self.time_left_s = torch.zeros(env.num_envs, device=env.device)

    def _sample_interval(self, n: int) -> torch.Tensor:
        return _uniform((n,), self.min_s, self.max_s, self.env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = _as_env_ids(self.env, env_ids)
        if ids.numel() > 0:
            self.time_left_s[ids] = self._sample_interval(ids.numel())

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        del env_ids
        self.time_left_s.sub_(env.step_dt)
        trigger_ids = torch.nonzero(self.time_left_s <= 1.0e-6, as_tuple=False).squeeze(-1)
        if trigger_ids.numel() == 0:
            return
        delta = _uniform_range(self.low, self.high, trigger_ids.numel())
        lin_vel = self.asset.data.root_link_lin_vel_w[trigger_ids] + delta[:, :3]
        ang_vel = self.asset.data.root_link_ang_vel_w[trigger_ids] + delta[:, 3:]
        self.asset.write_root_link_velocity_to_sim(
            torch.cat((lin_vel, ang_vel), dim=-1), env_ids=trigger_ids
        )
        self.time_left_s[trigger_ids] = self._sample_interval(trigger_ids.numel())


class perturb_body_wrench:
    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        self.enabled = bool(params.get("enabled", True))
        body_name = str(params.get("body_name", "torso_link"))
        body_ids, body_names = self.asset.find_bodies(body_name)
        if self.enabled and len(body_ids) != 1:
            raise ValueError(
                f"perturb_body_wrench.body_name must match one body, got {body_name!r}: {body_names}"
            )
        self.body_ids = torch.as_tensor(body_ids[:1], device=env.device, dtype=torch.long)
        self.total_duration_range_s = tuple(
            float(v) for v in params.get("total_duration_range_s", (3.0, 6.0))
        )
        self.active_duration_range_s = tuple(
            float(v) for v in params.get("active_duration_range_s", (1.0, 4.0))
        )
        self.force_magnitude_range_n = tuple(
            float(v) for v in params.get("force_magnitude_range_n", (0.0, 10.0))
        )
        self.lever_arm_length_range_m = tuple(
            float(v) for v in params.get("lever_arm_length_range_m", (0.0, 0.4))
        )
        self.total_time_left_s = torch.zeros(env.num_envs, device=env.device)
        self.active_time_left_s = torch.zeros(env.num_envs, device=env.device)
        self.current_force_w = torch.zeros((env.num_envs, 3), device=env.device)
        self.current_torque_w = torch.zeros((env.num_envs, 3), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = _as_env_ids(self.env, env_ids)
        self.total_time_left_s[ids] = 0.0
        self.active_time_left_s[ids] = 0.0
        self.current_force_w[ids] = 0.0
        self.current_torque_w[ids] = 0.0

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        del env_ids
        if not self.enabled:
            return
        self.total_time_left_s.sub_(env.step_dt).clamp_min_(0.0)
        self.active_time_left_s.sub_(env.step_dt).clamp_min_(0.0)
        resample_ids = torch.nonzero(self.total_time_left_s <= 1.0e-6, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            total_time = _uniform((resample_ids.numel(),), *self.total_duration_range_s, env.device)
            active_time = _uniform(
                (resample_ids.numel(),), *self.active_duration_range_s, env.device
            )
            self.total_time_left_s[resample_ids] = total_time
            self.active_time_left_s[resample_ids] = torch.minimum(active_time, total_time)
            force_mag = _uniform((resample_ids.numel(),), *self.force_magnitude_range_n, env.device)
            lever_mag = _uniform(
                (resample_ids.numel(),), *self.lever_arm_length_range_m, env.device
            )
            force_w = _rand_unit_vectors(
                (resample_ids.numel(), 3), env.device
            ) * force_mag.unsqueeze(-1)
            lever_w = _rand_unit_vectors(
                (resample_ids.numel(), 3), env.device
            ) * lever_mag.unsqueeze(-1)
            self.current_force_w[resample_ids] = force_w
            self.current_torque_w[resample_ids] = torch.cross(lever_w, force_w, dim=-1)

        active = self.active_time_left_s > 0.0
        force_w = torch.zeros_like(self.current_force_w)
        torque_w = torch.zeros_like(self.current_torque_w)
        force_w[active] = self.current_force_w[active]
        torque_w[active] = self.current_torque_w[active]
        self.asset.write_external_wrench_to_sim(
            forces=force_w.unsqueeze(1),
            torques=torque_w.unsqueeze(1),
            body_ids=self.body_ids,
        )


class body_force_pulse:
    """Apply independently timed short force pulses and expose the active wrench."""

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        params = cfg.params
        body_name = str(params.get("body_name", "torso_link"))
        body_ids, body_names = self.asset.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"body_force_pulse.body_name must match one body, got {body_name!r}: {body_names}"
            )
        self.body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
        self.interval_range_s = self._validate_range(
            "interval_range_s",
            params.get("interval_range_s", (3.0, 6.0)),
            positive=True,
        )
        self.duration_range_s = self._validate_range(
            "duration_range_s",
            params.get("duration_range_s", (0.3, 0.5)),
            positive=True,
        )
        self.force_abs_range_n = self._validate_range(
            "force_abs_range_n",
            params.get("force_abs_range_n", (0.0, 10.0)),
            nonnegative=True,
        )
        if self.duration_range_s[1] >= self.interval_range_s[0]:
            raise ValueError(
                "body_force_pulse requires max duration to be shorter than the "
                "minimum pulse interval."
            )

        self.time_to_next_pulse_s = self._sample(env.num_envs, self.interval_range_s)
        self.active_time_left_s = torch.zeros(env.num_envs, device=env.device)
        self.active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.current_force_w = torch.zeros((env.num_envs, 3), device=env.device)

    @staticmethod
    def _validate_range(
        name: str,
        values,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"body_force_pulse.{name} must contain two values.")
        low, high = (float(values[0]), float(values[1]))
        if high < low:
            raise ValueError(f"body_force_pulse.{name} must satisfy low <= high.")
        if positive and low <= 0.0:
            raise ValueError(f"body_force_pulse.{name} must be positive.")
        if nonnegative and low < 0.0:
            raise ValueError(f"body_force_pulse.{name} must be nonnegative.")
        return low, high

    def _sample(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        return _uniform((count,), *value_range, self.env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = _as_env_ids(self.env, env_ids)
        if ids.numel() == 0:
            return
        active_ids = ids[self.active[ids]]
        self._clear_wrench(active_ids)
        self.time_to_next_pulse_s[ids] = self._sample(ids.numel(), self.interval_range_s)
        self.active_time_left_s[ids] = 0.0
        self.active[ids] = False
        self.current_force_w[ids] = 0.0

    def _clear_wrench(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        zeros = torch.zeros((env_ids.numel(), 1, 3), device=self.env.device)
        self.asset.write_external_wrench_to_sim(
            forces=zeros,
            torques=zeros,
            env_ids=env_ids,
            body_ids=self.body_ids,
        )

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        del env_ids
        step_dt = float(env.step_dt)
        self.active_time_left_s[self.active] -= step_dt
        expired = self.active & (self.active_time_left_s <= 1.0e-6)
        if torch.any(expired):
            expired_ids = torch.nonzero(expired, as_tuple=False).squeeze(-1)
            self._clear_wrench(expired_ids)
            self.active[expired_ids] = False
            self.active_time_left_s[expired_ids] = 0.0
            self.current_force_w[expired_ids] = 0.0

        self.time_to_next_pulse_s.sub_(step_dt)
        trigger_ids = torch.nonzero(
            (~self.active) & (self.time_to_next_pulse_s <= 1.0e-6),
            as_tuple=False,
        ).squeeze(-1)
        if trigger_ids.numel() > 0:
            count = trigger_ids.numel()
            self.time_to_next_pulse_s[trigger_ids] = self._sample(count, self.interval_range_s)
            self.active_time_left_s[trigger_ids] = self._sample(count, self.duration_range_s)
            magnitude = _uniform((count, 3), *self.force_abs_range_n, env.device)
            sign = torch.where(
                torch.rand((count, 3), device=env.device) < 0.5,
                -torch.ones_like(magnitude),
                torch.ones_like(magnitude),
            )
            self.current_force_w[trigger_ids] = magnitude * sign
            self.active[trigger_ids] = True
            force_w = self.current_force_w[trigger_ids].unsqueeze(1)
            self.asset.write_external_wrench_to_sim(
                forces=force_w,
                torques=torch.zeros_like(force_w),
                env_ids=trigger_ids,
                body_ids=self.body_ids,
            )

    def observe(self, **_: Any) -> torch.Tensor:
        """Return the exact currently applied world-frame force."""
        return self.current_force_w


class perturb_gravity:
    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        params = cfg.params
        self.mean = torch.as_tensor(params.get("mean", (0.0, 0.0, -9.81)), device=env.device)
        self.std = float(params.get("std", 0.0))
        self._gravity = self.mean.unsqueeze(0).expand(env.num_envs, -1).clone()

    def _ensure_per_env_gravity_storage(self) -> None:
        gravity = self.env.sim.model.opt.gravity
        if gravity.shape[0] == self.env.num_envs and gravity.stride(0) != 0:
            return
        init_gravity = self.mean.unsqueeze(0).expand(self.env.num_envs, -1).contiguous()
        if wp is None:
            return
        with wp.ScopedDevice(self.env.sim.wp_device):
            self.env.sim.wp_model.opt.gravity = wp.from_torch(init_gravity, dtype=wp.vec3)
        self.env.sim.model.clear_cache()
        self.env.sim.create_graph()

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        ids = _as_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        self._ensure_per_env_gravity_storage()
        gravity = self.mean.unsqueeze(0).expand(ids.numel(), -1).clone()
        if self.std > 0.0:
            gravity = _add_spherical_noise(gravity, self.std)
        env.sim.model.opt.gravity[ids] = gravity
        self._gravity[ids] = gravity

    def observe(self, **_: Any) -> torch.Tensor:
        return self._gravity


class recorded_push_by_setting_velocity:
    """Instantaneous root-velocity kick with an observable per-step record."""

    _COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        velocity_range = dict(cfg.params.get("velocity_range", {}))
        ranges = [velocity_range.get(name, (0.0, 0.0)) for name in self._COMPONENTS]
        self.low = torch.as_tensor([float(value[0]) for value in ranges], device=env.device)
        self.high = torch.as_tensor([float(value[1]) for value in ranges], device=env.device)
        self.last_delta = torch.zeros((env.num_envs, 6), device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, **_: Any) -> None:
        ids = _as_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        delta = _uniform_range(self.low, self.high, ids.numel())
        velocity = self.asset.data.root_link_vel_w[ids] + delta
        self.asset.write_root_link_velocity_to_sim(velocity, env_ids=ids)
        self.last_delta[ids] = delta
        self.last_step[ids] = int(env.common_step_counter)

    def observe(self, **_: Any) -> torch.Tensor:
        active = self.last_step == int(self.env.common_step_counter)
        delta = torch.where(active.unsqueeze(-1), self.last_delta, 0.0)
        return torch.cat((delta, active.to(delta.dtype).unsqueeze(-1)), dim=-1)


def sp_tracking_progress(env: "ManagerBasedRlEnv", env_ids, **_: Any) -> dict[str, float]:
    del env_ids
    return dict(getattr(env, "_sp_tracking_curriculum_state", {"progress": 0.0}))
