from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if False:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ObservationHistoryJointPositionActionCfg(JointPositionActionCfg):
    """BFM joint-position action with optional policy order and mean history.

    Action processing and application are inherited unchanged from MJLab's
    ``JointPositionAction``.  ``joint_name_order`` only permutes the policy-facing
    vector together with its target IDs, scales, and offsets; it does not change
    which physical joint receives each command.  The extra history reproduces
    HEFT's ``prev_actions`` without enabling SP delay/smoothing/curriculum.
    """

    observation_history_steps: int = 8
    joint_name_order: tuple[str, ...] | None = None

    def build(self, env: "ManagerBasedRlEnv") -> "ObservationHistoryJointPositionAction":
        return ObservationHistoryJointPositionAction(self, env)


class ObservationHistoryJointPositionAction(JointPositionAction):
    cfg: ObservationHistoryJointPositionActionCfg

    def __init__(
        self,
        cfg: ObservationHistoryJointPositionActionCfg,
        env: "ManagerBasedRlEnv",
    ):
        super().__init__(cfg=cfg, env=env)
        self._apply_policy_joint_name_order(cfg.joint_name_order)
        canonical_names = tuple(getattr(getattr(self._entity, "cfg", None), "joint_name_order", ()))
        target_names = tuple(self._target_names)
        if not canonical_names:
            canonical_names = target_names
        if set(canonical_names) != set(target_names) or len(canonical_names) != len(target_names):
            raise ValueError(
                "Observation-history canonical joint order must contain every action "
                f"target exactly once; targets={target_names}, order={canonical_names}"
            )
        self._observation_order = torch.as_tensor(
            [target_names.index(name) for name in canonical_names],
            dtype=torch.long,
            device=self.device,
        )
        self._observation_history_steps = max(int(cfg.observation_history_steps), 1)
        self._policy_mean_history = torch.zeros(
            (
                self.num_envs,
                self._observation_history_steps,
                self.action_dim,
            ),
            dtype=self._raw_actions.dtype,
            device=self.device,
        )

    def _apply_policy_joint_name_order(self, ordered_names: tuple[str, ...] | None) -> None:
        if ordered_names is None:
            return
        ordered_names = tuple(ordered_names)
        target_names = tuple(self._target_names)
        if set(ordered_names) != set(target_names) or len(ordered_names) != len(target_names):
            raise ValueError(
                "joint_name_order must contain every BFM action target exactly once; "
                f"targets={target_names}, order={ordered_names}"
            )
        permutation = torch.as_tensor(
            [target_names.index(name) for name in ordered_names],
            dtype=torch.long,
            device=self.device,
        )
        self._target_ids = self._target_ids.index_select(0, permutation)
        self._target_names = list(ordered_names)
        for name in ("_raw_actions", "_processed_actions", "_offset", "_scale"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.index_select(-1, permutation))
        if hasattr(self, "_clip"):
            self._clip = self._clip.index_select(1, permutation)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._policy_mean_history[env_ids] = 0.0

    def record_policy_mean(self, mean_actions: torch.Tensor) -> None:
        if mean_actions.shape != self._raw_actions.shape:
            raise ValueError(
                "Policy mean action shape does not match the BFM action interface: "
                f"{tuple(mean_actions.shape)} != {tuple(self._raw_actions.shape)}"
            )
        canonical_mean = mean_actions.detach().index_select(-1, self._observation_order)
        self._policy_mean_history = torch.roll(self._policy_mean_history, shifts=1, dims=1)
        self._policy_mean_history[:, 0] = canonical_mean

    def get_recent_action_obs(self, steps: int) -> torch.Tensor:
        requested = max(int(steps), 0)
        available = min(requested, self._observation_history_steps)
        result = self._policy_mean_history[:, :available]
        if requested <= available:
            return result
        padding = torch.zeros(
            (self.num_envs, requested - available, self.action_dim),
            dtype=result.dtype,
            device=result.device,
        )
        return torch.cat((result, padding), dim=1)


@dataclass(kw_only=True)
class SpTrackingJointPositionActionCfg(JointPositionActionCfg):
    max_delay: int = 2
    delay_full_progress: float = 0.8
    alpha: tuple[float, float] = (0.8, 1.0)
    torque_limit_scale_range: tuple[float, float] = (1.0, 1.0)
    torque_limit_progress_range: tuple[float, float] = (0.0, 0.8)
    raw_action_clip: float | None = None
    boot_delay_steps: int = 0
    # ``full`` matches the reference student/finetune behavior: delay is
    # sampled uniformly from its full range and torque limits immediately use
    # their final scale.  ``progressive`` preserves the existing curriculum.
    curriculum_mode: Literal["progressive", "full"] = "progressive"
    # The reference task observes distribution means for previous actions while
    # keeping sampled actions for the action-rate penalty.  Defaults retain the
    # pre-existing behaviour for non-SP users of this reusable action term.
    prev_action_obs: Literal["sampled", "mean"] = "sampled"
    action_rate_source: Literal["sampled", "mean"] = "sampled"
    # Optional policy-facing order.  Simulator joint storage remains in MuJoCo
    # XML order; this only changes the action vector interface.
    joint_name_order: tuple[str, ...] | None = None

    def build(self, env: "ManagerBasedRlEnv") -> "SpTrackingJointPositionAction":
        return SpTrackingJointPositionAction(self, env)


class SpTrackingJointPositionAction(JointPositionAction):
    cfg: SpTrackingJointPositionActionCfg

    def __init__(self, cfg: SpTrackingJointPositionActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        self._apply_joint_name_order()
        if not isinstance(self._offset, torch.Tensor):
            self._offset = torch.full_like(self._processed_actions, float(self._offset))
        self._default_offset = self._offset.clone()
        self.joint_offset = torch.zeros_like(self._processed_actions)
        self.applied_action = torch.zeros_like(self._raw_actions)
        self.alpha = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)

        max_delay_steps = max(int(cfg.max_delay), 0)
        self.max_delay = max_delay_steps
        self._decimation = int(env.cfg.decimation)
        self._history_len = max(8, (max_delay_steps + self._decimation - 1) // self._decimation + 1)
        self._action_history = torch.zeros(
            (self.num_envs, self._history_len, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._action_mean_history = torch.zeros_like(self._action_history)
        self.delay = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
        self.delay_probs = torch.zeros(max_delay_steps + 1, dtype=torch.float32, device=self.device)
        self.boot_delay = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
        self.boot_target = self._default_offset.clone()
        self._substep = 0

        if "actuator_forcerange" not in env.sim.expanded_fields:
            env.sim.expand_model_fields(("actuator_forcerange",))
        self._ctrl_ids = self._entity.indexing.ctrl_ids[self._target_ids]
        self._default_forcerange = env.sim.get_default_field("actuator_forcerange")[
            self._ctrl_ids
        ].clone()
        self._torque_limit_scale: float | None = None
        self.step_schedule(0.0, None)

    def _apply_joint_name_order(self) -> None:
        """Permute action tensors into an optional canonical joint order."""
        ordered_names = self.cfg.joint_name_order
        if ordered_names is None:
            return
        ordered_names = tuple(ordered_names)
        target_names = tuple(self._target_names)
        if set(ordered_names) != set(target_names) or len(ordered_names) != len(target_names):
            raise ValueError(
                "joint_name_order must contain every actuated joint exactly once; "
                f"expected={target_names}, got={ordered_names}"
            )
        permutation = torch.as_tensor(
            [target_names.index(name) for name in ordered_names],
            dtype=torch.long,
            device=self.device,
        )
        self._target_ids = self._target_ids[permutation]
        self._target_names = list(ordered_names)
        for name in ("_raw_actions", "_processed_actions", "_offset", "_scale"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.index_select(-1, permutation))
        if hasattr(self, "_clip"):
            self._clip = self._clip.index_select(1, permutation)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._action_history[env_ids] = 0.0
        self._action_mean_history[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.boot_delay[env_ids] = 0
        self._substep = 0
        if isinstance(env_ids, slice):
            count = self.num_envs
        else:
            count = len(env_ids)
        sampled_delay = torch.multinomial(self.delay_probs, count, replacement=True)
        self.delay[env_ids] = sampled_delay.unsqueeze(-1).to(self.delay.dtype)
        alpha_low, alpha_high = self.cfg.alpha
        self.alpha[env_ids] = torch.empty((count, 1), device=self.device).uniform_(
            float(alpha_low), float(alpha_high)
        )

    def set_joint_offset(self, env_ids: torch.Tensor, offset: torch.Tensor) -> None:
        self.joint_offset[env_ids] = offset

    def set_boot_target(self, env_ids: torch.Tensor, target: torch.Tensor) -> None:
        """Hold the sampled reset pose for the configured physics substeps."""
        self.boot_target[env_ids] = target
        self.boot_delay[env_ids] = max(int(self.cfg.boot_delay_steps), 0)

    def get_recent_actions(self, steps: int) -> torch.Tensor:
        return self._recent_actions_from(self._action_history, steps)

    def _recent_actions_from(self, history: torch.Tensor, steps: int) -> torch.Tensor:
        if steps <= self._history_len:
            return history[:, :steps]
        pad = torch.zeros(
            (self.num_envs, steps - self._history_len, self.action_dim),
            dtype=history.dtype,
            device=self.device,
        )
        return torch.cat((history, pad), dim=1)

    def record_policy_mean(self, mean_actions: torch.Tensor) -> None:
        """Store the policy distribution mean for mean-based history consumers."""
        if self.cfg.prev_action_obs != "mean" and self.cfg.action_rate_source != "mean":
            return
        if mean_actions.shape != self._raw_actions.shape:
            raise ValueError(
                "Policy mean action shape does not match motion-tracking action shape: "
                f"{tuple(mean_actions.shape)} != {tuple(self._raw_actions.shape)}"
            )
        mean_actions = mean_actions.detach()
        if self.cfg.raw_action_clip is not None:
            mean_actions = mean_actions.clamp(
                min=-float(self.cfg.raw_action_clip), max=float(self.cfg.raw_action_clip)
            )
        self._action_mean_history = torch.roll(self._action_mean_history, shifts=1, dims=1)
        self._action_mean_history[:, 0] = mean_actions

    def get_recent_action_obs(self, steps: int) -> torch.Tensor:
        history = (
            self._action_mean_history
            if self.cfg.prev_action_obs == "mean"
            else self._action_history
        )
        return self._recent_actions_from(history, steps)

    def get_recent_action_rate_actions(self, steps: int) -> torch.Tensor:
        history = (
            self._action_mean_history
            if self.cfg.action_rate_source == "mean"
            else self._action_history
        )
        return self._recent_actions_from(history, steps)

    def step_schedule(self, progress: float, iters: int | None = None) -> dict[str, float]:
        del iters
        if self.cfg.curriculum_mode == "full":
            self.delay_probs.fill_(1.0 / float(self.max_delay + 1))
            scale = self._schedule_torque_limit(1.0)
        else:
            self._schedule_delay(progress)
            scale = self._schedule_torque_limit(progress)
        return {
            "progress": float(progress),
            "torque_limit_scale": float(scale),
            "max_delay_probability": float(self.delay_probs[-1].item()),
        }

    def _schedule_delay(self, progress: float) -> None:
        max_delay = int(self.max_delay)
        if max_delay <= 0:
            self.delay_probs.fill_(1.0)
            return
        full_progress = max(float(self.cfg.delay_full_progress), 1.0e-6)
        p = max(0.0, min(float(progress) / full_progress, 1.0))
        q = max(0.0, min(float(max_delay), p * float(max_delay + 1) - 1.0))
        k = int(q)
        alpha = q - float(k)
        kp1 = min(k + 1, max_delay)
        self.delay_probs.zero_()
        self.delay_probs[: k + 1] += (1.0 - alpha) / float(k + 1)
        if kp1 > k:
            self.delay_probs[: kp1 + 1] += alpha / float(kp1 + 1)
        self.delay_probs /= self.delay_probs.sum().clamp_min(1.0e-8)

    def _schedule_torque_limit(self, progress: float) -> float:
        start, end = self.cfg.torque_limit_progress_range
        start_scale, end_scale = self.cfg.torque_limit_scale_range
        p = max(0.0, min(float(progress), 1.0))
        if p <= float(start):
            scale = float(start_scale)
        elif p >= float(end) or abs(float(end) - float(start)) <= 1.0e-9:
            scale = float(end_scale)
        else:
            ratio = (p - float(start)) / (float(end) - float(start))
            scale = float(start_scale) + ratio * (float(end_scale) - float(start_scale))
        if self._torque_limit_scale is not None and abs(scale - self._torque_limit_scale) < 1.0e-6:
            return scale
        applied_scale = max(float(scale), 0.0)
        force_range = self._default_forcerange.unsqueeze(0) * applied_scale
        self._env.sim.model.actuator_forcerange[:, self._ctrl_ids] = force_range
        self._torque_limit_scale = applied_scale
        return applied_scale

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.raw_action_clip is not None:
            actions = actions.clamp(
                min=-float(self.cfg.raw_action_clip), max=float(self.cfg.raw_action_clip)
            )
        self._raw_actions[:] = actions
        self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
        self._action_history[:, 0] = self._raw_actions
        self._substep = 0

    def _update_processed_actions(self, substep: int) -> None:
        delay_env_steps = torch.div(
            self.delay.squeeze(-1) - int(substep) + self._decimation - 1,
            self._decimation,
            rounding_mode="floor",
        ).clamp(0, self._history_len - 1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed_action = self._action_history[env_ids, delay_env_steps]
        self.applied_action.lerp_(delayed_action, self.alpha)
        self._processed_actions = (
            self.applied_action * self._scale + self._default_offset + self.joint_offset
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        booting = self.boot_delay > 0
        self._processed_actions = torch.where(booting, self.boot_target, self._processed_actions)
        self.boot_delay.sub_(1).clamp_min_(0)

    def apply_actions(self) -> None:
        self._update_processed_actions(self._substep)
        super().apply_actions()
        self._substep = (self._substep + 1) % self._decimation
