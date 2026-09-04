"""Nominal-physics counterfactual rollouts for paired Forward training."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from intact_tracking.environment.runtime import prepare_rollout

from .mjlab_adapter import _clear_missing_motion_exclusions, _robot_raw_state


@dataclass(frozen=True)
class NominalPairRolloutConfig:
    """Build a lightweight, no-DR simulator used only for five-step targets."""

    checkpoint_file: str
    motion_path: str | None = None
    motion_file: str | None = None
    task_id: str | None = None
    num_envs: int = 64
    device: str | None = None
    seed: int = 0
    horizon: int = 5
    restore_atol: float = 1.0e-5
    failure_log_file: str | None = None

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("nominal pair num_envs must be positive")
        if self.horizon != 5:
            raise ValueError("nominal pair rollout is fixed to five steps")
        if self.restore_atol <= 0.0:
            raise ValueError("nominal restore_atol must be positive")
        if bool(self.motion_path) == bool(self.motion_file):
            raise ValueError("Provide exactly one of motion_path or motion_file")


def _make_nominal_dynamics_cfg(env_cfg: Any) -> dict[str, Any]:
    """Remove every task component that is unnecessary for open-loop physics.

    The scene and action interface stay checkpoint-identical.  Commands,
    observations and task managers are removed because the five final actions
    have already been produced by the real DR rollout.  Removing events before
    simulator construction leaves all model fields at their compiled nominal
    values and avoids allocating a second copy of the motion dataset.
    """

    action_overrides: dict[str, dict[str, dict[str, Any]]] = {}
    for term_name, term_cfg in env_cfg.actions.items():
        overrides: dict[str, dict[str, Any]] = {}
        deterministic_values = {
            "max_delay": 0,
            "alpha": (1.0, 1.0),
            "torque_limit_scale_range": (1.0, 1.0),
            "boot_delay_steps": 0,
        }
        for field_name, desired in deterministic_values.items():
            if not hasattr(term_cfg, field_name):
                continue
            previous = getattr(term_cfg, field_name)
            if previous != desired:
                setattr(term_cfg, field_name, desired)
                overrides[field_name] = {"from": previous, "to": desired}
        if overrides:
            action_overrides[str(term_name)] = overrides

    removed = {
        "events": sorted(str(name) for name in env_cfg.events),
        "commands": sorted(str(name) for name in env_cfg.commands),
        "observations": sorted(str(name) for name in env_cfg.observations),
        "rewards": sorted(str(name) for name in env_cfg.rewards),
        "terminations": sorted(str(name) for name in env_cfg.terminations),
        "curriculum": sorted(str(name) for name in env_cfg.curriculum),
        "metrics": sorted(str(name) for name in env_cfg.metrics),
        "recorders": sorted(str(name) for name in env_cfg.recorders),
        "action_randomization_overrides": action_overrides,
    }
    env_cfg.events = {}
    env_cfg.commands = {}
    env_cfg.observations = {}
    env_cfg.rewards = {}
    env_cfg.terminations = {}
    env_cfg.curriculum = {}
    env_cfg.metrics = {}
    env_cfg.recorders = {}
    env_cfg.auto_reset = False
    return removed


_STATE_COMPONENTS = (
    ("root_position", 0, 3),
    ("root_quaternion", 3, 7),
    ("root_linear_velocity", 7, 10),
    ("root_angular_velocity", 10, 13),
    ("joint_position", 13, 42),
    ("joint_velocity", 42, 71),
)


def _state_component(index: int) -> str:
    for name, start, stop in _STATE_COMPONENTS:
        if start <= index < stop:
            return name
    raise ValueError(f"Invalid 71-D state index {index}")


def _repeat_error_diagnostics(
    target: torch.Tensor,
    repeated: torch.Tensor,
    *,
    motion_ids: torch.Tensor | None = None,
    motion_steps: torch.Tensor | None = None,
    motion_files: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Locate repeat errors without mixing velocity into the pose metric."""

    difference = (repeated - target).abs()
    pose_indices = torch.cat(
        (
            torch.arange(0, 7, device=difference.device),
            torch.arange(13, 42, device=difference.device),
        )
    )
    pose_difference = difference.index_select(-1, pose_indices)

    def distribution(value: torch.Tensor) -> dict[str, float]:
        per_sample = value.flatten(start_dim=1).amax(dim=1).float()
        quantiles = torch.quantile(
            per_sample,
            torch.tensor((0.5, 0.95, 0.99), device=value.device),
        )
        return {
            "mean": float(per_sample.mean().item()),
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "max": float(per_sample.max().item()),
        }

    def worst(value: torch.Tensor, state_indices: torch.Tensor | None = None) -> dict[str, Any]:
        flat_index = int(value.reshape(-1).argmax().item())
        state_width = int(value.size(-1))
        horizon = (flat_index // state_width) % int(value.size(1))
        pair_index = flat_index // (int(value.size(1)) * state_width)
        local_state_index = flat_index % state_width
        state_index = (
            int(state_indices[local_state_index].item())
            if state_indices is not None
            else local_state_index
        )
        result: dict[str, Any] = {
            "pair_index": pair_index,
            "horizon": horizon + 1,
            "state_index": state_index,
            "state_component": _state_component(state_index),
            "abs_error": float(value.reshape(-1)[flat_index].item()),
        }
        if motion_ids is not None:
            motion_id = int(motion_ids[pair_index].item())
            result["motion_id"] = motion_id
            if motion_files is not None and 0 <= motion_id < len(motion_files):
                result["motion_path"] = str(motion_files[motion_id])
            else:
                result["motion_path"] = None
        if motion_steps is not None:
            result["motion_step"] = int(motion_steps[pair_index].item())
        return result

    return {
        "pose": distribution(pose_difference),
        "full_state": distribution(difference),
        "worst_pose": worst(pose_difference, pose_indices),
        "worst_full_state": worst(difference),
    }


class NominalPairRollout:
    """Restore replay states into nominal physics and replay identical actions."""

    def __init__(self, config: NominalPairRolloutConfig) -> None:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        self.config = config
        self.device = torch.device(
            config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = Path(config.checkpoint_file).expanduser().resolve()
        prepared = prepare_rollout(
            checkpoint_file=str(checkpoint),
            motion_path=config.motion_path,
            motion_file=config.motion_file,
            task_id=config.task_id,
            num_envs=config.num_envs,
        )
        prepared.env.seed = int(config.seed)
        cleared = _clear_missing_motion_exclusions(prepared.env)
        removed = _make_nominal_dynamics_cfg(prepared.env)
        self.env = ManagerBasedRlEnv(cfg=deepcopy(prepared.env), device=str(self.device))
        self.closed = False
        try:
            raw, _ = self.env.reset()
            if raw:
                raise RuntimeError(
                    "Nominal dynamics environment unexpectedly produced observations"
                )
            if self.env.event_manager.domain_randomization_fields:
                raise RuntimeError("Nominal dynamics environment still exposes DR model fields")
            self.robot = self.env.scene["robot"]
            self.action_dim = int(self.env.action_manager.total_action_dim)
            if self.action_dim != 29:
                raise RuntimeError(f"Nominal action width is {self.action_dim}, expected 29")
            if int(self.robot.num_joints) != 29:
                raise RuntimeError(f"Nominal robot has {self.robot.num_joints} joints, expected 29")
            self._env_ids = torch.arange(config.num_envs, dtype=torch.long, device=self.device)
            self._validated = False
            self._last_repeat_error = 0.0
            self._last_repeat_pose_error = 0.0
            self._last_repeat_full_state_p99_error = 0.0
            self._last_repeat_pose_p99_error = 0.0
            self._last_repeat_warning = 0.0
            self._last_restore_error = 0.0
            self.metadata = {
                "config": asdict(config),
                "checkpoint_task_id": prepared.checkpoint_task_id,
                "physics": "compiled checkpoint defaults with all DR events removed",
                "state_restore": (
                    "clear sim/entity/action state; write local 71-D robot state; "
                    "run zero-time sim.forward"
                ),
                "warmup": "none; warmup would change the paired initial state",
                "removed_managers": removed,
                "action_term": type(self.env.action_manager.get_term("joint_pos")).__name__,
                "cleared_missing_motion_exclusions": cleared,
            }
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    def _validate_inputs(
        self,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        expected_state = (self.num_envs, 71)
        expected_previous = (self.num_envs, self.action_dim)
        expected_actions = (self.num_envs, self.config.horizon, self.action_dim)
        expected = {
            "state": (state, expected_state),
            "previous_action": (previous_action, expected_previous),
            "actions": (actions, expected_actions),
        }
        for name, (value, shape) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"Nominal {name} has {tuple(value.shape)}, expected {shape}")
            if value.device != self.device:
                raise ValueError(f"Nominal {name} must be on {self.device}, got {value.device}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nominal {name} contains non-finite values")

    def _restore(self, state: torch.Tensor, previous_action: torch.Tensor) -> float:
        """Restore the Markov physics state without advancing simulation time."""

        env = self.env
        env.sim.reset(self._env_ids)
        env.scene.reset(self._env_ids)
        env.action_manager.reset(self._env_ids)

        root_state = state[:, :13].detach().clone()
        root_state[:, :3].add_(env.scene.env_origins)
        root_state[:, 3:7] = torch.nn.functional.normalize(root_state[:, 3:7], dim=-1, eps=1.0e-8)
        joint_position = state[:, 13:42].detach()
        joint_velocity = state[:, 42:71].detach()
        self.robot.write_root_state_to_sim(root_state, env_ids=self._env_ids)
        self.robot.write_joint_state_to_sim(
            joint_position,
            joint_velocity,
            env_ids=self._env_ids,
        )
        self.robot.set_joint_position_target(joint_position, env_ids=self._env_ids)

        # Preserve the replay sample's previous policy-coordinate action for
        # managers that expose it.  The checkpoint's BFM action is memoryless
        # physically, but this makes the reset contract explicit and portable.
        action_manager = env.action_manager
        action_manager._action.copy_(previous_action)
        action_manager._prev_action.copy_(previous_action)
        action_manager._prev_prev_action.copy_(previous_action)
        action_term = action_manager.get_term("joint_pos")
        raw_action = getattr(action_term, "_raw_actions", None)
        if isinstance(raw_action, torch.Tensor):
            raw_action.copy_(previous_action)
        policy_history = getattr(action_term, "_policy_mean_history", None)
        if isinstance(policy_history, torch.Tensor):
            policy_history.copy_(previous_action[:, None].expand_as(policy_history))

        # This is the only required "warmup": forward kinematics/contact
        # reconstruction at zero simulated time.  A physics step is forbidden.
        env.scene.write_data_to_sim()
        env.sim.forward()
        restored = _robot_raw_state(env)
        restore_error = float((restored - state).abs().max().item())
        self._last_restore_error = restore_error
        if restore_error > self.config.restore_atol:
            raise RuntimeError(
                "Nominal state restore changed the requested robot state: "
                f"max_abs_error={restore_error:.6g}, atol={self.config.restore_atol:.6g}"
            )
        return restore_error

    def _step_actions(self, actions: torch.Tensor) -> torch.Tensor:
        states: list[torch.Tensor] = []
        for index in range(self.config.horizon):
            _, _, terminated, truncated, _ = self.env.step(actions[:, index])
            if bool((terminated | truncated).any()):
                raise RuntimeError("Task-free nominal dynamics rollout unexpectedly terminated")
            states.append(_robot_raw_state(self.env).clone())
        return torch.stack(states, dim=1)

    def _step_joint_targets(self, joint_targets: torch.Tensor) -> torch.Tensor:
        """Advance nominal physics with already-resolved physical PD targets.

        ``joint_targets`` is in simulator robot-joint order.  It deliberately
        bypasses the policy-facing action term: scale, offset, clipping,
        encoder bias, delay and smoothing have already happened in rollout A.
        Reapplying that chain here would make A/B differ in both controller and
        physics and would invalidate the counterfactual target.
        """

        env = self.env
        states: list[torch.Tensor] = []
        for index in range(self.config.horizon):
            target = joint_targets[:, index]
            for _ in range(env.cfg.decimation):
                self.robot.set_joint_position_target(target, env_ids=self._env_ids)
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(dt=env.physics_dt)
            env.sim.forward()
            states.append(_robot_raw_state(env).clone())
        return torch.stack(states, dim=1)

    def rollout_joint_targets(
        self,
        state: torch.Tensor,
        joint_targets: torch.Tensor,
        *,
        motion_ids: torch.Tensor | None = None,
        motion_steps: torch.Tensor | None = None,
        motion_files: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the exact nominal-physics B batch for one A-batch trajectory.

        The starting qpos/qvel are restored from A and the five *physical* PD
        targets recorded in A are replayed directly.  The task, policy and
        policy-action transform are therefore absent from the B data path.
        """

        if self.closed:
            raise RuntimeError("Cannot use a closed nominal pair rollout")
        expected_state = (self.num_envs, 71)
        expected_targets = (self.num_envs, self.config.horizon, self.action_dim)
        if tuple(state.shape) != expected_state:
            raise ValueError(f"Nominal state has {tuple(state.shape)}, expected {expected_state}")
        if tuple(joint_targets.shape) != expected_targets:
            raise ValueError(
                "Nominal joint_targets has "
                f"{tuple(joint_targets.shape)}, expected {expected_targets}"
            )
        if state.device != self.device or joint_targets.device != self.device:
            raise ValueError(f"Nominal inputs must be on {self.device}")
        for name, value in (("motion_ids", motion_ids), ("motion_steps", motion_steps)):
            if value is None:
                continue
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(
                    f"Nominal {name} has {tuple(value.shape)}, expected {(self.num_envs,)}"
                )

        zero_previous_action = torch.zeros(
            (self.num_envs, self.action_dim),
            dtype=joint_targets.dtype,
            device=self.device,
        )
        restore_error = self._restore(state, zero_previous_action)
        target = self._step_joint_targets(joint_targets)
        if not self._validated:
            self._restore(state, zero_previous_action)
            repeated = self._step_joint_targets(joint_targets)
            diagnostics = _repeat_error_diagnostics(
                target,
                repeated,
                motion_ids=motion_ids,
                motion_steps=motion_steps,
                motion_files=motion_files,
            )
            self._last_repeat_error = float(diagnostics["full_state"]["max"])
            self._last_repeat_pose_error = float(diagnostics["pose"]["max"])
            self._last_repeat_full_state_p99_error = float(diagnostics["full_state"]["p99"])
            self._last_repeat_pose_p99_error = float(diagnostics["pose"]["p99"])
            if (
                self._last_repeat_pose_error > self.config.restore_atol
                or self._last_repeat_error > 10.0 * self.config.restore_atol
            ):
                self._last_repeat_warning = 1.0
            self._validated = True
        return target, {
            "restore_max_abs_error": restore_error,
            "repeat_max_abs_error": self._last_repeat_error,
            "repeat_warning": self._last_repeat_warning,
        }

    def rollout(
        self,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
        *,
        motion_ids: torch.Tensor | None = None,
        motion_steps: torch.Tensor | None = None,
        motion_files: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return nominal states for the same initial state and five actions.

        On the first call, repeat the complete restore-and-rollout operation and
        measure its reproducibility.  Rare contact-solver outliers are recorded
        as diagnostics instead of aborting training; exact 71-D state restore
        remains a hard requirement in :meth:`_restore`.
        """

        if self.closed:
            raise RuntimeError("Cannot use a closed nominal pair rollout")
        self._validate_inputs(state, previous_action, actions)
        for name, value in (("motion_ids", motion_ids), ("motion_steps", motion_steps)):
            if value is None:
                continue
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(
                    f"Nominal {name} has {tuple(value.shape)}, expected {(self.num_envs,)}"
                )
            if value.device != self.device or value.dtype != torch.long:
                raise ValueError(f"Nominal {name} must be long on {self.device}")
        restore_error = self._restore(state, previous_action)
        target = self._step_actions(actions)
        if not self._validated:
            self._restore(state, previous_action)
            repeated = self._step_actions(actions)
            diagnostics = _repeat_error_diagnostics(
                target,
                repeated,
                motion_ids=motion_ids,
                motion_steps=motion_steps,
                motion_files=motion_files,
            )
            repeat_error = float(diagnostics["full_state"]["max"])
            repeat_pose_error = float(diagnostics["pose"]["max"])
            self._last_repeat_error = repeat_error
            self._last_repeat_pose_error = repeat_pose_error
            self._last_repeat_full_state_p99_error = float(diagnostics["full_state"]["p99"])
            self._last_repeat_pose_p99_error = float(diagnostics["pose"]["p99"])
            if (
                repeat_pose_error > self.config.restore_atol
                or repeat_error > 10.0 * self.config.restore_atol
            ):
                self._last_repeat_warning = 1.0
                failure_record = {
                    "event": "nominal_repeat_validation_warning",
                    "restore_max_abs_error": restore_error,
                    "pose_atol": self.config.restore_atol,
                    "full_state_atol": 10.0 * self.config.restore_atol,
                    "action_term": type(self.env.action_manager.get_term("joint_pos")).__name__,
                    **diagnostics,
                }
                if self.config.failure_log_file:
                    failure_path = Path(self.config.failure_log_file)
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    with failure_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(failure_record, sort_keys=True) + "\n")
                print(json.dumps(failure_record, sort_keys=True), flush=True)
            self._validated = True
        return target, {
            "nominal_restore_state_max_abs_error": restore_error,
            "nominal_restore_repeat_max_abs_error": self._last_repeat_error,
            "nominal_restore_repeat_pose_max_abs_error": self._last_repeat_pose_error,
            "nominal_restore_repeat_full_state_p99_abs_error": (
                self._last_repeat_full_state_p99_error
            ),
            "nominal_restore_repeat_pose_p99_abs_error": (self._last_repeat_pose_p99_error),
            "nominal_restore_repeat_warning": self._last_repeat_warning,
        }

    def close(self) -> None:
        if not getattr(self, "closed", True):
            self.closed = True
            self.env.close()

    def __enter__(self) -> NominalPairRollout:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
