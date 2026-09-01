"""Nominal-physics counterfactual rollouts for paired Forward training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

    removed = {
        "events": sorted(str(name) for name in env_cfg.events),
        "commands": sorted(str(name) for name in env_cfg.commands),
        "observations": sorted(str(name) for name in env_cfg.observations),
        "rewards": sorted(str(name) for name in env_cfg.rewards),
        "terminations": sorted(str(name) for name in env_cfg.terminations),
        "curriculum": sorted(str(name) for name in env_cfg.curriculum),
        "metrics": sorted(str(name) for name in env_cfg.metrics),
        "recorders": sorted(str(name) for name in env_cfg.recorders),
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
                raise RuntimeError("Nominal dynamics environment unexpectedly produced observations")
            if self.env.event_manager.domain_randomization_fields:
                raise RuntimeError("Nominal dynamics environment still exposes DR model fields")
            self.robot = self.env.scene["robot"]
            self.action_dim = int(self.env.action_manager.total_action_dim)
            if self.action_dim != 29:
                raise RuntimeError(f"Nominal action width is {self.action_dim}, expected 29")
            if int(self.robot.num_joints) != 29:
                raise RuntimeError(
                    f"Nominal robot has {self.robot.num_joints} joints, expected 29"
                )
            self._env_ids = torch.arange(
                config.num_envs, dtype=torch.long, device=self.device
            )
            self._validated = False
            self._last_repeat_error = 0.0
            self._last_repeat_pose_error = 0.0
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
        root_state[:, 3:7] = torch.nn.functional.normalize(
            root_state[:, 3:7], dim=-1, eps=1.0e-8
        )
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

    def rollout(
        self,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return nominal states for the same initial state and five actions.

        On the first call, repeat the complete restore-and-rollout operation and
        require identical results.  This catches stale actuator state, solver
        buffers, or accidental physics warmup before training can consume pairs.
        """

        if self.closed:
            raise RuntimeError("Cannot use a closed nominal pair rollout")
        self._validate_inputs(state, previous_action, actions)
        restore_error = self._restore(state, previous_action)
        target = self._step_actions(actions)
        if not self._validated:
            self._restore(state, previous_action)
            repeated = self._step_actions(actions)
            repeat_error = float((repeated - target).abs().max().item())
            repeat_pose_error = float((repeated[..., :42] - target[..., :42]).abs().max().item())
            self._last_repeat_error = repeat_error
            self._last_repeat_pose_error = repeat_pose_error
            if (
                repeat_pose_error > self.config.restore_atol
                or repeat_error > 10.0 * self.config.restore_atol
            ):
                raise RuntimeError(
                    "Nominal restore is not deterministic over five steps: "
                    f"pose_max_abs_error={repeat_pose_error:.6g}, "
                    f"full_state_max_abs_error={repeat_error:.6g}, "
                    f"pose_atol={self.config.restore_atol:.6g}"
                )
            self._validated = True
        return target, {
            "nominal_restore_state_max_abs_error": restore_error,
            "nominal_restore_repeat_max_abs_error": self._last_repeat_error,
            "nominal_restore_repeat_pose_max_abs_error": self._last_repeat_pose_error,
        }

    def close(self) -> None:
        if not getattr(self, "closed", True):
            self.closed = True
            self.env.close()

    def __enter__(self) -> NominalPairRollout:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
