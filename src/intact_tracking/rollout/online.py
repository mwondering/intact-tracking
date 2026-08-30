"""Live frozen-tracker rollout with one immutable DR sample per vector slot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from intact_tracking.environment.runtime import create_runtime, prepare_rollout

from .mjlab_adapter import (
    _clear_missing_motion_exclusions,
    _policy_observations,
    _reset_all,
    _snapshot,
)


@dataclass(frozen=True)
class FixedDRRolloutConfig:
    """Inputs for a live rollout used directly by online training."""

    checkpoint_file: str
    motion_path: str | None = None
    motion_file: str | None = None
    task_id: str | None = None
    num_envs: int = 16
    device: str | None = None
    seed: int = 0
    stochastic_policy: bool = False

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if bool(self.motion_path) == bool(self.motion_file):
            raise ValueError("Provide exactly one of motion_path or motion_file")


def _keep_startup_events(env_cfg: Any) -> tuple[list[str], list[str]]:
    """Keep only construction-time DR, so reset cannot mutate a physics world."""
    startup: dict[str, Any] = {}
    removed: list[str] = []
    for name, term in env_cfg.events.items():
        if getattr(term, "mode", None) == "startup":
            startup[name] = term
        else:
            removed.append(str(name))
    if not startup:
        raise RuntimeError(
            "Pure online training requires at least one startup DR event in the checkpoint"
        )
    env_cfg.events = startup
    return sorted(str(name) for name in startup), sorted(removed)


def _disable_startup_reset_callbacks(env: Any) -> list[str]:
    """Prevent class-based startup DR terms from resampling on later resets.

    MJLab calls ``reset()`` on every class-based event term from
    ``EventManager.reset()``, independently of the term's event mode. The RSL
    wrapper has already performed its initialization reset before this helper is
    called. Removing startup callbacks here freezes their resulting values for
    all subsequent episode resets.
    """
    callbacks_by_mode = getattr(env.event_manager, "_mode_class_term_cfgs", None)
    if not isinstance(callbacks_by_mode, dict):
        raise RuntimeError("MJLab EventManager does not expose class reset callbacks")
    callbacks = list(callbacks_by_mode.get("startup", ()))
    callbacks_by_mode["startup"] = []
    return sorted(type(term.func).__name__ for term in callbacks)


def _capture_randomized_model_fields(env: Any) -> dict[str, torch.Tensor]:
    snapshots: dict[str, torch.Tensor] = {}
    for name in env.event_manager.domain_randomization_fields:
        value = getattr(env.sim.model, name, None)
        clone = getattr(value, "clone", None)
        if callable(clone):
            snapshot = clone()
            if isinstance(snapshot, torch.Tensor):
                snapshots[str(name)] = snapshot.detach()
    return snapshots


class FixedDRTrackerRollout:
    """Step MJLab with a frozen tracker and fixed per-environment physics DR.

    MJLab applies ``startup`` events while the environment is constructed. All
    other event modes are removed before construction, and this class never
    reapplies startup events. Manual resets therefore resample motion and robot
    state while preserving each vector slot's physics parameters.
    """

    def __init__(self, config: FixedDRRolloutConfig) -> None:
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        checkpoint = Path(config.checkpoint_file).expanduser().resolve()
        prepared = prepare_rollout(
            checkpoint_file=str(checkpoint),
            motion_path=config.motion_path,
            motion_file=config.motion_file,
            task_id=config.task_id,
            num_envs=config.num_envs,
        )
        prepared.env.seed = int(config.seed)
        prepared.env.auto_reset = False
        self.cleared_motion_exclusions = _clear_missing_motion_exclusions(prepared.env)
        self.startup_events, self.removed_non_startup_events = _keep_startup_events(prepared.env)
        self.device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.checkpoint_path = checkpoint
        self.checkpoint_task_id = prepared.checkpoint_task_id
        self._clip_actions = prepared.clip_actions
        self._runtime = create_runtime(
            prepared,
            device=self.device,
            stochastic_policy=config.stochastic_policy,
        )
        self.closed = False
        try:
            self.env = self._runtime.env
            self.disabled_startup_reset_callbacks = _disable_startup_reset_callbacks(self.env)
            self._fixed_dr_model_fields = _capture_randomized_model_fields(self.env)
            if not self._fixed_dr_model_fields:
                raise RuntimeError("Startup DR did not expose any randomized physics fields")
            self.dr_invariance_checks = 0
            self.wrapped = self._runtime.wrapped
            self.policy = self._runtime.policy
            self.observations = self.wrapped.get_observations()
            self.world_ids = torch.arange(config.num_envs, device=self.env.device, dtype=torch.long)
            self.episode_ids = torch.zeros_like(self.world_ids)
            self.episode_steps = torch.zeros_like(self.world_ids)
            self.env_ids = self.world_ids.clone()
            self.collector_step = 0
            self.synchronous_resets = 0
            self.motion_ids_seen: set[int] = set()

            actor = self._runtime.actor
            if actor.training or any(parameter.requires_grad for parameter in actor.parameters()):
                raise RuntimeError("Tracker must be frozen and in eval mode for online rollout")
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def transitions(self) -> int:
        return self.collector_step * self.num_envs

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_task_id": self.checkpoint_task_id,
            "tracker_frozen": True,
            "startup_events": self.startup_events,
            "removed_non_startup_events": self.removed_non_startup_events,
            "disabled_startup_reset_callbacks": self.disabled_startup_reset_callbacks,
            "fixed_dr_model_fields": sorted(self._fixed_dr_model_fields),
            "cleared_missing_motion_exclusions": self.cleared_motion_exclusions,
            "domain_randomization_contract": (
                "startup DR fixed per vector slot before rollout and never resampled"
            ),
            "motion_contract": "random motion resampling at initialization and reset",
            "reset_contract": "synchronous manual reset; startup events are never reapplied",
        }

    def _assert_fixed_dr(self) -> None:
        for name, expected in self._fixed_dr_model_fields.items():
            actual = getattr(self.env.sim.model, name).clone()
            if not torch.equal(actual, expected):
                raise RuntimeError(f"Physics DR field {name!r} changed during an episode reset")
        self.dr_invariance_checks += 1

    def step(self) -> dict[str, torch.Tensor]:
        """Collect one transition from every vector slot."""
        if self.closed:
            raise RuntimeError("Cannot step a closed rollout")
        before = _snapshot(self.env, self.observations)
        self.motion_ids_seen.update(int(value) for value in before["motion_id"].tolist())
        action = self.policy(self.observations)
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"Frozen tracker must return a Tensor, got {type(action).__name__}")
        if self._clip_actions is not None:
            action = action.clamp(-float(self._clip_actions), float(self._clip_actions))

        raw_next, reward, terminated, truncated, _ = self.env.step(action)
        next_observations = _policy_observations(raw_next, self.num_envs)
        after = _snapshot(self.env, next_observations)
        done = terminated | truncated
        synchronous_boundary = bool(done.any())
        reset_boundary = torch.full_like(done, synchronous_boundary)
        collector_steps = torch.full_like(self.episode_ids, self.collector_step)
        batch = {
            "proprio": before["proprio"],
            "next_proprio": after["proprio"],
            "observation": before["observation"],
            "next_observation": after["observation"],
            "reference_observation": before["reference_observation"],
            "next_reference_observation": after["reference_observation"],
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "reset_boundary": reset_boundary,
            "world_id": self.world_ids,
            "episode_id": self.episode_ids.clone(),
            "episode_step": self.episode_steps.clone(),
            "collector_step": collector_steps,
            "env_id": self.env_ids,
            "motion_id": before["motion_id"],
            "motion_step": before["motion_step"],
        }

        self.collector_step += 1
        self.episode_steps += 1
        if synchronous_boundary:
            self.observations = _reset_all(self.env, self.num_envs)
            self._assert_fixed_dr()
            self.episode_ids += 1
            self.episode_steps.zero_()
            self.synchronous_resets += 1
        else:
            self.observations = next_observations
        return batch

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._runtime.close()

    def __enter__(self) -> FixedDRTrackerRollout:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
