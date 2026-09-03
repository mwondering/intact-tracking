"""Collect portable transitions with the repository-owned MJLab runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from intact_tracking.data import RolloutDimensions, RolloutShardWriter
from intact_tracking.environment.runtime import create_runtime, prepare_rollout
from intact_tracking.forward_predictor_inputs import (
    CONTACT_BINARY_DIM,
    CONTACT_FORCE_DIM,
    FOOT_FEATURE_DIM,
    G1_FOOT_BODY_NAMES,
    g1_foot_features_from_link_state,
)

PROPRIO_TERM_DIMS = (29, 29, 3, 3, 29, 29)
PROPRIO_HISTORY = 50


@dataclass(frozen=True)
class MjlabCollectorConfig:
    checkpoint_file: str
    output_dir: str
    motion_path: str | None = None
    motion_file: str | None = None
    task_id: str | None = None
    num_envs: int = 1
    transitions: int = 100_000
    shard_size: int = 100_000
    device: str | None = None
    seed: int = 0
    stochastic_policy: bool = False
    include_disturbances: bool = False
    world_session_steps: int = 3_000
    world_id_offset: int = 0

    def __post_init__(self) -> None:
        for name in (
            "num_envs",
            "transitions",
            "shard_size",
            "world_session_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _latest_proprio(history: torch.Tensor) -> torch.Tensor:
    expected = PROPRIO_HISTORY * sum(PROPRIO_TERM_DIMS)
    if history.ndim != 2 or history.size(-1) != expected:
        raise ValueError(f"Estimator history must be [N,{expected}], got {tuple(history.shape)}")
    values = []
    cursor = 0
    for width in PROPRIO_TERM_DIMS:
        size = PROPRIO_HISTORY * width
        term = history[:, cursor : cursor + size].reshape(history.size(0), PROPRIO_HISTORY, width)
        values.append(term[:, -1])
        cursor += size
    proprio = torch.cat(values, dim=-1)
    if proprio.size(-1) != 122:
        raise RuntimeError(f"Extracted proprio width {proprio.size(-1)}, expected 122")
    return proprio


def _reference_snapshot(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from intact_tracking.environment.mdp import sp as tracking_mdp

    command = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    joint_pos = command.gather_reference("joint_pos", (0,))[:, 0]
    joint_vel = command.gather_reference("joint_vel", (0,))[:, 0]
    root_pos = command.gather_root_reference("body_pos_w", (0,))[:, 0]
    root_quat = command.gather_root_reference("body_quat_w", (0,))[:, 0]
    root_lin_vel = command.gather_root_reference("body_lin_vel_w", (0,))[:, 0]
    root_ang_vel = command.gather_root_reference("body_ang_vel_w", (0,))[:, 0]
    gravity = tracking_mdp._projected_gravity(root_quat)
    angular_velocity_body = tracking_mdp._quat_apply_inverse(root_quat, root_ang_vel)
    observation = torch.cat(
        (
            joint_pos - robot.data.default_joint_pos,
            joint_vel - robot.data.default_joint_vel,
            gravity,
            angular_velocity_body,
        ),
        dim=-1,
    )
    raw_state = torch.cat(
        (
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        ),
        dim=-1,
    )
    if observation.size(-1) != 64 or raw_state.size(-1) != 71:
        raise RuntimeError(f"Unexpected reference shapes: {observation.shape}, {raw_state.shape}")
    return observation, raw_state


def _robot_raw_state(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    data = robot.data
    root_position = data.root_link_pos_w - env.scene.env_origins
    value = torch.cat(
        (
            root_position,
            data.root_link_quat_w,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
            data.joint_pos,
            data.joint_vel,
        ),
        dim=-1,
    )
    if value.size(-1) != 71:
        raise RuntimeError(f"Unexpected robot raw-state shape {tuple(value.shape)}")
    return value


def _feet_motion_snapshot(env: Any) -> dict[str, torch.Tensor]:
    """Read sole height/velocity directly from simulator link kinematics."""

    robot = env.scene["robot"]
    body_names = tuple(str(name).split("/")[-1] for name in robot.body_names)
    missing = [name for name in G1_FOOT_BODY_NAMES if name not in body_names]
    if missing:
        raise RuntimeError(f"Forward Predictor foot bodies are missing: {missing}")
    indices = torch.as_tensor(
        [body_names.index(name) for name in G1_FOOT_BODY_NAMES],
        dtype=torch.long,
        device=robot.data.body_link_pos_w.device,
    )
    link_position = robot.data.body_link_pos_w.index_select(1, indices)
    link_position = link_position - env.scene.env_origins[:, None, :]
    foot = g1_foot_features_from_link_state(
        link_position,
        robot.data.body_link_quat_w.index_select(1, indices),
        robot.data.body_link_lin_vel_w.index_select(1, indices),
        robot.data.body_link_ang_vel_w.index_select(1, indices),
    )
    if foot.shape != (env.num_envs, FOOT_FEATURE_DIM):
        raise RuntimeError(
            "Forward Predictor foot feature must have shape "
            f"[{env.num_envs},{FOOT_FEATURE_DIM}], got {tuple(foot.shape)}"
        )
    return {"foot": foot}


def _feet_contact_snapshot(env: Any) -> dict[str, torch.Tensor]:
    """Read current left/right terrain contact without making it a policy input."""

    try:
        sensor = env.scene["contact_forces"]
    except (KeyError, TypeError):
        return {}
    force = getattr(sensor.data, "force", None)
    if not isinstance(force, torch.Tensor):
        return {}
    force_history = getattr(sensor.data, "force_history", None)
    if isinstance(force_history, torch.Tensor):
        force = force_history.mean(dim=2)
    primary_names = tuple(getattr(sensor, "primary_names", ()))
    expected_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    if set(primary_names) != set(expected_names):
        raise RuntimeError(
            "Forward Predictor contact sensor must resolve exactly the two feet; "
            f"got {primary_names}"
        )
    indices = torch.as_tensor(
        [primary_names.index(name) for name in expected_names],
        dtype=torch.long,
        device=force.device,
    )
    force = force.index_select(1, indices).reshape(env.num_envs, -1)
    if force.shape != (env.num_envs, CONTACT_FORCE_DIM):
        raise RuntimeError(
            "Forward Predictor contact force must have shape "
            f"[{env.num_envs},{CONTACT_FORCE_DIM}], got {tuple(force.shape)}"
        )

    from intact_tracking.environment.mdp.sp import feet_contact_binary_state

    binary = feet_contact_binary_state(env, "contact_forces").index_select(1, indices).bool()
    if binary.shape != (env.num_envs, CONTACT_BINARY_DIM):
        raise RuntimeError(
            "Forward Predictor contact state must have shape "
            f"[{env.num_envs},{CONTACT_BINARY_DIM}], got {tuple(binary.shape)}"
        )
    return {
        "contact_force": force,
        "contact_binary": binary,
    }


def _snapshot(env: Any, observations: Any) -> dict[str, torch.Tensor]:
    history = observations["estimator_history"]
    proprio = _latest_proprio(history)
    reference_observation, reference_state = _reference_snapshot(env)
    command = env.command_manager.get_term("motion")
    snapshot = {
        "proprio": proprio,
        # INTACT endpoints use the sensor/reference quantities common to both
        # sides. Previous action and torque remain in the interaction context.
        "observation": proprio[:, :64],
        "reference_observation": reference_observation,
        "robot_state": _robot_raw_state(env),
        "reference_state": reference_state,
        "motion_id": command.motion_idx.clone(),
        "motion_step": command.time_steps.clone(),
    }
    snapshot.update(_feet_motion_snapshot(env))
    snapshot.update(_feet_contact_snapshot(env))
    return snapshot


def _forward_predictor_snapshot(env: Any) -> dict[str, torch.Tensor]:
    """Read only simulator fields consumed by the online predictor replay."""

    command = env.command_manager.get_term("motion")
    snapshot = {
        "robot_state": _robot_raw_state(env),
        "motion_id": command.motion_idx.clone(),
        "motion_step": command.time_steps.clone(),
    }
    snapshot.update(_feet_motion_snapshot(env))
    snapshot.update(_feet_contact_snapshot(env))
    return snapshot


def _filter_disturbance_events(env_cfg: Any) -> list[str]:
    removed: list[str] = []
    kept = {}
    for name, term in env_cfg.events.items():
        if getattr(term, "mode", None) in {"step", "interval"}:
            removed.append(str(name))
        else:
            kept[name] = term
    env_cfg.events = kept
    return removed


def _clear_missing_motion_exclusions(env_cfg: Any) -> list[str]:
    """Drop stale machine-local exclusion manifests embedded in checkpoints."""
    command_cfg = env_cfg.commands["motion"]
    cleared: list[str] = []
    for name in ("motion_exclude_file", "excluded_motion_file"):
        value = getattr(command_cfg, name, None)
        if isinstance(value, str) and value and not Path(value).expanduser().is_file():
            cleared.append(value)
            setattr(command_cfg, name, "")
    for name in ("motion_exclude_files", "excluded_motion_files"):
        value = getattr(command_cfg, name, None)
        if not value:
            continue
        retained = []
        for item in value:
            path = Path(str(item)).expanduser()
            if path.is_file():
                retained.append(item)
            else:
                cleared.append(str(item))
        setattr(command_cfg, name, type(value)(retained))
    return cleared


def _policy_observations(raw: dict[str, torch.Tensor], num_envs: int):
    from tensordict import TensorDict

    return TensorDict(raw, batch_size=[num_envs])


def _reset_all(env: Any, num_envs: int):
    ids = torch.arange(num_envs, dtype=torch.long, device=env.device)
    raw, _ = env.reset(env_ids=ids)
    return _policy_observations(raw, num_envs)


def _resample_static_worlds(env: Any, num_envs: int):
    ids = torch.arange(num_envs, dtype=torch.long, device=env.device)
    if "startup" not in env.event_manager.available_modes:
        raise RuntimeError("The checkpoint environment has no startup world randomization")
    env.event_manager.apply(mode="startup", env_ids=ids)
    return _reset_all(env, num_envs)


def collect_mjlab_rollouts(config: MjlabCollectorConfig) -> Path:
    """Collect transitions while preserving true terminal observations.

    Resets are synchronous: if any vector slot terminates, all slots begin a new
    episode.  This avoids a second observation-history update on non-reset slots
    under ``auto_reset=False`` and makes every recorded boundary explicit.
    """
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.manual_seed(config.seed)
    checkpoint = Path(config.checkpoint_file).expanduser().resolve()
    prepared = prepare_rollout(
        task_id=config.task_id,
        checkpoint_file=str(checkpoint),
        motion_path=config.motion_path,
        motion_file=config.motion_file,
        num_envs=config.num_envs,
    )
    prepared.env.seed = int(config.seed)
    prepared.env.auto_reset = False
    cleared_motion_exclusions = _clear_missing_motion_exclusions(prepared.env)
    removed_events: list[str] = []
    if not config.include_disturbances:
        removed_events = _filter_disturbance_events(prepared.env)
    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = create_runtime(
        prepared,
        device=device,
        stochastic_policy=config.stochastic_policy,
    )
    env = runtime.env
    wrapped = runtime.wrapped
    policy = runtime.policy
    observations = wrapped.get_observations()

    source_root = Path(__file__).resolve().parents[3]
    metadata = {
        "collector": "in-repository MJLab runtime",
        "collector_config": asdict(config),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
        "checkpoint_task_id": prepared.checkpoint_task_id,
        "intact_tracking_commit": _git_commit(source_root),
        "mjlab_version": importlib.metadata.version("mjlab"),
        "control_dt_s": float(env.step_dt),
        "removed_disturbance_events": removed_events,
        "cleared_missing_motion_exclusions": cleared_motion_exclusions,
        "world_contract": "startup DR fixed within a world session",
        "reset_contract": "synchronous manual reset with terminal observation",
        "proprio_contract": {
            "history": PROPRIO_HISTORY,
            "term_dims": list(PROPRIO_TERM_DIMS),
            "frame_dim": sum(PROPRIO_TERM_DIMS),
        },
    }
    dimensions = RolloutDimensions()
    world_session = 0
    world_ids = (
        torch.arange(config.num_envs, device=env.device, dtype=torch.long) + config.world_id_offset
    )
    episode_ids = torch.zeros(config.num_envs, device=env.device, dtype=torch.long)
    episode_steps = torch.zeros_like(episode_ids)
    env_ids = torch.arange(config.num_envs, device=env.device, dtype=torch.long)

    try:
        with RolloutShardWriter(
            config.output_dir,
            dimensions=dimensions,
            shard_size=config.shard_size,
            include_diagnostics=True,
            metadata=metadata,
        ) as writer:
            collector_step = 0
            while writer.total_transitions < config.transitions:
                before = _snapshot(env, observations)
                with torch.inference_mode():
                    action = policy(observations)
                if not isinstance(action, torch.Tensor):
                    raise TypeError(
                        f"Frozen policy must return a Tensor, got {type(action).__name__}"
                    )
                if prepared.clip_actions is not None:
                    action = action.clamp(
                        -float(prepared.clip_actions),
                        float(prepared.clip_actions),
                    )
                raw_next, reward, terminated, truncated, _ = env.step(action)
                next_observations = _policy_observations(raw_next, config.num_envs)
                after = _snapshot(env, next_observations)
                done = terminated | truncated
                session_boundary = (collector_step + 1) % config.world_session_steps == 0
                synchronous_boundary = bool(done.any()) or session_boundary
                reset_boundary = torch.full_like(done, synchronous_boundary)
                action_term = env.action_manager.get_term("joint_pos")
                applied_action = getattr(action_term, "applied_action", None)
                if not isinstance(applied_action, torch.Tensor):
                    # The checkpoint's observation-history action has no delay or
                    # smoothing, so its applied policy-coordinate action is raw_action.
                    applied_action = action_term.raw_action
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
                    "world_id": world_ids,
                    "episode_id": episode_ids,
                    "episode_step": episode_steps,
                    "collector_step": torch.full_like(episode_ids, collector_step),
                    "env_id": env_ids,
                    "motion_id": before["motion_id"],
                    "motion_step": before["motion_step"],
                    "applied_action": applied_action,
                    "joint_target": action_term._processed_actions,
                    "joint_torque": after["proprio"][:, -dimensions.action :],
                    "robot_state": before["robot_state"],
                    "next_robot_state": after["robot_state"],
                    "reference_state": before["reference_state"],
                    "next_reference_state": after["reference_state"],
                }
                remaining = config.transitions - writer.total_transitions
                if remaining < config.num_envs:
                    batch = {name: value[:remaining] for name, value in batch.items()}
                writer.append(batch)
                collector_step += 1
                episode_steps += 1

                if session_boundary:
                    observations = _resample_static_worlds(env, config.num_envs)
                    world_session += 1
                    world_ids = (
                        torch.arange(config.num_envs, device=env.device, dtype=torch.long)
                        + config.world_id_offset
                        + world_session * config.num_envs
                    )
                    episode_ids.zero_()
                    episode_steps.zero_()
                elif synchronous_boundary:
                    observations = _reset_all(env, config.num_envs)
                    episode_ids += 1
                    episode_steps.zero_()
                else:
                    observations = next_observations
            return writer.close()
    finally:
        runtime.close()


def collector_config_json(config: MjlabCollectorConfig) -> str:
    return json.dumps(asdict(config), indent=2, sort_keys=True)
