"""Live frozen-tracker rollout with one immutable DR sample per vector slot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from intact_tracking.environment.runtime import create_runtime, prepare_rollout
from intact_tracking.forward_predictor_inputs import JointPositionTargetTransform

from .mjlab_adapter import (
    _clear_missing_motion_exclusions,
    _forward_predictor_snapshot,
    _policy_observations,
    _snapshot,
)

TRACKING_ERROR_NAMES = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
)


def _quaternion_error(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    dot = (first * second).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _tracking_error_snapshot(env: Any, snapshot: dict[str, torch.Tensor]) -> torch.Tensor:
    """Match SPTracking's instantaneous tracking-error names and units."""
    command = env.command_manager.get_term("motion")
    metrics = getattr(command, "metrics", {})
    if all(isinstance(metrics.get(name), torch.Tensor) for name in TRACKING_ERROR_NAMES):
        return torch.stack(tuple(metrics[name] for name in TRACKING_ERROR_NAMES), dim=-1)

    robot = snapshot["robot_state"]
    reference = snapshot["reference_state"]
    root_pos = torch.linalg.vector_norm(reference[:, :3] - robot[:, :3], dim=-1)
    root_rot = _quaternion_error(reference[:, 3:7], robot[:, 3:7])
    root_lin_vel = torch.linalg.vector_norm(reference[:, 7:10] - robot[:, 7:10], dim=-1)
    root_ang_vel = torch.linalg.vector_norm(reference[:, 10:13] - robot[:, 10:13], dim=-1)
    joint_pos = torch.linalg.vector_norm(reference[:, 13:42] - robot[:, 13:42], dim=-1)
    joint_vel = torch.linalg.vector_norm(reference[:, 42:71] - robot[:, 42:71], dim=-1)
    zeros = torch.zeros_like(root_pos)
    return torch.stack(
        (
            root_pos,
            root_rot,
            root_lin_vel,
            root_ang_vel,
            zeros,
            zeros,
            joint_pos,
            joint_vel,
        ),
        dim=-1,
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
    world_id_offset: int = 0
    stochastic_policy: bool = False
    randomize_initial_episode_phase: bool = True
    nominal_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.world_id_offset < 0:
            raise ValueError("world_id_offset must be non-negative")
        if not 0.0 <= self.nominal_fraction <= 1.0:
            raise ValueError("nominal_fraction must be in [0, 1]")
        nominal_count = self.num_envs * self.nominal_fraction
        if abs(nominal_count - round(nominal_count)) > 1.0e-8:
            raise ValueError(
                "num_envs * nominal_fraction must be an integer, got "
                f"{self.num_envs} * {self.nominal_fraction}"
            )
        if bool(self.motion_path) == bool(self.motion_file):
            raise ValueError("Provide exactly one of motion_path or motion_file")


@dataclass(frozen=True)
class PrivilegedDynamicsTargets:
    """Compact simulator-only labels for one fixed dynamics realization per world."""

    names: tuple[str, ...]
    values: torch.Tensor
    ignored_startup_events: tuple[str, ...]


def _entity_indices_and_names(
    env: Any,
    asset_cfg: Any,
    kind: str,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    asset = env.scene[asset_cfg.name]
    raw_ids = getattr(asset_cfg, f"{kind}_ids")
    all_names = tuple(str(name) for name in getattr(asset, f"{kind}_names"))
    if isinstance(raw_ids, slice):
        local_ids = torch.arange(len(all_names), device=env.device, dtype=torch.long)[raw_ids]
    else:
        local_ids = torch.as_tensor(raw_ids, device=env.device, dtype=torch.long)
    global_ids = getattr(asset.indexing, f"{kind}_ids").index_select(0, local_ids)
    names = tuple(all_names[int(index)] for index in local_ids.detach().cpu().tolist())
    return global_ids.to(dtype=torch.long), names


def _target_axes(params: Mapping[str, Any], default: tuple[int, ...]) -> tuple[int, ...]:
    axes = params.get("axes")
    if axes is not None:
        return tuple(int(axis) for axis in axes)
    ranges = params.get("ranges")
    if isinstance(ranges, Mapping):
        indexed = []
        for key in ranges:
            try:
                indexed.append(int(key))
            except (TypeError, ValueError):
                continue
        if indexed:
            return tuple(indexed)
    return default


def _expanded_and_default_field(env: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    actual = getattr(env.sim.model, name)
    default = env.sim.get_default_field(name).to(device=actual.device, dtype=actual.dtype)
    if actual.ndim == default.ndim:
        actual = actual.unsqueeze(0).expand(env.num_envs, *actual.shape)
    if actual.shape[0] != env.num_envs or actual.shape[1:] != default.shape:
        raise RuntimeError(
            f"Privileged dynamics field {name!r} has incompatible shapes: "
            f"{tuple(actual.shape)} vs default {tuple(default.shape)}"
        )
    return actual, default


def _safe_scale(actual: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    expanded_default = default.unsqueeze(0).expand_as(actual)
    return torch.where(
        expanded_default.abs() > 1.0e-8,
        actual / expanded_default,
        actual - expanded_default,
    )


def _capture_privileged_dynamics_targets(env: Any) -> PrivilegedDynamicsTargets:
    """Read compact causal DR factors after nominal-slot restoration.

    Action-chain nuisance variables such as encoder bias and joint-command
    offsets are intentionally excluded: the Forward Predictor already consumes
    the physical PD target produced after that chain.
    """

    blocks: list[torch.Tensor] = []
    names: list[str] = []
    ignored: list[str] = []

    def append(event_name: str, labels: list[str], values: torch.Tensor) -> None:
        values = values.detach().to(device=env.device, dtype=torch.float32)
        if values.ndim != 2 or values.shape != (env.num_envs, len(labels)):
            raise RuntimeError(
                f"Privileged event {event_name!r} produced {tuple(values.shape)} for "
                f"{len(labels)} labels"
            )
        if values.numel() and not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"Privileged event {event_name!r} produced non-finite labels")
        if labels:
            blocks.append(values)
            names.extend(f"{event_name}/{label}" for label in labels)

    startup_names = tuple(env.event_manager.active_terms.get("startup", ()))
    for event_name in startup_names:
        cfg = env.event_manager.get_term_cfg(event_name)
        func = cfg.func
        func_name = getattr(func, "__name__", type(func).__name__)
        params = cfg.params

        if func_name == "body_com_offset":
            asset_cfg = params["asset_cfg"]
            body_ids, body_names = _entity_indices_and_names(env, asset_cfg, "body")
            axes = _target_axes(params, (0, 1, 2))
            actual, default = _expanded_and_default_field(env, "body_ipos")
            selected = actual.index_select(1, body_ids) - default.index_select(0, body_ids)
            selected = selected[..., list(axes)]
            if bool(params.get("shared_random", False)):
                selected = selected[:, :1]
                body_names = ("shared",)
            labels = [
                f"com_offset/{body_name}/{axis_name}"
                for body_name in body_names
                for axis_name in ("xyz"[axis] for axis in axes)
            ]
            append(event_name, labels, selected.flatten(1))
            continue

        if func_name == "body_mass":
            asset_cfg = params["asset_cfg"]
            body_ids, body_names = _entity_indices_and_names(env, asset_cfg, "body")
            actual, default = _expanded_and_default_field(env, "body_mass")
            selected_actual = actual.index_select(1, body_ids)
            selected_default = default.index_select(0, body_ids)
            relative_delta = _safe_scale(selected_actual, selected_default) - 1.0
            if bool(params.get("shared_random", False)):
                relative_delta = relative_delta[:, :1]
                body_names = ("shared",)
            append(
                event_name,
                [f"relative_mass/{body_name}" for body_name in body_names],
                relative_delta,
            )
            continue

        if func_name == "geom_friction":
            asset_cfg = params["asset_cfg"]
            geom_ids, geom_names = _entity_indices_and_names(env, asset_cfg, "geom")
            axes = _target_axes(params, (0,))
            actual, _ = _expanded_and_default_field(env, "geom_friction")
            selected = actual.index_select(1, geom_ids)[..., list(axes)]
            if bool(params.get("shared_random", False)):
                selected = selected[:, :1]
                geom_names = ("shared",)
            labels = [
                f"friction/{geom_name}/{axis}"
                for geom_name in geom_names
                for axis in axes
            ]
            append(event_name, labels, selected.flatten(1))
            continue

        if func_name == "perturb_body_com":
            body_ids = func.global_body_ids.to(device=env.device, dtype=torch.long)
            actual, default = _expanded_and_default_field(env, "body_ipos")
            offsets = actual.index_select(1, body_ids) - default.index_select(0, body_ids)
            body_names = tuple(str(func.asset.body_names[index]) for index in func.body_ids)
            labels = [
                f"com_offset/{body_name}/{axis_name}"
                for body_name in body_names
                for axis_name in "xyz"
            ]
            append(event_name, labels, offsets.flatten(1))
            continue

        if func_name == "perturb_body_materials":
            geom_ids = func.geom_global_ids.to(device=env.device, dtype=torch.long)
            friction, _ = _expanded_and_default_field(env, "geom_friction")
            solref, _ = _expanded_and_default_field(env, "geom_solref")
            static = friction.index_select(1, geom_ids)[..., 0]
            time_constant = solref.index_select(1, geom_ids)[..., 0]
            damping_ratio = solref.index_select(1, geom_ids)[..., 1]
            geom_names = tuple(str(name) for name in func.geom_names)
            if func.homogeneous:
                static = static[:, :1]
                time_constant = time_constant[:, :1]
                damping_ratio = damping_ratio[:, :1]
                geom_names = ("shared",)
            append(
                event_name,
                [f"static_friction/{name}" for name in geom_names]
                + [f"solref_time_constant/{name}" for name in geom_names]
                + [f"solref_damping_ratio/{name}" for name in geom_names],
                torch.cat((static, time_constant, damping_ratio), dim=-1),
            )
            continue

        if func_name == "motor_params_implicit":
            motor_blocks: list[torch.Tensor] = []
            motor_labels: list[str] = []
            if func.kp_ctrl_ids.numel():
                gain, _ = _expanded_and_default_field(env, "actuator_gainprm")
                actual = gain[:, func.kp_ctrl_ids, 0]
                motor_blocks.append(_safe_scale(actual, func.kp_gain_def))
                motor_labels.extend(f"kp_scale/{name}" for name in func.kp_names)
            if func.kd_ctrl_ids.numel():
                bias, _ = _expanded_and_default_field(env, "actuator_biasprm")
                actual = bias[:, func.kd_ctrl_ids, 2]
                motor_blocks.append(_safe_scale(actual, func.kd_bias_def))
                motor_labels.extend(f"kd_scale/{name}" for name in func.kd_names)
            if func.arm_dof_ids.numel():
                armature, _ = _expanded_and_default_field(env, "dof_armature")
                actual = armature[:, func.arm_dof_ids]
                motor_blocks.append(_safe_scale(actual, func.arm_def))
                motor_labels.extend(f"armature_scale/{name}" for name in func.arm_names)
            if func.fric_dof_ids.numel():
                frictionloss, _ = _expanded_and_default_field(env, "dof_frictionloss")
                motor_blocks.append(frictionloss[:, func.fric_dof_ids])
                motor_labels.extend(f"frictionloss/{name}" for name in func.fric_names)
            if motor_blocks:
                append(event_name, motor_labels, torch.cat(motor_blocks, dim=-1))
            else:
                ignored.append(str(event_name))
            continue

        if func_name == "perturb_gravity":
            gravity = func.observe()
            append(event_name, ["gravity/x", "gravity/y", "gravity/z"], gravity)
            continue

        # These affect the policy-to-target or observation chain, not dynamics
        # after the physical PD target used by this predictor has been formed.
        if func_name in {"encoder_bias", "random_joint_offset"}:
            ignored.append(str(event_name))
            continue

        ignored.append(str(event_name))

    if not blocks:
        raise RuntimeError(
            "No supported persistent physics parameter is available for privileged "
            f"supervision; startup events={list(startup_names)}, ignored={ignored}"
        )
    values = torch.cat(blocks, dim=-1).contiguous().clone()
    if len(set(names)) != len(names):
        raise RuntimeError("Privileged dynamics target names must be unique")
    return PrivilegedDynamicsTargets(tuple(names), values, tuple(sorted(ignored)))


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


def _restore_nominal_physics(env: Any, env_ids: torch.Tensor) -> dict[str, float]:
    """Restore selected vector slots to the compiled model's nominal physics.

    Startup DR expands model fields along the environment dimension.  Writing
    the compiled defaults into only ``env_ids`` therefore creates nominal and
    DR worlds inside the same simulator without changing their task/reset
    behavior.  Encoder bias and action offsets live outside the model fields
    and are restored separately.
    """
    if env_ids.ndim != 1 or env_ids.dtype != torch.long:
        raise ValueError("nominal env_ids must be a 1-D long tensor")
    if env_ids.numel() == 0:
        return {
            "model_field_max_abs_error": 0.0,
            "encoder_bias_max_abs_error": 0.0,
            "dr_model_field_max_abs_difference": 0.0,
            "dr_encoder_bias_max_abs_difference": 0.0,
        }

    restored_fields: list[str] = []
    for raw_name in env.event_manager.domain_randomization_fields:
        name = str(raw_name)
        actual = getattr(env.sim.model, name, None)
        clone = getattr(actual, "clone", None)
        if not callable(clone):
            continue
        expanded = clone()
        if not isinstance(expanded, torch.Tensor):
            continue
        default = env.sim.get_default_field(name)
        if not isinstance(default, torch.Tensor):
            raise RuntimeError(f"Nominal default for model field {name!r} is not a Tensor")
        if expanded.ndim < 1 or expanded.shape[0] != env.num_envs:
            raise RuntimeError(
                f"DR model field {name!r} is not expanded over {env.num_envs} worlds: "
                f"{tuple(expanded.shape)}"
            )
        if expanded.shape[1:] != default.shape:
            raise RuntimeError(
                f"DR model field {name!r} has incompatible default shape: "
                f"{tuple(expanded.shape)} vs {tuple(default.shape)}"
            )
        actual[env_ids] = default
        restored_fields.append(name)

    robot = env.scene["robot"]
    encoder_bias = getattr(getattr(robot, "data", None), "encoder_bias", None)
    encoder_error = torch.zeros((), device=env_ids.device)
    if isinstance(encoder_bias, torch.Tensor):
        encoder_bias[env_ids] = 0.0
        if encoder_bias[env_ids].numel():
            encoder_error = encoder_bias[env_ids].abs().max()

    # Some checkpoints randomize a persistent joint command offset.  It is an
    # action-term buffer rather than an MuJoCo model field.
    try:
        action_term = env.action_manager.get_term("joint_pos")
    except (AttributeError, KeyError, ValueError):
        action_term = None
    joint_offset = getattr(action_term, "joint_offset", None)
    if isinstance(joint_offset, torch.Tensor) and joint_offset.shape[0] == env.num_envs:
        joint_offset[env_ids] = 0.0

    clear_cache = getattr(env.sim.model, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    forward = getattr(env.sim, "forward", None)
    if callable(forward):
        forward()

    if not restored_fields:
        raise RuntimeError("No expanded DR model field was available for nominal restoration")
    maximum_error = torch.zeros((), device=env_ids.device)
    dr_maximum_difference = torch.zeros((), device=env_ids.device)
    all_ids = torch.arange(env.num_envs, device=env_ids.device, dtype=torch.long)
    dr_ids = all_ids[~torch.isin(all_ids, env_ids)]
    for name in restored_fields:
        default = env.sim.get_default_field(name)
        expanded = getattr(env.sim.model, name).clone()
        restored = expanded[env_ids]
        if default.numel():
            maximum_error = torch.maximum(maximum_error, (restored - default).abs().max())
            if dr_ids.numel():
                dr_maximum_difference = torch.maximum(
                    dr_maximum_difference, (expanded[dr_ids] - default).abs().max()
                )
    if bool(maximum_error > 0.0) or bool(encoder_error > 0.0):
        raise RuntimeError(
            "Nominal physics restoration did not exactly match compiled defaults: "
            f"model={float(maximum_error):.6g}, encoder={float(encoder_error):.6g}"
        )
    return {
        "model_field_max_abs_error": float(maximum_error),
        "encoder_bias_max_abs_error": float(encoder_error),
        "dr_model_field_max_abs_difference": float(dr_maximum_difference),
        "dr_encoder_bias_max_abs_difference": (
            float(encoder_bias[dr_ids].abs().max())
            if isinstance(encoder_bias, torch.Tensor) and dr_ids.numel()
            else 0.0
        ),
    }


def _randomize_initial_episode_phases(env: Any, seed: int) -> dict[str, int]:
    """Desynchronize timeout resets without changing robot or reference state."""
    episode_length = getattr(env, "episode_length_buf", None)
    maximum = int(getattr(env, "max_episode_length", 0))
    if not isinstance(episode_length, torch.Tensor) or maximum < 1:
        raise RuntimeError("Rollout environment does not expose a valid episode timeout buffer")
    generator = torch.Generator(device=episode_length.device).manual_seed(int(seed))
    phases = torch.randint(
        low=0,
        high=maximum,
        size=episode_length.shape,
        dtype=episode_length.dtype,
        device=episode_length.device,
        generator=generator,
    )
    episode_length.copy_(phases)
    return {
        "minimum": int(phases.min().item()),
        "maximum": int(phases.max().item()),
        "unique": int(torch.unique(phases).numel()),
    }


def _verify_predictor_action_target(
    expected_target: torch.Tensor,
    simulator_target: torch.Tensor | None,
    transition_boundary: torch.Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> float | None:
    """Verify targets only for slots without an in-step discontinuity.

    MJLab clears or rewrites actuator targets during an in-step auto-reset or
    motion resample. Comparing those values with the target applied before the
    discontinuity would report a false action-chain mismatch. Returning ``None``
    defers the one-time check when every slot crossed a boundary on this step.
    """

    if not isinstance(simulator_target, torch.Tensor) or simulator_target.shape != (
        expected_target.shape
    ):
        raise RuntimeError(
            "Simulator does not expose the physical joint target needed to verify the "
            "Forward Predictor action chain"
        )
    if (
        transition_boundary.dtype != torch.bool
        or transition_boundary.shape != expected_target.shape[:1]
    ):
        raise RuntimeError(
            "Forward Predictor action-target verification requires one boolean boundary "
            f"flag per environment, got {tuple(transition_boundary.shape)}"
        )
    valid = ~transition_boundary
    if not bool(valid.any()):
        return None
    maximum_error = (simulator_target[valid] - expected_target[valid]).abs().max()
    if not torch.isfinite(maximum_error) or bool(maximum_error > tolerance):
        raise RuntimeError(
            "External Forward Predictor action transform does not match the simulator "
            "joint target on non-boundary environments; "
            f"max_abs_error={float(maximum_error):.6g}"
        )
    return float(maximum_error)


def _read_motion_resample_boundary(
    motion_command: Any,
    environment_boundary: torch.Tensor,
) -> torch.Tensor:
    """Snapshot the command's mid-episode motion-resample pulse.

    A motion can end and be replaced inside ``env.step`` without terminating or
    truncating the environment.  The command then teleports the robot/reference
    state and rewrites actuator targets, so that transition is just as much a
    causal boundary as an environment reset.
    """

    boundary = getattr(motion_command, "motion_resample_boundary", None)
    if boundary is None:
        return torch.zeros_like(environment_boundary)
    if (
        not isinstance(boundary, torch.Tensor)
        or boundary.dtype != torch.bool
        or boundary.shape != environment_boundary.shape
        or boundary.device != environment_boundary.device
    ):
        shape = tuple(boundary.shape) if isinstance(boundary, torch.Tensor) else None
        dtype = boundary.dtype if isinstance(boundary, torch.Tensor) else None
        device = boundary.device if isinstance(boundary, torch.Tensor) else None
        raise RuntimeError(
            "Motion command must expose one boolean motion_resample_boundary flag "
            "per environment on the rollout device; "
            f"shape={shape}, dtype={dtype}, device={device}"
        )
    # The command clears this pulse at the beginning of its next update.  Keep a
    # stable copy for replay and episode bookkeeping performed after env.step.
    return boundary.detach().clone()


class FixedDRTrackerRollout:
    """Step MJLab with a frozen tracker and fixed per-environment physics DR.

    MJLab applies ``startup`` events while the environment is constructed. All
    other event modes are removed before construction, and this class never
    reapplies startup events. Per-slot auto-resets therefore resample motion and
    robot state while preserving every slot's physics parameters.
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
        # Online replay discards transitions marked as reset boundaries, so it
        # does not need the true terminal observation. Let MJLab reset only the
        # completed slots inside ``step``. A synchronous all-environment reset
        # makes the probability of obtaining a full-context causal window
        # collapse as ``num_envs`` grows (for example at 4096 environments).
        prepared.env.auto_reset = True
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
            motion_command = self.env.command_manager.get_term("motion")
            self.motion_command = motion_command
            motion_files = getattr(motion_command, "motion_files", None)
            if motion_files is None:
                motion_store = getattr(motion_command, "motion_store", None)
                motion_files = getattr(motion_store, "motion_files", None)
            if not motion_files:
                raise RuntimeError("Motion command does not expose its motion-file mapping")
            self.motion_files = tuple(
                str(Path(path).expanduser().resolve()) for path in motion_files
            )
            self.disabled_startup_reset_callbacks = _disable_startup_reset_callbacks(self.env)
            self.nominal_count = int(round(config.num_envs * config.nominal_fraction))
            self.nominal_env_ids = torch.arange(
                self.nominal_count, device=self.env.device, dtype=torch.long
            )
            self.is_nominal = torch.zeros(config.num_envs, device=self.env.device, dtype=torch.bool)
            self.is_nominal[self.nominal_env_ids] = True
            self.nominal_restore_metrics: dict[str, float] | None = None
            if self.nominal_count:
                self.nominal_restore_metrics = _restore_nominal_physics(
                    self.env, self.nominal_env_ids
                )
            action_term = self.env.action_manager.get_term("joint_pos")
            self.predictor_action_transform: JointPositionTargetTransform | None = None
            self.predictor_action_transform_error: str | None = None
            try:
                self.predictor_action_transform = JointPositionTargetTransform.from_mjlab(
                    self.env, action_term
                )
            except ValueError as error:
                # Other training entry points still use this rollout class with
                # stateful SP action terms.  Only the dedicated Forward
                # Predictor requires the external memoryless transform.
                self.predictor_action_transform_error = str(error)
            self.predictor_action_target_verified = False
            self.predictor_action_target_max_abs_error: float | None = None
            self._fixed_dr_model_fields = _capture_randomized_model_fields(self.env)
            if not self._fixed_dr_model_fields:
                raise RuntimeError("Startup DR did not expose any randomized physics fields")
            privileged = _capture_privileged_dynamics_targets(self.env)
            self.privileged_dynamics_names = privileged.names
            self.privileged_dynamics = privileged.values
            self.ignored_privileged_startup_events = privileged.ignored_startup_events
            self.dr_invariance_checks = 0
            self.wrapped = self._runtime.wrapped
            self.policy = self._runtime.policy
            self.initial_episode_phase_summary: dict[str, int] | None = None
            if config.randomize_initial_episode_phase:
                self.initial_episode_phase_summary = _randomize_initial_episode_phases(
                    self.env, config.seed
                )
            self.observations = self.wrapped.get_observations()
            self.world_ids = torch.arange(
                config.world_id_offset,
                config.world_id_offset + config.num_envs,
                device=self.env.device,
                dtype=torch.long,
            )
            self.episode_ids = torch.zeros_like(self.world_ids)
            self.episode_steps = torch.zeros_like(self.world_ids)
            self.env_ids = torch.arange(config.num_envs, device=self.env.device, dtype=torch.long)
            self.collector_step = 0
            self.reset_events = 0
            self.environments_reset = 0
            # Retained in logs/checkpoints for backward compatibility. Online
            # rollout now uses asynchronous per-slot resets, so this stays zero.
            self.synchronous_resets = 0
            self._motion_ids_seen = torch.empty(0, dtype=torch.long, device=self.env.device)

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
    def motions_seen_count(self) -> int:
        return self._motion_ids_seen.numel()

    @property
    def motion_ids_seen(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._motion_ids_seen.tolist())

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_task_id": self.checkpoint_task_id,
            "tracker_frozen": True,
            "world_id_offset": self.config.world_id_offset,
            "startup_events": self.startup_events,
            "removed_non_startup_events": self.removed_non_startup_events,
            "disabled_startup_reset_callbacks": self.disabled_startup_reset_callbacks,
            "fixed_dr_model_fields": sorted(self._fixed_dr_model_fields),
            "privileged_dynamics_dim": len(self.privileged_dynamics_names),
            "privileged_dynamics_names": list(self.privileged_dynamics_names),
            "ignored_privileged_startup_events": list(
                self.ignored_privileged_startup_events
            ),
            "nominal_fraction": self.config.nominal_fraction,
            "nominal_world_count_per_rank": self.nominal_count,
            "dr_world_count_per_rank": self.num_envs - self.nominal_count,
            "nominal_world_local_ids": list(range(self.nominal_count)),
            "nominal_restore_metrics": self.nominal_restore_metrics,
            "predictor_action_transform": (
                self.predictor_action_transform.contract
                if self.predictor_action_transform is not None
                else {"available": False, "error": self.predictor_action_transform_error}
            ),
            "cleared_missing_motion_exclusions": self.cleared_motion_exclusions,
            "domain_randomization_contract": (
                "selected slots restored to compiled nominal physics; remaining startup DR "
                "fixed per vector slot; neither group is resampled"
            ),
            "motion_contract": "random motion resampling at initialization and reset",
            "motion_file_count_per_rank": len(self.motion_files),
            "reset_contract": (
                "asynchronous per-slot auto-reset; startup events are never reapplied"
            ),
            "initial_episode_phase_randomized": self.config.randomize_initial_episode_phase,
            "initial_episode_phase_summary": self.initial_episode_phase_summary,
            "reset_window_contract": (
                "the teleport boundary is excluded; the post-reset state may start a query"
            ),
            "tracking_error_names": list(TRACKING_ERROR_NAMES),
        }

    @property
    def policy_observation_dim(self) -> int:
        return int(self._runtime.actor.policy_input_dim)

    @property
    def action_clip(self) -> float | None:
        return self._clip_actions

    def _assert_fixed_dr(self, env_ids: torch.Tensor | None = None) -> None:
        unchanged: torch.Tensor | None = None
        for name, expected in self._fixed_dr_model_fields.items():
            actual = getattr(self.env.sim.model, name)
            if (
                env_ids is not None
                and actual.ndim > 0
                and expected.ndim > 0
                and actual.shape[0] == self.num_envs
                and expected.shape[0] == self.num_envs
            ):
                actual = actual[env_ids]
                expected = expected[env_ids]
            field_unchanged = torch.eq(actual, expected).all()
            unchanged = (
                field_unchanged
                if unchanged is None
                else torch.logical_and(unchanged, field_unchanged)
            )
        if unchanged is None or not bool(unchanged):
            raise RuntimeError("A fixed physics DR field changed during an episode reset")
        self.dr_invariance_checks += 1

    def _record_motion_ids(self, motion_ids: torch.Tensor) -> None:
        self._motion_ids_seen = torch.unique(
            torch.cat((self._motion_ids_seen, motion_ids.detach().to(dtype=torch.long).flatten()))
        )

    def step(
        self,
        residual_action_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        include_policy_observation: bool = True,
        predictor_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Collect one transition, optionally adding a learned residual action."""
        if self.closed:
            raise RuntimeError("Cannot step a closed rollout")
        if predictor_only and residual_action_fn is not None:
            raise ValueError("predictor_only collection cannot request a residual action")
        before = (
            _forward_predictor_snapshot(self.env)
            if predictor_only
            else _snapshot(self.env, self.observations)
        )
        if self.collector_step == 0:
            self._record_motion_ids(before["motion_id"])
        actor = getattr(getattr(self, "_runtime", None), "actor", None)
        needs_policy_observation = (
            include_policy_observation and not predictor_only
        ) or residual_action_fn is not None
        with torch.inference_mode():
            policy_observation = None
            if needs_policy_observation:
                policy_observation = (
                    actor.get_latent(self.observations)
                    if actor is not None
                    else before["observation"]
                )
            tracker_action = self.policy(self.observations)
        if not isinstance(tracker_action, torch.Tensor):
            raise TypeError(
                f"Frozen tracker must return a Tensor, got {type(tracker_action).__name__}"
            )
        residual_action = None if predictor_only else torch.zeros_like(tracker_action)
        if residual_action_fn is not None:
            assert policy_observation is not None
            with torch.inference_mode():
                residual_action = residual_action_fn(policy_observation, tracker_action)
            if not isinstance(residual_action, torch.Tensor):
                raise TypeError("Residual action callback must return a Tensor")
            if residual_action.shape != tracker_action.shape:
                raise ValueError(
                    "Residual action shape must match tracker action: "
                    f"{tuple(residual_action.shape)} vs {tuple(tracker_action.shape)}"
                )
        action = tracker_action if residual_action is None else tracker_action + residual_action
        if self._clip_actions is not None:
            action = action.clamp(-float(self._clip_actions), float(self._clip_actions))
        action_transform = getattr(self, "predictor_action_transform", None)
        joint_target = action_transform(action) if action_transform is not None else None

        raw_next, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated
        motion_resample_boundary = _read_motion_resample_boundary(
            getattr(self, "motion_command", None),
            done,
        )
        reset_boundary = done | motion_resample_boundary
        if joint_target is not None and not getattr(
            self, "predictor_action_target_verified", False
        ):
            simulator_target = getattr(self.env.scene["robot"].data, "joint_pos_target", None)
            maximum_error = _verify_predictor_action_target(
                joint_target,
                simulator_target,
                reset_boundary,
            )
            if maximum_error is not None:
                self.predictor_action_target_max_abs_error = maximum_error
                self.predictor_action_target_verified = True
        next_observations = _policy_observations(raw_next, self.num_envs)
        after = (
            _forward_predictor_snapshot(self.env)
            if predictor_only
            else _snapshot(self.env, next_observations)
        )
        batch = {
            "reset_boundary": reset_boundary,
            "world_id": self.world_ids,
            "dynamics_id": self.world_ids,
            "privileged_dynamics": self.privileged_dynamics,
            "episode_id": self.episode_ids.clone(),
            "episode_step": self.episode_steps.clone(),
            "motion_id": before["motion_id"],
            "motion_step": before["motion_step"],
        }
        if not predictor_only:
            assert residual_action is not None
            batch.update(
                {
                    "proprio": before["proprio"],
                    "next_proprio": after["proprio"],
                    "observation": before["observation"],
                    "next_observation": after["observation"],
                    "reference_observation": before["reference_observation"],
                    "next_reference_observation": after["reference_observation"],
                    "tracker_action": tracker_action,
                    "residual_action": residual_action,
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "is_nominal": self.is_nominal,
                    "collector_step": torch.full_like(self.episode_ids, self.collector_step),
                    "env_id": self.env_ids,
                }
            )
        if policy_observation is not None:
            batch["policy_observation"] = policy_observation
        if joint_target is not None:
            batch["joint_target"] = joint_target
        if "robot_state" in before and "robot_state" in after:
            batch.update(
                {
                    "robot_state": before["robot_state"],
                    "next_robot_state": after["robot_state"],
                }
            )
        if "reference_state" in before and "reference_state" in after:
            batch.update(
                {
                    "reference_state": before["reference_state"],
                    "next_reference_state": after["reference_state"],
                    "tracking_error": _tracking_error_snapshot(self.env, after),
                }
            )
        if "foot" in before and "foot" in after:
            batch.update(
                {
                    "foot": before["foot"],
                    "next_foot": after["foot"],
                }
            )
        contact_fields = ("contact_force", "contact_binary")
        if all(name in before and name in after for name in contact_fields):
            batch.update(
                {
                    "contact_force": before["contact_force"],
                    "next_contact_force": after["contact_force"],
                    "contact_binary": before["contact_binary"],
                    "next_contact_binary": after["contact_binary"],
                }
            )

        self.collector_step += 1
        self.episode_steps += 1
        done_ids = done.nonzero(as_tuple=False).flatten()
        done_count = int(done_ids.numel())
        boundary_ids = reset_boundary.nonzero(as_tuple=False).flatten()
        boundary_count = int(boundary_ids.numel())
        if done_count:
            # MJLab has already reset precisely these slots and returned their
            # post-reset observations. Startup DR must remain fixed across the
            # environment reset.
            self._assert_fixed_dr(done_ids)
            self.reset_events += 1
            self.environments_reset += done_count
        if boundary_count:
            # Both environment resets and in-step motion resampling teleport the
            # state. Exclude that transition from replay and start a fresh causal
            # episode at the already-returned post-boundary state.
            self._record_motion_ids(after["motion_id"][boundary_ids])
            self.episode_ids[boundary_ids] += 1
            self.episode_steps[boundary_ids] = 0
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
