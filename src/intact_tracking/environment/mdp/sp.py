from __future__ import annotations

from typing import Any, Literal, cast

import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_mul,
)
from mjlab.utils.lab_api.string import resolve_matching_names

from intact_tracking.environment.mdp.keypoints import SemanticKeypointResolver
from intact_tracking.environment.mdp.motion_fk import (
    quat_apply as fk_quat_apply,
)
from intact_tracking.environment.mdp.motion_fk import (
    quat_mul as fk_quat_mul,
)
from intact_tracking.environment.mdp.multi_commands import MultiMotionCommand

if False:
    from mjlab.envs import ManagerBasedRlEnv


TEACHER_STEPS = (0, 1, 2, 4, 8, 12, 16, 20, -1, -2, -4, -8)
STUDENT_STEPS = (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16)
POLICY_HISTORY_STEPS = (0, 1, 2, 3, 4, 8, 12, 16, 20)
PRIV_HISTORY_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8)

SP_REQUIRED_BODY_NAMES = (
    "pelvis",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_mimic",
    "left_shoulder_yaw_link",
    "left_wrist_roll_link",
    "left_hand_mimic",
    "right_shoulder_yaw_link",
    "right_wrist_roll_link",
    "right_hand_mimic",
)

SP_KEYPOINT_BODY_NAMES = (
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "head_mimic",
    "left_shoulder_yaw_link",
    "left_wrist_roll_link",
    "left_hand_mimic",
    "right_shoulder_yaw_link",
    "right_wrist_roll_link",
    "right_hand_mimic",
)

SP_FEET_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
SP_FEET_TOE_BODY_NAMES = (
    "left_ankle_roll_toe_link",
    "right_ankle_roll_toe_link",
)
SP_TERMINATION_BODY_NAMES = (
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "head_mimic",
    "left_hand_mimic",
    "right_hand_mimic",
)
SP_TERMINATION_KILL_FRAMES = 5


class substep_tracking_cache:
    """Source-compatible substep state shared by SP observations and rewards.

    the reference task samples joint position/velocity after every physics substep
    and uses the final two samples for uncorrupted histories and joint-velocity
    regularization.  It also derives foot contact with a per-control-step vote.
    MJLab exposes the same hook through a ``per_substep`` metrics term; keeping
    this as an opt-in term avoids changing non-SP environments.
    """

    def __init__(self, cfg: MetricsTermCfg, env: "ManagerBasedRlEnv"):
        self.env = env
        self.asset = env.scene["robot"]
        self.sensor = env.scene[str(cfg.params.get("sensor_name", "contact_forces"))]
        self.decimation = max(int(env.cfg.decimation), 1)
        self._joint_pos = torch.zeros(
            (env.num_envs, 2, len(self.asset.joint_names)),
            dtype=self.asset.data.joint_pos.dtype,
            device=env.device,
        )
        self._joint_vel = torch.zeros_like(self._joint_pos)
        primary_names = tuple(getattr(self.sensor, "primary_names", ()))
        self.contact_count = len(primary_names) or len(SP_FEET_BODY_NAMES)
        self._contact_found = torch.zeros(
            (env.num_envs, self.contact_count, self.decimation),
            dtype=torch.bool,
            device=env.device,
        )
        torque_sensor_prefix = str(cfg.params.get("joint_torque_sensor_prefix", ""))
        self._joint_torque_sensors = (
            tuple(
                env.scene[f"robot/{torque_sensor_prefix}{joint_name}"]
                for joint_name in self.asset.joint_names
            )
            if torque_sensor_prefix
            else ()
        )
        self._joint_torque_samples = (
            torch.zeros(
                (env.num_envs, len(self._joint_torque_sensors), self.decimation),
                dtype=self.asset.data.joint_pos.dtype,
                device=env.device,
            )
            if self._joint_torque_sensors
            else None
        )
        self.current_contact = torch.zeros(
            (env.num_envs, self.contact_count), dtype=torch.bool, device=env.device
        )
        self.first_contact = torch.zeros_like(self.current_contact)
        self.first_air = torch.zeros_like(self.current_contact)
        self._metric_value = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._substep_count = 0
        self._contact_finalized_step: int | None = None
        # Explicitly attach the cache to the environment rather than relying on
        # metric-manager internals; this makes it reusable by any SP term.
        env._sp_substep_tracking_cache = self

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._joint_pos[env_ids] = 0.0
        self._joint_vel[env_ids] = 0.0
        self._contact_found[env_ids] = False
        if self._joint_torque_samples is not None:
            self._joint_torque_samples[env_ids] = 0.0
        self.current_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.first_air[env_ids] = False
        self._contact_finalized_step = None

    def _read_contact_found(self) -> torch.Tensor:
        found = getattr(self.sensor.data, "found", None)
        if not isinstance(found, torch.Tensor):
            contact_time = getattr(self.sensor.data, "current_contact_time", None)
            if not isinstance(contact_time, torch.Tensor):
                raise RuntimeError("SP substep cache requires contact sensor found data.")
            return contact_time > self.env.physics_dt
        found = found > 0
        if found.ndim != 2:
            raise ValueError(
                "Contact sensor found data must have shape [num_envs, contacts], got "
                f"{tuple(found.shape)}"
            )
        if found.shape[1] == self.contact_count:
            return found
        if found.shape[1] % self.contact_count == 0:
            return found.reshape(self.env.num_envs, self.contact_count, -1).any(dim=-1)
        raise ValueError(
            "SP contact sensor width does not match its primary bodies: "
            f"{found.shape[1]} vs {self.contact_count}"
        )

    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        del env
        # The source implementation stores substep % 2, leaving precisely the
        # final two physical samples after a decimated control step.
        joint_slot = self._substep_count % 2
        contact_slot = self._substep_count % self.decimation
        self._joint_pos[:, joint_slot] = self.asset.data.joint_pos
        self._joint_vel[:, joint_slot] = self.asset.data.joint_vel
        self._contact_found[:, :, contact_slot] = self._read_contact_found()
        if self._joint_torque_samples is not None:
            torque = torch.cat(
                [
                    sensor.data.reshape(self.env.num_envs, -1)
                    for sensor in self._joint_torque_sensors
                ],
                dim=-1,
            )
            self._joint_torque_samples[:, :, contact_slot] = torque
        self._substep_count += 1
        return self._metric_value

    def joint_state_average(self, field_name: Literal["joint_pos", "joint_vel"]) -> torch.Tensor:
        if field_name == "joint_pos":
            return self._joint_pos.mean(dim=1)
        if field_name == "joint_vel":
            return self._joint_vel.mean(dim=1)
        raise ValueError(f"Unsupported joint state field: {field_name}")

    def contact_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source-style majority contact plus transition indicators."""
        step = int(self.env.common_step_counter)
        if self._contact_finalized_step != step:
            # Match the reference exactly: with even decimation a tied vote counts
            # as contact (``votes >= decimation // 2``).
            current = self._contact_found.sum(dim=-1) >= (self.decimation // 2)
            previous = self.current_contact.clone()
            self.first_contact[:] = (~previous) & current
            self.first_air[:] = previous & (~current)
            self.current_contact[:] = current
            self._contact_finalized_step = step
        return self.current_contact, self.first_contact, self.first_air

    def joint_torque_average(self) -> torch.Tensor:
        """Return the mean jointactuatorfrc measurement over one control step."""
        if self._joint_torque_samples is None:
            raise RuntimeError("SP substep cache was not configured with joint torque sensors.")
        return self._joint_torque_samples.mean(dim=-1)


def _substep_cache(env: "ManagerBasedRlEnv") -> substep_tracking_cache | None:
    cache = getattr(env, "_sp_substep_tracking_cache", None)
    return cache if isinstance(cache, substep_tracking_cache) else None


def _command(env: "ManagerBasedRlEnv", command_name: str) -> MultiMotionCommand:
    return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


def _steps(horizon: Literal["teacher", "student"]) -> tuple[int, ...]:
    return STUDENT_STEPS if horizon == "student" else TEACHER_STEPS


def _gather(
    env: "ManagerBasedRlEnv",
    command_name: str,
    field_name: str,
    steps: tuple[int, ...],
) -> torch.Tensor:
    cmd = _command(env, command_name)
    gather_reference = getattr(cmd, "gather_reference", None)
    if callable(gather_reference):
        return gather_reference(field_name, steps)
    time_steps = cmd.time_steps[:, None] + torch.as_tensor(
        steps, device=env.device, dtype=torch.long
    )
    return cmd._gather_motion_field(field_name, cmd.motion_idx, time_steps)


def _gather_horizon(
    env: "ManagerBasedRlEnv",
    command_name: str,
    field_name: str,
    steps: tuple[int, ...],
    horizon: Literal["teacher", "student"],
) -> torch.Tensor:
    cmd = _command(env, command_name)
    gather_student = getattr(cmd, "gather_student_reference", None)
    if horizon == "student" and callable(gather_student):
        return gather_student(field_name, steps)
    return _gather(env, command_name, field_name, steps)


def _gather_current(env: "ManagerBasedRlEnv", command_name: str, field_name: str) -> torch.Tensor:
    return _gather(env, command_name, field_name, (0,))[:, 0]


def _root_motion(
    env: "ManagerBasedRlEnv",
    command_name: str,
    field_name: str,
    steps: tuple[int, ...],
    horizon: Literal["teacher", "student"] = "teacher",
    root_body_name: str | None = None,
) -> torch.Tensor:
    cmd = _command(env, command_name)
    anchor_body_name = getattr(cmd.cfg, "anchor_body_name", None)
    if anchor_body_name is None:
        anchor_body_name = cmd.cfg.body_names[cmd.motion_anchor_body_index]
    is_primary_root = root_body_name is None or _basename(root_body_name) == _basename(
        anchor_body_name
    )
    if is_primary_root:
        method_name = (
            "gather_student_root_reference" if horizon == "student" else "gather_root_reference"
        )
        gather_root = getattr(cmd, method_name, None)
        if callable(gather_root):
            return gather_root(field_name, steps)
    data = _gather_horizon(env, command_name, field_name, steps, horizon)
    root_index = cmd.motion_anchor_body_index
    if root_body_name is not None:
        root_index = tuple(_basename(name) for name in cmd.cfg.body_names).index(
            _basename(root_body_name)
        )
    return data[:, :, root_index]


def _root_motion_current(
    env: "ManagerBasedRlEnv",
    command_name: str,
    field_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    return _root_motion(env, command_name, field_name, (0,), root_body_name=root_body_name)[:, 0]


def _default_keypoint_specs() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": name,
            "asset_body_name": name,
            "reference_body_name": name,
        }
        for name in SP_KEYPOINT_BODY_NAMES
    )


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
    # Match active_adaptation.utils.math.quat_to_rot6d exactly: encode the first
    # two matrix columns, with each column contiguous in the flattened tensor.
    # Reshaping the untransposed slice would instead interleave columns by row.
    return matrix_from_quat(quat)[..., :, :2].transpose(-2, -1).reshape(*quat.shape[:-1], 6)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_in_frame(frame_quat: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """Express ``quat_w`` in ``frame_quat`` (frame^-1 * world)."""
    if frame_quat.shape[:-1] != quat_w.shape[:-1]:
        shape = torch.broadcast_shapes(frame_quat.shape[:-1], quat_w.shape[:-1])
        frame_quat = frame_quat.expand(*shape, 4)
        quat_w = quat_w.expand(*shape, 4)
    return quat_mul(_quat_conjugate(frame_quat), quat_w)


def _quat_delta(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """Match active_adaptation quat_delta: target * current^-1."""
    if current_quat.shape[:-1] != target_quat.shape[:-1]:
        shape = torch.broadcast_shapes(current_quat.shape[:-1], target_quat.shape[:-1])
        current_quat = current_quat.expand(*shape, 4)
        target_quat = target_quat.expand(*shape, 4)
    return quat_mul(target_quat, _quat_conjugate(current_quat))


def _exp_sigma(error: torch.Tensor, sigma: list[float] | tuple[float, ...]) -> torch.Tensor:
    if not sigma:
        raise ValueError("sigma must contain at least one value.")
    rewards = [torch.exp(-error / float(s)) for s in sigma]
    return sum(rewards) / len(rewards)


def _body_indices(names: tuple[str, ...], selected: tuple[str, ...]) -> list[int]:
    return [names.index(name) for name in selected]


def _basename(name: str) -> str:
    return name.split("/")[-1]


def _termination_buffer(
    env: "ManagerBasedRlEnv", cmd: MultiMotionCommand, name: str
) -> torch.Tensor:
    buffer = getattr(cmd, name, None)
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.shape != (env.num_envs,)
        or buffer.device != torch.device(env.device)
    ):
        buffer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        setattr(cmd, name, buffer)
    return buffer


def _continuous_termination(
    env: "ManagerBasedRlEnv", trigger: torch.Tensor, buffer: torch.Tensor
) -> torch.Tensor:
    trigger = trigger.reshape(env.num_envs, -1).any(dim=1)
    buffer.add_(trigger.to(buffer.dtype))
    buffer.masked_fill_(~trigger, 0)
    buffer.clamp_(max=max(int(SP_TERMINATION_KILL_FRAMES), 1))
    return buffer >= int(SP_TERMINATION_KILL_FRAMES)


def _apply_termination_warmup(
    env: "ManagerBasedRlEnv", cmd: MultiMotionCommand, value: torch.Tensor
) -> torch.Tensor:
    """Mirror the reference task's initial failure-termination guard."""
    warmup_steps = max(int(getattr(cmd.cfg, "termination_warmup_steps", 0)), 0)
    if warmup_steps == 0:
        return value
    episode_length = getattr(env, "episode_length_buf", None)
    if not isinstance(episode_length, torch.Tensor):
        return value
    return value & (episode_length > warmup_steps)


def _body_z_values(
    env: "ManagerBasedRlEnv",
    command_name: str,
    body_z_terminate_patterns: tuple[str, ...],
) -> tuple[MultiMotionCommand, torch.Tensor, torch.Tensor]:
    cmd = _command(env, command_name)
    asset = env.scene["robot"]
    body_ids, body_names = asset.find_bodies(body_z_terminate_patterns, preserve_order=True)
    motion_body_names = tuple(cmd.cfg.body_names)
    motion_ids = [motion_body_names.index(_basename(name)) for name in body_names]
    motion_ids_t = torch.as_tensor(motion_ids, dtype=torch.long, device=env.device)
    asset_ids_t = torch.as_tensor(body_ids, dtype=torch.long, device=env.device)
    gather_body_pos = getattr(cmd, "gather_reference_body_pos_w", None)
    if callable(gather_body_pos):
        target_z = gather_body_pos(motion_ids_t, (0,), include_env_origins=False)[:, 0, :, 2]
    else:
        target_z = _gather_current(env, command_name, "body_pos_w")[:, motion_ids_t, 2]
    current_z = asset.data.body_link_pos_w[:, asset_ids_t, 2]
    return cmd, target_z, current_z


def _robot_body_indices(asset, selected: tuple[str, ...], device: str) -> torch.Tensor:
    ids = asset.find_bodies(selected, preserve_order=True)[0]
    return torch.as_tensor(ids, device=device, dtype=torch.long)


def _joint_target_ids(asset) -> torch.Tensor:
    return torch.arange(len(asset.joint_names), device=asset.data.joint_pos.device)


def _projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    gravity = quat.new_tensor((0.0, 0.0, -1.0))
    projected = _quat_apply_inverse(quat, gravity.expand(*quat.shape[:-1], 3))
    return projected / projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _uniform_noise(value: torch.Tensor, std: float) -> torch.Tensor:
    if float(std) <= 0.0:
        return value
    return value + (torch.rand_like(value) * 2.0 - 1.0) * float(std)


def _spherical_noise(value: torch.Tensor, std: float) -> torch.Tensor:
    if float(std) <= 0.0:
        return value
    direction = torch.randn_like(value)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    radius = torch.rand_like(value[..., :1]) * float(std)
    return value + direction * radius


def _perturb_quaternion(quat: torch.Tensor, angle_std: float) -> torch.Tensor:
    if float(angle_std) <= 0.0:
        return quat
    axis = torch.randn_like(quat[..., 1:])
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    angle = (torch.rand_like(quat[..., :1]) * 2.0 - 1.0) * float(angle_std)
    half_angle = angle * 0.5
    delta = torch.cat((torch.cos(half_angle), axis * torch.sin(half_angle)), dim=-1)
    return quat_mul(delta, quat)


def _canonical_joint_ids(env: "ManagerBasedRlEnv") -> torch.Tensor | None:
    try:
        asset = env.scene["robot"]
    except (AttributeError, KeyError):
        return None
    configured_order = tuple(getattr(getattr(asset, "cfg", None), "joint_name_order", ()))
    joint_names = tuple(asset.joint_names)
    if not configured_order:
        configured_order = joint_names
    if set(configured_order) != set(joint_names) or len(configured_order) != len(joint_names):
        raise ValueError(
            "Configured canonical joint order must match the robot joint names; "
            f"order={configured_order}, joints={joint_names}"
        )
    return torch.as_tensor(
        [joint_names.index(name) for name in configured_order],
        dtype=torch.long,
        device=env.device,
    )


def _canonical_joint_tensor(env: "ManagerBasedRlEnv", value: torch.Tensor) -> torch.Tensor:
    ids = _canonical_joint_ids(env)
    return value if ids is None else value.index_select(-1, ids)


def _current_joint_observation(
    env: "ManagerBasedRlEnv",
    command_name: str,
    field_name: Literal["joint_pos", "joint_vel"],
    noise_std: float,
) -> torch.Tensor:
    command_manager = getattr(env, "command_manager", None)
    get_term = getattr(command_manager, "get_term", None)
    command = get_term(command_name) if callable(get_term) else None
    getter = getattr(command, f"get_shared_noisy_{field_name}", None)
    if callable(getter):
        return getter(noise_std)
    return _uniform_noise(getattr(env.scene["robot"].data, field_name), noise_std)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if quat.shape[:-1] != vec.shape[:-1]:
        quat = quat.expand(*vec.shape[:-1], 4)
    return quat_apply_inverse(quat, vec)


def _action_offset_like(env: "ManagerBasedRlEnv", reference: torch.Tensor) -> torch.Tensor:
    term = _joint_action_term(env)
    joint_offset = getattr(term, "joint_offset", None)
    target_ids = getattr(term, "target_ids", None)
    if isinstance(joint_offset, torch.Tensor) and isinstance(target_ids, torch.Tensor):
        offset = torch.zeros_like(reference)
        target_ids = target_ids.to(device=reference.device, dtype=torch.long)
        offset[:, target_ids] = joint_offset.to(device=reference.device, dtype=reference.dtype)
        return offset
    offset = getattr(getattr(env, "action_manager", None), "offset", None)
    if offset is None:
        return torch.zeros_like(reference)
    if not isinstance(offset, torch.Tensor):
        return torch.full_like(reference, float(offset))
    offset = offset.to(device=reference.device, dtype=reference.dtype)
    if offset.shape == reference.shape:
        return offset
    if offset.ndim == 1 and offset.shape[0] == reference.shape[-1]:
        return offset.unsqueeze(0).expand_as(reference)
    if (
        offset.ndim == 2
        and reference.ndim == 2
        and offset.shape[0] == reference.shape[0]
        and offset.shape[1] >= reference.shape[1]
    ):
        return offset[:, : reference.shape[1]]
    return torch.zeros_like(reference)


def _action_tensor(env: "ManagerBasedRlEnv") -> torch.Tensor:
    try:
        target = getattr(env.scene["robot"].data, "joint_pos_target", None)
    except (AttributeError, KeyError):
        target = None
    if isinstance(target, torch.Tensor):
        return target
    action_manager = env.action_manager
    term = _joint_action_term(env)
    if term is not None and hasattr(term, "applied_action"):
        return term.applied_action
    if hasattr(action_manager, "applied_action"):
        return action_manager.applied_action
    return action_manager.action


def _joint_action_term(env: "ManagerBasedRlEnv"):
    action_manager = env.action_manager
    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        try:
            return get_term("joint_pos")
        except KeyError:
            return None
    return None


def _event_observation(
    env: "ManagerBasedRlEnv", term_name: str, fallback: torch.Tensor
) -> torch.Tensor:
    event_manager = getattr(env, "event_manager", None)
    if event_manager is None:
        return fallback
    try:
        term_cfg = event_manager.get_term_cfg(term_name)
    except (KeyError, ValueError):
        return fallback
    observe = getattr(term_cfg.func, "observe", None)
    if not callable(observe):
        return fallback
    try:
        return observe()
    except NotImplementedError:
        return fallback


def _target_feet_standing(
    env: "ManagerBasedRlEnv", command_name: str, steps: tuple[int, ...] = (0,)
) -> torch.Tensor:
    cmd = _command(env, command_name)
    standing_state = getattr(cmd, "feet_standing", None)
    if (
        steps == (0,)
        and isinstance(standing_state, torch.Tensor)
        and standing_state.shape == (env.num_envs, len(SP_FEET_BODY_NAMES))
    ):
        return standing_state
    feet_motion_ids = _body_indices(tuple(cmd.cfg.body_names), SP_FEET_BODY_NAMES)
    gather_body_pos = getattr(cmd, "gather_reference_body_pos_w", None)
    gather_body_vel = getattr(cmd, "gather_reference_body_lin_vel_w", None)
    if callable(gather_body_pos) and callable(gather_body_vel):
        feet_pos = gather_body_pos(feet_motion_ids, steps, include_env_origins=False)
        feet_vel = gather_body_vel(feet_motion_ids, steps)
    else:
        feet_pos = _gather(env, command_name, "body_pos_w", steps)[:, :, feet_motion_ids]
        feet_vel = _gather(env, command_name, "body_lin_vel_w", steps)[:, :, feet_motion_ids]
    root_vel = _root_motion(env, command_name, "body_lin_vel_w", steps)
    root_vxy = root_vel[..., :2].norm(dim=-1, keepdim=True).clamp_min(1.0)
    feet_vxy = feet_vel[..., :2].norm(dim=-1)
    feet_vz = feet_vel[..., 2].abs()
    feet_z = feet_pos[..., 2]
    standing = (feet_z < 0.18) & (feet_vxy < 0.2 * root_vxy) & (feet_vz < 0.15 * root_vxy)
    return standing[:, 0] if len(steps) == 1 else standing


class _HistoryObservation:
    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
        self.cfg = cfg
        self.env = env
        self.asset = env.scene["robot"]
        self.steps = tuple(int(s) for s in cfg.params.get("history_steps", (0,)))
        self.max_len = max(self.steps) + 1
        self.buffer: torch.Tensor | None = None
        self.noise_std = float(cfg.params.get("noise_std", 0.0))
        self.bias_noise_std = float(cfg.params.get("bias_noise_std", 0.0))

    def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        raise NotImplementedError

    def _ensure_buffer(self, sample: torch.Tensor) -> None:
        if self.buffer is not None:
            return
        self.buffer = torch.zeros(
            (self.env.num_envs, self.max_len, sample.shape[-1]),
            dtype=sample.dtype,
            device=sample.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.buffer is not None:
            self.buffer[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        sample = self._sample(env)
        self._ensure_buffer(sample)
        assert self.buffer is not None
        self.buffer = torch.roll(self.buffer, shifts=1, dims=1)
        self.buffer[:, 0] = sample
        idx = torch.as_tensor(self.steps, device=sample.device, dtype=torch.long)
        return self.buffer[:, idx].reshape(env.num_envs, -1)


class root_angvel_b_history(_HistoryObservation):
    def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        sensor = env.scene["robot/imu_ang_vel"]
        return _spherical_noise(sensor.data, self.noise_std)


class projected_gravity_history(_HistoryObservation):
    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.bias_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
        self.bias_quat[:, 0] = 1.0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            count = self.env.num_envs
        elif isinstance(env_ids, slice):
            count = self.env.num_envs
        else:
            count = len(env_ids)
        base = torch.zeros((count, 4), dtype=self.bias_quat.dtype, device=self.bias_quat.device)
        base[:, 0] = 1.0
        self.bias_quat[env_ids] = _perturb_quaternion(base, self.bias_noise_std)

    def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        root_quat = quat_mul(self.bias_quat, self.asset.data.root_link_quat_w)
        return _projected_gravity(_perturb_quaternion(root_quat, self.noise_std))


class root_linvel_b_history(_HistoryObservation):
    def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        return _uniform_noise(self.asset.data.root_link_lin_vel_b, self.noise_std)


class joint_pos_history(_HistoryObservation):
    def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        cache = _substep_cache(env)
        if self.noise_std <= 0.0 and cache is not None:
            joint_pos = cache.joint_state_average("joint_pos")
        else:
            joint_pos = _current_joint_observation(
                env,
                str(self.cfg.params.get("command_name", "motion")),
                "joint_pos",
                self.noise_std,
            )
        joint_pos = joint_pos - _action_offset_like(env, joint_pos)
        return _canonical_joint_tensor(env, joint_pos)


class joint_vel_history(_HistoryObservation):
    def _sample(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        cache = _substep_cache(env)
        if self.noise_std <= 0.0 and cache is not None:
            joint_vel = cache.joint_state_average("joint_vel")
        else:
            joint_vel = _current_joint_observation(
                env,
                str(self.cfg.params.get("command_name", "motion")),
                "joint_vel",
                self.noise_std,
            )
        return _canonical_joint_tensor(
            env,
            joint_vel,
        )


def boot_indicator_state_obs(
    env: "ManagerBasedRlEnv", command_name: str = "motion"
) -> torch.Tensor:
    cmd = _command(env, command_name)
    maximum = max(float(getattr(cmd.cfg, "boot_indicator_max", 0)), 1.0)
    return cmd.boot_indicator / maximum


def command_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    horizon: Literal["teacher", "student"] = "student",
    noise_std: float = 0.0,
    root_body_name: str | None = None,
) -> torch.Tensor:
    steps = _steps(horizon)
    root_quat = _perturb_quaternion(env.scene["robot"].data.root_link_quat_w, noise_std).unsqueeze(
        1
    )
    future_quat = _root_motion(env, command_name, "body_quat_w", steps, horizon, root_body_name)
    future_pos = _root_motion(env, command_name, "body_pos_w", steps, horizon, root_body_name)
    pos_diff = _quat_apply_inverse(future_quat[:, :1], future_pos[:, 1:] - future_pos[:, :1])
    quat_diff = _quat_in_frame(root_quat.expand_as(future_quat), future_quat)
    return torch.cat(
        (pos_diff.reshape(env.num_envs, -1), _rot6d(quat_diff).reshape(env.num_envs, -1)),
        dim=-1,
    )


def target_joint_pos_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    horizon: Literal["teacher", "student"],
    noise_std: float = 0.0,
) -> torch.Tensor:
    target = _canonical_joint_tensor(
        env,
        _gather_horizon(env, command_name, "joint_pos", _steps(horizon), horizon),
    )
    current = _current_joint_observation(env, command_name, "joint_pos", noise_std)
    current = _canonical_joint_tensor(env, current - _action_offset_like(env, current))
    current = current.unsqueeze(1)
    diff = target - current
    return torch.cat((target.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)), dim=-1)


def target_root_z_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    horizon: Literal["teacher", "student"],
    root_body_name: str | None = None,
) -> torch.Tensor:
    return _root_motion(env, command_name, "body_pos_w", _steps(horizon), horizon, root_body_name)[
        ..., 2
    ]


def target_projected_gravity_b_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    horizon: Literal["teacher", "student"],
    root_body_name: str | None = None,
) -> torch.Tensor:
    quat = _root_motion(env, command_name, "body_quat_w", _steps(horizon), horizon, root_body_name)
    return _projected_gravity(quat).reshape(env.num_envs, -1)


def prev_actions(env: "ManagerBasedRlEnv", steps: int) -> torch.Tensor:
    term = _joint_action_term(env)
    get_recent = getattr(term, "get_recent_action_obs", None)
    if callable(get_recent):
        return get_recent(steps).reshape(env.num_envs, -1)
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if steps <= 1:
        return action
    repeated = [action, prev]
    repeated.extend([torch.zeros_like(action) for _ in range(max(steps - 2, 0))])
    return torch.cat(repeated[:steps], dim=-1)


def target_pos_b_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    env_origins = getattr(env.scene, "env_origins", None)
    root_pos_w = env.scene["robot"].data.root_link_pos_w
    if isinstance(env_origins, torch.Tensor):
        root_pos_w = root_pos_w - env_origins.to(device=root_pos_w.device, dtype=root_pos_w.dtype)
    root_pos = root_pos_w.unsqueeze(1)
    root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
    target_pos = _root_motion(
        env, command_name, "body_pos_w", TEACHER_STEPS, root_body_name=root_body_name
    )
    return _quat_apply_inverse(root_quat, target_pos - root_pos).reshape(env.num_envs, -1)


def target_rot_b_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
    target_quat = _root_motion(
        env, command_name, "body_quat_w", TEACHER_STEPS, root_body_name=root_body_name
    )
    return _rot6d(_quat_in_frame(root_quat.expand_as(target_quat), target_quat)).reshape(
        env.num_envs, -1
    )


def target_linvel_b_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
    target = _root_motion(
        env, command_name, "body_lin_vel_w", TEACHER_STEPS, root_body_name=root_body_name
    )
    return _quat_apply_inverse(root_quat, target).reshape(env.num_envs, -1)


def target_angvel_b_obs(
    env: "ManagerBasedRlEnv",
    command_name: str,
    root_body_name: str | None = None,
) -> torch.Tensor:
    root_quat = env.scene["robot"].data.root_link_quat_w.unsqueeze(1)
    target = _root_motion(
        env, command_name, "body_ang_vel_w", TEACHER_STEPS, root_body_name=root_body_name
    )
    return _quat_apply_inverse(root_quat, target).reshape(env.num_envs, -1)


class _KeypointObservation:
    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
        self.cfg = cfg
        self.env = env
        self.asset = env.scene["robot"]
        self.command_name = str(cfg.params.get("command_name", "motion"))
        cmd = _command(env, self.command_name)
        raw_specs = cfg.params.get("keypoint_specs", _default_keypoint_specs())
        self.resolver = SemanticKeypointResolver(self.asset, tuple(cmd.cfg.body_names), raw_specs)

    def _current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos = self.asset.data.root_link_pos_w
        root_quat = self.asset.data.root_link_quat_w
        root_lin = self.asset.data.root_link_lin_vel_w
        root_ang = self.asset.data.root_link_ang_vel_w
        current = self.resolver.current(self.asset)
        pos = current.pos_w
        quat = current.quat_w
        lin = current.lin_vel_w
        ang = current.ang_vel_w
        pos_b = _quat_apply_inverse(root_quat.unsqueeze(1), pos - root_pos.unsqueeze(1))
        quat_b = _quat_in_frame(root_quat.unsqueeze(1).expand_as(quat), quat)
        lin_b = _quat_apply_inverse(root_quat.unsqueeze(1), lin - root_lin.unsqueeze(1))
        ang_b = _quat_apply_inverse(root_quat.unsqueeze(1), ang - root_ang.unsqueeze(1))
        return pos_b, quat_b, lin_b, ang_b

    def _reference(self, command_name: str, steps: tuple[int, ...]):
        return self.resolver.reference(
            _gather(self.env, command_name, "body_pos_w", steps),
            _gather(self.env, command_name, "body_quat_w", steps),
            _gather(self.env, command_name, "body_lin_vel_w", steps),
            _gather(self.env, command_name, "body_ang_vel_w", steps),
        )

    def _reference_pose_b(
        self,
        command_name: str,
        steps: tuple[int, ...],
        root_body_name: str | None,
    ):
        cmd = _command(self.env, command_name)
        gather_pose_b = getattr(cmd, "gather_reference_body_pose_b", None)
        if not callable(gather_pose_b):
            return None
        uses_actor_fk = getattr(cmd, "_uses_qpos_only_actor_fk", None)
        if callable(uses_actor_fk) and not uses_actor_fk():
            return None
        anchor_body_name = getattr(cmd.cfg, "anchor_body_name", None)
        if anchor_body_name is None:
            anchor_body_name = cmd.cfg.body_names[cmd.motion_anchor_body_index]
        if root_body_name is not None and _basename(root_body_name) != _basename(anchor_body_name):
            return None
        pos_b, quat_b = gather_pose_b(steps)
        zero = torch.zeros_like(pos_b)
        return self.resolver.reference(pos_b, quat_b, zero, zero)


class current_keypoint_pos_b_obs(_KeypointObservation):
    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        return self._current()[0].reshape(env.num_envs, -1)


class current_keypoint_rot_b_obs(_KeypointObservation):
    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        return _rot6d(self._current()[1]).reshape(env.num_envs, -1)


class current_keypoint_linvel_b_obs(_KeypointObservation):
    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        return self._current()[2].reshape(env.num_envs, -1)


class current_keypoint_angvel_b_obs(_KeypointObservation):
    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        return self._current()[3].reshape(env.num_envs, -1)


class target_keypoints_pos_b_obs(_KeypointObservation):
    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        required_steps: int,
        include_diff: bool,
        root_body_name: str | None = None,
        **_: Any,
    ) -> torch.Tensor:
        steps = TEACHER_STEPS[:required_steps]
        reference_b = self._reference_pose_b(command_name, steps, root_body_name)
        if reference_b is not None:
            target_b = reference_b.pos_w
            if not include_diff:
                return target_b.reshape(env.num_envs, -1)
            root_pos = _root_motion(
                env, command_name, "body_pos_w", steps, root_body_name=root_body_name
            )
            root_quat = _root_motion(
                env, command_name, "body_quat_w", steps, root_body_name=root_body_name
            )
            root_delta_b = _quat_apply_inverse(root_quat[:, :1], root_pos - root_pos[:, :1])
            target_ref_b = root_delta_b.unsqueeze(2) + _quat_apply_inverse(
                root_quat[:, :1].unsqueeze(2),
                fk_quat_apply(root_quat.unsqueeze(2), target_b),
            )
            diff = target_ref_b - self._current()[0].unsqueeze(1)
            return torch.cat(
                (target_b.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)),
                dim=-1,
            )
        target_w = self._reference(command_name, steps).pos_w
        root_pos = _root_motion(
            env, command_name, "body_pos_w", steps, root_body_name=root_body_name
        )
        root_quat = _root_motion(
            env, command_name, "body_quat_w", steps, root_body_name=root_body_name
        )
        target_b = _quat_apply_inverse(root_quat.unsqueeze(2), target_w - root_pos.unsqueeze(2))
        if not include_diff:
            return target_b.reshape(env.num_envs, -1)
        target_ref_b = _quat_apply_inverse(
            root_quat[:, :1].unsqueeze(2), target_w - root_pos[:, :1].unsqueeze(2)
        )
        diff = target_ref_b - self._current()[0].unsqueeze(1)
        return torch.cat(
            (target_b.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)), dim=-1
        )


class target_keypoints_rot_b_obs(_KeypointObservation):
    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        required_steps: int,
        include_diff: bool,
        root_body_name: str | None = None,
        **_: Any,
    ) -> torch.Tensor:
        steps = TEACHER_STEPS[:required_steps]
        reference_b = self._reference_pose_b(command_name, steps, root_body_name)
        if reference_b is not None:
            target_b = reference_b.quat_w
            target_rot = _rot6d(target_b)
            if not include_diff:
                return target_rot.reshape(env.num_envs, -1)
            root_quat = _root_motion(
                env, command_name, "body_quat_w", steps, root_body_name=root_body_name
            )
            target_w = fk_quat_mul(root_quat.unsqueeze(2), target_b)
            target_ref_b = _quat_in_frame(root_quat[:, :1].unsqueeze(2), target_w)
            diff = _rot6d(_quat_delta(self._current()[1].unsqueeze(1), target_ref_b))
            return torch.cat(
                (target_rot.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)),
                dim=-1,
            )
        target_w = self._reference(command_name, steps).quat_w
        root_quat = _root_motion(
            env, command_name, "body_quat_w", steps, root_body_name=root_body_name
        )
        target_b = _quat_in_frame(root_quat.unsqueeze(2).expand_as(target_w), target_w)
        target_rot = _rot6d(target_b)
        if not include_diff:
            return target_rot.reshape(env.num_envs, -1)
        target_ref_b = _quat_in_frame(root_quat[:, :1].unsqueeze(2), target_w)
        diff = _rot6d(_quat_delta(self._current()[1].unsqueeze(1), target_ref_b))
        return torch.cat(
            (target_rot.reshape(env.num_envs, -1), diff.reshape(env.num_envs, -1)), dim=-1
        )


def applied_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _canonical_joint_tensor(env, _action_tensor(env))


def applied_torque(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset = env.scene["robot"]
    ordered_names = tuple(
        getattr(getattr(asset, "cfg", None), "joint_name_order", asset.joint_names)
    )
    actuator_names = tuple(asset.actuator_names)
    if all(name in actuator_names for name in ordered_names):
        ids = torch.as_tensor(
            [actuator_names.index(name) for name in ordered_names],
            dtype=torch.long,
            device=env.device,
        )
        return asset.data.actuator_force.index_select(-1, ids)
    return _canonical_joint_tensor(env, asset.data.actuator_force)


def body_z_termination_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    cmd, target_z, current_z = _body_z_values(env, command_name, SP_TERMINATION_BODY_NAMES)
    buffer = _termination_buffer(env, cmd, "_body_z_termination_buffer").float().unsqueeze(1)
    return torch.cat((current_z, target_z, buffer), dim=-1)


def gravity_dir_termination_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    cmd = _command(env, command_name)
    robot_g = _projected_gravity(env.scene["robot"].data.root_link_quat_w)
    motion_g = _projected_gravity(_root_motion_current(env, command_name, "body_quat_w"))
    buffer = _termination_buffer(env, cmd, "_gravity_dir_termination_buffer").float().unsqueeze(1)
    return torch.cat((robot_g, motion_g, buffer), dim=-1)


def body_z_termination(
    env: "ManagerBasedRlEnv",
    command_name: str,
    body_z_terminate_thres: tuple[float, float],
    body_z_terminate_patterns: tuple[str, ...],
) -> torch.Tensor:
    low, high = body_z_terminate_thres
    cmd, target_z, current_z = _body_z_values(env, command_name, body_z_terminate_patterns)
    target_z_min_thres = target_z + float(low)
    target_z_max_thres = target_z + float(high)
    target_z_min = target_z.amin(dim=1, keepdim=True)
    lower_relax = ((target_z_min - 0.1) / 0.2).clamp(0.0, 1.0) * 0.2
    target_z_min_thres = target_z_min_thres - lower_relax
    exceed = (current_z < target_z_min_thres) | (current_z > target_z_max_thres)
    buffer = _termination_buffer(env, cmd, "_body_z_termination_buffer")
    return _apply_termination_warmup(env, cmd, _continuous_termination(env, exceed, buffer))


def gravity_dir_termination(
    env: "ManagerBasedRlEnv",
    command_name: str,
    gravity_terminate_thres: float,
) -> torch.Tensor:
    cmd = _command(env, command_name)
    obs = gravity_dir_termination_obs(env, command_name)
    robot_g = obs[:, :3]
    motion_g = obs[:, 3:6]
    exceed = torch.norm(robot_g - motion_g, dim=-1) > float(gravity_terminate_thres)
    buffer = _termination_buffer(env, cmd, "_gravity_dir_termination_buffer")
    return _apply_termination_warmup(env, cmd, _continuous_termination(env, exceed, buffer))


def motion_timeout(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    cmd = _command(env, command_name)
    return cmd.time_steps >= (cmd.motion_length - 1)


def motion_xy_range_termination(
    env: "ManagerBasedRlEnv",
    command_name: str,
    motion_xy_max_offset: float | None = None,
) -> torch.Tensor:
    maximum = float("inf") if motion_xy_max_offset is None else float(motion_xy_max_offset)
    target_xy = _root_motion_current(env, command_name, "body_pos_w")[:, :2]
    return (target_xy.abs() > maximum).any(dim=1)


def _robot_mass(env: "ManagerBasedRlEnv") -> float | torch.Tensor | None:
    robot_cfg = getattr(getattr(env, "cfg", None), "robot", None)
    mass = getattr(robot_cfg, "mass", None) if robot_cfg is not None else None
    if mass is not None:
        return mass
    asset = None
    if hasattr(env, "scene"):
        try:
            asset = env.scene["robot"]
        except KeyError:
            asset = None
    data = getattr(asset, "data", None)
    for name in ("body_mass", "default_body_mass"):
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            return value.sum(dim=-1) if value.ndim > 1 else value.sum()
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    value = getattr(model, "body_mass", None)
    if isinstance(value, torch.Tensor):
        return value.sum(dim=-1) if value.ndim > 1 else value.sum()
    return None


def _contact_force_denominator(
    env: "ManagerBasedRlEnv", force: torch.Tensor, divide_by_mass: bool
) -> torch.Tensor:
    if not divide_by_mass:
        return force.new_tensor(1.0)
    mass = _robot_mass(env)
    if mass is None:
        mass = 33.341142
    denom = torch.as_tensor(mass, dtype=force.dtype, device=force.device) * 9.81
    denom = denom.clamp_min(1.0e-6)
    if denom.ndim == 1 and force.ndim >= 2 and denom.shape[0] == force.shape[0]:
        denom = denom.reshape(force.shape[0], *([1] * (force.ndim - 1)))
    return denom


def feet_contact_state(
    env: "ManagerBasedRlEnv", sensor_name: str, divide_by_mass: bool = True
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if sensor.data.force_history is not None:
        force = sensor.data.force_history.mean(dim=2)
    assert force is not None
    contact_time = sensor.data.current_contact_time
    air_time = sensor.data.current_air_time
    assert contact_time is not None and air_time is not None
    in_contact = (contact_time > env.physics_dt).float()
    denom = _contact_force_denominator(env, force, divide_by_mass)
    return torch.cat(
        (
            (force / denom).clamp(-10.0, 10.0).reshape(env.num_envs, -1),
            in_contact,
            contact_time,
            air_time,
        ),
        dim=-1,
    )


def feet_contact_binary_state(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Return one binary supervision target for each SP foot.

    Prefer the control-step majority vote collected by ``substep_tracking_cache``
    so the target uses the same physical samples as the proprioceptive history.
    The contact sensor fallback keeps the term usable in task variants that do
    not install that optional cache.
    """
    cache = _substep_cache(env)
    if cache is not None:
        in_contact, _, _ = cache.contact_state()
    else:
        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time
        if contact_time is None:
            raise RuntimeError("feet_contact_binary_state requires contact timing data")
        in_contact = contact_time > env.physics_dt
    expected = (env.num_envs, len(SP_FEET_BODY_NAMES))
    if in_contact.shape != expected:
        raise RuntimeError(
            "feet_contact_binary_state must contain one value per SP foot; "
            f"expected={expected}, got={tuple(in_contact.shape)}"
        )
    return in_contact.float()


def target_feet_contact_state_obs(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    standing = _target_feet_standing(env, command_name).float()
    expected = (env.num_envs, len(SP_FEET_BODY_NAMES))
    if standing.shape != expected:
        raise RuntimeError(
            "target_feet_contact_state_obs must contain one value per SP foot; "
            f"expected={expected}, got={tuple(standing.shape)}"
        )
    return standing


def domain_motor_params_implicit(env: "ManagerBasedRlEnv") -> torch.Tensor:
    joints = len(env.scene["robot"].joint_names)
    fallback = torch.ones((env.num_envs, joints * 3), device=env.device)
    return _event_observation(env, "motor_params_implicit", fallback)


def domain_perturb_body_materials(env: "ManagerBasedRlEnv") -> torch.Tensor:
    fallback = torch.ones((env.num_envs, 3), device=env.device)
    return _event_observation(env, "perturb_body_materials", fallback)


def domain_random_joint_offset(env: "ManagerBasedRlEnv") -> torch.Tensor:
    fallback = torch.zeros((env.num_envs, len(env.scene["robot"].joint_names)), device=env.device)
    # The HEFT event exposes values in its resolved randomization order, which
    # follows the articulation/XML joint order for the catch-all range used by
    # this task.  Do not apply the policy-facing canonical permutation here.
    return _event_observation(env, "random_joint_offset", fallback)


def domain_perturb_gravity(env: "ManagerBasedRlEnv") -> torch.Tensor:
    fallback = torch.tensor((0.0, 0.0, -9.81), device=env.device).repeat(env.num_envs, 1)
    return _event_observation(env, "perturb_gravity", fallback)


class _RewardBase:
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        self.cfg = cfg
        self.env = env
        self.asset = env.scene["robot"]


class loco_reward_group_schedule(_RewardBase):
    """Apply the source pretrain's linear multiplier to the whole loco group."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.term_names = tuple(str(v) for v in cfg.params["term_names"])
        self.base_weights = {
            str(name): float(weight) for name, weight in cfg.params["base_weights"].items()
        }
        self.progress_range = tuple(float(v) for v in cfg.params.get("progress_range", (0.0, 1.0)))
        self.factor_range = tuple(float(v) for v in cfg.params.get("factor_range", (0.5, 1.0)))
        self.current_factor = float(self.factor_range[0])
        state = dict(getattr(env, "_sp_tracking_curriculum_state", {"progress": 0.0}))
        state["reward/loco_reward_group_schedule/factor"] = self.current_factor
        env._sp_tracking_curriculum_state = state

    def __call__(self, env: "ManagerBasedRlEnv", **_: Any) -> torch.Tensor:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def step_schedule(self, progress: float, iters: int | None = None) -> dict[str, float]:
        del iters
        start, end = self.progress_range
        low, high = self.factor_range
        fraction = min(max((float(progress) - start) / max(end - start, 1.0e-8), 0.0), 1.0)
        self.current_factor = low + fraction * (high - low)
        manager = getattr(self.env, "reward_manager", None)
        if manager is not None:
            for name in self.term_names:
                manager.get_term_cfg(name).weight = self.base_weights[name] * self.current_factor
        return {"factor": self.current_factor}


class _KeypointReward(_RewardBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        raw_specs = cfg.params.get("keypoint_specs")
        self.resolver: SemanticKeypointResolver | None = None
        self.asset_ids: torch.Tensor | None = None
        if raw_specs is None:
            # Preserve the complete SP task's direct-body fast path exactly.
            self.asset_ids = _robot_body_indices(self.asset, SP_KEYPOINT_BODY_NAMES, env.device)
        else:
            command_name = str(cfg.params.get("command_name", "motion"))
            cmd = _command(env, command_name)
            self.resolver = SemanticKeypointResolver(
                self.asset, tuple(cmd.cfg.body_names), raw_specs
            )

    def _motion_ids(self, command_name: str) -> list[int]:
        cmd = _command(self.env, command_name)
        return _body_indices(tuple(cmd.cfg.body_names), SP_KEYPOINT_BODY_NAMES)

    def _current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos = self.asset.data.root_link_pos_w
        root_quat = self.asset.data.root_link_quat_w
        root_lin = self.asset.data.root_link_lin_vel_w
        root_ang = self.asset.data.root_link_ang_vel_w
        if self.resolver is None:
            assert self.asset_ids is not None
            pos = self.asset.data.body_link_pos_w[:, self.asset_ids]
            quat = self.asset.data.body_link_quat_w[:, self.asset_ids]
            lin = self.asset.data.body_link_lin_vel_w[:, self.asset_ids]
            ang = self.asset.data.body_link_ang_vel_w[:, self.asset_ids]
        else:
            current = self.resolver.current(self.asset)
            pos = current.pos_w
            quat = current.quat_w
            lin = current.lin_vel_w
            ang = current.ang_vel_w
        pos_b = _quat_apply_inverse(root_quat.unsqueeze(1), pos - root_pos.unsqueeze(1))
        quat_b = _quat_in_frame(root_quat.unsqueeze(1).expand_as(quat), quat)
        lin_b = _quat_apply_inverse(root_quat.unsqueeze(1), lin - root_lin.unsqueeze(1))
        ang_b = _quat_apply_inverse(root_quat.unsqueeze(1), ang - root_ang.unsqueeze(1))
        return pos_b, quat_b, lin_b, ang_b

    def _semantic_reference(
        self, command_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.resolver is not None
        cmd = _command(self.env, command_name)
        gather_state_b = getattr(cmd, "gather_reference_body_state_b", None)
        if callable(gather_state_b):
            uses_actor_fk = getattr(cmd, "_uses_qpos_only_actor_fk", None)
            body = None if callable(uses_actor_fk) and not uses_actor_fk() else gather_state_b((0,))
            if body is not None:
                reference_b = self.resolver.reference(
                    body.pos_b[:, 0],
                    body.quat_b[:, 0],
                    body.lin_vel_b[:, 0],
                    body.ang_vel_b[:, 0],
                )
                return (
                    reference_b.pos_w,
                    reference_b.quat_w,
                    reference_b.lin_vel_w,
                    reference_b.ang_vel_w,
                )
        reference = self.resolver.reference(
            _gather_current(self.env, command_name, "body_pos_w"),
            _gather_current(self.env, command_name, "body_quat_w"),
            _gather_current(self.env, command_name, "body_lin_vel_w"),
            _gather_current(self.env, command_name, "body_ang_vel_w"),
        )
        root_pos = _root_motion_current(self.env, command_name, "body_pos_w")
        root_quat = _root_motion_current(self.env, command_name, "body_quat_w")
        root_lin = _root_motion_current(self.env, command_name, "body_lin_vel_w")
        root_ang = _root_motion_current(self.env, command_name, "body_ang_vel_w")
        pos_b = _quat_apply_inverse(root_quat.unsqueeze(1), reference.pos_w - root_pos.unsqueeze(1))
        quat_b = _quat_in_frame(
            root_quat.unsqueeze(1).expand_as(reference.quat_w), reference.quat_w
        )
        lin_b = _quat_apply_inverse(
            root_quat.unsqueeze(1), reference.lin_vel_w - root_lin.unsqueeze(1)
        )
        ang_b = _quat_apply_inverse(
            root_quat.unsqueeze(1), reference.ang_vel_w - root_ang.unsqueeze(1)
        )
        return pos_b, quat_b, lin_b, ang_b


class root_pos_tracking(_RewardBase):
    def __call__(
        self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
    ) -> torch.Tensor:
        cmd = _command(env, command_name)
        target = getattr(cmd, "reward_root_pos_w", None)
        if not isinstance(target, torch.Tensor):
            target = _root_motion_current(env, command_name, "body_pos_w")
            env_origins = getattr(env.scene, "env_origins", None)
            if isinstance(env_origins, torch.Tensor):
                target = target + env_origins
        error = (target - self.asset.data.root_link_pos_w).norm(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class global_link_height_tracking(_RewardBase):
    """Track the mean world-frame height error of selected reference links."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        command_name = str(cfg.params.get("command_name", "motion"))
        cmd = _command(env, command_name)
        configured_names = tuple(_basename(name) for name in cmd.cfg.body_names)
        selected_names = tuple(_basename(name) for name in cfg.params.get("body_names", ()))
        missing_names = tuple(name for name in selected_names if name not in configured_names)
        if not selected_names or missing_names:
            raise ValueError(
                "global_link_height_tracking requires non-empty body_names present in "
                f"the command reference; missing={missing_names}."
            )
        self.body_ids = torch.as_tensor(
            [configured_names.index(name) for name in selected_names],
            dtype=torch.long,
            device=env.device,
        )

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        body_names: tuple[str, ...],
        sigma: list[float],
    ) -> torch.Tensor:
        del body_names
        cmd = _command(env, command_name)
        gather_body_pos = getattr(cmd, "gather_reference_body_pos_w", None)
        if callable(gather_body_pos):
            target_z = gather_body_pos(self.body_ids, (0,))[:, 0, :, 2]
        else:
            target_z = cmd.body_pos_w[:, self.body_ids, 2]
        current_z = cmd.robot_body_pos_w[:, self.body_ids, 2]
        error = (target_z - current_z).abs().mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class root_rot_tracking(_RewardBase):
    def __call__(
        self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
    ) -> torch.Tensor:
        cmd = _command(env, command_name)
        target = getattr(cmd, "reward_root_quat_w", None)
        if not isinstance(target, torch.Tensor):
            target = _root_motion_current(env, command_name, "body_quat_w")
        error = quat_error_magnitude(self.asset.data.root_link_quat_w, target).unsqueeze(-1)
        return _exp_sigma(error, sigma).squeeze(-1)


class root_vel_tracking(_RewardBase):
    def __call__(
        self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
    ) -> torch.Tensor:
        current = _quat_apply_inverse(
            self.asset.data.root_link_quat_w, self.asset.data.root_link_lin_vel_w
        )
        target_quat = _root_motion_current(env, command_name, "body_quat_w")
        target = _quat_apply_inverse(
            target_quat, _root_motion_current(env, command_name, "body_lin_vel_w")
        )
        return _exp_sigma((target - current).norm(dim=-1, keepdim=True), sigma).squeeze(-1)


class root_ang_vel_tracking(_RewardBase):
    def __call__(
        self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
    ) -> torch.Tensor:
        current = _quat_apply_inverse(
            self.asset.data.root_link_quat_w, self.asset.data.root_link_ang_vel_w
        )
        target_quat = _root_motion_current(env, command_name, "body_quat_w")
        target = _quat_apply_inverse(
            target_quat, _root_motion_current(env, command_name, "body_ang_vel_w")
        )
        return _exp_sigma((target - current).norm(dim=-1, keepdim=True), sigma).squeeze(-1)


class keypoint_pos_tracking(_KeypointReward):
    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        sigma: list[float],
        **_: Any,
    ) -> torch.Tensor:
        if self.resolver is not None:
            target = self._semantic_reference(command_name)[0]
        else:
            ids = self._motion_ids(command_name)
            root_pos = _root_motion_current(env, command_name, "body_pos_w")
            root_quat = _root_motion_current(env, command_name, "body_quat_w")
            target_w = _gather_current(env, command_name, "body_pos_w")[:, ids]
            target = _quat_apply_inverse(root_quat.unsqueeze(1), target_w - root_pos.unsqueeze(1))
        error = (target - self._current()[0]).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class keypoint_vel_tracking(_KeypointReward):
    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        sigma: list[float],
        **_: Any,
    ) -> torch.Tensor:
        if self.resolver is not None:
            target = self._semantic_reference(command_name)[2]
        else:
            ids = self._motion_ids(command_name)
            root_quat = _root_motion_current(env, command_name, "body_quat_w")
            root_vel = _root_motion_current(env, command_name, "body_lin_vel_w")
            target_w = _gather_current(env, command_name, "body_lin_vel_w")[:, ids]
            target = _quat_apply_inverse(root_quat.unsqueeze(1), target_w - root_vel.unsqueeze(1))
        error = (target - self._current()[2]).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class keypoint_rot_tracking(_KeypointReward):
    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        sigma: list[float],
        **_: Any,
    ) -> torch.Tensor:
        if self.resolver is not None:
            target = self._semantic_reference(command_name)[1]
        else:
            ids = self._motion_ids(command_name)
            root_quat = _root_motion_current(env, command_name, "body_quat_w")
            target_w = _gather_current(env, command_name, "body_quat_w")[:, ids]
            target = _quat_in_frame(root_quat.unsqueeze(1).expand_as(target_w), target_w)
        error = axis_angle_from_quat(_quat_delta(self._current()[1], target)).norm(dim=-1)
        error = error.mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class keypoint_angvel_tracking(_KeypointReward):
    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        sigma: list[float],
        **_: Any,
    ) -> torch.Tensor:
        if self.resolver is not None:
            target = self._semantic_reference(command_name)[3]
        else:
            ids = self._motion_ids(command_name)
            root_quat = _root_motion_current(env, command_name, "body_quat_w")
            root_ang = _root_motion_current(env, command_name, "body_ang_vel_w")
            target_w = _gather_current(env, command_name, "body_ang_vel_w")[:, ids]
            target = _quat_apply_inverse(root_quat.unsqueeze(1), target_w - root_ang.unsqueeze(1))
        error = (target - self._current()[3]).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class joint_pos_tracking(_RewardBase):
    def __call__(
        self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
    ) -> torch.Tensor:
        target = _gather_current(env, command_name, "joint_pos")
        error = (target - self.asset.data.joint_pos).abs().mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


class joint_vel_tracking(_RewardBase):
    def __call__(
        self, env: "ManagerBasedRlEnv", command_name: str, sigma: list[float]
    ) -> torch.Tensor:
        target = _gather_current(env, command_name, "joint_vel")
        error = (target - self.asset.data.joint_vel).abs().mean(dim=-1, keepdim=True)
        return _exp_sigma(error, sigma).squeeze(-1)


def survival(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def joint_vel_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cache = _substep_cache(env)
    joint_vel = (
        cache.joint_state_average("joint_vel")
        if cache is not None
        else env.scene["robot"].data.joint_vel
    )
    return -joint_vel.square().sum(dim=-1)


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    term = _joint_action_term(env)
    get_recent = getattr(term, "get_recent_action_rate_actions", None)
    if callable(get_recent):
        action_buf = get_recent(2)
        diff = action_buf[:, 0] - action_buf[:, 1]
        return -diff.square().sum(dim=-1)
    diff = env.action_manager.action - env.action_manager.prev_action
    return -diff.square().sum(dim=-1)


class feet_air_time_ref(_RewardBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.reward_time = torch.zeros(env.num_envs, len(SP_FEET_BODY_NAMES), device=env.device)
        self.last_contact = torch.zeros_like(self.reward_time, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        sensor_name: str,
        thres: float,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        cache = _substep_cache(env)
        if cache is not None:
            current, first_contact, _ = cache.contact_state()
        else:
            contact_time = sensor.data.current_contact_time
            assert contact_time is not None
            current = contact_time > env.physics_dt
            first_contact = (~self.last_contact) & current
            self.last_contact[:] = current
        target = _target_feet_standing(env, command_name)
        mismatch = target ^ current
        self.reward_time += torch.where(mismatch, -env.step_dt, env.step_dt)
        reward = ((self.reward_time - float(thres)).clamp_max(0.0) * first_contact).sum(1)
        self.reward_time *= ~current
        return reward


class feet_air_time_ref_dense(_RewardBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.feet_ids = _robot_body_indices(self.asset, SP_FEET_BODY_NAMES, env.device)
        raw_toe_specs = cfg.params.get("toe_specs")
        self.toe_resolver: SemanticKeypointResolver | None = None
        self.toe_ids: torch.Tensor | None = None
        if raw_toe_specs is None:
            # Preserve the complete SP task's dedicated toe-body path.
            self.toe_ids = _robot_body_indices(self.asset, SP_FEET_TOE_BODY_NAMES, env.device)
        else:
            command_name = str(cfg.params.get("command_name", "motion"))
            cmd = _command(env, command_name)
            self.toe_resolver = SemanticKeypointResolver(
                self.asset, tuple(cmd.cfg.body_names), raw_toe_specs
            )

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        sensor_name: str,
        air_h_low: float,
        air_h_high: float,
        contact_h_low: float,
        contact_h_high: float,
        **_: Any,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        cache = _substep_cache(env)
        if cache is not None:
            current, _, _ = cache.contact_state()
        else:
            contact_time = sensor.data.current_contact_time
            assert contact_time is not None
            current = contact_time > env.physics_dt
        target = _target_feet_standing(env, command_name)
        mismatch = current ^ target
        both_air = (~current) & (~target)
        both_contact = current & target
        penalty = torch.zeros_like(current, dtype=torch.float32)
        penalty[mismatch] = -1.0

        feet_z = self.asset.data.body_link_pos_w[:, self.feet_ids, 2]
        if self.toe_resolver is None:
            assert self.toe_ids is not None
            toe_z = self.asset.data.body_link_pos_w[:, self.toe_ids, 2]
        else:
            toe_z = self.toe_resolver.current(self.asset).pos_w[..., 2]
        air_height = torch.minimum(feet_z, toe_z)
        air_span = max(float(air_h_high) - float(air_h_low), 1.0e-6)
        air_ratio = ((air_height - float(air_h_low)) / air_span).clamp(0.0, 1.0)
        penalty = torch.where(both_air, -(1.0 - air_ratio), penalty)

        contact_height = torch.maximum(feet_z, toe_z)
        contact_span = max(float(contact_h_high) - float(contact_h_low), 1.0e-6)
        contact_ratio = ((contact_height - float(contact_h_low)) / contact_span).clamp(0.0, 1.0)
        penalty = torch.where(both_contact, -contact_ratio, penalty)
        return penalty.mean(dim=1)


class joint_pos_limits(_RewardBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        joint_names = cfg.params.get("joint_names", ".*")
        ids, _ = resolve_matching_names(joint_names, self.asset.joint_names)
        self.joint_ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)

    def __call__(self, env: "ManagerBasedRlEnv", soft_factor: float, **_: Any) -> torch.Tensor:
        limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        mean = (limits[..., 0] + limits[..., 1]) * 0.5
        span = limits[..., 1] - limits[..., 0]
        lower = mean - 0.5 * span * float(soft_factor)
        upper = mean + 0.5 * span * float(soft_factor)
        pos = self.asset.data.joint_pos[:, self.joint_ids]
        violation = (lower - pos).clamp_min(0.0) + (pos - upper).clamp_min(0.0)
        return -violation.sum(dim=1) / max(1.0 - float(soft_factor), 1.0e-6)


class joint_torque_limits(_RewardBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        joint_names = cfg.params.get("joint_names", ".*")
        _, names = resolve_matching_names(joint_names, self.asset.joint_names)
        actuator_names = list(self.asset.actuator_names)
        self.act_ids = torch.as_tensor(
            [actuator_names.index(name) for name in names],
            device=env.device,
            dtype=torch.long,
        )

    def __call__(self, env: "ManagerBasedRlEnv", soft_factor: float, **_: Any) -> torch.Tensor:
        force_range = env.sim.model.actuator_forcerange[:, self.asset.indexing.ctrl_ids]
        limits = torch.maximum(force_range[..., 0].abs(), force_range[..., 1].abs())
        soft_limits = limits[:, self.act_ids].clamp_min(1.0e-6) * float(soft_factor)
        torque = self.asset.data.actuator_force[:, self.act_ids]
        high = (torque / soft_limits - 1.0).clamp_min(0.0)
        low = (-torque / soft_limits - 1.0).clamp_min(0.0)
        return -(high + low).sum(dim=1)
