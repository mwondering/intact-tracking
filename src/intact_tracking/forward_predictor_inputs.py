"""Differentiable privileged inputs for the nominal G1 Forward Predictor."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

ROBOT_STATE_DIM = 71
ACTION_DIM = 29
FOOT_COUNT = 2
FOOT_FEATURE_DIM = 8
CONTACT_FORCE_DIM = 6
CONTACT_BINARY_DIM = 2

G1_FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
G1_FOOT_SITE_POSITIONS = (
    (0.04, 0.0, -0.037),
    (0.04, 0.0, -0.037),
)

G1_XML_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_FOOT_FEATURE_NAMES = (
    "left_foot_height_m",
    "left_foot_velocity_x_mps",
    "left_foot_velocity_y_mps",
    "left_foot_velocity_z_mps",
    "right_foot_height_m",
    "right_foot_velocity_x_mps",
    "right_foot_velocity_y_mps",
    "right_foot_velocity_z_mps",
)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with scalar-first unit quaternions."""

    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=1.0e-8)
    quaternion_vector = quaternion[..., 1:]
    quaternion_vector, vector = torch.broadcast_tensors(quaternion_vector, vector)
    first_cross = torch.linalg.cross(quaternion_vector, vector, dim=-1)
    second_cross = torch.linalg.cross(quaternion_vector, first_cross, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * first_cross + second_cross)


def g1_foot_features_from_link_state(
    link_position: torch.Tensor,
    link_quaternion: torch.Tensor,
    link_linear_velocity: torch.Tensor,
    link_angular_velocity: torch.Tensor,
) -> torch.Tensor:
    """Read sole height/velocity from simulator-provided ankle-link state.

    This is a constant-time rigid-point transform, not articulated forward
    kinematics. ``link_position`` must already be relative to the environment
    origin so its z component uses the same ground-height convention as the
    Forward Predictor state.
    """

    expected_prefix = link_position.shape[:-1]
    expected = {
        "link_position": (*expected_prefix, 3),
        "link_quaternion": (*expected_prefix, 4),
        "link_linear_velocity": (*expected_prefix, 3),
        "link_angular_velocity": (*expected_prefix, 3),
    }
    actual = {
        "link_position": tuple(link_position.shape),
        "link_quaternion": tuple(link_quaternion.shape),
        "link_linear_velocity": tuple(link_linear_velocity.shape),
        "link_angular_velocity": tuple(link_angular_velocity.shape),
    }
    invalid = {
        name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape
    }
    if invalid or len(expected_prefix) < 1 or expected_prefix[-1] != FOOT_COUNT:
        raise ValueError(
            "G1 foot link state must have matching [...,2,3/4] shapes; "
            f"invalid={invalid}, position={tuple(link_position.shape)}"
        )

    local_site = link_position.new_tensor(G1_FOOT_SITE_POSITIONS)
    while local_site.ndim < link_position.ndim:
        local_site = local_site.unsqueeze(0)
    site_offset = _quaternion_rotate(link_quaternion, local_site.expand_as(link_position))
    site_position = link_position + site_offset
    site_velocity = link_linear_velocity + torch.linalg.cross(
        link_angular_velocity,
        site_offset,
        dim=-1,
    )
    per_foot = torch.cat((site_position[..., 2:3], site_velocity), dim=-1)
    return per_foot.flatten(start_dim=-2)


class G1FootKinematics(nn.Module):
    """Compute sole-site height and velocity from the 71-D G1 state.

    The fixed transforms match ``g1_tracking_bfm/g1.xml``.  The two site
    origins are the XML ``left_foot`` and ``right_foot`` sites.  Because the
    nominal training terrain is a plane and root position is stored relative
    to each environment origin, site world-z is also signed ground clearance.
    Every operation is native PyTorch, so recursive foot features remain
    differentiable with respect to predicted root pose, q and qdot.
    """

    def __init__(self, ground_height: float = 0.0) -> None:
        super().__init__()
        body_position = torch.tensor(
            (
                (
                    (0.0, 0.064452, -0.1027),
                    (0.0, 0.052, -0.030465),
                    (0.025001, 0.0, -0.12412),
                    (-0.078273, 0.0021489, -0.17734),
                    (0.0, -9.4445e-05, -0.30001),
                    (0.0, 0.0, -0.017558),
                ),
                (
                    (0.0, -0.064452, -0.1027),
                    (0.0, -0.052, -0.030465),
                    (0.025001, 0.0, -0.12412),
                    (-0.078273, -0.0021489, -0.17734),
                    (0.0, 9.4445e-05, -0.30001),
                    (0.0, 0.0, -0.017558),
                ),
            ),
            dtype=torch.float32,
        )
        identity = (1.0, 0.0, 0.0, 0.0)
        hip_roll = (0.996179, 0.0, -0.0873386, 0.0)
        knee = (0.996179, 0.0, 0.0873386, 0.0)
        body_quaternion = torch.tensor(
            ((identity, hip_roll, identity, knee, identity, identity),) * FOOT_COUNT,
            dtype=torch.float32,
        )
        joint_axis = torch.tensor(
            (
                (
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ),
            )
            * FOOT_COUNT,
            dtype=torch.float32,
        )
        joint_indices = torch.tensor(
            ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)),
            dtype=torch.long,
        )
        site_position = torch.tensor(
            ((0.04, 0.0, -0.037), (0.04, 0.0, -0.037)),
            dtype=torch.float32,
        )
        self.register_buffer("body_position", body_position)
        self.register_buffer("body_quaternion", body_quaternion)
        self.register_buffer("joint_axis", joint_axis)
        self.register_buffer("joint_indices", joint_indices)
        self.register_buffer("site_position", site_position)
        self.register_buffer("ground_height", torch.tensor(float(ground_height)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.size(-1) != ROBOT_STATE_DIM:
            raise ValueError(
                f"G1 foot kinematics expects [...,{ROBOT_STATE_DIM}], got {tuple(state.shape)}"
            )
        dtype = state.dtype
        body_position = self.body_position.to(dtype=dtype)
        body_quaternion = self.body_quaternion.to(dtype=dtype)
        joint_axis = self.joint_axis.to(dtype=dtype)
        site_position = self.site_position.to(dtype=dtype)
        joint_position = state[..., 13:42]
        joint_velocity = state[..., 42:71]
        root_position = state[..., :3]
        root_quaternion = torch.nn.functional.normalize(state[..., 3:7], dim=-1, eps=1.0e-8)
        root_linear_velocity = state[..., 7:10]
        root_angular_velocity = state[..., 10:13]

        features: list[torch.Tensor] = []
        for side in range(FOOT_COUNT):
            position = root_position
            quaternion = root_quaternion
            linear_velocity = root_linear_velocity
            angular_velocity = root_angular_velocity
            for link in range(6):
                offset = _quaternion_rotate(quaternion, body_position[side, link])
                next_position = position + offset
                next_linear_velocity = linear_velocity + torch.linalg.cross(
                    angular_velocity, offset, dim=-1
                )
                pre_joint_quaternion = torch.nn.functional.normalize(
                    _quaternion_multiply(quaternion, body_quaternion[side, link]),
                    dim=-1,
                    eps=1.0e-8,
                )
                axis = joint_axis[side, link]
                axis_world = _quaternion_rotate(pre_joint_quaternion, axis)
                joint_index = self.joint_indices[side, link]
                angle = joint_position[..., joint_index]
                half_angle = 0.5 * angle
                joint_quaternion = torch.cat(
                    (
                        torch.cos(half_angle).unsqueeze(-1),
                        axis * torch.sin(half_angle).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                quaternion = torch.nn.functional.normalize(
                    _quaternion_multiply(pre_joint_quaternion, joint_quaternion),
                    dim=-1,
                    eps=1.0e-8,
                )
                angular_velocity = angular_velocity + axis_world * joint_velocity[
                    ..., joint_index
                ].unsqueeze(-1)
                position = next_position
                linear_velocity = next_linear_velocity

            site_offset = _quaternion_rotate(quaternion, site_position[side])
            site_position_world = position + site_offset
            site_velocity_world = linear_velocity + torch.linalg.cross(
                angular_velocity, site_offset, dim=-1
            )
            height = site_position_world[..., 2:3] - self.ground_height.to(dtype=dtype)
            features.append(torch.cat((height, site_velocity_world), dim=-1))
        return torch.cat(features, dim=-1)


def _collapse_identical_rows(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2 and value.size(0) > 1:
        first = value[:1]
        if torch.equal(value, first.expand_as(value)):
            return first.squeeze(0)
    return value


class JointPositionTargetTransform(nn.Module):
    """External differentiable policy-action to physical PD-target mapping."""

    def __init__(
        self,
        *,
        scale: torch.Tensor,
        offset: torch.Tensor,
        encoder_bias: torch.Tensor,
        target_reindex: torch.Tensor,
        clip_lower: torch.Tensor | None = None,
        clip_upper: torch.Tensor | None = None,
        raw_action_clip: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("scale", _collapse_identical_rows(scale.detach().clone().float()))
        self.register_buffer("offset", _collapse_identical_rows(offset.detach().clone().float()))
        self.register_buffer(
            "encoder_bias",
            _collapse_identical_rows(encoder_bias.detach().clone().float()),
        )
        self.register_buffer("target_reindex", target_reindex.detach().clone().long())
        if (clip_lower is None) != (clip_upper is None):
            raise ValueError("clip_lower and clip_upper must either both be set or both be None")
        self.register_buffer(
            "clip_lower",
            None if clip_lower is None else _collapse_identical_rows(clip_lower.detach().clone()),
        )
        self.register_buffer(
            "clip_upper",
            None if clip_upper is None else _collapse_identical_rows(clip_upper.detach().clone()),
        )
        self.raw_action_clip = raw_action_clip
        self.contract = dict(metadata or {})

    @classmethod
    def from_mjlab(cls, env: Any, action_term: Any) -> "JointPositionTargetTransform":
        """Capture an exact memoryless MJLab joint-position action chain.

        Stateful delay, smoothing and boot-target overrides are rejected.  They
        need their own explicit recurrent controller state rather than a static
        affine transform.
        """

        action_dim = int(getattr(action_term, "action_dim", 0))
        if action_dim != ACTION_DIM:
            raise ValueError(f"Expected a {ACTION_DIM}-D joint action, got {action_dim}")
        max_delay = int(getattr(action_term, "max_delay", 0))
        alpha = getattr(action_term, "alpha", None)
        boot_delay = getattr(action_term, "boot_delay", None)
        boot_delay_steps = int(getattr(getattr(action_term, "cfg", None), "boot_delay_steps", 0))
        if max_delay > 0:
            raise ValueError(f"Action term has stateful delay (max_delay={max_delay})")
        if isinstance(alpha, torch.Tensor) and not torch.equal(alpha, torch.ones_like(alpha)):
            raise ValueError("Action term has stateful alpha smoothing")
        if boot_delay_steps > 0 or (
            isinstance(boot_delay, torch.Tensor) and bool((boot_delay > 0).any())
        ):
            raise ValueError("Action term has a stateful boot-target override")

        raw_action = getattr(action_term, "raw_action", None)
        if not isinstance(raw_action, torch.Tensor) or raw_action.shape[-1] != ACTION_DIM:
            raise ValueError("Action term does not expose a compatible raw-action tensor")
        scale_value = getattr(action_term, "_scale", None)
        offset_value = getattr(action_term, "_offset", None)
        scale = torch.as_tensor(scale_value, device=raw_action.device, dtype=raw_action.dtype)
        offset = torch.as_tensor(offset_value, device=raw_action.device, dtype=raw_action.dtype)
        if scale.ndim == 0:
            scale = scale.expand(ACTION_DIM)
        if offset.ndim == 0:
            offset = offset.expand(ACTION_DIM)
        joint_offset = getattr(action_term, "joint_offset", None)
        if isinstance(joint_offset, torch.Tensor):
            offset = offset + joint_offset

        robot = env.scene["robot"]
        target_ids = getattr(action_term, "target_ids", None)
        target_names = tuple(getattr(action_term, "target_names", ()))
        robot_names = tuple(robot.joint_names)
        if not isinstance(target_ids, torch.Tensor) or set(target_names) != set(robot_names):
            raise ValueError("Predictor requires one action target for every robot joint")
        target_reindex = torch.as_tensor(
            [target_names.index(name) for name in robot_names],
            device=raw_action.device,
            dtype=torch.long,
        )
        encoder_bias = robot.data.encoder_bias[:, target_ids]

        clip_lower: torch.Tensor | None = None
        clip_upper: torch.Tensor | None = None
        cfg = getattr(action_term, "cfg", None)
        if getattr(cfg, "clip", None) is not None:
            clip = getattr(action_term, "_clip", None)
            if not isinstance(clip, torch.Tensor) or clip.shape[-2:] != (ACTION_DIM, 2):
                raise ValueError("Action term has clip settings but no compatible clip tensor")
            clip_lower = clip[..., 0]
            clip_upper = clip[..., 1]
        raw_clip_value = getattr(cfg, "raw_action_clip", None)
        raw_action_clip = None if raw_clip_value is None else float(raw_clip_value)
        metadata = {
            "action_term_class": type(action_term).__name__,
            "action_delay": False,
            "alpha_smoothing": False,
            "boot_delay": False,
            "predictor_action": "physical_pd_joint_target_rad",
            "policy_action": "raw_normalized_joint_position_action",
            "target_names": list(robot_names),
        }
        return cls(
            scale=scale,
            offset=offset,
            encoder_bias=encoder_bias,
            target_reindex=target_reindex,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            raw_action_clip=raw_action_clip,
            metadata=metadata,
        )

    @staticmethod
    def _select_world_rows(
        value: torch.Tensor,
        env_ids: torch.Tensor | None,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        if value.ndim < 2:
            return value
        if env_ids is None:
            if value.size(0) not in (1, batch_size):
                raise ValueError(
                    f"Per-world action-transform {name} has {value.size(0)} rows but "
                    f"the action batch has {batch_size}; pass env_ids when sub-sampling worlds"
                )
            return value
        env_ids = env_ids.to(device=value.device, dtype=torch.long)
        if env_ids.shape != (batch_size,):
            raise ValueError(f"env_ids must have shape [{batch_size}], got {tuple(env_ids.shape)}")
        if value.size(0) == 1:
            return value
        if env_ids.numel() and (
            bool((env_ids < 0).any()) or bool((env_ids >= value.size(0)).any())
        ):
            raise IndexError(
                f"env_ids must index [0,{value.size(0)}), got range "
                f"[{int(env_ids.min())},{int(env_ids.max())}]"
            )
        return value.index_select(0, env_ids)

    def forward(
        self,
        policy_action: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if policy_action.size(-1) != ACTION_DIM:
            raise ValueError(
                f"Policy action must end in {ACTION_DIM} values, got {tuple(policy_action.shape)}"
            )
        action = policy_action
        if self.raw_action_clip is not None:
            action = action.clamp(-self.raw_action_clip, self.raw_action_clip)
        batch_size = policy_action.numel() // ACTION_DIM
        if policy_action.ndim != 2:
            if env_ids is not None:
                raise ValueError("env_ids sub-sampling requires a two-dimensional action batch")
            batch_size = policy_action.size(0)
        scale = self._select_world_rows(self.scale, env_ids, batch_size, "scale")
        offset = self._select_world_rows(self.offset, env_ids, batch_size, "offset")
        encoder_bias = self._select_world_rows(
            self.encoder_bias, env_ids, batch_size, "encoder_bias"
        )
        target = action * scale + offset
        if self.clip_lower is not None and self.clip_upper is not None:
            clip_lower = self._select_world_rows(self.clip_lower, env_ids, batch_size, "clip_lower")
            clip_upper = self._select_world_rows(self.clip_upper, env_ids, batch_size, "clip_upper")
            target = torch.clamp(target, min=clip_lower, max=clip_upper)
        target = target - encoder_bias
        return target.index_select(-1, self.target_reindex)
