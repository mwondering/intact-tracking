"""Current-frame HEFT key-body observations for the privileged SPV4 actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from intact_tracking.environment.mdp.keypoints import (
    KeypointKinematics,
    SemanticKeypointResolver,
)

from . import sp as sp_mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


SPV4_KEY_BODY_COUNT = 13
SPV4_KEY_BODY_STATE_DIM = SPV4_KEY_BODY_COUNT * (3 + 6 + 3 + 3)


@dataclass(frozen=True)
class RootFrameKeyBodyState:
    """Semantic key-body state expressed relative to one root frame."""

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor


def _root_frame_state(
    keypoints: KeypointKinematics,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
) -> RootFrameKeyBodyState:
    root_quat = root_quat_w.unsqueeze(1)
    return RootFrameKeyBodyState(
        pos=sp_mdp._quat_apply_inverse(root_quat, keypoints.pos_w - root_pos_w.unsqueeze(1)),
        quat=sp_mdp._quat_in_frame(root_quat.expand_as(keypoints.quat_w), keypoints.quat_w),
        lin_vel=sp_mdp._quat_apply_inverse(
            root_quat, keypoints.lin_vel_w - root_lin_vel_w.unsqueeze(1)
        ),
        ang_vel=sp_mdp._quat_apply_inverse(
            root_quat, keypoints.ang_vel_w - root_ang_vel_w.unsqueeze(1)
        ),
    )


def _pack_state(state: RootFrameKeyBodyState) -> torch.Tensor:
    num_envs = state.pos.shape[0]
    return torch.cat(
        (
            state.pos.reshape(num_envs, -1),
            sp_mdp._rot6d(state.quat).reshape(num_envs, -1),
            state.lin_vel.reshape(num_envs, -1),
            state.ang_vel.reshape(num_envs, -1),
        ),
        dim=-1,
    )


def _key_body_error(
    robot: RootFrameKeyBodyState,
    reference: RootFrameKeyBodyState,
    robot_from_reference_root: torch.Tensor,
) -> torch.Tensor:
    """Reference-minus-robot error, entirely in the current robot root frame."""
    relative_root = robot_from_reference_root.unsqueeze(1).expand_as(reference.quat)
    reference_pos = quat_apply(relative_root, reference.pos)
    reference_quat = quat_mul(relative_root, reference.quat)
    reference_lin_vel = quat_apply(relative_root, reference.lin_vel)
    reference_ang_vel = quat_apply(relative_root, reference.ang_vel)

    rotation_error_quat = sp_mdp._quat_in_frame(robot.quat, reference_quat)
    rotation_error = sp_mdp._rot6d(rotation_error_quat)
    identity_rot6d = rotation_error.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    rotation_error = rotation_error - identity_rot6d
    num_envs = robot.pos.shape[0]
    return torch.cat(
        (
            (reference_pos - robot.pos).reshape(num_envs, -1),
            rotation_error.reshape(num_envs, -1),
            (reference_lin_vel - robot.lin_vel).reshape(num_envs, -1),
            (reference_ang_vel - robot.ang_vel).reshape(num_envs, -1),
        ),
        dim=-1,
    )


class _SPV4KeyBodyObservation:
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        self.cfg = cfg
        self.env = env
        self.asset = env.scene["robot"]
        self.command_name = str(cfg.params.get("command_name", "motion"))
        self.root_body_name = cfg.params.get("root_body_name")
        command = sp_mdp._command(env, self.command_name)
        raw_specs = cfg.params.get("keypoint_specs", sp_mdp._default_keypoint_specs())
        self.resolver = SemanticKeypointResolver(
            self.asset, tuple(command.cfg.body_names), raw_specs
        )
        if len(self.resolver.names) != SPV4_KEY_BODY_COUNT:
            raise ValueError(
                f"SPV4 requires {SPV4_KEY_BODY_COUNT} HEFT semantic key bodies, "
                f"got {len(self.resolver.names)}: {self.resolver.names}"
            )
        self._cache_key = (self.resolver.specs, self.root_body_name)

    def _robot_state(self) -> tuple[RootFrameKeyBodyState, torch.Tensor]:
        data = self.asset.data
        state = _root_frame_state(
            self.resolver.current(self.asset),
            data.root_link_pos_w,
            data.root_link_quat_w,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
        )
        return state, data.root_link_quat_w

    def _reference_state(self) -> tuple[RootFrameKeyBodyState, torch.Tensor]:
        steps = (0,)
        keypoints = self.resolver.reference(
            sp_mdp._gather(self.env, self.command_name, "body_pos_w", steps),
            sp_mdp._gather(self.env, self.command_name, "body_quat_w", steps),
            sp_mdp._gather(self.env, self.command_name, "body_lin_vel_w", steps),
            sp_mdp._gather(self.env, self.command_name, "body_ang_vel_w", steps),
        )
        root_pos = sp_mdp._root_motion(
            self.env,
            self.command_name,
            "body_pos_w",
            steps,
            root_body_name=self.root_body_name,
        )
        root_quat = sp_mdp._root_motion(
            self.env,
            self.command_name,
            "body_quat_w",
            steps,
            root_body_name=self.root_body_name,
        )
        root_lin_vel = sp_mdp._root_motion(
            self.env,
            self.command_name,
            "body_lin_vel_w",
            steps,
            root_body_name=self.root_body_name,
        )
        root_ang_vel = sp_mdp._root_motion(
            self.env,
            self.command_name,
            "body_ang_vel_w",
            steps,
            root_body_name=self.root_body_name,
        )
        state = _root_frame_state(
            KeypointKinematics(
                pos_w=keypoints.pos_w[:, 0],
                quat_w=keypoints.quat_w[:, 0],
                lin_vel_w=keypoints.lin_vel_w[:, 0],
                ang_vel_w=keypoints.ang_vel_w[:, 0],
            ),
            root_pos[:, 0],
            root_quat[:, 0],
            root_lin_vel[:, 0],
            root_ang_vel[:, 0],
        )
        return state, root_quat[:, 0]

    def _values(self) -> dict[str, torch.Tensor]:
        command = sp_mdp._command(self.env, self.command_name)
        cache = getattr(command, "_shared_spv4_key_body_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            command._shared_spv4_key_body_cache = cache
        cached = cache.get(self._cache_key)
        if cached is not None:
            return cached

        robot, robot_root_quat = self._robot_state()
        reference, reference_root_quat = self._reference_state()
        robot_from_reference_root = sp_mdp._quat_in_frame(robot_root_quat, reference_root_quat)
        values = {
            "robot": _pack_state(robot),
            "reference": _pack_state(reference),
            "error": _key_body_error(robot, reference, robot_from_reference_root),
        }
        for name, value in values.items():
            if value.shape[-1] != SPV4_KEY_BODY_STATE_DIM:
                raise RuntimeError(
                    f"SPV4 {name} key-body state has {value.shape[-1]} values, "
                    f"expected {SPV4_KEY_BODY_STATE_DIM}"
                )
        cache[self._cache_key] = values
        return values


class robot_key_body_state(_SPV4KeyBodyObservation):
    def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
        del env
        return self._values()["robot"]


class ref_key_body_state(_SPV4KeyBodyObservation):
    def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
        del env
        return self._values()["reference"]


class key_body_error(_SPV4KeyBodyObservation):
    def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
        del env
        return self._values()["error"]
