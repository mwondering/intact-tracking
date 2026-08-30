"""Deploy-equivalent noisy-FK robot key-body observations for SPV5-2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

from intact_tracking.environment.mdp.keypoints import (
    KeypointSpec,
    _rigid_points,
    parse_keypoint_specs,
)

from . import spv1 as spv1_mdp
from .motion_fk import MotionFKHelper
from .spv4 import (
    SPV4_KEY_BODY_COUNT,
    SPV4_KEY_BODY_STATE_DIM,
    RootFrameKeyBodyState,
    _pack_state,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _basename(name: str) -> str:
    return name.split("/")[-1]


class SPV52RobotKeyBodyKinematics:
    """Rebuild the 13 semantic key bodies from measured joint state and gyro."""

    def __init__(
        self,
        helper: MotionFKHelper,
        raw_specs: Iterable[KeypointSpec | Mapping[str, Any]],
    ):
        self.helper = helper
        self.specs = parse_keypoint_specs(raw_specs)
        if len(self.specs) != SPV4_KEY_BODY_COUNT:
            raise ValueError(
                f"SPV5-2 requires {SPV4_KEY_BODY_COUNT} HEFT semantic key bodies, "
                f"got {len(self.specs)}"
            )

        body_index = {name: index for index, name in enumerate(self.helper.output_body_names)}
        self.parent_ids = torch.tensor(
            [body_index[spec.asset_body_name] for spec in self.specs],
            device=self.helper.device,
            dtype=torch.long,
        )
        self.correction_ids = torch.tensor(
            [
                body_index[spec.asset_correction_body_name or spec.asset_body_name]
                for spec in self.specs
            ],
            device=self.helper.device,
            dtype=torch.long,
        )
        self.local_pos = torch.tensor(
            [spec.asset_local_pos for spec in self.specs],
            device=self.helper.device,
            dtype=torch.float32,
        )
        self.local_quat = torch.tensor(
            [spec.asset_local_quat for spec in self.specs],
            device=self.helper.device,
            dtype=torch.float32,
        )
        self.correction_local_pos = torch.tensor(
            [spec.asset_correction_local_pos for spec in self.specs],
            device=self.helper.device,
            dtype=torch.float32,
        )

    @staticmethod
    def physical_body_names(
        specs: tuple[KeypointSpec, ...],
    ) -> tuple[str, ...]:
        names: list[str] = []
        for spec in specs:
            for name in (
                spec.asset_body_name,
                spec.asset_correction_body_name,
            ):
                if name is not None and name not in names:
                    names.append(name)
        return tuple(names)

    @classmethod
    def from_mjlab_asset(
        cls,
        *,
        asset: Any,
        raw_specs: Iterable[KeypointSpec | Mapping[str, Any]],
    ) -> SPV52RobotKeyBodyKinematics:
        specs = parse_keypoint_specs(raw_specs)
        helper = MotionFKHelper.from_mjlab_asset(
            asset=asset,
            dataset_joint_names=tuple(_basename(name) for name in asset.joint_names),
            output_body_names=cls.physical_body_names(specs),
        )
        return cls(helper, specs)

    def __call__(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
    ) -> torch.Tensor:
        body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b = self.helper.body_kinematics(
            joint_pos,
            joint_vel,
            root_ang_vel_b,
        )
        parent_ids = self.parent_ids.to(body_pos_b.device)
        correction_ids = self.correction_ids.to(body_pos_b.device)
        keypoints = _rigid_points(
            body_pos_b.index_select(-2, parent_ids),
            body_quat_b.index_select(-2, parent_ids),
            body_lin_vel_b.index_select(-2, parent_ids),
            body_ang_vel_b.index_select(-2, parent_ids),
            self.local_pos.to(body_pos_b),
            self.local_quat.to(body_quat_b),
            body_quat_b.index_select(-2, correction_ids),
            body_ang_vel_b.index_select(-2, correction_ids),
            self.correction_local_pos.to(body_pos_b),
        )
        packed = _pack_state(
            RootFrameKeyBodyState(
                pos=keypoints.pos_w,
                quat=keypoints.quat_w,
                lin_vel=keypoints.lin_vel_w,
                ang_vel=keypoints.ang_vel_w - root_ang_vel_b.unsqueeze(-2),
            )
        )
        if packed.shape[-1] != SPV4_KEY_BODY_STATE_DIM:
            raise RuntimeError(
                f"SPV5-2 noisy-FK key-body state has {packed.shape[-1]} values, "
                f"expected {SPV4_KEY_BODY_STATE_DIM}"
            )
        return packed


class robot_key_body_state:
    """Compute robot key bodies from the same noisy sensors used at deployment."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        self.asset = env.scene["robot"]
        self.command_name = str(cfg.params.get("command_name", "motion"))
        self.biased = bool(cfg.params.get("biased", True))
        self.joint_pos_noise_std = float(cfg.params.get("joint_pos_noise_std", 0.0))
        self.joint_vel_noise_std = float(cfg.params.get("joint_vel_noise_std", 0.0))
        self.gyro_sensor_name = str(cfg.params.get("gyro_sensor_name", "robot/imu_ang_vel"))
        self.gyro_noise_std = float(cfg.params.get("gyro_noise_std", 0.0))
        self.kinematics = SPV52RobotKeyBodyKinematics.from_mjlab_asset(
            asset=self.asset,
            raw_specs=cfg.params["keypoint_specs"],
        )

    def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
        data = self.asset.data
        joint_pos = (
            spv1_mdp.joint_pos(
                env,
                command_name=self.command_name,
                biased=self.biased,
                noise_std=self.joint_pos_noise_std,
            )
            + data.default_joint_pos
        )
        joint_vel = (
            spv1_mdp.joint_vel(
                env,
                command_name=self.command_name,
                noise_std=self.joint_vel_noise_std,
            )
            + data.default_joint_vel
        )
        root_ang_vel_b = spv1_mdp.base_ang_vel(
            env,
            command_name=self.command_name,
            sensor_name=self.gyro_sensor_name,
            noise_std=self.gyro_noise_std,
        )
        return self.kinematics(joint_pos, joint_vel, root_ang_vel_b)
