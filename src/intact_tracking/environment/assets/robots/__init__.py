"""G1 robot configuration used by the rollout runtime."""

from .g1_tracking_bfm import (
    SPV1_JOINT_TORQUE_SENSOR_PREFIX,
    get_g1_tracking_bfm_robot_cfg,
    get_g1_tracking_bfm_spv1_robot_cfg,
)

__all__ = [
    "SPV1_JOINT_TORQUE_SENSOR_PREFIX",
    "get_g1_tracking_bfm_robot_cfg",
    "get_g1_tracking_bfm_spv1_robot_cfg",
]
