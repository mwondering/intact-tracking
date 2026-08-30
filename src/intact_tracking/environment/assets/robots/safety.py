from __future__ import annotations

from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION
from mjlab.entity import EntityArticulationInfoCfg

SAFETY_EFFORT_LIMITS = {
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 35.0,
    "left_ankle_roll_joint": 35.0,
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 35.0,
    "right_ankle_roll_joint": 35.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 35.0,
    "waist_pitch_joint": 35.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}


def get_safe_g1_articulation() -> EntityArticulationInfoCfg:
    """Return G1 articulation with sim2real-safe minimum effort limits."""
    actuators = []
    for actuator in G1_ARTICULATION.actuators:
        names = tuple(actuator.target_names_expr)
        effort_limit = actuator.effort_limit
        if names == ("waist_pitch_joint", "waist_roll_joint"):
            effort_limit = 35.0
        elif names == (".*_ankle_pitch_joint", ".*_ankle_roll_joint"):
            effort_limit = 35.0
        actuators.append(replace(actuator, effort_limit=effort_limit))
    return replace(G1_ARTICULATION, actuators=tuple(actuators))
