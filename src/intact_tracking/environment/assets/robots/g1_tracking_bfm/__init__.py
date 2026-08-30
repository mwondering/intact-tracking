from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg

from intact_tracking.environment.assets.robots.safety import get_safe_g1_articulation

G1_TRACKING_BFM_XML = Path(__file__).with_name("g1.xml")
SPV1_JOINT_TORQUE_SENSOR_PREFIX = "spv1_joint_torque_"


def get_g1_tracking_bfm_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(G1_TRACKING_BFM_XML))


def get_g1_tracking_bfm_spv1_spec() -> mujoco.MjSpec:
    """Return the BFM model with joint-side actuator-force sensors for SPV1."""
    spec = get_g1_tracking_bfm_spec()
    for joint in spec.joints:
        if int(joint.type) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        spec.add_sensor(
            name=f"{SPV1_JOINT_TORQUE_SENSOR_PREFIX}{joint.name}",
            type=mujoco.mjtSensor.mjSENS_JOINTACTFRC,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint.name,
        )
    return spec


def get_g1_tracking_bfm_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_g1_tracking_bfm_spec,
        articulation=get_safe_g1_articulation(),
    )


def get_g1_tracking_bfm_spv1_robot_cfg() -> EntityCfg:
    """Keep BFM dynamics while exposing task-local measured joint torques."""
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_g1_tracking_bfm_spv1_spec,
        articulation=get_safe_g1_articulation(),
    )
