from __future__ import annotations

from dataclasses import replace
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


def get_g1_tracking_bfm_mesh_arm_spec() -> mujoco.MjSpec:
    """Reproduce the SP mesh-arm collision variant from the shared BFM MJCF."""

    spec = get_g1_tracking_bfm_spec()
    spec.modelname = "g1_29dof_rev_1_0_mesh_arm"
    collision_default = spec.find_default("collision")
    wrist_fromto = (
        -0.01799265,
        0.0,
        -0.000247097,
        0.04150020,
        0.0,
        -0.000247097,
    )
    for side in ("left", "right"):
        wrist = spec.geom(f"{side}_wrist_yaw_collision")
        wrist_body = wrist.parent
        spec.delete(wrist)
        wrist_body.add_geom(
            default=collision_default,
            name=f"{side}_wrist_yaw_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(0.030247096,),
            fromto=wrist_fromto,
        )
        for suffix in ("palm", "knuckle", "finger_root"):
            spec.delete(spec.geom(f"{side}_hand_{suffix}_collision"))
    return spec


def get_g1_tracking_bfm_mesh_arm_spv1_spec() -> mujoco.MjSpec:
    """Return the mesh-arm collision model with SPV joint torque sensors."""

    spec = get_g1_tracking_bfm_mesh_arm_spec()
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


def get_g1_tracking_bfm_mesh_arm_spv1_robot_cfg() -> EntityCfg:
    """Use the mesh-arm collision model required by recent SPV5 checkpoints."""

    return replace(
        get_g1_tracking_bfm_spv1_robot_cfg(),
        spec_fn=get_g1_tracking_bfm_mesh_arm_spv1_spec,
    )
