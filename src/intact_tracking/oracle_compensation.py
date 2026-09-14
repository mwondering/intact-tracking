"""Privileged diagnostic: compensate gravity/Coriolis torque relative to nominal.

This controller is a feasibility probe, not a learned policy or a deployment
claim. It never changes the real world's model; a separate nominal simulator
is evaluated at the same state to expose the missing generalized torque.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from intact_tracking.adaptation_policy import expanded_model_field
from intact_tracking.rollout.nominal import _make_nominal_dynamics_cfg
from intact_tracking.rollout.online import DEFAULT_PAYLOAD_POSITION_BODY_M


def payload_gravity_joint_torque(center, anchors, axes, mass, gravity, ancestors):
    """Virtual-work gravity torque for hinge ancestors of a rigid payload.

    The actuator must oppose the payload's gravity wrench: -J_position.T m g.
    Nonancestor joints receive exactly zero; no contact or inertial terms are
    approximated here. All arguments are observational tensors.
    """
    force = -mass[..., None] * gravity
    moment = torch.linalg.cross(center[..., None, :] - anchors, force[..., None, :], dim=-1)
    return (moment * axes).sum(-1) * ancestors


class PayloadGravityCompensation:
    """Payload-only analytical DIAGNOSTIC, not a trained policy or acceptance.

    Uses current kinematics instead of comparing a possibly lagged bias-force
    buffer with a separately forwarded nominal twin. This does not establish
    that timing was the cause of earlier bias-compensation failures.
    """

    def __init__(self, real_env):
        self.real = real_env
        model = real_env.sim.mj_model
        bodies = [i for i in range(model.nbody) if model.body(i).name.split("/")[-1] == "right_wrist_yaw_link"]
        if len(bodies) != 1:
            raise ValueError("Expected exactly one right-hand payload body")
        self.body = bodies[0]
        event = real_env.cfg.events.get("right_hand_payload")
        if event is not None and tuple(event.params["position_body_m"]) != DEFAULT_PAYLOAD_POSITION_BODY_M:
            raise ValueError("Payload gravity probe offset differs from the configured payload")
        robot = real_env.scene["robot"]
        action = real_env.action_manager.get_term("joint_pos")
        self.joints = robot.indexing.joint_ids[action.target_ids].long()
        joint_ids = self.joints.cpu().tolist()
        ancestors = set()
        body = self.body
        while body:
            ancestors.add(body)
            body = int(model.body_parentid[body])
        self.ancestors = torch.tensor([int(model.jnt_bodyid[j]) in ancestors for j in joint_ids], device=real_env.device)
        controls = []
        for joint in joint_ids:
            # Only hinge joints have the scalar rotational Jacobian used here.
            if int(model.jnt_type[joint]) != 3:
                raise ValueError("Payload probe requires hinge-joint action targets")
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError("Expected one position actuator per action target")
            controls.append(int(matches[0]))
        gain, _ = expanded_model_field(real_env, "actuator_gainprm")
        self.torque_per_action = gain[:, controls, 0] * torch.as_tensor(action._scale, device=real_env.device)
        if not bool((self.torque_per_action > 0).all()):
            raise ValueError("Payload correction needs positive position gains and action scales")
        self.offset = torch.tensor(DEFAULT_PAYLOAD_POSITION_BODY_M, device=real_env.device).expand(real_env.num_envs, -1)
        self.gravity = torch.as_tensor(model.opt.gravity.copy(), device=real_env.device, dtype=torch.float32)

    @torch.no_grad()
    def correction(self):
        data = self.real.sim.data
        center = data.xpos[:, self.body] + quat_apply(data.xquat[:, self.body], self.offset)
        mass, nominal_mass = expanded_model_field(self.real, "body_mass")
        payload = mass[:, self.body] - nominal_mass[self.body]
        torque = payload_gravity_joint_torque(
            center, data.xanchor[:, self.joints], data.xaxis[:, self.joints],
            payload, self.gravity, self.ancestors,
        )
        return (torque / self.torque_per_action).clamp(-3.0, 3.0)

    def close(self):
        pass  # Owns no simulator and never modifies the real model/state.


class NominalBiasCompensation:
    def __init__(self, real_env):
        self.real = real_env
        cfg = deepcopy(real_env.cfg)
        _make_nominal_dynamics_cfg(cfg)
        self.nominal = ManagerBasedRlEnv(cfg=cfg, device=real_env.device)
        robot = real_env.scene["robot"]
        action = real_env.action_manager.get_term("joint_pos")
        self.dofs = robot.indexing.joint_v_adr[action.target_ids].long()
        joints = robot.indexing.joint_ids[action.target_ids].cpu().tolist()
        model = real_env.sim.mj_model
        controls = []
        for joint in joints:
            candidates = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one position actuator for joint {joint}")
            controls.append(int(candidates[0]))
        gain, _ = expanded_model_field(real_env, "actuator_gainprm")
        self.kp = gain[:, controls, 0].clone()
        self.scale = torch.as_tensor(action._scale, device=real_env.device)
        if not bool((self.kp > 0).all()) or not bool((self.scale.abs() > 0).all()):
            raise ValueError("Compensation needs nonzero action scales and positive position gains")
        if real_env.sim.data.qpos.shape != self.nominal.sim.data.qpos.shape:
            raise ValueError("Nominal twin topology mismatch")

    @torch.no_grad()
    def correction(self):
        actual = self.real.sim.data
        nominal = self.nominal.sim.data
        nominal.qpos.copy_(actual.qpos)
        nominal.qvel.copy_(actual.qvel)
        self.nominal.sim.forward()
        torque = (actual.qfrc_bias - nominal.qfrc_bias)[:, self.dofs]
        return (torque / (self.kp * self.scale)).clamp(-3.0, 3.0)

    def close(self):
        self.nominal.close()
