"""Explicit physics for future PPO runs; historical experiments retain their physics."""

from __future__ import annotations

import copy
from dataclasses import replace
import re

import numpy as np
import torch

from intact_tracking.fixed_dr_profiles import configure_fixed_dr, audit_fixed_dr, representative_bank
from intact_tracking.limb_context_dr import TRACKER_DR, configure_limb_dr, audit_limb_dr, _event_contract
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT, UniformLimbPayload
from intact_tracking.memory350_moe_policy import HardMoEActor
from intact_tracking.memory350_moe_critic_action import configure_critic_action_models

VERSION = "residual_uniform_wrist10_hand2p5_unbounded_v1"
EVAL_VERSION = VERSION + "_evaluation"
ACTOR_CLASS = "intact_tracking.residual_uniform_protocol:UnboundedResidualActor"
WRIST_JOINTS = tuple(f"{side}_wrist_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "yaw"))
MAX_MASSES = (2.5, 2.5, 4.0, 4.0)
WRIST_TORQUE_NM = 10.0


def physics_contract():
    return {"version": VERSION, "motion_sampling": "uniform", "limb_max_masses_kg": list(MAX_MASSES),
            "wrist_pitch_yaw_effort_limit_nm": WRIST_TORQUE_NM,
            "tracker_action_scale": "preserved from frozen tracker", "pd_gains": "preserved"}


def sample_masses(num_envs, seed, fixed_masses=None):
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    limits = torch.tensor(MAX_MASSES)
    if fixed_masses is not None:
        masses = torch.as_tensor(fixed_masses, dtype=torch.float32)
        if masses.shape != (4,) or not torch.isfinite(masses).all() or not ((masses >= 0) & (masses <= limits)).all():
            raise ValueError("Payload must be within hand [0,2.5] and shin [0,4] kg")
        return masses.expand(num_envs, -1).clone()
    return torch.rand(num_envs, 4, generator=torch.Generator().manual_seed(seed)) * limits


class CappedLimbPayload(UniformLimbPayload):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mass = sample_masses(env.num_envs, cfg.params["seed"], cfg.params.get("fixed_masses")).to(env.device)


def wrist_articulation(articulation):
    actuators, changed = [], set()
    for actuator in articulation.actuators:
        patterns = tuple(actuator.target_names_expr)
        selected = {joint for joint in WRIST_JOINTS if any(re.fullmatch(pattern, joint) for pattern in patterns)}
        if selected:
            # The current G1 groups exactly pitch/yaw. Never expand a mixed
            # actuator group's other joints as an unintended side effect.
            allowed = {".*_wrist_pitch_joint", ".*_wrist_yaw_joint", *WRIST_JOINTS}
            if not set(patterns) <= allowed:
                raise ValueError("Unexpected mixed wrist actuator group")
            actuator = replace(actuator, effort_limit=WRIST_TORQUE_NM)
            changed.update(selected)
        actuators.append(actuator)
    if changed != set(WRIST_JOINTS):
        raise ValueError("Expected all four wrist pitch/yaw actuators")
    return replace(articulation, actuators=tuple(actuators))


def configure_physics(env_cfg, seed, *, profile=TRACKER_DR, fixed_masses=None, bank_path=None, profile_id=None):
    if profile != TRACKER_DR:
        raise ValueError("This PPO protocol uses original tracker DR plus limb payloads")
    robot = env_cfg.scene.entities["robot"]
    original_scale = copy.deepcopy(env_cfg.actions["joint_pos"].scale)
    robot.articulation = wrist_articulation(robot.articulation)
    if bank_path is not None:
        if fixed_masses is not None or profile_id is None:
            raise ValueError("Use the fixed DR bank without a second payload override")
        metadata = configure_fixed_dr(env_cfg, seed, bank_path=bank_path, profile_id=profile_id, profile=profile)
        sample_masses(1, seed, metadata["fixed_masses"])
    else:
        sample_masses(1, seed, fixed_masses)
        metadata = configure_limb_dr(env_cfg, seed, profile=profile, fixed_masses=fixed_masses)
    env_cfg.events[PAYLOAD_EVENT].func = CappedLimbPayload
    metadata.update(residual_physics_contract=physics_contract(), total_added_mass_range_kg=[0, sum(MAX_MASSES)])
    if bank_path is None:
        metadata.update(profile=VERSION, sampling="Independent hand U(0,2.5) and shin U(0,4) kg per world; fixed at startup")
    if "effective_events" in metadata:
        metadata["effective_events"] = {name: _event_contract(term) for name, term in env_cfg.events.items()}
    if env_cfg.actions["joint_pos"].scale != original_scale:
        raise RuntimeError("Changing force limits must not change the frozen tracker's action scale")
    return metadata


def audit_physics(env, metadata):
    if metadata.get("residual_physics_contract") != physics_contract():
        raise ValueError("Residual physics differs from its saved contract")
    result = audit_fixed_dr(env, metadata) if "fixed_dr" in metadata else audit_limb_dr(env, metadata)
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    actual = payload.observe()
    limits = actual.new_tensor(MAX_MASSES)
    if not ((actual >= -1e-5) & (actual <= limits + 1e-5)).all():
        raise RuntimeError("Runtime payload exceeded the per-limb cap")
    robot = env.scene["robot"]
    local, _ = robot.find_joints(list(WRIST_JOINTS), preserve_order=True)
    joint_ids = robot.indexing.joint_ids[torch.as_tensor(local, device=env.device)].cpu().numpy()
    model = env.sim.mj_model
    controls = [np.flatnonzero((model.actuator_trnid[:, 0] == joint_id) & (model.actuator_trntype == 0)) for joint_id in joint_ids]
    if any(len(ids) != 1 for ids in controls):
        raise RuntimeError("Expected exactly one joint actuator for each wrist axis")
    controls = torch.tensor([int(ids[0]) for ids in controls], device=env.device)
    force = env.sim.model.actuator_forcerange
    observed = force[:, controls] if force.ndim == 3 else force[controls][None]
    torch.testing.assert_close(observed, observed.new_tensor([-WRIST_TORQUE_NM, WRIST_TORQUE_NM]).expand_as(observed), atol=0, rtol=0)
    result["residual_physics_contract"] = physics_contract()
    result["wrist_force_ranges_nm"] = observed[0].cpu().tolist()
    action = env.action_manager.get_term("joint_pos")
    names = tuple(action._target_names)
    targets = robot.indexing.joint_ids[action.target_ids.long()].cpu().numpy()
    ids = [np.flatnonzero((model.actuator_trnid[:, 0] == j) & (model.actuator_trntype == 0)) for j in targets]
    if any(len(value) != 1 for value in ids):
        raise RuntimeError("Expected one position actuator per action joint")
    ids = torch.tensor([int(value[0]) for value in ids], device=env.device)
    gains = env.sim.model.actuator_gainprm
    gains = gains[:, ids, 0] if gains.ndim == 3 else gains[ids, 0][None]
    kp = env.sim.get_default_field("actuator_gainprm")[ids, 0]
    torch.testing.assert_close(gains, kp.to(gains).expand_as(gains), atol=1e-5, rtol=1e-6)
    actual_scale = torch.as_tensor(action._scale, device=env.device).expand(env.num_envs, len(names))
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE
    scale = actual_scale.new_tensor([next(v for p, v in G1_ACTION_SCALE.items() if re.fullmatch(p, name)) for name in names])
    torch.testing.assert_close(actual_scale, scale.expand_as(actual_scale), atol=1e-6, rtol=1e-6)
    if action.cfg.clip is not None:
        raise RuntimeError("A position-target clamp would invalidate the requested residual correction space")
    result["action_order"] = list(names)
    result["pd_gains_and_tracker_action_scale_verified"] = True
    return result


def representative_profiles(seed=20260914):
    bank = representative_bank(seed)
    bank["residual_physics_contract"] = physics_contract()
    bank["selection"] = "Eight fixed DRs with hand cap 2.5 kg and shin cap 4 kg"
    names = ("all_0kg", "all_1kg", "all_2kg", "hands_2p5_shins_4kg",
             "hands_2p5kg", "shins_4kg", "left_hand_2p5_shin_4kg", "right_hand_2p5_shin_4kg")
    for entry, name in zip(bank["profiles"], names, strict=True):
        entry["name"] = name
        entry["masses_kg"] = [min(mass, limit) for mass, limit in zip(entry["masses_kg"], MAX_MASSES, strict=True)]
    return bank


class UnboundedResidualActor(HardMoEActor):
    def _residual(self, value):
        return self.residual_mlp(value)

    @torch.no_grad()
    def policy_metrics(self, obs):
        result = super().policy_metrics(obs)
        result.pop("residual_saturation_fraction", None)
        result["residual_output_bounded"] = 0.0
        return result


def configure_models(train, fusion, **kwargs):
    result = configure_critic_action_models(train, fusion, **kwargs)
    result["actor"].update(class_name=ACTOR_CLASS)
    return result
