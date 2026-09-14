"""Read-only PD diagnostics; direct bias recovery is invalid for this controller.

MJLab subtracts encoder bias from commanded position targets and adds it to
measured positions. Consequently that bias cancels from the ideal PD residual.
The earlier numerical probe is retained as failed historical evidence, not a
usable estimator. No action path imports this module.
"""

from __future__ import annotations

import torch


def nominal_pd_contract(env):
    """Extract fixed, nominal controller constants; never randomized physics labels."""
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if type(action).__name__ != "ObservationHistoryJointPositionAction":
        raise ValueError("PD diagnostic requires the unchanged, undelayed position-action interface")
    targets = action.target_ids.long()
    if targets.numel() != 29 or not torch.equal(targets.sort().values, torch.arange(29, device=targets.device)):
        raise ValueError("PD diagnostic requires all29 uniquely mapped robot joints")
    controls = []
    model = env.sim.mj_model
    for joint in robot.indexing.joint_ids[targets].tolist():
        matches = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == joint]
        if len(matches) != 1:
            raise ValueError("PD diagnostic requires exactly one actuator per joint")
        controls.append(matches[0])
    gain = env.sim.get_default_field("actuator_gainprm")[controls, 0]
    bias = env.sim.get_default_field("actuator_biasprm")[controls]
    torch.testing.assert_close(bias[:, 1], -gain)
    if not bool((gain > 0).all()):
        raise ValueError("Nominal position gains must be positive")
    scale = torch.as_tensor(action._scale, device=gain.device).expand(env.num_envs, 29)
    torch.testing.assert_close(scale, scale[:1].expand_as(scale))
    offset = torch.as_tensor(action._offset, device=gain.device).expand(env.num_envs, 29)
    torch.testing.assert_close(offset, robot.data.default_joint_pos[:, targets])
    force_range = env.sim.get_default_field("actuator_forcerange")[controls]
    inverse = targets.argsort().to(gain.device)
    return {
        "kp": gain[inverse],
        "kd": -bias[inverse, 2],
        "action_scale": scale[0, inverse],
        "action_to_joint": inverse,
        "force_limit": force_range[inverse].abs().amin(-1),
        "encoder_bias_subtracted_from_target": True,
    }


def sensor_pd_bias_proxy(history, contract):
    """Toy diagnostic valid ONLY for an uncompensated position-target interface.

    Besides an uncompensated target, sensor torque and joint state must obey
    the nominal PD law at a common timestamp. No truth is read by this
    calculation. The actual MJLab action contract is explicitly rejected.
    """
    if contract.get("encoder_bias_subtracted_from_target", False):
        raise ValueError(
            "Encoder bias cancels from the PD relation because the controller subtracts it from targets"
        )
    if history.shape[-1] != 6100:
        raise ValueError("PD history diagnostic requires the existing50-frame6100D group")
    terms = [term.reshape(-1, 50, dim) for term, dim in zip(history.split([1450, 1450, 150, 150, 1450, 1450], -1), (29, 29, 3, 3, 29, 29), strict=True)]
    position, velocity, _, _, actions, torque = terms
    actions = actions.index_select(-1, contract["action_to_joint"])
    residual = position - contract["action_scale"] * actions + (torque + contract["kd"] * velocity) / contract["kp"]
    present = history.new_zeros(position.shape[:2], dtype=torch.bool)
    for value in terms:
        present |= value.abs().sum(-1) > 0
    # Full known torque-noise support is2Nm. Avoid falsely accepting clipped
    # actuator readings as linear PD measurements; no true saturation flag.
    valid = present[..., None] & (torque.abs() < (contract["force_limit"] - 2).clamp_min(0))
    count = valid.sum(1)
    mean = (residual * valid).sum(1) / count.clamp_min(1)
    variance = ((residual - mean[:, None]).square() * valid).sum(1) / (count - 1).clamp_min(1)
    prior_variance = 0.01**2 / 3
    shrink = prior_variance / (prior_variance + variance / count.clamp_min(1))
    estimate = mean * shrink * (count > 1)
    return estimate, mean, count
