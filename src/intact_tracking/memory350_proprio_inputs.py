"""Control-rate Memory350 inputs shared by collection and policy inference."""

from dataclasses import dataclass

import torch

from intact_tracking.memory350_model import Memory350Config


PROPRIO_TERMS = ("joint_pos", "joint_vel", "projected_gravity", "base_ang_vel", "last_action", "joint_torque")
PROPRIO_WIDTHS = (29, 29, 3, 3, 29, 29)
PROPRIO_DIM = sum(PROPRIO_WIDTHS)
PROPRIO_ARCHITECTURE = "memory350_noisy_proprio122_control_action_v1"
INPUT_CONTRACT = {
    "version": 1,
    "observation_group": "estimator_history",
    "state_dim": PROPRIO_DIM,
    "state_terms": list(PROPRIO_TERMS),
    "state_term_dims": list(PROPRIO_WIDTHS),
    "state_sampling": "latest frame of each term in cached policy observations; reuse the same noise sample",
    "action_dim": 29,
    "action_source": "joint_pos action term.raw_action",
    "action_sampling": "one commanded action per completed control step; before delay/smoothing; no substep averaging",
    "action_units": "policy command units after optional clipping and tracker/residual composition",
    "action_order": "action term target joint order",
    "interaction_dim": 273,
    "reset": "exclude reset/motion-switch edges; retain complete old-memory chunks",
    "privileged_inputs": False,
}


@dataclass(frozen=True)
class ProprioMemory350Config(Memory350Config):
    architecture_version: str = PROPRIO_ARCHITECTURE
    context_state_dim: int = PROPRIO_DIM
    chunk_depth: int = 2
    memory_depth: int = 4
    context_depth: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.context_state_dim != PROPRIO_DIM or self.architecture_version != PROPRIO_ARCHITECTURE:
            raise ValueError("Noisy proprio Memory350 requires its 122D input architecture")


def current_proprio(observations):
    """Extract term-major history correctly; never generate new sensor noise."""
    history = observations["estimator_history"]
    if history.ndim != 2 or history.shape[-1] != 50 * PROPRIO_DIM:
        raise ValueError(f"Expected 50-frame term-major proprio history [N,6100], got {history.shape}")
    offset, values = 0, []
    for width in PROPRIO_WIDTHS:
        offset += 50 * width
        values.append(history[:, offset - width:offset])
    return torch.cat(values, dim=-1).detach()


def control_action(env):
    """The command sent once per control step, before SP delay and smoothing.

    Read the retained raw command after stepping to include optional clipping.
    It stays constant while SP computes its substep applied actions/PD targets.
    Reset-edge values are excluded by the common memory validity protocol.
    """
    value = env.action_manager.get_term("joint_pos").raw_action
    if value.shape != (env.num_envs, 29):
        raise ValueError("Expected a 29D control command for each environment")
    return value.detach().clone()


def validate_proprio_observations(env):
    manager = env.observation_manager
    if not manager.cfg["estimator_history"].enable_corruption:
        raise ValueError("The noisy-proprio encoder requires observation corruption to remain enabled")
    if tuple(manager.active_terms.get("estimator_history", ())) != PROPRIO_TERMS:
        raise ValueError("The encoder must reuse the selected tracker's exact six proprio terms")
    for name in PROPRIO_TERMS:
        cfg = manager.get_term_cfg("estimator_history", name)
        if cfg.history_length != 50 or not cfg.flatten_history_dim:
            raise ValueError("Proprio history layout changed")
    noises = {"joint_pos": .01, "joint_vel": .5, "projected_gravity": .05, "base_ang_vel": .2}
    for name, magnitude in noises.items():
        if manager.get_term_cfg("estimator_history", name).params.get("noise_std") != magnitude:
            raise ValueError(f"The source checkpoint's {name} observation noise changed")
    if manager.get_term_cfg("estimator_history", "joint_pos").params.get("biased") is not True:
        raise ValueError("The encoder must observe the configured joint encoder bias")
    torque = manager.get_term_cfg("estimator_history", "joint_torque")
    if (torque.params.get("sample_mode") != "latest" or torque.noise is None
            or (torque.noise.n_min, torque.noise.n_max) != (-2., 2.)):
        raise ValueError("The source checkpoint's latest noisy joint-torque observation changed")
    current_proprio(manager.compute())
    return {"uniform_noise_half_ranges": {**noises, "joint_torque": 2., "last_action": 0.},
            "joint_encoder_bias": True, "same_cached_sample_as_tracker": True}


def validate_input_checkpoint(state):
    if (state.get("architecture_version") != PROPRIO_ARCHITECTURE
            or state.get("context_input_contract") != INPUT_CONTRACT
            or state.get("model_config", {}).get("context_state_dim") != PROPRIO_DIM):
        raise ValueError("A noisy-proprio122 checkpoint is required; old truth71 checkpoints cannot be reused")
