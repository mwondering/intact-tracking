"""Minimal noisy-reference windows and clean SPV5 supervision targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

from . import sp as sp_mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


SPV5_REFERENCE_INPUT_STEPS = tuple(range(-42, 8))
SPV5_REFERENCE_SUPPORT_STEPS = tuple(range(-3, 8))
SPV5_REFERENCE_FRAME_DIM = 3 + 6 + 29
SPV5_REFERENCE_INPUT_DIM = len(SPV5_REFERENCE_INPUT_STEPS) * SPV5_REFERENCE_FRAME_DIM
SPV5_REFERENCE_TARGET_DIM = len(SPV5_REFERENCE_SUPPORT_STEPS) * SPV5_REFERENCE_FRAME_DIM
SPV5_ROBOT_ROOT_QUAT_DIM = 4


def _root_reference(
    env: ManagerBasedRlEnv,
    command_name: str,
    field_name: str,
    steps: tuple[int, ...],
    *,
    noisy: bool,
) -> torch.Tensor:
    command = sp_mdp._command(env, command_name)
    method_name = "gather_student_root_reference" if noisy else "gather_root_reference"
    method = getattr(command, method_name, None)
    if callable(method):
        return method(field_name, steps)
    return sp_mdp._root_motion(
        env,
        command_name,
        field_name,
        steps,
        horizon="student" if noisy else "teacher",
    )


def _pack_minimal_reference(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
) -> torch.Tensor:
    frames = torch.cat((root_pos, sp_mdp._rot6d(root_quat), joint_pos), dim=-1)
    return frames.reshape(frames.shape[0], -1)


def _reference_values(
    env: ManagerBasedRlEnv,
    command_name: str,
    root_quat_noise_std: float,
) -> dict[str, torch.Tensor]:
    command = sp_mdp._command(env, command_name)
    cache = getattr(command, "_shared_spv5_reference_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        command._shared_spv5_reference_cache = cache
    key = float(root_quat_noise_std)
    cached = cache.get(key)
    if cached is not None:
        return cached

    input_steps = SPV5_REFERENCE_INPUT_STEPS
    clean_root_pos_full = _root_reference(env, command_name, "body_pos_w", input_steps, noisy=False)
    clean_root_quat_full = _root_reference(
        env, command_name, "body_quat_w", input_steps, noisy=False
    )
    clean_joint_pos_full = sp_mdp._gather_horizon(
        env, command_name, "joint_pos", input_steps, "teacher"
    )
    corrupt_root = getattr(command, "apply_student_root_reference_randomization", None)
    corrupt_reference = getattr(command, "apply_student_reference_randomization", None)
    if callable(corrupt_root) and callable(corrupt_reference):
        noisy_root_pos = corrupt_root("body_pos_w", input_steps, clean_root_pos_full.clone())
        noisy_root_quat = corrupt_root("body_quat_w", input_steps, clean_root_quat_full.clone())
        noisy_joint_pos = corrupt_reference("joint_pos", input_steps, clean_joint_pos_full.clone())
    else:
        noisy_root_pos = _root_reference(env, command_name, "body_pos_w", input_steps, noisy=True)
        noisy_root_quat = _root_reference(env, command_name, "body_quat_w", input_steps, noisy=True)
        noisy_joint_pos = sp_mdp._gather_horizon(
            env, command_name, "joint_pos", input_steps, "student"
        )

    support_length = len(SPV5_REFERENCE_SUPPORT_STEPS)
    clean_root_pos = clean_root_pos_full[:, -support_length:]
    clean_root_quat = clean_root_quat_full[:, -support_length:]
    clean_joint_pos = clean_joint_pos_full[:, -support_length:]

    values = {
        "input": _pack_minimal_reference(noisy_root_pos, noisy_root_quat, noisy_joint_pos),
        "target": _pack_minimal_reference(clean_root_pos, clean_root_quat, clean_joint_pos),
        # This is consumed only inside the actor to form robot-to-reference
        # rotations and never concatenated into the final policy feature vector.
        "robot_root_quat": sp_mdp._perturb_quaternion(
            env.scene["robot"].data.root_link_quat_w, root_quat_noise_std
        ),
    }
    expected = {
        "input": SPV5_REFERENCE_INPUT_DIM,
        "target": SPV5_REFERENCE_TARGET_DIM,
        "robot_root_quat": SPV5_ROBOT_ROOT_QUAT_DIM,
    }
    for name, value in values.items():
        if value.shape[-1] != expected[name]:
            raise RuntimeError(
                f"SPV5 {name} has {value.shape[-1]} values, expected {expected[name]}"
            )
    cache[key] = values
    return values


class _SPV5ReferenceObservation:
    output_name: str

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        self.command_name = str(cfg.params.get("command_name", "motion"))
        self.root_quat_noise_std = float(cfg.params.get("root_quat_noise_std", 0.0))

    def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
        return _reference_values(env, self.command_name, self.root_quat_noise_std)[self.output_name]


class reference_encoder_input(_SPV5ReferenceObservation):
    output_name = "input"


class reference_encoder_target(_SPV5ReferenceObservation):
    output_name = "target"


class robot_root_quat(_SPV5ReferenceObservation):
    output_name = "robot_root_quat"
