"""SPV7-only observations for policy-core routing ablations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import sp as sp_mdp
from .spv5 import _pack_minimal_reference, _root_reference

if TYPE_CHECKING:
    import torch
    from mjlab.envs import ManagerBasedRlEnv


SPV7_PMOE_REFERENCE_INPUT_STEPS = tuple(range(-99, 1))
SPV7_PMOE_REFERENCE_FRAME_DIM = 3 + 6 + 29
SPV7_PMOE_REFERENCE_INPUT_DIM = len(SPV7_PMOE_REFERENCE_INPUT_STEPS) * SPV7_PMOE_REFERENCE_FRAME_DIM
# The routing patch contains 25 current-inclusive history frames and 25 future
# frames. Two extra frames on both sides supply the four adjacent weak views.
SPV7_V26_ROUTING_WINDOW_STEPS = tuple(range(-24, 26))
SPV7_V26_REFERENCE_INPUT_STEPS = tuple(range(-26, 28))
SPV7_V26_REFERENCE_FRAME_DIM = 3 + 6 + 29
SPV7_V26_REFERENCE_INPUT_DIM = len(SPV7_V26_REFERENCE_INPUT_STEPS) * SPV7_V26_REFERENCE_FRAME_DIM


def pmoe_reference_input(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Return a noisy, current-inclusive 100-frame past reference window."""
    steps = SPV7_PMOE_REFERENCE_INPUT_STEPS
    command = sp_mdp._command(env, command_name)
    root_pos = _root_reference(env, command_name, "body_pos_w", steps, noisy=False)
    root_quat = _root_reference(env, command_name, "body_quat_w", steps, noisy=False)
    joint_pos = sp_mdp._gather_horizon(env, command_name, "joint_pos", steps, "teacher")

    corrupt_root = getattr(command, "apply_student_root_reference_randomization", None)
    corrupt_reference = getattr(command, "apply_student_reference_randomization", None)
    if callable(corrupt_root) and callable(corrupt_reference):
        root_pos = corrupt_root("body_pos_w", steps, root_pos.clone())
        root_quat = corrupt_root("body_quat_w", steps, root_quat.clone())
        joint_pos = corrupt_reference("joint_pos", steps, joint_pos.clone())
    else:
        root_pos = _root_reference(env, command_name, "body_pos_w", steps, noisy=True)
        root_quat = _root_reference(env, command_name, "body_quat_w", steps, noisy=True)
        joint_pos = sp_mdp._gather_horizon(env, command_name, "joint_pos", steps, "student")

    return _pack_minimal_reference(root_pos, root_quat, joint_pos)


def v26_reference_input(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Return the clean [-26,+27] qpos union used by SPV7-6."""
    steps = SPV7_V26_REFERENCE_INPUT_STEPS
    root_pos = _root_reference(env, command_name, "body_pos_w", steps, noisy=False)
    root_quat = _root_reference(env, command_name, "body_quat_w", steps, noisy=False)
    joint_pos = sp_mdp._gather_horizon(env, command_name, "joint_pos", steps, "teacher")
    return _pack_minimal_reference(root_pos, root_quat, joint_pos)
