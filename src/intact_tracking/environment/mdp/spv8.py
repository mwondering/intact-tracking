"""SPV8-only observations for online motion-cluster routing."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from . import sp as sp_mdp

SPV8_V29_REFERENCE_INPUT_STEPS = tuple(range(-9, 11))
SPV8_V29_REFERENCE_FRAME_DIM = 3 + 6 + 29
SPV8_V29_REFERENCE_INPUT_DIM = len(SPV8_V29_REFERENCE_INPUT_STEPS) * SPV8_V29_REFERENCE_FRAME_DIM


def v29_reference_input(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Return the clean compact-qpos ``[-9,+10]`` routing window.

    MultiMotionLoader exposes qpos in one resident tensor: SPV8-1 opts into it
    beside the full corpus, while SPV8-1A makes it the only resident motion
    representation. Canonicalization is intentionally performed by the runtime
    router, exactly where the standalone v2-9 corpus canonicalizes each view.
    """
    command = sp_mdp._command(env, command_name)
    gather = getattr(command, "gather_compact_qpos_reference", None)
    if not callable(gather):
        raise TypeError("SPV8-1 requires a command with gather_compact_qpos_reference()")
    qpos = gather(SPV8_V29_REFERENCE_INPUT_STEPS)
    if qpos.shape[-1] != 36:
        raise ValueError(f"SPV8-1 compact qpos has {qpos.shape[-1]} values, expected 36")
    frames = torch.cat(
        (qpos[..., :3], sp_mdp._rot6d(qpos[..., 3:7]), qpos[..., 7:]),
        dim=-1,
    )
    return frames.reshape(frames.shape[0], -1)
