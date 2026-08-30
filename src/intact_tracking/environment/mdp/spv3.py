from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def root_height_gt(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged root height target relative to the environment ground plane."""
    root_pos_w = env.scene["robot"].data.root_link_pos_w
    env_origins = getattr(env.scene, "env_origins", None)
    if isinstance(env_origins, torch.Tensor):
        root_height = root_pos_w[:, 2] - env_origins[:, 2].to(
            device=root_pos_w.device,
            dtype=root_pos_w.dtype,
        )
    else:
        root_height = root_pos_w[:, 2]
    return root_height.unsqueeze(-1)


def root_lin_vel_b_gt(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged root linear-velocity target in the current robot frame."""
    return env.scene["robot"].data.root_link_lin_vel_b
