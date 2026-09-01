"""Stateful execution of residual action trunks in vectorized simulator rollout."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .residual_model import ResidualTrackingModel

ContextProvider = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class ResidualTrunkController:
    """Generate five residuals once, then consume exactly one slot per env step.

    The frozen tracker remains closed loop and is evaluated separately on every
    real simulator observation.  This controller caches only the residual part.
    A reset invalidates the unconsumed suffix for the affected vector slots.
    """

    def __init__(
        self,
        model: ResidualTrackingModel,
        *,
        num_worlds: int,
        context_provider: ContextProvider,
        device: torch.device | str,
    ) -> None:
        if num_worlds < 1:
            raise ValueError("num_worlds must be positive")
        self.model = model
        self.num_worlds = int(num_worlds)
        self.context_provider = context_provider
        self.device = torch.device(device)
        self.horizon = model.config.horizon
        self.action_dim = model.config.action_dim
        self._trunk = torch.zeros(
            self.num_worlds,
            self.horizon,
            self.action_dim,
            device=self.device,
        )
        self._cursor = torch.full(
            (self.num_worlds,), self.horizon, dtype=torch.long, device=self.device
        )
        self._world = torch.zeros(self.num_worlds, model.config.context_dim, device=self.device)
        self.last_step = torch.full_like(self._cursor, -1)
        self.last_world = torch.zeros_like(self._world)
        self.trunks_generated = 0
        self.trunks_invalidated = 0

    @property
    def cursor(self) -> torch.Tensor:
        return self._cursor

    def invalidate(self, mask: torch.Tensor) -> None:
        if tuple(mask.shape) != (self.num_worlds,) or mask.dtype != torch.bool:
            raise ValueError(
                "Residual trunk invalidation mask must be bool[num_worlds], got "
                f"{mask.dtype} {tuple(mask.shape)}"
            )
        invalid = mask.to(device=self.device)
        self.trunks_invalidated += int((invalid & (self._cursor < self.horizon)).sum())
        self._cursor[invalid] = self.horizon
        self._trunk[invalid] = 0.0
        self._world[invalid] = 0.0

    def invalidate_all(self) -> None:
        self.invalidate(torch.ones(self.num_worlds, dtype=torch.bool, device=self.device))

    def __call__(
        self,
        policy_observation: torch.Tensor,
        tracker_action: torch.Tensor,
    ) -> torch.Tensor:
        expected_observation = (self.num_worlds, self.model.config.policy_observation_dim)
        expected_action = (self.num_worlds, self.action_dim)
        if tuple(policy_observation.shape) != expected_observation:
            raise ValueError(
                f"Residual policy observation must have shape {expected_observation}, "
                f"got {tuple(policy_observation.shape)}"
            )
        if tuple(tracker_action.shape) != expected_action:
            raise ValueError(
                f"Tracker action must have shape {expected_action}, "
                f"got {tuple(tracker_action.shape)}"
            )

        needs_trunk = self._cursor >= self.horizon
        if needs_trunk.any():
            pending_ids = needs_trunk.nonzero(as_tuple=False).flatten()
            context, context_ready = self.context_provider(pending_ids)
            if tuple(context_ready.shape) != (pending_ids.numel(),):
                raise ValueError("Context ready mask must match requested world IDs")
            ready = context_ready.to(device=self.device, dtype=torch.bool)
            env_ids = pending_ids[ready]
            if env_ids.numel():
                world = self.model.encode_context(context[ready])
                trunk = self.model.residual_action_trunk(
                    world,
                    policy_observation.index_select(0, env_ids),
                )
                expected_trunk = (env_ids.numel(), self.horizon, self.action_dim)
                if tuple(trunk.shape) != expected_trunk:
                    raise RuntimeError(
                        f"Residual policy returned {tuple(trunk.shape)}, expected {expected_trunk}"
                    )
                self._trunk.index_copy_(0, env_ids, trunk)
                self._world.index_copy_(0, env_ids, world)
                self._cursor[env_ids] = 0
                self.trunks_generated += int(env_ids.numel())

        active = self._cursor < self.horizon
        self.last_step.fill_(-1)
        self.last_world.zero_()
        self.last_step[active] = self._cursor[active]
        self.last_world[active] = self._world[active]
        residual = torch.zeros_like(tracker_action)
        env_ids = active.nonzero(as_tuple=False).flatten()
        if env_ids.numel():
            residual[env_ids] = self._trunk[env_ids, self._cursor[env_ids]]
            self._cursor[env_ids] += 1
        return residual
