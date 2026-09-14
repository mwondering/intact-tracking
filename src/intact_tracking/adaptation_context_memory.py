"""Causal, sensor-only context accumulation; no simulator labels enter memory."""

from __future__ import annotations

import torch

CONTEXT_MEAN = "adaptation_causal_context_mean"


def physics_mean_only_configuration(actor, horizon):
    """An inference architecture change, preserving every learned parameter."""
    if horizon < 1:
        raise ValueError("Context mean horizon must be positive")
    if (
        not actor.get("class_name", "").endswith(":ContextAdaptationActor")
        or actor.get("adaptation_latent_dim", 64) != 7
        or not (actor.get("physics_bilinear", False) or actor.get("physics_low_rank", False))
        or any(actor.get(key, False) for key in (
            "context_latent_mean", "context_feature_adapter", "train_base_policy",
            "train_reference_encoder", "correction_use_privilege",
        ))
    ):
        raise ValueError("Mean-only conversion requires a stateless seven-physics deployable student with frozen frontend")
    return dict(actor, context_latent_mean=True, context_mean_only=True,
                context_mean_horizon=int(horizon), train_context_encoder=False,
                train_residual_policy=False)


def update_context_mean(previous, count, current, reset, horizon=250):
    """Prefix mean for the first horizon samples, then EMA with gain1/horizon."""
    if horizon < 1:
        raise ValueError("Context mean horizon must be positive")
    reset = reset.reshape(-1, 1).bool()
    previous = torch.where(reset, torch.zeros_like(previous), previous)
    count = torch.where(reset, torch.zeros_like(count), count)
    count = (count + 1).clamp_max(horizon)
    return previous + (current - previous) / count, count


class CausalContextMeanWrapper:
    """Augment a wrapper with a past-and-current latent mean.

    Existing noisy measurements are reused; no extra sensor samples or noise
    draws are generated. The producer encoder MUST remain frozen, so replayed
    context means and their current latent coordinate system stay consistent.
    """

    def __init__(self, wrapped, latent_dim=64, horizon=250):
        self.wrapped = wrapped
        self.actor = None
        self.horizon = int(horizon)
        self.mean = torch.zeros(wrapped.num_envs, latent_dim, device=wrapped.device)
        self.count = torch.zeros(wrapped.num_envs, 1, device=wrapped.device)
        self.last_step = None

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def bind(self, actor):
        if any(parameter.requires_grad for parameter in actor.context_encoder.parameters()):
            raise ValueError("Causal latent memory requires a frozen producer encoder")
        self.actor = actor

    @torch.no_grad()
    def attach(self, obs, reset=None):
        step = int(self.unwrapped.common_step_counter)
        if self.actor is not None and step != self.last_step:
            if reset is None:
                reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            inputs = obs.select(*(
                name for name in self.actor.deployable_observation_groups if name != CONTEXT_MEAN
            ))
            current = self.actor.instant_context_latent(inputs)
            self.mean, self.count = update_context_mean(
                self.mean, self.count, current, reset, self.horizon
            )
            self.last_step = step
        # Snapshot, never an alias that a later update can change in replay.
        obs.set(CONTEXT_MEAN, self.mean.clone())
        return obs

    def get_observations(self):
        return self.attach(self.wrapped.get_observations())

    def step(self, actions):
        obs, reward, done, extras = self.wrapped.step(actions)
        return self.attach(obs, reset=done), reward, done, extras

    def reset(self):
        obs, extras = self.wrapped.reset()
        self.mean.zero_()
        self.count.zero_()
        self.last_step = None
        return self.attach(obs), extras
