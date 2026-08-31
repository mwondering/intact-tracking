"""Paired physical/deployment INTACT objective for tracking windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import SIGReg, TrackingINTACT


@dataclass(frozen=True)
class INTACTLossConfig:
    """Weights match the roles in the reference INTACT single-task objective."""

    forward_weight: float = 1.0
    sigreg_weight: float = 0.02
    physical_weight: float = 0.1
    goal_weight: float = 0.05
    physical_start: int = 0
    goal_start: int = 0

    def __post_init__(self) -> None:
        for name in (
            "forward_weight",
            "sigreg_weight",
            "physical_weight",
            "goal_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.physical_start < 0 or self.goal_start < 0:
            raise ValueError("Intent start indices must be non-negative")


def predict_adjacent_latents(
    model: TrackingINTACT,
    embeddings: torch.Tensor,
    action_embeddings: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Match INTACT's causal adjacent-latent prediction schedule."""
    transitions = embeddings.size(1) - 1
    if transitions < 1:
        raise ValueError("A training window needs at least one transition")
    if action_embeddings.shape[:2] != (embeddings.size(0), transitions):
        raise ValueError(
            "Action embeddings must align with latent transitions: "
            f"{tuple(action_embeddings.shape)} vs {tuple(embeddings.shape)}"
        )
    prefix = min(history_size, transitions)
    predictions = [model.predict(embeddings[:, :prefix], action_embeddings[:, :prefix])]
    for index in range(prefix, transitions):
        start = max(0, index - history_size + 1)
        prediction = model.predict(
            embeddings[:, start : index + 1],
            action_embeddings[:, start : index + 1],
        )[:, -1:]
        predictions.append(prediction)
    return torch.cat(predictions, dim=1)


def construct_intents(
    embeddings: torch.Tensor,
    goal_embedding: torch.Tensor,
    *,
    physical_start: int = 0,
    goal_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct attached physical intent and stop-gradient deployment intent.

    ``goal_embedding[:, k]`` is the reference endpoint aligned with physical
    successor ``embeddings[:, k + 1]``.  Only the reference occurrence is
    detached; the physical successor and every current-state occurrence remain
    attached.
    """
    max_start = embeddings.size(1) - 2
    for name, value in (("physical_start", physical_start), ("goal_start", goal_start)):
        if value < 0 or value > max_start:
            raise ValueError(
                f"{name} must be in [0,{max_start}], got {value} for T={embeddings.size(1)}"
            )
    physical_current = embeddings[:, physical_start:-1]
    physical_successor = embeddings[:, physical_start + 1 :]
    physical_intent = physical_successor - physical_current

    deployment_current = embeddings[:, goal_start:-1]
    expected_goal_shape = (
        embeddings.size(0),
        embeddings.size(1) - 1,
        embeddings.size(-1),
    )
    if goal_embedding.ndim != 3 or goal_embedding.shape != expected_goal_shape:
        raise ValueError(
            "Aligned goal embeddings must be [B,T-1,D], got "
            f"{tuple(goal_embedding.shape)}; expected {expected_goal_shape}"
        )
    aligned_goal = goal_embedding[:, goal_start:].detach()
    goal_intent = aligned_goal - deployment_current
    return physical_intent, goal_intent


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    weights = mask.to(device=value.device, dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1)


def intact_objective(
    model: TrackingINTACT,
    batch: dict[str, torch.Tensor],
    loss_config: INTACTLossConfig | None = None,
    sigreg: SIGReg | None = None,
) -> dict[str, torch.Tensor]:
    """Run Forward MSE + SIGReg + the paired shared-actor likelihoods."""
    cfg = loss_config or INTACTLossConfig()
    required = {
        "observation",
        "goal_observation",
        "forward_action",
        "action",
        "previous_action",
        "context",
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"Missing INTACT batch fields: {missing}")
    for name in required:
        if not torch.isfinite(batch[name]).all():
            raise ValueError(f"Batch field {name!r} contains non-finite values")

    context_mask = batch.get("context_mask")
    world = model.encode_context(batch["context"], context_mask)
    embeddings = model.encode_observation(batch["observation"], world)
    goal_embedding = model.encode_observation(batch["goal_observation"], world)
    action_embeddings = model.forward_action_encoder(batch["forward_action"])
    predictions = predict_adjacent_latents(
        model,
        embeddings,
        action_embeddings,
        model.config.forward_history,
    )
    targets = embeddings[:, 1:]
    transition_mask = batch.get("transition_mask")
    forward_per_step = (predictions - targets).square().mean(dim=-1)
    forward_loss = _masked_mean(forward_per_step, transition_mask)

    physical_intent, goal_intent = construct_intents(
        embeddings,
        goal_embedding,
        physical_start=cfg.physical_start,
        goal_start=cfg.goal_start,
    )
    physical_mask = batch.get("physical_mask", transition_mask)
    if physical_mask is not None:
        physical_mask = physical_mask[:, cfg.physical_start :]
    goal_mask = batch.get("goal_mask", transition_mask)
    if goal_mask is not None:
        goal_mask = goal_mask[:, cfg.goal_start :]

    physical_stats = model.action_nll(
        z=embeddings[:, cfg.physical_start : -1],
        intent=physical_intent,
        previous_action=batch["previous_action"][:, cfg.physical_start :],
        target_action=batch["action"][:, cfg.physical_start :],
        mask=physical_mask,
    )
    goal_stats = model.action_nll(
        z=embeddings[:, cfg.goal_start : -1],
        intent=goal_intent,
        previous_action=batch["previous_action"][:, cfg.goal_start :],
        target_action=batch["action"][:, cfg.goal_start :],
        mask=goal_mask,
    )

    sigreg_module = sigreg
    if sigreg_module is None:
        sigreg_module = SIGReg().to(device=embeddings.device)
    sigreg_loss = sigreg_module(embeddings.transpose(0, 1))
    action_loss = (
        cfg.physical_weight * physical_stats["loss"] + cfg.goal_weight * goal_stats["loss"]
    )
    weighted_forward = cfg.forward_weight * forward_loss
    weighted_sigreg = cfg.sigreg_weight * sigreg_loss
    weighted_physical = cfg.physical_weight * physical_stats["loss"]
    weighted_goal = cfg.goal_weight * goal_stats["loss"]
    total = weighted_forward + weighted_sigreg + weighted_physical + weighted_goal

    # Scale-aware diagnostics make latent MSE and Gaussian NLL interpretable.
    # These values are observational only and do not alter the INTACT objective.
    flat_embeddings = embeddings.detach().reshape(-1, embeddings.size(-1)).float()
    latent_std = flat_embeddings.std(dim=0, unbiased=False)
    target_variance = targets.detach().float().var(unbiased=False)
    return {
        "loss": total,
        "forward_loss": forward_loss,
        "forward_nmse": (forward_loss.detach().float() / target_variance.clamp_min(1e-8)),
        "forward_target_variance": target_variance,
        "sigreg_loss": sigreg_loss,
        "action_loss": action_loss,
        "weighted_forward_loss": weighted_forward.detach(),
        "weighted_sigreg_loss": weighted_sigreg.detach(),
        "weighted_physical_nll": weighted_physical.detach(),
        "weighted_goal_nll": weighted_goal.detach(),
        "physical_nll": physical_stats["loss"],
        "goal_nll": goal_stats["loss"],
        "physical_mae": _masked_mean(physical_stats["mae"], physical_mask).detach(),
        "goal_mae": _masked_mean(goal_stats["mae"], goal_mask).detach(),
        "physical_log_std": _masked_mean(
            physical_stats["log_std"].mean(dim=-1), physical_mask
        ).detach(),
        "goal_log_std": _masked_mean(goal_stats["log_std"].mean(dim=-1), goal_mask).detach(),
        "latent_mean_abs": flat_embeddings.mean(dim=0).abs().mean(),
        "latent_rms": flat_embeddings.square().mean().sqrt(),
        "latent_std_mean": latent_std.mean(),
        "latent_std_min": latent_std.min(),
        "latent_std_max": latent_std.max(),
        "latent_collapsed_fraction": (latent_std < 0.1).float().mean(),
        "predictions": predictions,
        "embeddings": embeddings,
        "goal_embedding": goal_embedding,
        "world_embedding": world,
    }


class TrackingINTACTObjective(torch.nn.Module):
    """Module-form objective so DDP observes the complete training forward pass.

    Calling :func:`intact_objective` directly through ``DDP.module`` would bypass
    DDP's forward bookkeeping.  This wrapper preserves the original objective
    while making gradient synchronization correct and explicit.
    """

    def __init__(
        self,
        model: TrackingINTACT,
        loss_config: INTACTLossConfig | None = None,
        sigreg: SIGReg | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_config = loss_config or INTACTLossConfig()
        self.sigreg = sigreg or SIGReg()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return intact_objective(
            self.model,
            batch,
            loss_config=self.loss_config,
            sigreg=self.sigreg,
        )
