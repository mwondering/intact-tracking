"""Memory350 response10 supervision against one frozen nominal direction."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from intact_tracking.forward_predictor_objective import _cross_world_partner, _pearson_correlation
from intact_tracking.memory350_response_window import ResponseWindowObjective
from intact_tracking.memory350_weak_pairs import WeakPairLossConfig, masked_mean


@dataclass(frozen=True)
class NominalDirectionLossConfig(WeakPairLossConfig):
    representation_weight: float = 0.01
    representation_relation_weight: float = 4.0
    response_distance_scale: float = 0.3
    weak_positive_weight: float = 0.008
    weak_negative_weight: float = 0.0
    weak_negative_margin: float = 1.1  # Inert compatibility metadata; no negative pairs are computed.
    nominal_anchor_weight: float = 0.01
    nominal_direction_objective_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.weak_negative_weight != 0:
            raise ValueError("Nominal-direction supervision excludes weak negatives")
        if not math.isfinite(self.nominal_anchor_weight) or self.nominal_anchor_weight < 0:
            raise ValueError("nominal_anchor_weight must be finite and nonnegative")
        if self.nominal_direction_objective_version != 1:
            raise ValueError("Unsupported nominal-direction objective version")


def direction_response_loss(latent, anchor, response, valid, *, distance_scale=0.3):
    """Each row's own A-B RMS supervises its unit distance to the fixed anchor."""
    if response.shape != (len(latent), 10, 70):
        raise ValueError("Nominal-direction response labels must be [batch,10,70]")
    if anchor.shape != (latent.shape[-1],) or valid.shape != (len(latent),):
        raise ValueError("Invalid nominal anchor or row validity shape")
    if valid.dtype != torch.bool or distance_scale <= 0:
        raise ValueError("Validity must be boolean and response scale positive")
    unit = F.normalize(latent.float(), dim=-1, eps=1e-8)
    distance = torch.linalg.vector_norm(unit - anchor.detach(), dim=-1)
    response_rms = response.detach().float().square().mean(dim=(1, 2)).sqrt()
    target = 2 * response_rms / (response_rms + distance_scale)
    error = F.smooth_l1_loss(distance, target, beta=0.25, reduction="none")
    return masked_mean(error, valid), distance, target, response_rms


def positive_direction_loss(latent, positive, valid):
    unit = F.normalize(latent.float(), dim=-1, eps=1e-8)
    other = F.normalize(positive.float(), dim=-1, eps=1e-8)
    cosine = (unit * other).sum(-1).clamp(-1, 1)
    return masked_mean(1 - cosine, valid), cosine


class NominalDirectionObjective(ResponseWindowObjective):
    def __init__(self, model, loss_config=None, *, anchor=None):
        super().__init__(model, loss_config or NominalDirectionLossConfig())
        self.register_buffer("nominal_anchor", torch.zeros(
            model.config.dynamics_latent_dim, device=next(model.parameters()).device))
        self.anchor_ready = False
        if anchor is not None:
            self.set_anchor(anchor)

    @torch.no_grad()
    def set_anchor(self, anchor):
        value = torch.as_tensor(anchor, device=self.nominal_anchor.device, dtype=torch.float32)
        if (value.shape != self.nominal_anchor.shape or not torch.isfinite(value).all()
                or not torch.isclose(value.norm(), value.new_tensor(1.), atol=1e-5, rtol=0)):
            raise ValueError("The frozen nominal direction must be a finite unit vector")
        self.nominal_anchor.copy_(value)
        self.anchor_ready = True

    def _representation_terms(self, batch, views, target, *, compute_metrics):
        if not self.anchor_ready:
            raise RuntimeError("Calibrate or restore the fixed nominal direction before training")
        if "label_response" not in batch:
            raise ValueError("This objective requires ten-step labels, including validation")
        response, response_valid = self._representation_response(batch, target)
        usable = batch["history_valid"].any(1) | batch["memory_valid"].any(1)
        positive_usable = (batch["positive_history_valid"].any(1)
                           | batch["positive_memory_valid"].any(1))
        positive_valid = batch["positive_pair_valid"].bool() & usable & positive_usable
        local, cosine = positive_direction_loss(views[0], views[1], positive_valid)
        relation_valid = usable & response_valid.bool() & ~batch["is_nominal"].bool()
        relation, distance, targets, rms = direction_response_loss(
            views[0], self.nominal_anchor, response, relation_valid,
            distance_scale=self.loss_config.response_distance_scale)
        loss = local + self.loss_config.representation_relation_weight * relation
        metrics = {
            "representation_positive_loss": local.detach(),
            "representation_relation_loss": relation.detach(),
            "dr_nominal_relation_loss": relation.detach(),
            "dr_nominal_weighted_loss": (self.loss_config.representation_weight
                                          * self.loss_config.representation_relation_weight * relation).detach(),
            "latent_relation_pairs": relation_valid.sum().float(),
            "latent_positive_cosine": masked_mean(cosine.detach(), positive_valid),
            "latent_distance_mean": masked_mean(distance.detach(), relation_valid),
            "latent_target_distance_mean": masked_mean(targets, relation_valid),
            "response_label_valid_fraction": response_valid.float().mean(),
        }
        if compute_metrics:
            metrics["latent_response_correlation"] = _pearson_correlation(
                distance.detach(), rms, relation_valid)
            for name, values in (("latent_distance", distance.detach()),
                                 ("latent_target_distance", targets)):
                selected = values[relation_valid]
                quantiles = (torch.quantile(selected, values.new_tensor([.1, .5, .9]))
                             if selected.numel() else values.new_zeros(3))
                for index, percentile in enumerate((10, 50, 90)):
                    metrics[f"{name}_batch_p{percentile}"] = quantiles[index]
        # Pairing remains only for the predictor shuffle diagnostic.
        partner, _ = _cross_world_partner(batch["world_id"])
        return loss, metrics, partner, response, response_valid

    def _extra_representation_loss(self, batch, views):
        usable = batch["history_valid"].any(1) | batch["memory_valid"].any(1)
        nominal_valid = batch["is_nominal"].bool() & usable
        unit = F.normalize(views[0].float(), dim=-1, eps=1e-8)
        nominal_cosine = (unit * self.nominal_anchor).sum(-1).clamp(-1, 1)
        nominal_loss = masked_mean(1 - nominal_cosine, nominal_valid)
        full_dr = (~batch["is_nominal"].bool() & batch["history_valid"].all(1)
                   & batch["memory_valid"].all(1))
        if len(views) == 3:
            valid = (full_dr & batch["weak_pair_valid"]
                     & (batch["world_id"] == batch["weak_world_id"])
                     & (batch["motion_id"] != batch["weak_motion_id"])
                     & (batch["physics_session"] == batch["weak_session"])
                     & batch["weak_history_valid"].all(1) & batch["weak_memory_valid"].all(1))
            positive = views[2]
        else:
            valid = torch.zeros_like(full_dr)
            positive = views[0]
        weak_loss, _ = positive_direction_loss(views[0], positive, valid)
        weighted_nominal = self.loss_config.nominal_anchor_weight * nominal_loss
        weighted_weak = self.loss_config.weak_positive_weight * weak_loss
        return weighted_nominal + weighted_weak, {
            "nominal_anchor_loss": nominal_loss.detach(),
            "nominal_anchor_weighted_loss": weighted_nominal.detach(),
            "nominal_anchor_samples": nominal_valid.sum().float(),
            "nominal_anchor_cosine": masked_mean(nominal_cosine.detach(), nominal_valid),
            "weak_positive_loss": weak_loss.detach(),
            "weak_positive_weighted_loss": weighted_weak.detach(),
            "weak_positive_pairs": valid.sum().float(),
            "weak_positive_fraction_of_full_dr": valid.sum() / full_dr.sum().clamp_min(1),
        }
