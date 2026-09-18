"""DR-parameter center geometry with optional same-DR, cross-motion positives."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from intact_tracking.forward_predictor_objective import (
    ForwardPredictorLossConfig, _cross_world_partner, _pearson_correlation,
)
from intact_tracking.memory350_objective import Memory350Objective
from intact_tracking.memory350_weak_pairs import WeakPairReplayBuffer, masked_mean


@dataclass(frozen=True)
class DRCenterLossConfig(ForwardPredictorLossConfig):
    # Total center-relation coefficient. The optional positive coefficient is
    # independent, not multiplied by this value a second time.
    representation_weight: float = 0.02
    representation_relation_weight: float = 0.0
    dr_distance_scale: float = 0.3
    dr_relation_beta: float = 0.25
    dr_positive_weight: float = 0.0
    dr_center_objective_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.representation_relation_weight != 0:
            raise ValueError("DR-center-only training requires the old response relation weight to be zero")
        if not math.isfinite(self.representation_weight) or self.representation_weight < 0:
            raise ValueError("representation_weight must be finite and nonnegative")
        if not math.isfinite(self.dr_positive_weight) or self.dr_positive_weight < 0:
            raise ValueError("dr_positive_weight must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.dr_distance_scale, self.dr_relation_beta)):
            raise ValueError("DR distance scale and SmoothL1 beta must be finite and positive")
        if self.dr_center_objective_version != 1:
            raise ValueError("Unsupported DR-center objective version")


def dr_center_relation_loss(latent, archived_latent, dr_metric, world_id, session,
                            valid, is_nominal, *, distance_scale=0.3, beta=0.25,
                            compute_metrics=True):
    """A stochastic center uses two current-encoder, distinct-motion views.

    Repeated rows of a world/session are pooled into one center, with equal
    weight per eligible row. No normalization is applied AFTER averaging unit
    latents. Every unordered pair of eligible centers has equal weight, even
    when its physical parameter distance is zero. Pairing is rank/microbatch
    local; neither cached latents nor a learned parameter encoder are used.

    dr_metric already contains range-normalized coordinates times sqrt(weight).
    Thus Euclidean distance here is a weighted RMS of physical DR differences.
    """
    if latent.ndim != 2 or archived_latent.shape != latent.shape:
        raise ValueError("Center views must be equal [batch, latent_dim] tensors")
    n = len(latent)
    if n < 1 or dr_metric.ndim != 2 or len(dr_metric) != n:
        raise ValueError("DR metric must have one coordinate vector per nonempty batch row")
    if any(x.shape != (n,) for x in (world_id, session, valid, is_nominal)):
        raise ValueError("World/session/valid/nominal fields must be [batch]")
    if distance_scale <= 0 or beta <= 0:
        raise ValueError("Distance scale and SmoothL1 beta must be positive")
    unit = F.normalize(latent.float(), dim=-1, eps=1e-8)
    other = F.normalize(archived_latent.float(), dim=-1, eps=1e-8)
    identities, inverse = torch.unique(torch.stack((world_id, session), -1), dim=0,
                                       return_inverse=True)
    count = unit.new_zeros(len(identities)).index_add(0, inverse, valid.float())

    def center_mean(rows):
        rows = rows.masked_fill(~valid[:, None], 0)
        return rows.new_zeros(len(identities), rows.shape[-1]).index_add(
            0, inverse, rows) / count.clamp_min(1)[:, None]

    centers = center_mean((unit + other) * 0.5)
    parameters = center_mean(dr_metric.detach().float())
    nominal = center_mean(is_nominal.float()[:, None]).squeeze(1) > 0.5
    ready = count > 0
    pairs = ready[:, None] & ready[None, :]
    pairs &= torch.ones_like(pairs).triu(diagonal=1)
    distances = torch.cdist(centers, centers, compute_mode="donot_use_mm_for_euclid_dist")
    dr_distances = torch.cdist(parameters, parameters, compute_mode="donot_use_mm_for_euclid_dist")
    targets = 2 * dr_distances / (dr_distances + distance_scale)
    loss = masked_mean(F.smooth_l1_loss(distances, targets, beta=beta, reduction="none"), pairs)
    metrics = {
        "dr_center_relation_loss": loss.detach(),
        "dr_center_pairs": pairs.sum().float(),
        "dr_center_valid_worlds": ready.sum().float(),
        "dr_center_eligible_fraction": valid.float().mean(),
    }
    if compute_metrics:
        with torch.no_grad():
            # This relation term does not itself penalize within-center drift.
            # The optional positive term is computed separately below.
            row_centers = centers[inverse]
            within = (torch.linalg.vector_norm(unit - row_centers, dim=-1)
                      + torch.linalg.vector_norm(other - row_centers, dim=-1)) * 0.5
            metrics.update(
                dr_center_distance_mean=masked_mean(distances, pairs),
                dr_center_target_distance_mean=masked_mean(targets, pairs),
                dr_parameter_distance_mean=masked_mean(dr_distances, pairs),
                dr_center_parameter_correlation=_pearson_correlation(
                    distances.flatten(), dr_distances.flatten(), pairs.flatten()),
                dr_center_norm_mean=masked_mean(torch.linalg.vector_norm(centers, dim=-1), ready),
                dr_within_center_distance_mean=masked_mean(within, valid),
                dr_cross_motion_distance_mean=masked_mean(
                    torch.linalg.vector_norm(unit - other, dim=-1), valid),
            )
            for name, population in (("dr", ~nominal), ("nominal", nominal)):
                subset = pairs & population[:, None] & population[None, :]
                metrics[f"dr_center_{name}_pairs"] = subset.sum().float()
                metrics[f"dr_center_{name}_distance_mean"] = masked_mean(distances, subset)
                metrics[f"dr_center_{name}_target_distance_mean"] = masked_mean(targets, subset)
            for name, value in (("distance", distances), ("target_distance", targets)):
                eligible = value[pairs]
                quantiles = (torch.quantile(eligible, value.new_tensor([.1, .5, .9]))
                             if eligible.numel() else value.new_zeros(3))
                for i, q in enumerate((10, 50, 90)):
                    metrics[f"dr_center_{name}_batch_p{q}"] = quantiles[i]
    return loss, {key: value.detach() for key, value in metrics.items()}


CENTER_BATCH_FIELDS = frozenset({
    "center_" + name for name in ("history_state", "history_action", "history_next_state",
                                 "history_valid", "memory_interactions", "memory_valid",
                                 "pair_valid", "world_id", "motion_id", "session", "age_steps")
} | {"physics_session", "dr_metric"})


def dr_center_pair_valid(batch):
    """Raw archive validity includes disjoint histories and unchanged physics."""
    return (batch["center_pair_valid"]
            & (batch["world_id"] == batch["center_world_id"])
            & (batch["motion_id"] != batch["center_motion_id"])
            & (batch["physics_session"] == batch["center_session"])
            & batch["history_valid"].all(1) & batch["memory_valid"].all(1)
            & batch["center_history_valid"].all(1) & batch["center_memory_valid"].all(1))


def dr_cross_motion_positive_loss(latent, archived_latent, valid, is_nominal):
    """Squared unit-latent distance, averaged over eligible DR pairs only.

    Both views are encoded by the current encoder and receive gradients.
    The 64 coordinates are summed, not averaged; nominal rows do not dilute
    the coefficient. A batch without eligible DR pairs contributes zero.
    """
    unit = F.normalize(latent.float(), dim=-1, eps=1e-8)
    other = F.normalize(archived_latent.float(), dim=-1, eps=1e-8)
    eligible = valid & ~is_nominal.bool()
    difference = unit - other
    loss = masked_mean(difference.square().sum(-1), eligible)
    with torch.no_grad():
        metrics = {
            "dr_positive_loss": loss.detach(),
            "dr_positive_pairs": eligible.sum().float(),
            "dr_positive_fraction_of_dr": eligible.sum() / (~is_nominal.bool()).sum().clamp_min(1),
            "dr_positive_unit_distance": masked_mean(torch.linalg.vector_norm(difference, dim=-1), eligible),
        }
    return loss, metrics


class DRCenterReplayBuffer(WeakPairReplayBuffer):
    """Reuse only the raw archive mechanism; no weak-pair losses are inherited."""

    def __init__(self, *, dr_metric_dim=38, center_archive_slots=4,
                 center_archive_interval=200, require_center_probe=True, **kwargs):
        if dr_metric_dim < 1:
            raise ValueError("DR metric dimension must be positive")
        self.dr_metric_dim = dr_metric_dim
        self.require_center_probe = require_center_probe
        self._current_dr_metric = None
        super().__init__(weak_archive_slots=center_archive_slots,
                         weak_archive_interval=center_archive_interval, **kwargs)

    def _allocate(self):
        if self._history:
            return
        super()._allocate()
        self._samples["dr_metric"] = torch.empty(self.capacity, self.dr_metric_dim, device=self.device)

    @property
    def storage_bytes(self):
        return super().storage_bytes + (0 if self._current_dr_metric is None else
                                        self._current_dr_metric.numel() * 4)

    @property
    def estimated_storage_bytes(self):
        return super().estimated_storage_bytes + (self.capacity + self.num_worlds) * self.dr_metric_dim * 4

    def _initialize_weak_archive(self, batch):
        # Nominal worlds need cross-motion center estimates too. This affects
        # archive allocation only; replay keeps their true nominal labels.
        super()._initialize_weak_archive({**batch, "is_nominal": torch.zeros_like(batch["is_nominal"])})

    def add_step(self, batch):
        metric = batch["dr_metric"]
        if metric.shape != (self.num_worlds, self.dr_metric_dim) or metric.device != self.device:
            raise ValueError("DR metric shape/device differs from the replay schema")
        if self._current_dr_metric is not None:
            changed = (metric != self._current_dr_metric).any(-1)
            if "parameters_changed" in batch:
                changed = changed | batch["parameters_changed"]
            batch = {**batch, "parameters_changed": changed}
        # Copy, since simulator-owned static-label tensors can be modified in place.
        self._current_dr_metric = metric.detach().float().clone()
        return super().add_step(batch)

    def _append_samples(self, samples, count):
        samples["dr_metric"] = self._current_dr_metric[samples["env_id"]]
        # Base memory records the anchor's session. Windows crossing a physics
        # change are rejected by its active-sample filter, before any sampling.
        super()._append_samples(samples, count)

    def _positive_ready_indices(self):
        if not self.require_center_probe:
            return super()._positive_ready_indices()
        active = self._active_sample_indices()
        if not len(active) or not self.weak_archive:
            return active[:0]
        s = {name: value[active] for name, value in self._samples.items()
             if name in ("env_id", "context_full", "motion_id", "memory_session",
                         "memory_total", "memory_start", "collector_step")}
        a = self.weak_archive
        rows = a["lookup"][s["env_id"]]
        full = s["context_full"] & (s["memory_total"] - s["memory_start"] >= 30)
        candidates = (full[:, None]
                      & (a["collector_step"][rows] >= 0)
                      & (a["session"][rows] == s["memory_session"][:, None])
                      & (a["motion_id"][rows] != s["motion_id"][:, None])
                      & (s["memory_total"][:, None] - 30 >= a["excluded_until_chunk"][rows])
                      & (s["collector_step"][:, None] - 4 - a["collector_step"][rows] >= 50))
        return active[candidates.any(1)]

    def _choose_positive_indices(self, indices):
        # Local +/-5 views are unused by this objective. Keep tensor fields for
        # the shared five-step predictor's input contract, without extra RNG.
        return indices, torch.zeros_like(indices, dtype=torch.bool)

    def _materialize_context(self, selected):
        result = super()._materialize_context(selected)
        query_slot = (selected["collector_step"] - 4).remainder(self.ring_steps)
        count = self._memory_snapshots["short_count"][query_slot, selected["env_id"]]
        # A physics change can keep the same episode/motion counters. Respect
        # the query-time bank's short count as well, so pre-change interactions
        # cannot leak into prediction or center views through the replay ring.
        result["valid"] &= torch.arange(50, device=self.device)[None] >= 50 - count[:, None]
        return result

    def _extra_sample_fields(self, selected, context, normalization):
        archived = super()._extra_sample_fields(selected, context, normalization)
        result = {name.replace("weak_", "center_", 1): value for name, value in archived.items()}
        result["dr_metric"] = selected["dr_metric"]
        return result


class DRCenterObjective(Memory350Objective):
    def _encode_views(self, batch):
        # Missing fields are an error even with validate_batch=False or eval().
        # Old frozen response probes cannot masquerade as DR-center validation.
        missing = CENTER_BATCH_FIELDS - batch.keys()
        if missing:
            raise KeyError(f"Missing DR-center batch fields: {sorted(missing)}")

        def cat(name):
            return torch.cat((batch[name], batch["center_" + name]), dim=0)

        combined = self.model.encode_context(
            cat("history_state"), cat("history_action"),
            torch.cat((batch["state"][:, 0], batch["center_history_next_state"][:, -1]), dim=0),
            cat("history_valid"), history_next_state=cat("history_next_state"),
            memory_interactions=cat("memory_interactions"), memory_valid=cat("memory_valid"))
        return combined.split(batch["state"].size(0), dim=0)

    def _representation_terms(self, batch, views, target, *, compute_metrics):
        valid = dr_center_pair_valid(batch)
        loss, metrics = dr_center_relation_loss(
            views[0], views[1], batch["dr_metric"], batch["world_id"],
            batch["physics_session"], valid, batch["is_nominal"],
            distance_scale=self.loss_config.dr_distance_scale,
            beta=self.loss_config.dr_relation_beta, compute_metrics=compute_metrics)
        metrics["dr_center_weighted_loss"] = loss.detach() * self.loss_config.representation_weight
        partner, _ = _cross_world_partner(batch["world_id"])
        # A-B is retained only as a numerical diagnostic on evaluation, never
        # as a target or a term in the optimization path.
        response, response_valid = (self._representation_response(batch, target)
                                    if compute_metrics else (None, None))
        return loss, metrics, partner, response, response_valid

    def _extra_representation_loss(self, batch, views):
        loss, metrics = dr_cross_motion_positive_loss(
            views[0], views[1], dr_center_pair_valid(batch), batch["is_nominal"])
        weighted = self.loss_config.dr_positive_weight * loss
        metrics["dr_positive_weighted_loss"] = weighted.detach()
        return weighted, metrics
