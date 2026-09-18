"""Nominal response geometry plus ordinal supervision of DR environment centers."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from intact_tracking.memory350_nominal_direction import (
    NominalDirectionLossConfig,
    NominalDirectionObjective,
)
from intact_tracking.memory350_response_window import ResponseWindowReplay


RANK_LOSS_FIELDS = (
    "dr_center_rank_weight", "dr_rank_temperature", "dr_rank_min_gap",
    "dr_center_rank_version",
)
RANK_METRICS = (
    "dr_center_rank_loss", "dr_center_rank_weighted_loss", "dr_center_rank_worlds",
    "dr_center_rank_pairs", "dr_center_rank_comparisons", "dr_center_rank_accuracy",
    "dr_center_rank_tie_fraction", "dr_center_rank_parameter_gap",
    "dr_center_rank_latent_distance", "dr_center_rank_parameter_distance",
)


@dataclass(frozen=True)
class NominalDRRankLossConfig(NominalDirectionLossConfig):
    dr_center_rank_weight: float = 0.002
    dr_rank_temperature: float = 0.1
    dr_rank_min_gap: float = 0.01
    dr_center_rank_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.dr_center_rank_weight) or self.dr_center_rank_weight < 0:
            raise ValueError("DR center rank weight must be finite and nonnegative")
        if not math.isfinite(self.dr_rank_temperature) or self.dr_rank_temperature <= 0:
            raise ValueError("DR rank temperature must be finite and positive")
        if not math.isfinite(self.dr_rank_min_gap) or not 0 <= self.dr_rank_min_gap < 1:
            raise ValueError("DR rank minimum parameter gap must be in [0,1)")
        if self.dr_center_rank_version != 1:
            raise ValueError("Unsupported DR center rank objective version")


def dr_center_rank_loss(latent, archived_latent, dr_metric, world_id, session,
                        valid, is_nominal, *, temperature=0.1, min_gap=0.01):
    """Order center-pair distances, without assigning an absolute latent distance.

    Pool two current-encoder unit views per row by (world, physics session),
    excluding nominal and invalid rows. Do not normalize the resulting mean.
    Sort all unordered center pairs by their detached physical distance. Compare
    pairs at rank offsets 1,2,4,...: O(P log P), including local and broad ranks.
    Ties/near-ties in physical distance do not prescribe an order. The loss is
    gap-weighted temperature*softplus((near_latent-far_latent)/temperature).
    Thus two near-nominal centers need not reach a fixed margin such as 1.1.
    """
    n = len(latent)
    if (latent.ndim != 2 or archived_latent.shape != latent.shape
            or dr_metric.ndim != 2 or len(dr_metric) != n
            or any(x.shape != (n,) for x in (world_id, session, valid, is_nominal))):
        raise ValueError("Invalid DR center rank tensor shapes")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= min_gap < 1:
        raise ValueError("Invalid DR center rank temperature or gap")
    # Keep a differentiable zero even when this microbatch has no center pairs.
    zero = (latent.float().sum() + archived_latent.float().sum()) * 0
    metrics = {name: zero.detach() for name in RANK_METRICS
               if name != "dr_center_rank_weighted_loss"}
    eligible = valid.bool() & ~is_nominal.bool()
    identities, inverse = torch.unique(
        torch.stack((world_id[eligible], session[eligible]), -1), dim=0, return_inverse=True)
    count = len(identities)
    metrics["dr_center_rank_worlds"] = zero.detach().new_tensor(count)
    if count < 2:
        return zero, metrics
    unit = F.normalize(latent[eligible].float(), dim=-1, eps=1e-8)
    other = F.normalize(archived_latent[eligible].float(), dim=-1, eps=1e-8)
    counts = torch.bincount(inverse, minlength=count).float()[:, None]

    def pool(x):
        return x.new_zeros(count, x.shape[-1]).index_add(0, inverse, x) / counts

    centers = pool((unit + other) * 0.5)
    parameters = pool(dr_metric[eligible].detach().float())
    pairs = torch.triu_indices(count, count, offset=1, device=latent.device)
    distances = torch.linalg.vector_norm(centers[pairs[0]] - centers[pairs[1]], dim=-1)
    physical = torch.linalg.vector_norm(parameters[pairs[0]] - parameters[pairs[1]], dim=-1)
    order = physical.argsort(stable=True)
    physical, distances = physical[order], distances[order]
    metrics.update(
        dr_center_rank_pairs=zero.detach().new_tensor(len(physical)),
        dr_center_rank_latent_distance=distances.detach().mean(),
        dr_center_rank_parameter_distance=physical.mean(),
    )
    numerator, denominator = zero, zero.detach()
    comparisons, correct, tied, gap_sum = (zero.detach() for _ in range(4))
    offset = 1
    while offset < len(physical):
        gap = physical[offset:] - physical[:-offset]
        mask = gap > min_gap
        weight = gap * mask
        delta = distances[:-offset] - distances[offset:]
        numerator = numerator + (weight * temperature * F.softplus(delta / temperature)).sum()
        denominator = denominator + weight.sum()
        comparisons = comparisons + mask.sum()
        correct = correct + ((delta.detach() < -1e-7) & mask).sum()
        tied = tied + ((delta.detach().abs() <= 1e-7) & mask).sum()
        gap_sum = gap_sum + weight.sum()
        offset *= 2
    loss = numerator / denominator.clamp_min(1e-12)
    metrics.update(
        dr_center_rank_loss=loss.detach(),
        dr_center_rank_comparisons=comparisons,
        dr_center_rank_accuracy=correct / comparisons.clamp_min(1),
        dr_center_rank_tie_fraction=tied / comparisons.clamp_min(1),
        dr_center_rank_parameter_gap=gap_sum / comparisons.clamp_min(1),
    )
    return loss, metrics


def rank_view_valid(batch):
    return (~batch["is_nominal"].bool() & batch["weak_pair_valid"].bool()
            & (batch["world_id"] == batch["weak_world_id"])
            & (batch["physics_session"] == batch["weak_session"])
            & (batch["motion_id"] != batch["weak_motion_id"])
            & batch["history_valid"].all(1) & batch["memory_valid"].all(1)
            & batch["weak_history_valid"].all(1) & batch["weak_memory_valid"].all(1))


class NominalDRRankObjective(NominalDirectionObjective):
    def __init__(self, model, loss_config=None, *, anchor=None):
        super().__init__(model, loss_config or NominalDRRankLossConfig(), anchor=anchor)

    def _extra_representation_loss(self, batch, views):
        if "dr_metric" not in batch or len(views) != 3:
            raise ValueError("DR ranking requires physical labels and archived cross-motion views")
        original, metrics = super()._extra_representation_loss(batch, views)
        rank_loss, rank_metrics = dr_center_rank_loss(
            views[0], views[2], batch["dr_metric"], batch["world_id"],
            batch["physics_session"], rank_view_valid(batch), batch["is_nominal"],
            temperature=self.loss_config.dr_rank_temperature,
            min_gap=self.loss_config.dr_rank_min_gap)
        weighted = self.loss_config.dr_center_rank_weight * rank_loss
        metrics.update(rank_metrics, dr_center_rank_weighted_loss=weighted.detach())
        return original + weighted, metrics


class NominalDRRankReplay(ResponseWindowReplay):
    """Ten-step labels plus physical labels; reuse the existing causal DR archive."""

    def __init__(self, *, dr_metric_dim=38, require_rank_probe=True, **kwargs):
        if dr_metric_dim < 1:
            raise ValueError("DR metric dimension must be positive")
        self.dr_metric_dim = dr_metric_dim
        self.require_rank_probe = require_rank_probe
        self._current_dr_metric = None
        super().__init__(**kwargs)

    def _allocate(self):
        initialized = bool(self._history)
        super()._allocate()
        if not initialized:
            self._samples["dr_metric"] = torch.empty(
                self.capacity, self.dr_metric_dim, device=self.device)

    @property
    def storage_bytes(self):
        return super().storage_bytes + (0 if self._current_dr_metric is None else
                                        self._current_dr_metric.numel() * 4)

    @property
    def estimated_storage_bytes(self):
        return super().estimated_storage_bytes + (self.capacity + self.num_worlds) * self.dr_metric_dim * 4

    def add_step(self, batch):
        metric = batch["dr_metric"]
        if metric.shape != (self.num_worlds, self.dr_metric_dim) or metric.device != self.device:
            raise ValueError("DR labels must match this replay's worlds and schema")
        if self._current_dr_metric is not None:
            changed = (metric != self._current_dr_metric).any(-1)
            changed |= batch.get("parameters_changed", torch.zeros_like(changed))
            batch = {**batch, "parameters_changed": changed}
        self._current_dr_metric = metric.detach().float().clone()
        return super().add_step(batch)

    def _append_samples(self, samples, count):
        samples["dr_metric"] = self._current_dr_metric[samples["env_id"]]
        super()._append_samples(samples, count)

    def _extra_sample_fields(self, selected, context, normalization):
        return {**super()._extra_sample_fields(selected, context, normalization),
                "dr_metric": selected["dr_metric"]}

    def _materialize_context(self, selected):
        context = super()._materialize_context(selected)
        # A physics-session change need not reset the episode counters.
        slot = (selected["collector_step"] - 4).remainder(self.ring_steps)
        count = self._memory_snapshots["short_count"][slot, selected["env_id"]]
        context["valid"] &= torch.arange(50, device=self.device)[None] >= 50 - count[:, None]
        return context

    def _positive_ready_indices(self):
        active = super()._positive_ready_indices()
        if not self.require_rank_probe or not len(active) or not self.weak_archive:
            return active
        s, archive = self._samples, self.weak_archive
        rows = archive["lookup"][s["env_id"][active]]
        safe = rows.clamp_min(0)
        if not len(archive["env_ids"]):
            return active[:0]
        full = (s["context_full"][active]
                & (s["memory_total"][active] - s["memory_start"][active] >= 30))
        candidates = ((rows >= 0)[:, None] & full[:, None]
                      & (archive["collector_step"][safe] >= 0)
                      & (archive["session"][safe] == s["memory_session"][active, None])
                      & (archive["motion_id"][safe] != s["motion_id"][active, None])
                      & (s["memory_total"][active, None] - 30 >= archive["excluded_until_chunk"][safe])
                      & (s["collector_step"][active, None] - 4 - archive["collector_step"][safe] >= 50))
        # Preserve nominal in the fixed probe for the original anchor/prediction metrics.
        return active[s["is_nominal"][active] | candidates.any(1)]

    def can_sample_positive_pairs(self, count=1):
        if not super().can_sample_positive_pairs(count):
            return False
        if not self.require_rank_probe:
            return True
        ready = self._positive_ready_indices()
        dr = ready[~self._samples["is_nominal"][ready]]
        return torch.unique(self._samples["world_id"][dr]).numel() >= 3
