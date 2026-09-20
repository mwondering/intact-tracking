"""Weak cross-motion invariance and direct DR separation for Memory350."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from intact_tracking.forward_predictor_objective import ForwardPredictorLossConfig
from intact_tracking.memory350_objective import Memory350Objective
from intact_tracking.memory350_replay import Memory350ReplayBuffer


@dataclass(frozen=True)
class WeakPairLossConfig(ForwardPredictorLossConfig):
    weak_positive_weight: float = 0.002
    weak_negative_weight: float = 0.002
    weak_negative_margin: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if any(not math.isfinite(x) or x < 0 for x in
               (self.weak_positive_weight, self.weak_negative_weight)):
            raise ValueError("Weak pair weights must be finite and nonnegative")
        if not 0 < self.weak_negative_margin < 2:
            raise ValueError("Weak negative margin must lie strictly between zero and two")


def masked_mean(value, valid):
    return value.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def weak_pair_losses(latent, positive_latent, positive_valid, dr_full, world_id, *, margin):
    """All distinct DR worlds are negatives, with no motion/phase conditions."""
    unit = F.normalize(latent.float(), dim=-1, eps=1e-8)
    positive = F.normalize(positive_latent.float(), dim=-1, eps=1e-8)
    positive_valid = positive_valid & dr_full
    positive_cosine = (unit * positive).sum(-1).clamp(-1, 1)
    positive_distance = torch.linalg.vector_norm(unit - positive, dim=-1)
    positive_loss = masked_mean(1 - positive_cosine, positive_valid)
    # The direct kernel avoids cancellation in the squared-distance identity
    # and has finite gradients for identical vectors and the masked diagonal.
    distance = torch.cdist(unit, unit, compute_mode="donot_use_mm_for_euclid_dist")
    negative_valid = (dr_full[:, None] & dr_full[None, :]
                      & (world_id[:, None] != world_id[None, :]))
    negative_valid &= torch.ones_like(negative_valid).triu(diagonal=1)
    negative_loss = masked_mean((margin - distance).clamp_min(0).square(), negative_valid)
    metrics = {
        "weak_positive_loss": positive_loss.detach(),
        "weak_negative_loss": negative_loss.detach(),
        "weak_positive_pairs": positive_valid.sum().float(),
        "weak_negative_pairs": negative_valid.sum().float(),
        "weak_positive_fraction_of_full_dr": positive_valid.sum() / dr_full.sum().clamp_min(1),
        "weak_positive_unit_distance": masked_mean(positive_distance, positive_valid).detach(),
        "weak_negative_unit_distance": masked_mean(distance, negative_valid).detach(),
        "weak_negative_active_fraction": masked_mean((distance < margin).float(), negative_valid),
    }
    return positive_loss, negative_loss, metrics


class WeakPairReplayBuffer(Memory350ReplayBuffer):
    """Keep a small float32 raw-history archive beyond the recent replay ring.

    Each snapshot is causal at its own query time. Positive views are older,
    from a different motion in the same physics session, with no reused raw
    interactions even after short histories are flushed into long memory.
    """

    def __init__(self, *, weak_archive_slots=4, weak_archive_interval=200, **kwargs):
        super().__init__(**kwargs)
        if weak_archive_slots < 2 or weak_archive_interval < 1:
            raise ValueError("Weak archive needs at least two slots and a positive interval")
        self.weak_archive_slots = weak_archive_slots
        self.weak_archive_interval = weak_archive_interval
        self.weak_archive = {}
        self._weak_generator = torch.Generator(device=self.device)
        self._weak_generator.manual_seed(int(kwargs.get("seed", 0)) + 590117)

    @property
    def storage_bytes(self):
        return super().storage_bytes + sum(x.numel() * x.element_size()
                                          for x in self.weak_archive.values())

    @property
    def estimated_storage_bytes(self):
        # Nominal50 uses half the worlds; this estimate is an upper bound.
        return super().estimated_storage_bytes + self.num_worlds * self.weak_archive_slots * 350 * self.memory.short.shape[-1] * 4

    def _initialize_weak_archive(self, batch):
        ids = (~batch["is_nominal"].bool()).nonzero(as_tuple=False).flatten()
        count, slots = len(ids), self.weak_archive_slots
        lookup = torch.full((self.num_worlds,), -1, dtype=torch.long, device=self.device)
        lookup[ids] = torch.arange(count, device=self.device)
        self.weak_archive = {
            "env_ids": ids, "lookup": lookup,
            "raw": torch.empty(count, slots, 350, self.memory.short.shape[-1], dtype=torch.float32, device=self.device),
            "next_slot": torch.zeros(count, dtype=torch.long, device=self.device),
            "last_capture": torch.full((count,), -self.weak_archive_interval, dtype=torch.long, device=self.device),
            **{name: torch.full((count, slots), -1, dtype=torch.long, device=self.device)
               for name in ("motion_id", "session", "collector_step", "excluded_until_chunk")},
        }

    def add_step(self, batch):
        count = super().add_step(batch)
        if not self.weak_archive:
            self._initialize_weak_archive(batch)
        # Snapshot only completed outcomes, after the base memory accepted the
        # step. A reset edge leaves short_count=0 and cannot enter this archive.
        if self.collector_step % 10:
            return count
        archive, bank = self.weak_archive, self.memory
        ids = archive["env_ids"]
        full = ((bank.short_count[ids] == 50)
                & (bank.total_chunks[ids] - bank.session_start[ids] >= 30)
                & (self.collector_step - archive["last_capture"] >= self.weak_archive_interval))
        rows = full.nonzero(as_tuple=False).flatten()
        ids = ids[rows]
        if not len(ids):
            return count
        short, short_valid = bank.ordered_short(ids)
        long, long_valid = bank.read_chunks(ids)
        if not bool(short_valid.all() & long_valid.all()):
            raise RuntimeError("Weak positive snapshots must contain all 350 interactions")
        slots = archive["next_slot"][rows]
        archive["raw"][rows, slots] = torch.cat((short, long.flatten(1, 2)), dim=1)
        archive["motion_id"][rows, slots] = batch["motion_id"][ids]
        archive["session"][rows, slots] = bank.session[ids]
        archive["collector_step"][rows, slots] = self.collector_step
        reserved = (bank.short_count[ids] + bank.pending_count[ids] + 9) // 10
        archive["excluded_until_chunk"][rows, slots] = bank.total_chunks[ids] + reserved
        archive["next_slot"][rows] = (slots + 1).remainder(self.weak_archive_slots)
        archive["last_capture"][rows] = self.collector_step
        return count

    def _extra_sample_fields(self, selected, context, normalization):
        archive = self.weak_archive
        batch_size = len(selected["env_id"])
        full = context["valid"].all(1) & context["memory_valid"].all(1)
        rows = archive["lookup"][selected["env_id"]]
        if len(archive["env_ids"]):
            safe_rows = rows.clamp_min(0)
            query_time = selected["collector_step"] - 4
            valid = ((rows >= 0)[:, None] & full[:, None]
                     & (archive["collector_step"][safe_rows] >= 0)
                     & (archive["session"][safe_rows] == selected["memory_session"][:, None])
                     & (archive["motion_id"][safe_rows] != selected["motion_id"][:, None])
                     & (selected["memory_total"][:, None] - 30 >= archive["excluded_until_chunk"][safe_rows])
                     & (query_time[:, None] - archive["collector_step"][safe_rows] >= 50))
            scores = torch.rand(valid.shape, device=self.device, generator=self._weak_generator).masked_fill(~valid, -1)
            slots = scores.argmax(1)
            pair_valid = valid.any(1)
            raw = archive["raw"][safe_rows, slots].masked_fill(~pair_valid[:, None, None], 0)
            motion = archive["motion_id"][safe_rows, slots]
            session = archive["session"][safe_rows, slots]
            gap = query_time - archive["collector_step"][safe_rows, slots]
        else:
            pair_valid = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            raw = torch.zeros(batch_size, 350, self.memory.short.shape[-1], device=self.device)
            motion = session = gap = torch.zeros_like(selected["world_id"])
        state_mean, state_std, action_mean, action_std = self._context_normalization_tensors(normalization)
        s, a = self.context_state_dim, self.context_state_dim + 29
        normalized = torch.cat(((raw[..., :s] - state_mean) / state_std,
                                (raw[..., s:a] - action_mean) / action_std,
                                (raw[..., a:] - state_mean) / state_std), -1)
        normalized = normalized.masked_fill(~pair_valid[:, None, None], 0)
        return {
            "weak_history_state": normalized[:, :50, :s],
            "weak_history_action": normalized[:, :50, s:a],
            "weak_history_next_state": normalized[:, :50, a:],
            "weak_history_valid": pair_valid[:, None].expand(-1, 50),
            "weak_memory_interactions": normalized[:, 50:].reshape(batch_size, 30, 10, 2 * s + 29),
            "weak_memory_valid": pair_valid[:, None].expand(-1, 30),
            "weak_pair_valid": pair_valid,
            "weak_world_id": selected["world_id"],
            "weak_motion_id": motion, "weak_session": session,
            "weak_age_steps": gap.masked_fill(~pair_valid, 0),
            "physics_session": selected["memory_session"],
        }


WEAK_BATCH_FIELDS = frozenset({
    "weak_" + name for name in ("history_state", "history_action", "history_next_state",
                               "history_valid", "memory_interactions", "memory_valid",
                               "pair_valid", "world_id", "motion_id", "session", "age_steps")
} | {"physics_session"})


class WeakPairObjective(Memory350Objective):
    def _encode_views(self, batch):
        if "weak_pair_valid" not in batch:
            return super()._encode_views(batch)
        prefixes = ("", "positive_", "weak_")
        def cat(name):
            return torch.cat([batch[prefix + name] for prefix in prefixes], dim=0)
        combined = self.model.encode_context(
            cat("history_state"), cat("history_action"),
            cat("history_next_state")[:, -1],
            cat("history_valid"), history_next_state=cat("history_next_state"),
            memory_interactions=cat("memory_interactions"), memory_valid=cat("memory_valid"))
        return combined.split(batch["state"].size(0), dim=0)

    def _extra_representation_loss(self, batch, views):
        full_dr = (~batch["is_nominal"].bool() & batch["history_valid"].all(1)
                   & batch["memory_valid"].all(1))
        if len(views) == 3:
            valid = (batch["weak_pair_valid"] & (batch["world_id"] == batch["weak_world_id"])
                     & (batch["motion_id"] != batch["weak_motion_id"])
                     & (batch["physics_session"] == batch["weak_session"])
                     & batch["weak_history_valid"].all(1) & batch["weak_memory_valid"].all(1))
            positive = views[2]
            age = batch["weak_age_steps"].float()
        else:
            valid = torch.zeros_like(full_dr)
            positive = views[0]
            age = torch.zeros_like(full_dr, dtype=torch.float32)
        positive_loss, negative_loss, metrics = weak_pair_losses(
            views[0], positive, valid, full_dr, batch["world_id"],
            margin=self.loss_config.weak_negative_margin)
        loss = (self.loss_config.weak_positive_weight * positive_loss
                + self.loss_config.weak_negative_weight * negative_loss)
        metrics.update(weak_pair_weighted_loss=loss.detach(),
                       weak_positive_age_steps=masked_mean(age, valid & full_dr))
        return loss, metrics
