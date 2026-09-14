"""Replay snapshots of disjoint short history and raw cross-trial memory."""

from __future__ import annotations

import math

import torch

from intact_tracking.data.predictor_online import ForwardPredictorReplayBuffer
from intact_tracking.memory350_bank import InteractionMemory


class Memory350ReplayBuffer(ForwardPredictorReplayBuffer):
    def __init__(self, **kwargs):
        kwargs.setdefault("context_history_steps", 50)
        if kwargs["context_history_steps"] != 50:
            raise ValueError("Memory350 replay uses short history 50")
        super().__init__(**kwargs)
        # In any elapsed interval at most floor((elapsed+59)/10) chunks can be
        # committed: the extra 59 covers a pending tail and a reset flush.
        archive = 30 + math.ceil(self.ring_steps / 10) + 6
        self.memory = InteractionMemory(self.num_worlds, device=self.device, archive_chunks=archive)
        self._memory_snapshots = {}

    def _allocate(self):
        if self._history:
            return
        super()._allocate()
        self._history["next_state"] = torch.zeros_like(self._history["state"])
        self._memory_snapshots = {key: torch.zeros(self.ring_steps, self.num_worlds,
                                                  dtype=torch.long, device=self.device)
                                  for key in ("memory_total", "memory_start", "memory_session", "short_count")}
        for key in ("memory_total", "memory_start", "memory_session"):
            self._samples[key] = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        self._samples["context_available"] = torch.empty(self.capacity, dtype=torch.bool, device=self.device)

    @property
    def storage_bytes(self):
        return super().storage_bytes + self.memory.storage_bytes + sum(
            x.numel() * x.element_size() for x in self._memory_snapshots.values())

    @property
    def estimated_storage_bytes(self):
        return (super().estimated_storage_bytes + self.memory.storage_bytes
                + self.ring_steps * self.num_worlds * (71 * 4 + 4 * 8)
                + self.capacity * (3 * 8 + 1))

    def add_step(self, batch):
        self._allocate()
        self.memory.begin_step(batch["episode_id"], batch["episode_step"], batch["motion_id"],
                               batch["motion_step"], batch.get("parameters_changed"))
        position = self.collector_step % self.ring_steps
        for name, value in (("memory_total", self.memory.total_chunks),
                            ("memory_start", self.memory.session_start),
                            ("memory_session", self.memory.session),
                            ("short_count", self.memory.short_count)):
            self._memory_snapshots[name][position].copy_(value)
        self._history["next_state"][position].copy_(batch["next_robot_state"])
        count = super().add_step(batch)
        raw = torch.cat((batch["robot_state"], batch["joint_target"], batch["next_robot_state"]), dim=-1)
        self.memory.finish_step(raw, batch["reset_boundary"])
        self._invalidate_caches()
        return count

    def _append_samples(self, samples, count):
        start = (samples["collector_step"] - (self.horizon - 1)).remainder(self.ring_steps)
        ids = samples["env_id"]
        for name in ("memory_total", "memory_start", "memory_session"):
            samples[name] = self._memory_snapshots[name][start, ids]
        samples["context_available"] = ((self._memory_snapshots["short_count"][start, ids] > 0)
                                         | (samples["memory_total"] > samples["memory_start"]))
        super()._append_samples(samples, count)

    def _active_sample_indices(self):
        active = super()._active_sample_indices()
        if active.numel():
            ids = self._samples["env_id"][active]
            total = self._samples["memory_total"][active]
            first = torch.maximum(total - self.memory.long_chunks, self._samples["memory_start"][active])
            valid = ((self._samples["memory_session"][active] == self.memory.session[ids])
                     & ((first >= self.memory.total_chunks[ids] - self.memory.archive_chunks)
                        | (first == total)))
            active = active[valid]
            self._active_indices = active
        return active

    def _positive_ready_indices(self):
        anchors, candidates, valid, _ = self._local_positive_candidates()
        if not anchors.numel():
            return anchors
        usable = self._samples["context_available"]
        return anchors[(valid & usable[anchors, None] & usable[candidates]).any(1)]

    def _choose_positive_indices(self, indices):
        anchors, candidates, candidate_valid, lookup = self._local_positive_candidates()
        if not anchors.numel():
            return indices, torch.zeros_like(indices, dtype=torch.bool)
        rows = lookup[indices]
        choices = candidates[rows.clamp_min(0)]
        usable = self._samples["context_available"]
        valid = (candidate_valid[rows.clamp_min(0)] & (rows >= 0)[:, None]
                 & usable[indices, None] & usable[choices])
        scores = torch.rand(valid.shape, generator=self._generator, device=self.device).masked_fill(~valid, -1)
        selected = choices.gather(1, scores.argmax(1)[:, None]).squeeze(1)
        return torch.where(valid.any(1), selected, indices), valid.any(1)

    def _materialize_context(self, selected):
        result = super()._materialize_context(selected)
        absolute = selected["collector_step"][:, None] - 4 - 50 + torch.arange(50, device=self.device)[None]
        slots, ids = absolute.remainder(self.ring_steps), selected["env_id"][:, None]
        # Explicit outcomes prevent an episode boundary from becoming a fake
        # state-action-next-state token, even when adjacent archive rows differ.
        result["next_state"] = self._history["next_state"][slots, ids]
        expected_motion_step = selected["motion_step"][:, None] - 50 + torch.arange(50, device=self.device)[None]
        result["valid"] &= ((~self._reset_history[slots, ids])
                            & (self._history["motion_id"][slots, ids] == selected["motion_id"][:, None])
                            & (self._history["motion_step"][slots, ids] == expected_motion_step))
        result["memory"], result["memory_valid"] = self.memory.read_chunks(
            selected["env_id"], total=selected["memory_total"],
            session_start=selected["memory_start"], session=selected["memory_session"])
        return result

    def sample_batch(self, batch_size, normalization, *, positive_ready_only=False):
        indices = self._sample_indices(batch_size, positive_ready_only=positive_ready_only)
        positive_indices, pair_valid = self._choose_positive_indices(indices)
        selected = {name: value[indices] for name, value in self._samples.items()}
        positive = {name: value[positive_indices] for name, value in self._samples.items()}
        context, positive_context = self._materialize_context(selected), self._materialize_context(positive)
        values = self._normalization_tensors(normalization, self.device)
        (state_mean, state_std, action_mean, action_std, foot_mean, foot_std,
         contact_force_mean, contact_force_std, delta_mean, delta_std) = values

        def normalize_context(ctx, prefix=""):
            memory = ctx["memory"]
            normalized_memory = torch.cat(((memory[..., :71] - state_mean) / state_std,
                                            (memory[..., 71:100] - action_mean) / action_std,
                                            (memory[..., 100:] - state_mean) / state_std), dim=-1)
            return {
                prefix + "history_state": self._normalize_masked(ctx["state"], state_mean, state_std, ctx["valid"]),
                prefix + "history_action": self._normalize_masked(ctx["action"], action_mean, action_std, ctx["valid"]),
                prefix + "history_next_state": self._normalize_masked(ctx["next_state"], state_mean, state_std, ctx["valid"]),
                prefix + "history_valid": ctx["valid"],
                prefix + "memory_interactions": normalized_memory.masked_fill(~ctx["memory_valid"][..., None, None], 0),
                prefix + "memory_valid": ctx["memory_valid"],
            }

        result = {
            "state": (selected["state"] - state_mean) / state_std,
            "nominal_state": (selected["nominal_state"] - state_mean) / state_std,
            "action": (selected["action"] - action_mean) / action_std,
            **normalize_context(context), **normalize_context(positive_context, "positive_"),
            "positive_current_state": (positive["state"][:, 0] - state_mean) / state_std,
            "positive_pair_valid": pair_valid,
            "foot": (selected["foot"] - foot_mean) / foot_std,
            "history_foot": (context["foot"] - foot_mean) / foot_std,
            "contact_force": (selected["contact_force"] - contact_force_mean) / contact_force_std,
            "contact_binary": selected["contact_binary"],
            "history_contact_force": (context["contact_force"] - contact_force_mean) / contact_force_std,
            "history_contact_binary": context["contact_binary"],
            "is_nominal": selected["is_nominal"], "world_id": selected["world_id"],
            "motion_id": selected["motion_id"], "context_full": context["valid"].all(1),
            **dict(zip(("state_mean", "state_std", "action_mean", "action_std", "foot_mean", "foot_std",
                        "contact_force_mean", "contact_force_std", "delta_mean", "delta_std"), values)),
        }
        result.update(self._extra_sample_fields(selected, context, normalization))
        return result

    def _extra_sample_fields(self, selected, context, normalization):
        """Optional experiment views, materialized at the same anchor query."""
        return {}
