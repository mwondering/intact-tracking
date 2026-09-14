"""Causal, disjoint short history and reset-safe raw interaction chunks.

No learned tensors are cached here. Training re-encodes raw chunks with the
current parameters; saved query counters prevent replay from seeing the future.
"""

from __future__ import annotations

import torch


class InteractionMemory:
    def __init__(self, num_worlds, *, device="cpu", short_steps=50, chunk_steps=10,
                 long_chunks=30, archive_chunks=None, token_dim=171):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        self.short_steps, self.chunk_steps, self.long_chunks = short_steps, chunk_steps, long_chunks
        self.archive_chunks = archive_chunks or long_chunks
        if min(num_worlds, short_steps, chunk_steps, long_chunks) < 1 or self.archive_chunks < long_chunks:
            raise ValueError("Invalid interaction-memory capacity")
        self.short = torch.zeros(num_worlds, short_steps, token_dim, device=self.device)
        self.pending = torch.zeros(num_worlds, chunk_steps, token_dim, device=self.device)
        self.chunks = torch.zeros(num_worlds, self.archive_chunks, chunk_steps, token_dim, device=self.device)
        self.chunk_stamp = torch.full((num_worlds, self.archive_chunks), -1, dtype=torch.long, device=self.device)
        self.short_count = torch.zeros(num_worlds, dtype=torch.long, device=self.device)
        self.short_cursor = torch.zeros_like(self.short_count)
        self.pending_count = torch.zeros_like(self.short_count)
        self.total_chunks = torch.zeros_like(self.short_count)
        self.session_start = torch.zeros_like(self.short_count)
        self.session = torch.zeros_like(self.short_count)
        self._previous = torch.full((num_worlds, 4), -1, dtype=torch.long, device=self.device)
        self._worlds = torch.arange(num_worlds, device=self.device)
        self._short_offsets = torch.arange(short_steps, device=self.device)
        self.flushed_chunks = 0
        self.discarded_tail_transitions = 0
        self.reset_boundaries = 0
        self.parameter_invalidations = 0

    @property
    def storage_bytes(self):
        return sum(x.numel() * x.element_size() for x in vars(self).values() if isinstance(x, torch.Tensor))

    def ordered_short(self, env_ids=None):
        ids = self._worlds if env_ids is None else env_ids
        # Left-padding gives the predictor its latest ten actual interactions.
        logical = self._short_offsets[None] - (self.short_steps - self.short_count[ids, None])
        first = torch.where(self.short_count[ids] == self.short_steps, self.short_cursor[ids], 0)
        slots = (logical + first[:, None]).remainder(self.short_steps)
        valid = logical >= 0
        value = self.short[ids[:, None], slots]
        return value.masked_fill(~valid[..., None], 0), valid

    def _commit(self, ids, chunks):
        slots = self.total_chunks[ids].remainder(self.archive_chunks)
        self.chunks[ids, slots] = chunks
        self.chunk_stamp[ids, slots] = self.total_chunks[ids]
        self.total_chunks[ids] += 1

    def _flush(self, ids):
        if ids.numel() == 0:
            return
        count = self.pending_count[ids] + self.short_count[ids]
        # Build the old trial in chronological order: pending eviction tail,
        # then every completed short interaction. Never include the reset edge.
        short, _ = self.ordered_short(ids)
        offsets = torch.arange(self.chunk_steps - 1 + self.short_steps, device=self.device)[None]
        from_pending = offsets < self.pending_count[ids, None]
        short_index = (offsets - self.pending_count[ids, None]
                       + self.short_steps - self.short_count[ids, None]).clamp(0, self.short_steps - 1)
        pending_index = offsets.expand(len(ids), -1).clamp_max(self.chunk_steps - 1)
        sequence = torch.where(
            from_pending[..., None],
            self.pending[ids[:, None], pending_index],
            short[torch.arange(len(ids), device=self.device)[:, None], short_index],
        )
        for index in range((self.chunk_steps - 1 + self.short_steps) // self.chunk_steps):
            usable = count >= (index + 1) * self.chunk_steps
            self._commit(ids[usable], sequence[usable, index * self.chunk_steps:(index + 1) * self.chunk_steps])
        self.flushed_chunks += int((count // self.chunk_steps).sum())
        self.discarded_tail_transitions += int((count % self.chunk_steps).sum())
        self.short_count[ids] = self.short_cursor[ids] = self.pending_count[ids] = 0

    def invalidate(self, changed):
        ids = self._worlds[changed.bool()]
        self.short_count[ids] = self.short_cursor[ids] = self.pending_count[ids] = 0
        self.session_start[ids] = self.total_chunks[ids]
        self.session[ids] += 1
        self._previous[ids] = -1
        self.parameter_invalidations += int(ids.numel())

    def begin_step(self, episode_id, episode_step, motion_id, motion_step, parameters_changed=None):
        if parameters_changed is not None:
            self.invalidate(parameters_changed)
        now = torch.stack((episode_id, episode_step, motion_id, motion_step), dim=1)
        previous = self._previous
        discontinuity = (previous[:, 0] >= 0) & (
            (now[:, 0] != previous[:, 0]) | (now[:, 1] != previous[:, 1] + 1)
            | (now[:, 2] != previous[:, 2]) | (now[:, 3] != previous[:, 3] + 1))
        self._flush(self._worlds[discontinuity])
        self._previous.copy_(now)

    def finish_step(self, interaction, reset_boundary):
        if interaction.shape != (self.num_worlds, self.short.shape[-1]):
            raise ValueError("Expected one explicit completed (state, target, next-state) interaction per world")
        boundary = reset_boundary.bool()
        self._flush(self._worlds[boundary])
        self.reset_boundaries += int(boundary.sum())
        full = self._worlds[(~boundary) & (self.short_count == self.short_steps)]
        self.pending[full, self.pending_count[full]] = self.short[full, self.short_cursor[full]]
        self.pending_count[full] += 1
        commit = full[self.pending_count[full] == self.chunk_steps]
        self._commit(commit, self.pending[commit])
        self.pending_count[commit] = 0
        ids = self._worlds[~boundary]
        self.short[ids, self.short_cursor[ids]] = interaction[ids]
        self.short_cursor[ids] = (self.short_cursor[ids] + 1).remainder(self.short_steps)
        self.short_count[ids] = (self.short_count[ids] + 1).clamp_max(self.short_steps)

    def read_chunks(self, env_ids=None, *, total=None, session_start=None, session=None):
        ids = self._worlds if env_ids is None else env_ids
        total = self.total_chunks[ids] if total is None else total
        start = self.session_start[ids] if session_start is None else session_start
        version = self.session[ids] if session is None else session
        sequence = total[:, None] - self.long_chunks + torch.arange(self.long_chunks, device=self.device)[None]
        valid = (sequence >= start[:, None]) & (version == self.session[ids])[:, None]
        slots = sequence.remainder(self.archive_chunks)
        present = self.chunk_stamp[ids[:, None], slots] == sequence
        if bool((valid & ~present).any()):
            raise RuntimeError("Replay query references an overwritten memory chunk")
        raw = self.chunks[ids[:, None], slots]
        return raw.masked_fill(~valid[..., None, None], 0), valid

    def metrics(self):
        counts = (self.total_chunks - self.session_start).clamp_max(self.long_chunks)
        return {
            "short_valid_steps_mean": float(self.short_count.float().mean()),
            "short_full_fraction": float((self.short_count == self.short_steps).float().mean()),
            "long_valid_chunks_mean": float(counts.float().mean()),
            "long_available_fraction": float((counts > 0).float().mean()),
            "long_full_fraction": float((counts == self.long_chunks).float().mean()),
            "pending_steps_mean": float(self.pending_count.float().mean()),
            "flushed_chunks": self.flushed_chunks,
            "discarded_tail_transitions": self.discarded_tail_transitions,
            "reset_boundaries": self.reset_boundaries,
            "parameter_invalidations": self.parameter_invalidations,
        }
