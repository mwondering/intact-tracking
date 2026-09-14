"""Isolated performance prototypes; no production trainer imports this module.

The pair optimization requires same-world/session local positives at +/-5 steps,
so the long window can move by at most one 10-step chunk. Replay would supply
positive.memory_total - anchor.memory_total as ``local_chunk_shift``. The
benchmark reconstructs that metadata from exact saved raw-chunk equality.
"""

import torch

from intact_tracking.memory350_bank import InteractionMemory
from intact_tracking.memory350_weak_pairs import WeakPairObjective


def encode_local_shared(encoder, history_state, history_action, history_next_state,
                        history_valid, interactions, valid, local_chunk_shift):
    """Encode anchor/positive/weak views, sharing only local-pair chunk graphs.

    All three views still run memory and final attention independently. One
    boundary chunk per positive is always encoded to keep tensor shapes fixed,
    even when the two long windows are equal. No learned values survive a call.
    """
    if encoder.config.dropout != 0:
        raise ValueError('Sharing local chunk graphs requires dropout=0')
    batch = local_chunk_shift.shape[0]
    chunks, steps = interactions.shape[1:3]
    short = torch.cat((history_state, history_action, history_next_state), -1)
    short = encoder.interaction_projection(short.masked_fill(~history_valid[..., None], 0))
    short = short.masked_fill(~history_valid[..., None], 0)

    anchor, positive, weak = interactions.split(batch)
    av, pv, wv = valid.split(batch)
    rows = torch.arange(batch, device=interactions.device)
    boundary = torch.where(local_chunk_shift < 0, 0, chunks - 1)
    unique = torch.cat((anchor, weak, positive[rows, boundary][:, None]), dim=1)
    unique_valid = torch.cat((av, wv, pv[rows, boundary][:, None]), dim=1)
    unique_count = 2 * chunks + 1
    clean = unique.masked_fill(~unique_valid[..., None, None], 0)
    projected = encoder.interaction_projection(clean).reshape(batch * unique_count, steps, -1)
    tokens = torch.cat((encoder.chunk_cls.expand(batch * unique_count, -1, -1), projected), 1)
    hidden = encoder.chunk_encoder(tokens + encoder.chunk_position)
    summary = encoder.chunk_output(hidden[:, 0]).reshape(batch, unique_count, -1)
    summary = summary.masked_fill(~unique_valid[..., None], 0)
    a_summary, w_summary, edge = summary.split((chunks, chunks, 1), dim=1)
    index = torch.arange(chunks, device=interactions.device)[None] + local_chunk_shift[:, None]
    shared = a_summary.gather(1, index.clamp(0, chunks - 1)[..., None].expand(-1, -1, summary.shape[-1]))
    p_summary = torch.where(((index >= 0) & (index < chunks))[..., None], shared, edge)
    p_summary = p_summary.masked_fill(~pv[..., None], 0)
    summaries = torch.cat((a_summary, p_summary, w_summary), dim=0)

    n = 3 * batch
    tokens = torch.cat((encoder.memory_cls.expand(n, -1, -1), summaries), dim=1)
    padding = torch.cat((torch.zeros(n, 1, dtype=torch.bool, device=valid.device), ~valid), 1)
    hidden = encoder.memory_encoder(tokens + encoder.memory_position, src_key_padding_mask=padding)
    memory = encoder.memory_output(hidden[:, 0]).masked_fill(~valid.any(1, keepdim=True), 0)
    tokens = torch.cat((encoder.cls_token.expand(n, -1, -1), memory[:, None] + encoder.memory_type, short), 1)
    padding = torch.cat((torch.zeros(n, 1, dtype=torch.bool, device=valid.device),
                         ~valid.any(1, keepdim=True), ~history_valid), 1)
    encoded = encoder.transformer(tokens + encoder.position, src_key_padding_mask=padding)
    return encoder.output(encoded[:, 0])


class BenchmarkWeakObjective(WeakPairObjective):
    def __init__(self, model, loss_config):
        super().__init__(model, loss_config)
        self.view_function = None

    def _encode_views(self, batch):
        if self.view_function is None:
            return super()._encode_views(batch)
        def cat(name):
            return torch.cat([batch[p + name] for p in ('', 'positive_', 'weak_')], dim=0)
        args = [cat(n) for n in ('history_state', 'history_action', 'history_next_state',
                                'history_valid', 'memory_interactions', 'memory_valid')]
        result = self.view_function(*args, batch['local_chunk_shift'])
        return result.split(batch['state'].shape[0])


class DeferredStatsMemory(InteractionMemory):
    """Only defer GPU scalar statistics; all history operations are unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flushed_chunks = torch.zeros((), dtype=torch.int64, device=self.device)
        self.discarded_tail_transitions = torch.zeros_like(self.flushed_chunks)
        self.reset_boundaries = torch.zeros_like(self.flushed_chunks)

    def _flush(self, ids):
        if ids.numel() == 0:
            return
        count = self.pending_count[ids] + self.short_count[ids]
        short, _ = self.ordered_short(ids)
        offsets = torch.arange(self.chunk_steps - 1 + self.short_steps, device=self.device)[None]
        from_pending = offsets < self.pending_count[ids, None]
        short_index = (offsets - self.pending_count[ids, None]
                       + self.short_steps - self.short_count[ids, None]).clamp(0, self.short_steps - 1)
        pending_index = offsets.expand(len(ids), -1).clamp_max(self.chunk_steps - 1)
        sequence = torch.where(from_pending[..., None], self.pending[ids[:, None], pending_index],
                               short[torch.arange(len(ids), device=self.device)[:, None], short_index])
        for index in range((self.chunk_steps - 1 + self.short_steps) // self.chunk_steps):
            usable = count >= (index + 1) * self.chunk_steps
            self._commit(ids[usable], sequence[usable, index * self.chunk_steps:(index + 1) * self.chunk_steps])
        self.flushed_chunks.add_((count // self.chunk_steps).sum())
        self.discarded_tail_transitions.add_((count % self.chunk_steps).sum())
        self.short_count[ids] = self.short_cursor[ids] = self.pending_count[ids] = 0

    def finish_step(self, interaction, reset_boundary):
        if interaction.shape != (self.num_worlds, self.short.shape[-1]):
            raise ValueError('Expected one explicit completed interaction per world')
        boundary = reset_boundary.bool()
        self._flush(self._worlds[boundary])
        self.reset_boundaries.add_(boundary.sum())
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

    def metrics(self):
        counts = (self.total_chunks - self.session_start).clamp_max(self.long_chunks)
        statistics = torch.stack((self.short_count.float().mean(),
                                  (self.short_count == self.short_steps).float().mean(),
                                  counts.float().mean(), (counts > 0).float().mean(),
                                  (counts == self.long_chunks).float().mean(),
                                  self.pending_count.float().mean()))
        names = ('short_valid_steps_mean', 'short_full_fraction', 'long_valid_chunks_mean',
                 'long_available_fraction', 'long_full_fraction', 'pending_steps_mean')
        # Integer counters must retain int64 precision rather than cast to float.
        counters = torch.stack((self.flushed_chunks, self.discarded_tail_transitions,
                                self.reset_boundaries)).cpu().tolist()
        return {**dict(zip(names, statistics.cpu().tolist())),
                **dict(zip(('flushed_chunks', 'discarded_tail_transitions', 'reset_boundaries'), counters)),
                'parameter_invalidations': self.parameter_invalidations}
