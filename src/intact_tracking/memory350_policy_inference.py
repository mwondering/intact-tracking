"""Frozen Memory350 for PPO, caching only parameter-invariant chunk summaries."""

from contextlib import nullcontext

import torch

from intact_tracking.memory350_bank import InteractionMemory


class CachedMemory350Inference:
    def __init__(self, checkpoint, num_worlds, *, batch_size=512, chunk_batch_size=8192,
                 use_bfloat16=True):
        if checkpoint.encoder.training or any(p.requires_grad for p in checkpoint.encoder.parameters()):
            raise ValueError("Chunk caching requires a frozen encoder in eval mode")
        if min(batch_size, chunk_batch_size) < 1:
            raise ValueError("Inference batch sizes must be positive")
        self.checkpoint = checkpoint
        self.state_dim = checkpoint.state_mean.numel()
        self.action_dim = checkpoint.action_mean.numel()
        self.memory = InteractionMemory(num_worlds, device=checkpoint.state_mean.device,
                                        token_dim=2 * self.state_dim + self.action_dim)
        self.batch_size, self.chunk_batch_size = batch_size, chunk_batch_size
        self.use_bfloat16 = use_bfloat16 and self.memory.device.type == "cuda"
        width = checkpoint.config.context_dim
        self.chunk_features = torch.zeros(num_worlds, 30, width, device=self.memory.device)
        self.encoded_stamp = torch.full_like(self.memory.chunk_stamp, -1)
        self.memory_features = torch.zeros(num_worlds, width, device=self.memory.device)
        self.encoded_total = torch.full_like(self.memory.total_chunks, -1)
        self.encoded_session = torch.full_like(self.memory.session, -1)
        self.last_chunks_encoded = 0
        self.last_memories_encoded = 0

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bfloat16 else nullcontext()

    def _normalize(self, raw):
        c = self.checkpoint
        s, a = self.state_dim, self.state_dim + self.action_dim
        return torch.cat(((raw[..., :s] - c.state_mean) / c.state_std,
                          (raw[..., s:a] - c.action_mean) / c.action_std,
                          (raw[..., a:] - c.state_mean) / c.state_std), dim=-1)

    @torch.no_grad()
    def append(self, batch):
        self.memory.begin_step(batch["episode_id"], batch["episode_step"], batch["motion_id"],
                               batch["motion_step"], batch.get("parameters_changed"))
        interaction = torch.cat((batch["robot_state"], batch["joint_target"], batch["next_robot_state"]), -1)
        self.memory.finish_step(interaction, batch["reset_boundary"])

    @torch.no_grad()
    def episode_reset(self, env_ids=None):
        """Keep complete old-trial chunks, discard its incomplete tail, clear short history."""
        ids = self.memory._worlds if env_ids is None else env_ids
        self.memory._flush(ids)
        self.memory._previous[ids] = -1

    @torch.no_grad()
    def invalidate_parameters(self, changed):
        self.memory.invalidate(changed)

    @property
    def history_valid(self):
        # Compatibility with the existing evaluation's [time, environment] mask.
        return (self.memory._short_offsets[None] >=
                (self.memory.short_steps - self.memory.short_count[:, None])).T

    @property
    def metrics(self):
        return {**self.memory.metrics(), "cached_chunks_encoded_last_step": self.last_chunks_encoded,
                "cached_memories_encoded_last_step": self.last_memories_encoded,
                "memory_cache_storage_gib": sum(t.numel() * t.element_size() for t in
                    (self.chunk_features, self.encoded_stamp, self.memory_features,
                     self.encoded_total, self.encoded_session)) / 2**30}

    def _refresh_cache(self):
        bank, encoder = self.memory, self.checkpoint.encoder
        valid = bank.chunk_stamp >= bank.session_start[:, None]
        new = (valid & (bank.chunk_stamp != self.encoded_stamp)).nonzero(as_tuple=False)
        self.last_chunks_encoded = len(new)
        for start in range(0, len(new), self.chunk_batch_size):
            indices = new[start:start + self.chunk_batch_size]
            worlds, slots = indices.unbind(1)
            raw = self._normalize(bank.chunks[worlds, slots])
            tokens = encoder.interaction_projection(raw)
            tokens = torch.cat((encoder.chunk_cls.expand(len(indices), -1, -1), tokens), 1)
            hidden = encoder.chunk_encoder(tokens + encoder.chunk_position)
            self.chunk_features[worlds, slots] = encoder.chunk_output(hidden[:, 0]).float()
            self.encoded_stamp[worlds, slots] = bank.chunk_stamp[worlds, slots]
        dirty = ((bank.total_chunks != self.encoded_total) |
                 (bank.session != self.encoded_session)).nonzero(as_tuple=False).flatten()
        self.last_memories_encoded = len(dirty)
        offsets = torch.arange(bank.long_chunks, device=bank.device)[None]
        for start in range(0, len(dirty), self.batch_size):
            ids = dirty[start:start + self.batch_size]
            sequence = bank.total_chunks[ids, None] - bank.long_chunks + offsets
            mask = sequence >= bank.session_start[ids, None]
            slots = sequence.remainder(bank.archive_chunks)
            chunks = self.chunk_features[ids[:, None], slots].masked_fill(~mask[..., None], 0)
            tokens = torch.cat((encoder.memory_cls.expand(len(ids), -1, -1), chunks), 1)
            padding = torch.cat((torch.zeros(len(ids), 1, dtype=torch.bool, device=bank.device), ~mask), 1)
            hidden = encoder.memory_encoder(tokens + encoder.memory_position, src_key_padding_mask=padding)
            summary = encoder.memory_output(hidden[:, 0]).masked_fill(~mask.any(1, keepdim=True), 0)
            self.memory_features[ids] = summary.float()
            self.encoded_total[ids] = bank.total_chunks[ids]
            self.encoded_session[ids] = bank.session[ids]

    @torch.inference_mode()
    def encode(self):
        bank, encoder = self.memory, self.checkpoint.encoder
        results = []
        with self._autocast():
            self._refresh_cache()
            for start in range(0, bank.num_worlds, self.batch_size):
                ids = bank._worlds[start:start + self.batch_size]
                raw, valid = bank.ordered_short(ids)
                raw = self._normalize(raw).masked_fill(~valid[..., None], 0)
                short = encoder.interaction_projection(raw).masked_fill(~valid[..., None], 0)
                tokens = torch.cat((encoder.cls_token.expand(len(ids), -1, -1),
                                    self.memory_features[ids, None] + encoder.memory_type, short), 1)
                long_available = bank.total_chunks[ids] > bank.session_start[ids]
                padding = torch.cat((torch.zeros(len(ids), 1, dtype=torch.bool, device=bank.device),
                                     ~long_available[:, None], ~valid), 1)
                hidden = encoder.transformer(tokens + encoder.position, src_key_padding_mask=padding)
                results.append(encoder.output(hidden[:, 0]).float())
        return torch.cat(results)
