"""Independent hierarchical context version; the original predictor is unchanged."""

from dataclasses import dataclass

import torch
from torch import nn

from intact_tracking.forward_predictor import ForwardPredictorConfig, ForwardDynamicsTransformer


@dataclass(frozen=True)
class Memory350Config(ForwardPredictorConfig):
    architecture_version: str = "nominal_counterfactual_short50_chunk10_long30_context_v1"
    context_history_steps: int = 50
    memory_chunk_steps: int = 10
    memory_chunks: int = 30
    chunk_depth: int = 1
    memory_depth: int = 2

    def __post_init__(self):
        super().__post_init__()
        if (self.context_history_steps, self.memory_chunk_steps, self.memory_chunks) != (50, 10, 30):
            raise ValueError("This version fixes disjoint short50 plus 30 chunks of 10 interactions")


def _attention(width, heads, depth, dropout):
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class HierarchicalContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.context_dim
        state_dim = getattr(config, "context_state_dim", config.state_dim)
        interaction_dim = 2 * state_dim + config.action_dim
        self.interaction_projection = nn.Sequential(nn.Linear(interaction_dim, width), nn.GELU(), nn.LayerNorm(width))
        self.chunk_cls = nn.Parameter(torch.empty(1, 1, width))
        self.chunk_position = nn.Parameter(torch.empty(1, config.memory_chunk_steps + 1, width))
        self.chunk_encoder = _attention(width, config.context_heads, config.chunk_depth, config.dropout)
        self.chunk_output = nn.LayerNorm(width)
        self.memory_cls = nn.Parameter(torch.empty(1, 1, width))
        self.memory_position = nn.Parameter(torch.empty(1, config.memory_chunks + 1, width))
        self.memory_encoder = _attention(width, config.context_heads, config.memory_depth, config.dropout)
        self.memory_output = nn.LayerNorm(width)
        self.memory_type = nn.Parameter(torch.empty(1, 1, width))
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, config.context_history_steps + 2, width))
        self.transformer = _attention(width, config.context_heads, config.context_depth, config.dropout)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.dynamics_latent_dim),
                                    nn.LayerNorm(config.dynamics_latent_dim))
        for name in ("chunk_cls", "chunk_position", "memory_cls", "memory_position", "memory_type", "cls_token", "position"):
            nn.init.trunc_normal_(getattr(self, name), std=.02)

    def encode_memory(self, interactions, valid):
        batch, chunks, steps, _ = interactions.shape
        valid = valid.bool()
        # Even empty slots take the same graph, then are masked. This keeps DDP
        # gradients defined when a rank starts with no long memory at all.
        clean = interactions.masked_fill(~valid[..., None, None], 0)
        tokens = self.interaction_projection(clean).reshape(batch * chunks, steps, -1)
        chunk_tokens = torch.cat((self.chunk_cls.expand(batch * chunks, -1, -1), tokens), dim=1)
        chunk_hidden = self.chunk_encoder(chunk_tokens + self.chunk_position)
        chunk_summary = self.chunk_output(chunk_hidden[:, 0]).reshape(batch, chunks, -1)
        chunk_summary = chunk_summary.masked_fill(~valid[..., None], 0)
        memory_tokens = torch.cat((self.memory_cls.expand(batch, -1, -1), chunk_summary), dim=1)
        mask = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=valid.device), ~valid), dim=1)
        hidden = self.memory_encoder(memory_tokens + self.memory_position, src_key_padding_mask=mask)
        memory = self.memory_output(hidden[:, 0])
        return memory.masked_fill(~valid.any(1, keepdim=True), 0)

    def forward(self, history_state, history_action, history_next_state, history_valid,
                memory_interactions, memory_valid):
        valid = history_valid.bool()
        short = torch.cat((history_state, history_action, history_next_state), dim=-1)
        short = self.interaction_projection(short.masked_fill(~valid[..., None], 0))
        short = short.masked_fill(~valid[..., None], 0)
        memory = self.encode_memory(memory_interactions, memory_valid)
        batch = short.shape[0]
        # Exactly one summarized memory token reaches the final context encoder.
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1),
                            memory[:, None] + self.memory_type, short), dim=1)
        padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=valid.device),
                             ~memory_valid.any(1, keepdim=True), ~valid), dim=1)
        encoded = self.transformer(tokens + self.position, src_key_padding_mask=padding)
        return self.output(encoded[:, 0])


class Memory350Predictor(ForwardDynamicsTransformer):
    def __init__(self, config=None):
        super().__init__(config or Memory350Config())
        self.context_encoder = HierarchicalContextEncoder(self.config)

    def encode_context(self, history_state, history_action, normalized_state, history_valid,
                       *, history_next_state=None, memory_interactions=None, memory_valid=None):
        if history_next_state is None or memory_interactions is None or memory_valid is None:
            raise ValueError("Hierarchical encoding needs explicit completed outcomes and query-time long memory")
        return self.context_encoder(history_state, history_action, history_next_state, history_valid,
                                    memory_interactions, memory_valid)
