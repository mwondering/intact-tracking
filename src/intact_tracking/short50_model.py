"""From-scratch short-only control with Memory350-matched common initialization."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from intact_tracking.forward_predictor import ForwardPredictorConfig
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor


@dataclass(frozen=True)
class Short50Config(ForwardPredictorConfig):
    architecture_version: str = "nominal_counterfactual_short50_only_context_v1"
    context_history_steps: int = 50

    def __post_init__(self):
        super().__post_init__()
        if self.context_history_steps != 50:
            raise ValueError("The matched short-only control uses exactly 50 steps")


class Short50ContextEncoder(nn.Module):
    def __init__(self, reference, config):
        super().__init__()
        self.config = config
        # Keep shared fresh random weights and positions identical. No pretrained
        # parameters are loaded here, and no extra random draws are consumed.
        self.interaction_projection = reference.interaction_projection
        self.cls_token = reference.cls_token
        indices = [0, *range(2, config.context_history_steps + 2)]
        self.position = nn.Parameter(reference.position[:, indices].detach().clone())
        self.transformer = reference.transformer
        self.output = reference.output

    def forward(self, history_state, history_action, history_next_state, history_valid):
        valid = history_valid.bool()
        raw = torch.cat((history_state, history_action, history_next_state), dim=-1)
        short = self.interaction_projection(raw.masked_fill(~valid[..., None], 0))
        short = short.masked_fill(~valid[..., None], 0)
        tokens = torch.cat((self.cls_token.expand(len(short), -1, -1), short), dim=1)
        padding = torch.cat((torch.zeros(len(short), 1, dtype=torch.bool, device=short.device), ~valid), dim=1)
        encoded = self.transformer(tokens + self.position, src_key_padding_mask=padding)
        return self.output(encoded[:, 0])


class Short50Predictor(Memory350Predictor):
    def __init__(self, config=None):
        config = config or Short50Config()
        reference_config = Memory350Config(**{k: v for k, v in asdict(config).items()
                                             if k != "architecture_version"})
        # Same construction/RNG consumption as a fresh Memory350 model, then
        # remove the entire long-memory path before creating the optimizer/DDP.
        super().__init__(reference_config)
        self.config = config
        self.context_encoder = Short50ContextEncoder(self.context_encoder, config)

    def encode_context(self, history_state, history_action, normalized_state, history_valid,
                       *, history_next_state=None, memory_interactions=None, memory_valid=None):
        if history_next_state is None:
            raise ValueError("Short50 requires explicit completed transition outcomes")
        # The common collector retains memory to match sample selection and
        # positive-pair eligibility. None of it is read by this model.
        return self.context_encoder(history_state, history_action, history_next_state, history_valid)
