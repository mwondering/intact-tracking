"""Double Memory350 attention depth without changing predictor initialization."""

from dataclasses import asdict, dataclass

import torch

from intact_tracking.memory350_model import (
    HierarchicalContextEncoder, Memory350Config, Memory350Predictor,
)


@dataclass(frozen=True)
class Memory350ScaleConfig(Memory350Config):
    chunk_depth: int = 2
    memory_depth: int = 4
    context_depth: int = 4

    def __post_init__(self):
        super().__post_init__()
        if (self.chunk_depth, self.memory_depth, self.context_depth) != (2, 4, 4):
            raise ValueError("The encoder-depth comparison fixes chunk/memory/context depths to 2/4/4")


class Memory350ScalePredictor(Memory350Predictor):
    def __init__(self, config=None):
        config = config or Memory350ScaleConfig()
        reference_config = Memory350Config(**{
            **asdict(config), "chunk_depth": 1, "memory_depth": 2, "context_depth": 2,
        })
        # Build a fresh original-size model, exactly as in the reference run.
        # No trained checkpoint is used. Predictor and retained context weights
        # therefore match at the same seed, as does the subsequent RNG stream.
        super().__init__(reference_config)
        original_context = self.context_encoder.state_dict()
        with torch.random.fork_rng(devices=[]):
            expanded = HierarchicalContextEncoder(config)
            missing, unexpected = expanded.load_state_dict(original_context, strict=False)
        expected_missing = {
            name for name in expanded.state_dict() if name not in original_context
        }
        if unexpected or set(missing) != expected_missing:
            raise RuntimeError("Unexpected mismatch while expanding fresh Memory350 context")
        self.config = config
        self.context_encoder = expanded


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    context = sum(parameter.numel() for parameter in model.context_encoder.parameters())
    return {"total": total, "context_encoder": context, "predictor": total - context}
