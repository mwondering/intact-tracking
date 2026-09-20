"""Separate observation normalization; privileged targets stay in predictor replay."""

from dataclasses import asdict, dataclass

import torch

from intact_tracking.data.predictor_online import (
    ForwardPredictorNormalization, ForwardPredictorNormalizationStats, _RunningMoments, _float_tuple,
)
from intact_tracking.memory350_nominal_dr_rank import NominalDRRankReplay
from intact_tracking.memory350_proprio_inputs import PROPRIO_DIM


@dataclass(frozen=True)
class ProprioNormalizationStats(ForwardPredictorNormalizationStats):
    context_state_mean: tuple[float, ...] = ()
    context_state_std: tuple[float, ...] = ()
    context_action_mean: tuple[float, ...] = ()
    context_action_std: tuple[float, ...] = ()


class ProprioNormalization:
    def __init__(self, dimensions, *, device):
        self.predictor = ForwardPredictorNormalization(dimensions, device=device)
        self.context_state = _RunningMoments(PROPRIO_DIM, self.predictor.device)
        self.context_action = _RunningMoments(29, self.predictor.device)

    @property
    def frozen(self):
        return self.predictor.frozen

    def freeze(self):
        self.predictor.freeze()

    def update(self, batch, valid):
        if self.frozen:
            return
        self.predictor.update(batch, valid)
        self.context_state.update(torch.cat((batch["encoder_state"][valid], batch["encoder_next_state"][valid])))
        self.context_action.update(batch["encoder_action"][valid])

    @property
    def packed_size(self):
        return self.predictor.packed_size + 1 + 2 * PROPRIO_DIM + 1 + 2 * 29

    def packed_statistics(self, device=None):
        return torch.cat((self.predictor.packed_statistics(), self.context_state.packed(),
                          self.context_action.packed())).to(device=device or self.predictor.device)

    def snapshot_from_packed(self, packed, world_ids):
        packed = packed.to(device=self.predictor.device, dtype=torch.float64).flatten()
        if packed.numel() != self.packed_size:
            raise ValueError("Missing proprio normalization moments in distributed reduction")
        offset = self.predictor.packed_size
        base = self.predictor.snapshot_from_packed(packed[:offset], world_ids)
        extra = {}
        for name, width in (("context_state", PROPRIO_DIM), ("context_action", 29)):
            count = float(packed[offset])
            if count < 1 or not count.is_integer():
                raise ValueError(f"Invalid {name} normalization count")
            mean = packed[offset + 1:offset + 1 + width] / count
            variance = (packed[offset + 1 + width:offset + 1 + 2 * width] / count - mean.square())
            extra[name + "_mean"] = _float_tuple(mean.float())
            extra[name + "_std"] = _float_tuple(variance.clamp_min(base.epsilon**2).sqrt().float())
            offset += 1 + 2 * width
        return ProprioNormalizationStats(**asdict(base), **extra)


class ProprioMemory350Replay(NominalDRRankReplay):
    def __init__(self, **kwargs):
        super().__init__(context_state_dim=PROPRIO_DIM, **kwargs)
        self.normalizer = ProprioNormalization(self.dimensions, device=self.device)
