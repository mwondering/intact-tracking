"""Frozen latent-only environment partition, with optional causal smoothing.

This does not change any policy. Artifacts bind the metric to an encoder SHA.
The causal path assumes fixed batch row identities and full-history validity;
reset a row when its physics session changes. Distances are not probabilities.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LatentEnvironmentPartition(nn.Module):
    def __init__(self, artifact: str | Path, *, encoder_sha256: str):
        super().__init__()
        with np.load(artifact, allow_pickle=False) as saved:
            values = {name: saved[name] for name in saved.files}
        if str(values['format_version']) != 'latent_environment_blend_v1':
            raise ValueError('Unsupported environment partition artifact')
        if str(values['encoder_sha256']) != encoder_sha256:
            raise ValueError('Environment partition was calibrated for a different encoder')
        self.encoder_sha256 = encoder_sha256
        self.tau_control_steps = float(values['tau_control_steps'])
        self.nominal_class = int(values['nominal_class'])
        if self.tau_control_steps <= 0:
            raise ValueError('Smoothing time constant must be positive')
        names = ('response_mean', 'response_transform', 'physical_mean', 'physical_scale',
                 'physical_coef', 'physical_intercept', 'lower', 'upper', 'centers')
        for name in names:
            if not np.isfinite(values[name]).all():
                raise ValueError(f'Nonfinite partition parameter: {name}')
            self.register_buffer(name, torch.from_numpy(values[name].copy()).float())
        weight = float(values['physical_weight'])
        if not 0 <= weight <= 1:
            raise ValueError('Invalid physical metric weight')
        self.response_gain = (1 - weight) ** .5 / float(values['response_scale'])
        self.physical_gain = weight ** .5 / float(values['physical_scale_total'])
        if self.response_transform.shape != (64, 64) or self.physical_coef.shape != (64, 38):
            raise ValueError('Unexpected latent metric dimensions')
        if self.centers.ndim != 2 or self.centers.shape[1] != 102 or len(self.centers) < 2:
            raise ValueError('Unexpected cluster center dimensions')
        self.register_buffer('_state', None, persistent=False)
        self.register_buffer('_ready', None, persistent=False)
        self.register_buffer('_pending_steps', None, persistent=False)
        self.eval()

    @torch.inference_mode()
    def features(self, latent: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != 64:
            raise ValueError('Expected latent [batch, 64]')
        if latent.device != self.centers.device:
            raise ValueError('Latent and partition must be on the same device')
        if not torch.isfinite(latent).all():
            raise ValueError('Nonfinite latent')
        if not normalized and (torch.linalg.vector_norm(latent, dim=-1) < 1e-8).any():
            raise ValueError('A zero latent is not a valid full-history observation')
        z = latent.float() if normalized else F.normalize(latent.float(), dim=-1)
        response = (z - self.response_mean) @ self.response_transform * self.response_gain
        physical = ((z - self.physical_mean) / self.physical_scale) @ self.physical_coef
        physical = torch.maximum(torch.minimum(physical + self.physical_intercept, self.upper), self.lower)
        return torch.cat((response, physical * self.physical_gain), dim=-1)

    @torch.inference_mode()
    def assign(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        squared = (features.square().sum(-1, keepdim=True)
                   + self.centers.square().sum(-1)[None] - 2 * features @ self.centers.T)
        distance, index = squared.clamp_min(0).sqrt().topk(2, dim=-1, largest=False)
        return {'class_id': index[:, 0], 'distance': distance[:, 0],
                'margin': distance[:, 1] - distance[:, 0]}

    @torch.inference_mode()
    def forward(self, latent: torch.Tensor, *, normalized: bool = False) -> dict[str, torch.Tensor]:
        return self.assign(self.features(latent, normalized=normalized))

    def clear_history(self) -> None:
        self._state = self._ready = self._pending_steps = None

    @torch.inference_mode()
    def route_causal(self, latent: torch.Tensor, *, elapsed_control_steps: float,
                     valid: torch.Tensor, reset: torch.Tensor | None = None,
                     normalized: bool = False) -> dict[str, torch.Tensor]:
        """Call on fixed world rows; valid means the encoder has full history.

        Invalid rows retain earlier state but do not ingest a new observation.
        Rows without any valid observation return class_id=-1 and ready=False.
        Only the 100-control-step query cadence has been evaluated so far.
        """
        n = len(latent)
        if valid.shape != (n,) or valid.dtype != torch.bool or valid.device != latent.device:
            raise ValueError('valid must be a Boolean mask on the latent device')
        if elapsed_control_steps <= 0 or not np.isfinite(elapsed_control_steps):
            raise ValueError('Elapsed control steps must be finite and positive')
        if self._state is None:
            self._state = self.centers.new_zeros((n, 102))
            self._ready = torch.zeros(n, device=latent.device, dtype=torch.bool)
            self._pending_steps = self.centers.new_zeros(n)
        elif len(self._state) != n:
            raise ValueError('Batch changed: clear_history before changing world rows')
        if reset is not None:
            if reset.shape != (n,) or reset.dtype != torch.bool or reset.device != latent.device:
                raise ValueError('reset must be a Boolean mask on the latent device')
            self._ready[reset] = False
            self._state[reset] = 0
            self._pending_steps[reset] = 0
        self._pending_steps += elapsed_control_steps
        if valid.any():
            idx = torch.nonzero(valid).squeeze(-1)
            value = self.features(latent[idx], normalized=normalized)
            fresh = ~self._ready[idx]
            alpha = torch.exp(-self._pending_steps[idx] / self.tau_control_steps)
            alpha[fresh] = 0
            self._state[idx] = alpha[:, None] * self._state[idx] + (1-alpha[:, None]) * value
            self._ready[idx] = True
            self._pending_steps[idx] = 0
        result = self.assign(self._state)
        result['class_id'] = result['class_id'].masked_fill(~self._ready, -1)
        result['ready'] = self._ready.clone()
        return result
