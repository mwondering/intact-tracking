"""DR soft-neighborhood variant; inference remains a history-only encoder."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from intact_tracking.memory350_nominal_direction import NominalDirectionLossConfig, NominalDirectionObjective
from intact_tracking.memory350_nominal_dr_rank import rank_view_valid

SOFT_FIELDS = ('dr_soft_weight', 'dr_soft_h', 'dr_soft_temperature', 'dr_soft_version')
SOFT_METRICS = ('dr_soft_loss', 'dr_soft_weighted_loss', 'dr_soft_worlds',
                'dr_soft_target_self_mass', 'dr_soft_target_entropy',
                'dr_soft_effective_neighbors', 'dr_soft_predicted_self_mass')


@dataclass(frozen=True)
class NominalDRSoftLossConfig(NominalDirectionLossConfig):
    dr_soft_weight: float = .02
    dr_soft_h: float = .15
    dr_soft_temperature: float = .1
    dr_soft_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.dr_soft_weight) or self.dr_soft_weight < 0:
            raise ValueError('DR soft weight must be finite and nonnegative')
        for value in (self.dr_soft_h, self.dr_soft_temperature):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('DR soft bandwidth and temperature must be finite and positive')
        if self.dr_soft_version != 1:
            raise ValueError('Unsupported DR soft version')


def dr_soft_targets(parameters, h=.15):
    """Detached row-normalized Gaussian: unnormalized weight at distance h = .5.

    Diagonal is included: it represents the OTHER motion view of the same world.
    """
    if not math.isfinite(h) or h <= 0:
        raise ValueError('Bandwidth must be finite and positive')
    distances = torch.cdist(parameters.detach().float(), parameters.detach().float(),
                            compute_mode='donot_use_mm_for_euclid_dist')
    return (-math.log(2) * (distances / h).square()).softmax(-1)


def dr_center_soft_loss(latent, archived_latent, dr_metric, world_id, session,
                        valid, is_nominal, *, h=.15, temperature=.1):
    """Symmetric cross-view prototype KL, with equal weight per world/session.

    Pool current and archived unit views separately; normalize each prototype
    for cosine logits. Never compare a prototype to itself in the same view.
    All prototypes/gradients are local to a rank's microbatch. Labels are detached.
    Full, disjoint, different-motion eligibility is enforced by rank_view_valid.
    """
    n = len(latent)
    if (latent.ndim != 2 or archived_latent.shape != latent.shape
            or dr_metric.ndim != 2 or len(dr_metric) != n
            or any(x.shape != (n,) for x in (world_id, session, valid, is_nominal))):
        raise ValueError('Invalid DR soft tensor shapes')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(h) or h <= 0:
        raise ValueError('Invalid DR soft bandwidth or temperature')
    zero = (latent.float().sum() + archived_latent.float().sum()) * 0
    metrics = {k: zero.detach() for k in SOFT_METRICS if k != 'dr_soft_weighted_loss'}
    eligible = valid.bool() & ~is_nominal.bool()
    identities, inverse = torch.unique(torch.stack((world_id[eligible], session[eligible]), -1),
                                       dim=0, return_inverse=True)
    count = len(identities)
    metrics['dr_soft_worlds'] = zero.detach().new_tensor(count)
    if count < 2:
        return zero, metrics
    counts = torch.bincount(inverse, minlength=count).float()[:, None]

    def pool(x):
        return x.new_zeros(count, x.shape[-1]).index_add(0, inverse, x) / counts

    centers = [F.normalize(pool(F.normalize(z[eligible].float(), dim=-1)), dim=-1)
               for z in (latent, archived_latent)]
    target = dr_soft_targets(pool(dr_metric[eligible].detach().float()), h)
    logits = centers[0] @ centers[1].T / temperature
    log_probs = [F.log_softmax(logits, -1), F.log_softmax(logits.T, -1)]
    # KL and soft cross entropy have identical encoder gradients; KL removes
    # target entropy from the reported scalar (entropy varies with world count).
    loss = sum(F.kl_div(p, target, reduction='batchmean') for p in log_probs) * .5
    entropy = -(target * target.clamp_min(1e-30).log()).sum(-1)
    metrics.update(dr_soft_loss=loss.detach(),
                   dr_soft_target_self_mass=target.diag().mean(),
                   dr_soft_target_entropy=entropy.mean(),
                   dr_soft_effective_neighbors=entropy.exp().mean(),
                   dr_soft_predicted_self_mass=sum(p.detach().exp().diag().mean() for p in log_probs)*.5)
    return loss, metrics


class NominalDRSoftObjective(NominalDirectionObjective):
    def __init__(self, model, loss_config=None, *, anchor=None):
        super().__init__(model, loss_config or NominalDRSoftLossConfig(), anchor=anchor)

    def _extra_representation_loss(self, batch, views):
        if 'dr_metric' not in batch or len(views) != 3:
            raise ValueError('DR soft loss requires physical labels and archived cross-motion views')
        original, metrics = super()._extra_representation_loss(batch, views)
        loss, extra = dr_center_soft_loss(
            views[0], views[2], batch['dr_metric'], batch['world_id'], batch['physics_session'],
            rank_view_valid(batch), batch['is_nominal'], h=self.loss_config.dr_soft_h,
            temperature=self.loss_config.dr_soft_temperature)
        weighted = self.loss_config.dr_soft_weight * loss
        metrics.update(extra, dr_soft_weighted_loss=weighted.detach())
        return original + weighted, metrics
