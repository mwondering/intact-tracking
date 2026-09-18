"""Offline router comparisons with world-separated fitting and validation.

These functions never update a policy. Physical labels are used only for
decoders and the explicitly supervised metric; inference consumes latent only.
"""

from __future__ import annotations

import numpy as np
import torch

STRONG_COORDINATES = [0, 1, 2, 3, 5, 6, 7, 8]


def affine_ridge(x, y, alpha):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-8)
    ym = y.mean(0)
    normalized = (x-xm)/xs
    coefficient = np.linalg.solve(normalized.T@normalized+alpha*np.eye(x.shape[1]), normalized.T@(y-ym))/xs[:, None]
    return {'matrix': coefficient, 'offset': ym-xm@coefficient, 'alpha': alpha}


def affine_predict(x, model):
    return np.asarray(x)@model['matrix']+model['offset']


def regression_metrics(y, prediction, groups):
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    errors = prediction-y
    mse = np.square(errors).mean(0)
    r2 = 1-mse/np.maximum(y.var(0), 1e-20)
    return {'r2': r2.tolist(), 'normalized_mae': np.abs(errors).mean(0).tolist(),
            'normalized_rmse': np.sqrt(mse).tolist(),
            'strong8_r2': float(r2[STRONG_COORDINATES].mean()),
            'load_r2': float(r2[:4].mean()),
            'groups_r2': {name: float(r2[columns].mean()) for name, columns in groups.items()}}


def select_readout(x, y, val_x, val_y, groups):
    best, grid = None, []
    for alpha in (.01, .1, 1., 10., 100., 1000.):
        model = affine_ridge(x, y, alpha)
        metrics = regression_metrics(val_y, affine_predict(val_x, model), groups)
        score = metrics['strong8_r2']
        grid.append({'alpha': alpha, 'validation_strong8_r2': score})
        if best is None or score > best[0]:
            best = score, model, metrics
    return best[1], best[2], grid


def fit_projection(x, worlds, kind, *, floor=1e-4, dimension=64):
    """Fit ordinary/reliability whitening without numerical DR labels."""
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    if kind == 'identity':
        return {'matrix': np.eye(x.shape[1]), 'offset': np.zeros(x.shape[1]), 'kind': kind}
    covariance = np.cov(x.T, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    transform = vectors/np.sqrt(np.maximum(values, values.max()*floor))[None, :]
    result = {'kind': kind, 'eigenvalue_floor_fraction': floor}
    if kind == 'reliability':
        unique, inverse = np.unique(worlds, return_inverse=True)
        centers = np.zeros((len(unique), x.shape[1]), dtype=np.float64)
        np.add.at(centers, inverse, x)
        centers /= np.bincount(inverse)[:, None]
        between = np.cov(centers.T, bias=True)
        reliability, axes = np.linalg.eigh(transform.T@between@transform)
        order = np.argsort(reliability)[::-1][:dimension]
        # Between-world means contain finite-sample motion noise. The small
        # 1/m correction is a heuristic, not an independence-based confidence bound.
        m = float(np.bincount(inverse).mean())
        signal = np.clip((reliability[order]-1/m)/(1-1/m), 0, 1)
        transform = (transform@axes[:, order])*np.sqrt(signal)[None, :]
        result.update(reliability_eigenvalues=reliability[order].tolist(), output_dimension=dimension)
    elif kind != 'whiten':
        raise ValueError(kind)
    result.update(matrix=transform, offset=-mean@transform)
    return result


def project(x, projection):
    features = affine_predict(x, projection)
    if projection.get('clip_unit_range', False):
        features = np.clip(features, 0, 1)
    if projection.get('normalize_after_projection', False):
        features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    return features.astype(np.float32)


@torch.no_grad()
def fit_euclidean_centers(x, k=16, *, seed=731, restarts=3, iterations=60):
    """KMeans++/Lloyd; preserves the scale of the supplied metric coordinates."""
    value = torch.as_tensor(x, dtype=torch.float32)
    if value.ndim != 2 or len(value) < k or not torch.isfinite(value).all():
        raise ValueError('Expected enough finite feature vectors')
    generator = torch.Generator().manual_seed(seed)
    squared_norm = value.square().sum(-1, keepdim=True)

    def distances(centers):
        return (squared_norm+centers.square().sum(-1)[None]-2*value@centers.T).clamp_min(0)

    best = None
    for _ in range(restarts):
        centers = [value[torch.randint(len(value), (), generator=generator)]]
        minimum = distances(torch.stack(centers)).squeeze(-1)
        for _ in range(1, k):
            if float(minimum.sum()) < 1e-10:
                raise ValueError('Fewer distinct feature vectors than centers')
            index = torch.multinomial(minimum, 1, generator=generator)[0]
            centers.append(value[index])
            minimum = torch.minimum(minimum, distances(value[index:index+1]).squeeze(-1))
        centers = torch.stack(centers)
        for _ in range(iterations):
            distance = distances(centers)
            ids = distance.argmin(-1)
            counts = torch.bincount(ids, minlength=k)
            candidate = torch.zeros_like(centers).index_add_(0, ids, value)/counts.clamp_min(1)[:, None]
            empty = (counts == 0).nonzero().flatten()
            if len(empty):
                candidate[empty] = value[distance.min(-1).values.topk(len(empty)).indices]
            movement = float((candidate-centers).square().sum(-1).max())
            centers = candidate
            if movement < 1e-10:
                break
        inertia = float(distances(centers).min(-1).values.mean())
        if best is None or inertia < best[0]:
            best = inertia, centers.clone()
    return best[1].numpy(), best[0]


def squared_distances(x, centers):
    x = np.asarray(x, np.float32)
    return np.maximum(np.square(x).sum(-1, keepdims=True)+np.square(centers).sum(-1)[None]-2*x@centers.T, 0)


def gate_weights(distances, *, top_k, temperature):
    distances = np.asarray(distances, np.float32)
    n = distances.shape[-1]
    if not 1 <= top_k <= n or temperature <= 0:
        raise ValueError('Invalid gate settings')
    if top_k == 1:
        return np.eye(n, dtype=np.float32)[distances.argmin(-1)]
    score = -(distances-distances.min(-1, keepdims=True))/temperature
    if top_k < n:
        selected = np.argpartition(score, n-top_k, axis=-1)[..., -top_k:]
        keep = np.zeros_like(score, dtype=bool)
        np.put_along_axis(keep, selected, True, axis=-1)
        score = np.where(keep, score, -np.inf)
    weight = np.exp(score)
    return weight/weight.sum(-1, keepdims=True)


def gate_summary(weights):
    marginal = weights.mean(0)
    entropy = -(weights*np.log(np.maximum(weights, 1e-30))).sum(-1)
    return {'mean_weight_by_expert': marginal.tolist(),
            'minimum_expert_weight': float(marginal.min()),
            'maximum_expert_weight': float(marginal.max()),
            'effective_experts_per_point': float(np.exp(entropy).mean()),
            'active_experts_per_point': float((weights > 0).sum(-1).mean()),
            'mean_maximum_weight': float(weights.max(-1).mean()),
            'mean_weight_coordinate_std': float(weights.std(0).mean())}


def readout_amplification(model):
    # Since weights sum to one, these are predictions at each simplex vertex.
    vertices = model['matrix']+model['offset'][None]
    spans = np.ptp(vertices, axis=0)
    return {'strong8_mean_prototype_span_in_DR_ranges': float(spans[STRONG_COORDINATES].mean()),
            'per_coordinate_prototype_span_in_DR_ranges': spans.tolist(),
            'strong8_max_abs_prototype_value': float(np.abs(vertices[:, STRONG_COORDINATES]).max())}


def causal_ema(weights, valid, *, dt, time_constant):
    """Hold last weights while context is incomplete; no future-world averaging."""
    if time_constant < 0:
        raise ValueError('time_constant must be nonnegative')
    result = np.empty_like(weights)
    memory = weights[0].copy()
    initialized = np.zeros(weights.shape[1], dtype=bool)
    alpha = 1. if time_constant == 0 else -np.expm1(-dt/time_constant)
    for step in range(len(weights)):
        first = valid[step] & ~initialized
        continuing = valid[step] & initialized
        memory[first] = weights[step, first]
        memory[continuing] += alpha*(weights[step, continuing]-memory[continuing])
        initialized |= valid[step]
        result[step] = memory
    return result


def sparse_renormalize(weights, k):
    if k == weights.shape[-1]:
        return weights
    selected = np.argpartition(weights, weights.shape[-1]-k, axis=-1)[..., -k:]
    result = np.zeros_like(weights)
    np.put_along_axis(result, selected, np.take_along_axis(weights, selected, axis=-1), axis=-1)
    return result/result.sum(-1, keepdims=True)
