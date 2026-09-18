"""Offline statistics for recorded hard routes; reset edges are explicit."""

from __future__ import annotations

import numpy as np


def describe(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None, "p95": None}
    return {"count": len(x), "mean": float(x.mean()),
            **dict(zip(("p10", "median", "p90", "p95"), map(float, np.quantile(x, [.1, .5, .9, .95]))))}


def transition_masks(trace, *, start=500):
    r, ep, motion, frames = (trace[k] for k in ("routes", "episode_ids", "motion_ids", "motion_steps"))
    changed = r[1:] != r[:-1]
    boundary = ((ep[1:] != ep[:-1]) | (motion[1:] != motion[:-1]) | (frames[1:] != frames[:-1] + 1))
    steady = np.broadcast_to((np.arange(1, len(r)) >= start + 1)[:, None], changed.shape)
    full = (trace["short_count"] == 50) & (trace["long_count"] == 30)
    return changed, {
        "all_policy_steps": np.ones_like(changed),
        "initial_policy_steps_before_steady": np.broadcast_to((np.arange(1, len(r)) < start)[:, None], changed.shape),
        "steady_all": steady,
        "steady_within_motion": steady & ~boundary,
        "steady_full_memory_within_motion": steady & ~boundary & full[1:] & full[:-1],
        "steady_boundary": steady & boundary,
        "steady_short_refilling_within_motion": steady & ~boundary & (trace["short_count"][1:] < 50),
    }, full


def temporal_statistics(trace, dt, *, start=500, k=16):
    r = trace["routes"]
    changed, masks, full = transition_masks(trace, start=start)
    rates = {}
    for key, mask in masks.items():
        opportunities, switches = int(mask.sum()), int((changed & mask).sum())
        per_world_den = mask.sum(0)
        per_world_rate = (changed & mask).sum(0) / np.maximum(per_world_den, 1)
        rates[key] = {"opportunities": opportunities, "switches": switches,
                      "switch_fraction": switches / opportunities if opportunities else None,
                      "switches_per_second": switches / opportunities / dt if opportunities else None,
                      "per_world_switch_fraction": describe(per_world_rate[per_world_den > 0])}
    valid = full & (np.arange(len(r))[:, None] >= start)
    counts = np.stack([((r == i) & valid).sum(0) for i in range(k)], -1)
    total = counts.sum(-1)
    p = counts / np.maximum(total[:, None], 1)
    dominant = counts.argmax(-1)
    purity = p.max(-1)
    effective = np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum(-1))
    worlds = total > 0
    coverage = {"worlds": r.shape[1], "worlds_with_full_memory_samples": int(worlds.sum()),
                "steady_full_memory_observation_fraction": float(full[start:].mean()) if start < len(r) else None,
                "dominant_expert_fraction": describe(purity[worlds]),
                "effective_experts_per_world": describe(effective[worlds]),
                "distinct_experts_per_world": describe((counts > 0).sum(-1)[worlds]),
                "dominant_at_least_90_percent": float((purity[worlds] >= .9).mean()),
                "dominant_at_least_80_percent": float((purity[worlds] >= .8).mean())}
    # Episode/motion dominance compares whole trials, not only adjacent frames.
    trial_modes, trial_worlds, trial_motions, trial_purity, trial_counts = [], [], [], [], []
    for world in range(r.shape[1]):
        ep = trace["episode_ids"][:, world]
        motion = trace["motion_ids"][:, world]
        edges = np.r_[0, np.flatnonzero((ep[1:] != ep[:-1]) | (motion[1:] != motion[:-1])) + 1, len(r)]
        for left, right in zip(edges[:-1], edges[1:]):
            selected = r[left:right, world][valid[left:right, world]]
            if len(selected) < 50:
                continue
            histogram = np.bincount(selected, minlength=k)
            trial_modes.append(int(histogram.argmax()))
            trial_worlds.append(world)
            trial_motions.append(int(motion[left]))
            trial_purity.append(float(histogram.max() / len(selected)))
            trial_counts.append(len(selected))
    trial_modes, trial_worlds, trial_motions = map(np.asarray, (trial_modes, trial_worlds, trial_motions))
    agreements, different_motion_pairs, distinct_motions = [], 0, []
    for world in range(r.shape[1]):
        selected = trial_worlds == world
        modes, motions = trial_modes[selected], trial_motions[selected]
        distinct_motions.append(len(np.unique(motions)))
        a, b = np.triu_indices(len(modes), 1)
        different = motions[a] != motions[b]
        if different.any():
            agreements.append(float((modes[a[different]] == modes[b[different]]).mean()))
            different_motion_pairs += int(different.sum())
    cross_motion = {"minimum_full_memory_samples_per_trial": 50, "usable_trials": len(trial_modes),
                    "trial_dominant_fraction": describe(trial_purity),
                    "distinct_motions_per_world": describe(distinct_motions),
                    "worlds_with_multiple_motions": len(agreements), "different_motion_trial_pairs": different_motion_pairs,
                    "same_expert_across_motion_fraction_world_weighted": describe(agreements)}
    main = masks["steady_full_memory_within_motion"]
    matrix = np.bincount((r[:-1][main].astype(int) * k + r[1:][main]), minlength=k*k).reshape(k, k)
    switch_details = {}
    if "center_squared_margin" in trace:
        margin = trace["center_squared_margin"]
        switch_details["new_point_margin_on_switch"] = describe(margin[1:][main & changed])
        switch_details["new_point_margin_without_switch"] = describe(margin[1:][main & ~changed])
    if len(r) >= 3:
        eligible = main[:-1] & main[1:] & changed[:-1]
        returned = (r[:-2] == r[2:]) & (r[:-2] != r[1:-1])
        switch_details["immediate_A_B_A_return_fraction"] = float(returned[eligible].mean()) if eligible.any() else None
    lagged = {}
    for lag in (1, 5, 10, 50, 100, 350):
        if lag >= len(r):
            continue
        eligible = valid[:-lag] & valid[lag:]
        same_trial = (trace["episode_ids"][:-lag] == trace["episode_ids"][lag:])
        same_trial &= (trace["motion_ids"][:-lag] == trace["motion_ids"][lag:])
        same_trial &= (trace["motion_steps"][lag:] == trace["motion_steps"][:-lag] + lag)
        for label, mask in (("same_world", eligible), ("same_motion_contiguous", eligible & same_trial)):
            lagged[f"{label}_{lag}_steps"] = {"pairs": int(mask.sum()),
                "different_expert_fraction": float((r[:-lag][mask] != r[lag:][mask]).mean()) if mask.any() else None}
    return {"start_policy_step": start, "rates": rates, "per_world": coverage,
            "cross_motion": cross_motion, "lagged": lagged,
            "switch_details": switch_details,
            "transition_matrix_full_memory": matrix.tolist()}, counts, dominant


def collect_latent_samples(trace, *, start=500, per_world=32, seed=981):
    """Equal maximum samples per world; no adjacent-frame pair inflation."""
    steps = trace["sampled_steps"]
    valid = ((steps[:, None] >= start) & (trace["short_count"][steps] == 50)
             & (trace["long_count"][steps] == 30))
    rng = np.random.default_rng(seed)
    rows, worlds = [], []
    for world in range(valid.shape[1]):
        candidates = np.flatnonzero(valid[:, world])
        if len(candidates) > per_world:
            candidates = rng.choice(candidates, per_world, replace=False)
        rows.extend(candidates)
        worlds.extend([world] * len(candidates))
    rows, worlds = np.asarray(rows, dtype=int), np.asarray(worlds, dtype=int)
    z = trace["unit_latent_samples"][rows, worlds]
    labels = trace["routes"][steps[rows], worlds]
    return {"latent": z, "routes": labels, "worlds": worlds,
            "motion_ids": trace["motion_ids"][steps[rows], worlds],
            "raw_norm": trace["raw_latent_norm"][steps[rows], worlds],
            "margin": trace["center_squared_margin"][steps[rows], worlds]}


def physics_shuffle_reference(dr, memberships, schema, *, permutations=100, seed=764):
    """Shuffle complete DR-world identities, preserving every route trajectory."""
    x = (dr - schema["lower"]) / (np.asarray(schema["upper"]) - schema["lower"])
    w = np.asarray(memberships, dtype=np.float64)
    population = w.sum(1)
    cluster = w.sum(0)
    rng, scores = np.random.default_rng(seed), []
    for _ in range(permutations):
        shuffled = x[rng.permutation(len(x))]
        first = population @ shuffled
        second = population @ np.square(shuffled)
        total_ss = second - np.square(first) / population.sum()
        cluster_sums = w.T @ shuffled
        within_ss = second - (np.square(cluster_sums) / cluster.clip(1)[:, None]).sum(0)
        explained = 1 - within_ss / total_ss.clip(1e-20)
        scores.append([float(explained[cols].mean()) for cols in schema["groups"].values()])
    scores = np.asarray(scores)
    return {name: describe(scores[:, i]) for i, name in enumerate(schema["groups"])}


def weighted_physics_statistics(dr, memberships, schema):
    """Time-weighted P(DR | expert) and explained variance, with actual units."""
    dr = np.asarray(dr, dtype=np.float64)
    weights = np.asarray(memberships, dtype=np.float64)
    world_weights = weights.sum(1)
    normal = (dr - np.asarray(schema["lower"])) / (np.asarray(schema["upper"]) - schema["lower"])
    global_mean = np.average(normal, axis=0, weights=world_weights)
    total_variance = np.average((normal - global_mean)**2, axis=0, weights=world_weights)
    rows, within_variance = [], np.zeros(normal.shape[1])
    for i in range(weights.shape[1]):
        w, size = weights[:, i], weights[:, i].sum()
        if size == 0:
            rows.append({"expert": i, "samples": 0})
            continue
        center = np.average(normal, axis=0, weights=w)
        variance = np.average((normal - center)**2, axis=0, weights=w)
        within_variance += size * variance
        raw_center = np.average(dr, axis=0, weights=w)
        raw_std = np.sqrt(np.average((dr - raw_center)**2, axis=0, weights=w))
        quantiles = []
        for col in range(dr.shape[1]):
            order = np.flatnonzero(w > 0)
            order = order[np.argsort(dr[order, col])]
            cum = np.cumsum(w[order])
            indices = np.searchsorted(cum, np.asarray([.1, .5, .9]) * size)
            quantiles.append(dr[order[indices], col].tolist())
        rows.append({"expert": i, "samples": float(size), "worlds_with_membership": int((w > 0).sum()),
                     "raw_mean": raw_center.tolist(), "raw_std": raw_std.tolist(), "raw_q10_q50_q90": quantiles,
                     "normalized_within_std": np.sqrt(variance).tolist()})
    within_variance /= max(weights.sum(), 1)
    explained = 1 - within_variance / np.maximum(total_variance, 1e-20)
    groups = {name: {"variance_explained": float(explained[cols].mean()),
                     "within_std_over_global_std": float(np.sqrt(within_variance[cols].sum() / total_variance[cols].sum()))}
              for name, cols in schema["groups"].items()}
    return {"groups": groups, "per_coordinate_variance_explained": explained.tolist(), "experts": rows}


def geometry_statistics(samples, dr, schema, centers, *, pair_count=400000, seed=771):
    z, labels, worlds = (samples[k] for k in ("latent", "routes", "worlds"))
    rng = np.random.default_rng(seed)
    a, b = rng.integers(len(z), size=(2, pair_count))
    keep = worlds[a] != worlds[b]
    a, b = a[keep], b[keep]
    within = labels[a] == labels[b]
    dz, raw_dz = np.empty(len(a)), np.empty(len(a))
    groups = schema["groups"]
    normalized = (dr - schema["lower"]) / (np.asarray(schema["upper"]) - schema["lower"])
    distances = {name: np.empty(len(a)) for name in groups}
    absolute = np.empty((len(a), 9), dtype=np.float32)
    for offset in range(0, len(a), 16384):
        sl = slice(offset, offset + 16384)
        ia, ib = a[sl], b[sl]
        za, zb = z[ia], z[ib]
        dz[sl] = np.linalg.norm(za-zb, axis=-1)
        raw_dz[sl] = np.linalg.norm(za*samples["raw_norm"][ia, None]-zb*samples["raw_norm"][ib, None], axis=-1)
        da, db = normalized[worlds[ia]], normalized[worlds[ib]]
        absolute[sl] = np.abs(dr[worlds[ia], :9] - dr[worlds[ib], :9])
        for name, cols in groups.items():
            distances[name][sl] = np.sqrt(np.square(da[:, cols] - db[:, cols]).mean(-1))
    distances["equal_weight_six_groups"] = np.sqrt(np.square(np.stack(list(distances.values()), -1)).mean(-1))
    conditions = {"within_expert_different_worlds": within,
                  "between_experts_different_worlds": ~within, "random_different_worlds": np.ones(len(a), dtype=bool)}
    result = {"sample_count": len(z), "different_world_pairs": len(a),
              "distance_definition": "Euclidean L2 between unit-normalized 64-D latent vectors; not per-coordinate RMS",
              "DR_distance_definition": "Per-family RMS of coordinate differences divided by each full training range; six-family score weights families equally",
              "latent": {}, "raw_latent": {}, "dr": {name: {} for name in distances}, "absolute_DR_differences": {}}
    for name, mask in conditions.items():
        result["latent"][name] = describe(dz[mask])
        result["raw_latent"][name] = describe(raw_dz[mask])
        result["absolute_DR_differences"][name] = {schema["names"][i]: describe(absolute[mask, i]) for i in range(9)}
        for group, values in distances.items():
            result["dr"][group][name] = describe(values[mask])
    result["within_over_between_latent"] = float(dz[within].mean() / dz[~within].mean())
    result["per_expert"] = []
    for i in range(len(centers)):
        selected = labels == i
        mask = within & (labels[a] == i)
        result["per_expert"].append({"expert": i, "latent_samples": int(selected.sum()),
            "distance_to_saved_center": describe(np.linalg.norm(z[selected]-centers[i], axis=-1)),
            "within_pair_distance": describe(dz[mask]),
            "DR_pairs": {name: describe(values[mask]) for name, values in distances.items()}})
    cdist = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    result["center_pair_distance"] = describe(cdist[np.triu_indices(len(centers), 1)])
    result["center_nearest_distance"] = describe(np.where(np.eye(len(centers), dtype=bool), np.inf, cdist).min(-1))
    result["assignment_squared_margin"] = describe(samples["margin"])
    # A separate repeated-DR control: compare the same world across motions.
    # The cross-world cluster statistics above deliberately exclude these pairs.
    buckets = [np.flatnonzero(worlds == w) for w in np.unique(worlds)]
    buckets = [v for v in buckets if len(v) >= 2]
    ia, ib = [], []
    for bucket in buckets:
        pairs = rng.choice(bucket, size=(2, 100), replace=True)
        eligible = ((pairs[0] != pairs[1]) &
                    (samples["motion_ids"][pairs[0]] != samples["motion_ids"][pairs[1]]))
        ia.extend(pairs[0, eligible]); ib.extend(pairs[1, eligible])
    ia, ib = np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)
    result["same_world_different_motion"] = {
        "latent_distance": describe(np.linalg.norm(z[ia]-z[ib], axis=-1)),
        "different_expert_fraction": float((labels[ia] != labels[ib]).mean()) if len(ia) else None}
    return result
