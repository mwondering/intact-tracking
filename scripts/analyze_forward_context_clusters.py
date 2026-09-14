"""Analyze and plot full-context latent samples without fitting the encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalized(z):
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def describe(z):
    z = z.astype(np.float64)
    u = normalized(z)
    mean = z.mean(0)
    centered = u - u.mean(0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(z)).clip(0)
    radius = np.linalg.norm(centered, axis=1)
    return {
        "n": len(z),
        "raw_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
        "raw_centroid_norm": float(np.linalg.norm(mean)),
        "raw_radius_rms": float(np.sqrt(np.mean(np.sum((z - mean) ** 2, axis=1)))),
        "unit_centroid_norm": float(np.linalg.norm(u.mean(0))),
        "unit_radius_rms": float(np.sqrt(np.mean(radius ** 2))),
        "unit_radius_p50": float(np.quantile(radius, .5)),
        "unit_radius_p95": float(np.quantile(radius, .95)),
        "unit_mean_pair_cosine": float((np.sum(u.sum(0) ** 2) - len(u)) / (len(u) * (len(u) - 1))),
        "centered_effective_rank": float(eigenvalues.sum() ** 2 / max(np.sum(eigenvalues ** 2), 1e-20)),
        "centered_top2_variance_fraction": float(eigenvalues[-2:].sum() / max(eigenvalues.sum(), 1e-20)),
    }


def grouping(values):
    return {key: np.flatnonzero(values == key) for key in np.unique(values)}


def select_pairs(data, name, rng):
    world, motion, phase, nominal = (data[k] for k in ("world", "motion", "phase", "nominal"))
    anchors = np.flatnonzero(nominal if name.startswith("nominal") else ~nominal)
    by_world, by_motion = grouping(world), grouping(motion)
    # At most one partner per query: many Monte Carlo pairs are not independent
    # observations. Confidence intervals below resample complete query worlds.
    left, right = [], []
    for i in rng.permutation(anchors):
        if name == "dr_same_world_cross_motion":
            candidates = by_world[world[i]]
            keep = motion[candidates] != motion[i]
        elif name == "nominal_cross_motion":
            candidates = anchors[rng.integers(0, len(anchors), size=64)]
            keep = (motion[candidates] != motion[i]) & (world[candidates] != world[i])
        elif name == "nominal_same_motion_far_phase":
            candidates = by_motion[motion[i]]
            keep = nominal[candidates] & (world[candidates] != world[i]) & (np.abs(phase[candidates] - phase[i]) >= .25)
        elif name == "nominal_dr_same_motion_near_phase":
            candidates = by_motion[motion[i]]
            keep = ~nominal[candidates] & (np.abs(phase[candidates] - phase[i]) <= .02)
        elif name == "dr_different_world_same_motion_near_phase":
            candidates = by_motion[motion[i]]
            keep = ~nominal[candidates] & (world[candidates] != world[i]) & (np.abs(phase[candidates] - phase[i]) <= .02)
        elif name == "dr_different_world_cross_motion":
            candidates = anchors[rng.integers(0, len(anchors), size=64)]
            keep = (motion[candidates] != motion[i]) & (world[candidates] != world[i])
        else:
            raise ValueError(name)
        choices = candidates[keep]
        if len(choices):
            left.append(i)
            right.append(rng.choice(choices))
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def pair_stats(z, left, right, worlds, rng):
    if not len(left):
        return {"n": 0}, np.empty(0)
    u = normalized(z)
    cosine = np.clip(np.sum(u[left] * u[right], axis=1), -1, 1)
    distance = np.linalg.norm(u[left] - u[right], axis=1)
    raw_distance = np.linalg.norm(z[left] - z[right], axis=1)
    blocks = grouping(worlds[left])
    sums = np.array([np.sum(distance[ids] ** 2) for ids in blocks.values()])
    counts = np.array([len(ids) for ids in blocks.values()])
    bootstrap = rng.integers(len(blocks), size=(400, len(blocks)))
    rms_bootstrap = np.sqrt(sums[bootstrap].sum(1) / counts[bootstrap].sum(1))
    return {
        "n": len(left), "query_worlds": len(blocks),
        "mean_cosine": float(cosine.mean()),
        "mean_angle_degrees": float(np.rad2deg(np.arccos(cosine)).mean()),
        "unit_distance_rms": float(np.sqrt(np.mean(distance ** 2))),
        "unit_distance_rms_query_world_bootstrap95": np.quantile(rms_bootstrap, [.025, .975]).tolist(),
        "unit_distance_p50": float(np.quantile(distance, .5)),
        "unit_distance_p95": float(np.quantile(distance, .95)),
        "raw_distance_rms": float(np.sqrt(np.mean(raw_distance ** 2))),
    }, distance


def crossvalidated_group_r2(z, label, world, rng):
    """Predict nominal latent from motion/phase, testing on different worlds."""
    unique_worlds = rng.permutation(np.unique(world))
    train = np.isin(world, unique_worlds[:len(unique_worlds) // 2])
    baseline = z[train].mean(0)
    prediction = np.broadcast_to(baseline, z[~train].shape).copy()
    for key in np.unique(label[train]):
        prediction[label[~train] == key] = z[train & (label == key)].mean(0)
    squared_error = np.sum((z[~train] - prediction) ** 2)
    baseline_error = np.sum((z[~train] - baseline) ** 2)
    return {"heldout_world_r2": float(1 - squared_error / max(baseline_error, 1e-20)),
            "train_samples": int(train.sum()), "test_samples": int((~train).sum()),
            "groups": len(np.unique(label)), "split": "half of complete physical-world trajectories held out"}


def binary_probe(u, nominal, world, family, rng):
    """Fixed ridge readout, with both worlds and motion families held out."""
    train_worlds = []
    for is_nominal in (True, False):
        ids = rng.permutation(np.unique(world[nominal == is_nominal]))
        train_worlds.extend(ids[:len(ids) // 2])
    families = rng.permutation(np.unique(family))
    train_family = np.isin(family, families[:len(families) // 2])
    train_world = np.isin(world, train_worlds)
    train, test = train_family & train_world, ~train_family & ~train_world
    mean, std = u[train].mean(0), np.maximum(u[train].std(0), 1e-5)
    x = np.column_stack(((u[train] - mean) / std, np.ones(train.sum())))
    xt = np.column_stack(((u[test] - mean) / std, np.ones(test.sum())))
    y = nominal[train].astype(float) * 2 - 1
    weight = np.where(y == 1, .5 / (y == 1).sum(), .5 / (y == -1).sum())
    penalty = .1 * np.eye(x.shape[1])
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(x.T @ (weight[:, None] * x) + penalty, x.T @ (weight * y))
    score = xt @ coefficients
    truth = nominal[test]
    from scipy.stats import rankdata
    rank = rankdata(score)
    positives, negatives = truth.sum(), (~truth).sum()
    auc = (rank[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return {
        "balanced_accuracy": float(.5 * ((score[truth] > 0).mean() + (score[~truth] <= 0).mean())),
        "roc_auc": float(auc), "train_samples": int(train.sum()), "test_samples": int(test.sum()),
        "test_nominal": int(positives), "test_dr": int(negatives),
        "train_motion_families": families[:len(families) // 2].tolist(),
        "test_motion_families": families[len(families) // 2:].tolist(),
        "note": "Separability diagnostic, not evidence of compactness; ridge=0.1 fixed in advance.",
    }


def physics_prototype_probe(u, world, family, rng):
    """Test whether an environment's center transfers to other motions."""
    families = rng.permutation(np.unique(family))
    train = np.isin(family, families[:len(families) // 2])
    ids = sorted(set(world[train]).intersection(world[~train]))
    test = ~train & np.isin(world, ids)
    centers = np.stack([u[train & (world == key)].mean(0) for key in ids])
    mean = u[train].mean(0)
    truth = np.searchsorted(ids, world[test])
    distances = np.sum(u[test] ** 2, 1)[:, None] + np.sum(centers ** 2, 1)[None] - 2 * u[test] @ centers.T
    best5 = np.argpartition(distances, kth=min(4, len(ids) - 1), axis=1)[:, :5]
    return {
        "worlds": len(ids), "test_samples": int(test.sum()),
        "top1_accuracy": float((np.argmin(distances, 1) == truth).mean()),
        "top5_accuracy": float((best5 == truth[:, None]).any(1).mean()),
        "top1_uniform_chance": 1 / len(ids),
        "heldout_motion_center_r2": float(1 - np.sum((u[test] - centers[truth]) ** 2) / np.sum((u[test] - mean) ** 2)),
        "note": "Only DR worlds; physical-world centers fitted on one half of motion families, evaluated on the other half.",
    }


def make_plots(output, data, metadata, pair_distances, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = data["latent"].astype(np.float64)
    u, nominal = normalized(z), data["nominal"]
    centered = u - u.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    xy = centered @ vectors[:, -2:][:, ::-1]
    nominal_centered = u[nominal] - u[nominal].mean(0)
    _, nominal_vectors = np.linalg.eigh(nominal_centered.T @ nominal_centered)
    nominal_xy = nominal_centered @ nominal_vectors[:, -2:][:, ::-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for mask, color, label in ((~nominal, "#d45d35", "Fixed DR"), (nominal, "#2878b5", "Nominal")):
        axes[0, 0].scatter(*xy[mask].T, s=3, alpha=.22, color=color, rasterized=True, label=label)
    axes[0, 0].legend(markerscale=3)
    axes[0, 0].set_title("All full-context samples: PCA of unit latents")
    p = axes[0, 1].scatter(*nominal_xy.T, c=data["motion"][nominal], cmap="turbo", s=4, alpha=.5, rasterized=True)
    fig.colorbar(p, ax=axes[0, 1], label="Motion ID")
    axes[0, 1].set_title("Nominal only: colored by motion (own PCA)")
    p = axes[1, 0].scatter(*nominal_xy.T, c=data["phase"][nominal], cmap="viridis", s=4, alpha=.5, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(p, ax=axes[1, 0], label="Normalized phase")
    axes[1, 0].set_title("Nominal only: colored by phase (same PCA)")
    labels = {
        "nominal_cross_motion": "Nominal: different motions",
        "nominal_same_motion_far_phase": "Nominal: distant phases",
        "nominal_dr_same_motion_near_phase": "Nominal vs DR: matched motion/phase",
        "dr_same_world_cross_motion": "Same DR world: different motions",
        "dr_different_world_same_motion_near_phase": "Different DR worlds: matched motion/phase",
    }
    for key, label in labels.items():
        axes[1, 1].hist(pair_distances[key], bins=65, density=True, histtype="step", linewidth=1.8, label=label)
    axes[1, 1].set_xlabel("Euclidean distance between unit latents")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Distances in the original 64 dimensions")
    axes[1, 1].legend(fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("Forward context v12 / update 8000 — 100-frame histories, nonoverlapping main samples", fontsize=14)
    fig.savefig(output / "latent_clusters.png", dpi=180)
    fig.savefig(output / "latent_clusters.pdf")
    plt.close(fig)

    motion_ids = np.unique(data["motion"][nominal])
    centers = np.stack([u[nominal & (data["motion"] == key)].mean(0) for key in motion_ids])
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    p = ax.imshow(distances, cmap="magma", vmin=0)
    labels = [Path(metadata["motion_files"][key]).name.removesuffix(".motion.npz") for key in motion_ids]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_title("Nominal motion-centroid distances (unit latents, no re-normalization of centers)")
    fig.colorbar(p, ax=ax)
    fig.savefig(output / "nominal_motion_centroids.png", dpi=180)
    plt.close(fig)

    # Deterministic world IDs, selected independently of measured separation.
    selected_worlds = np.unique(data["world"][~nominal])[:8]
    rng = np.random.default_rng(810)
    nominal_ids = np.flatnonzero(nominal)
    nominal_ids = rng.choice(nominal_ids, min(200, len(nominal_ids)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(*xy[nominal_ids].T, s=12, color="black", alpha=.3, label="Nominal (200 samples)")
    for i, world_id in enumerate(selected_worlds):
        mask = data["world"] == world_id
        points = xy[mask]
        color = plt.get_cmap("tab10")(i)
        ax.scatter(*points.T, s=24, color=color, alpha=.7, label=f"DR world {world_id} ({mask.sum()} samples)")
        ax.scatter(*points.mean(0), s=130, marker="X", edgecolors="black", color=color)
    ax.legend(fontsize=9)
    ax.set(xlabel="PC1 (global unit-latent PCA)", ylabel="PC2 (global unit-latent PCA)",
           title="First 8 DR worlds across motions/phases; X = each world center")
    fig.savefig(output / "fixed_dr_world_clusters.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.output / "metadata.json").read_text())
    with np.load(args.output / "latents.npz") as stored:
        all_data = {key: stored[key] for key in stored.files}
    main_mask = ~all_data["neighbor"]
    data = {key: value[main_mask] for key, value in all_data.items()}
    z = data["latent"].astype(np.float64)
    u = normalized(z)
    nominal, world, motion, phase = (data[key] for key in ("nominal", "world", "motion", "phase"))
    if not np.isfinite(z).all() or len(z) < 2:
        raise ValueError("Need finite latent samples")
    families = np.array([Path(path).name.split("_subject")[0] for path in metadata["motion_files"]])
    family = families[motion]
    rng = np.random.default_rng(192608)
    result = {
        "checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["checkpoint_sha256"],
        "collection_complete": metadata.get("complete", False),
        "main_samples": len(z), "nominal": describe(z[nominal]), "dr": describe(z[~nominal]),
        "motion_count": len(np.unique(motion)), "motion_family_count": len(np.unique(family)),
        "worlds": len(np.unique(world)), "nominal_worlds": len(np.unique(world[nominal])),
        "dr_worlds": len(np.unique(world[~nominal])),
        "phase_range": [float(phase.min()), float(phase.max())],
        "pairs": {},
    }
    pair_distances = {}
    for name in (
        "nominal_cross_motion", "nominal_same_motion_far_phase", "nominal_dr_same_motion_near_phase",
        "dr_same_world_cross_motion", "dr_different_world_same_motion_near_phase", "dr_different_world_cross_motion",
    ):
        left, right = select_pairs(data, name, rng)
        result["pairs"][name], pair_distances[name] = pair_stats(z, left, right, world, rng)
    index = {(int(w), int(s)): i for i, (w, s) in enumerate(zip(all_data["world"], all_data["step"], strict=True))}
    for is_nominal, name in ((True, "nominal_local_plus5"), (False, "dr_local_plus5")):
        left, right = [], []
        for i in np.flatnonzero(main_mask & (all_data["nominal"] == is_nominal)):
            offset = metadata["arguments"]["neighbor_offset"]
            j = index.get((int(all_data["world"][i]), int(all_data["step"][i]) + offset))
            if j is not None and all_data["episode"][i] == all_data["episode"][j] and all_data["motion"][i] == all_data["motion"][j]:
                left.append(i)
                right.append(j)
        result["pairs"][name], _ = pair_stats(all_data["latent"].astype(float), np.array(left), np.array(right), all_data["world"], rng)
    result["nominal_motion_dependence"] = crossvalidated_group_r2(u[nominal], motion[nominal], world[nominal], np.random.default_rng(321))
    phase_bins = np.minimum((phase * 10).astype(int), 9)
    result["nominal_motion_phase_dependence"] = crossvalidated_group_r2(u[nominal], motion[nominal] * 10 + phase_bins[nominal], world[nominal], np.random.default_rng(321))
    result["nominal_dr_readout"] = binary_probe(u, nominal, world, family, rng)
    result["dr_world_center_transfer"] = physics_prototype_probe(u[~nominal], world[~nominal], family[~nominal], rng)
    result["nominal_radius_over_matched_nominal_dr_distance"] = result["nominal"]["unit_radius_rms"] / result["pairs"]["nominal_dr_same_motion_near_phase"]["unit_distance_rms"]
    result["dr_within_crossmotion_over_between_matched_distance"] = result["pairs"]["dr_same_world_cross_motion"]["unit_distance_rms"] / result["pairs"]["dr_different_world_same_motion_near_phase"]["unit_distance_rms"]
    result["nominal_per_motion"] = [
        {"motion_id": int(key), "file": metadata["motion_files"][key], **describe(z[nominal & (motion == key)])}
        for key in np.unique(motion[nominal])
    ]
    result["limitations"] = [
        "New rollouts and physics seed within the training motion directory; not a held-out motion benchmark.",
        "Only complete 100-frame histories are analyzed; terminated or short histories are excluded and coverage is reported separately.",
        "High cosine alone can reflect a common offset. Reported distances/radii and probes also use centered variation.",
        "Matched pairs share motion identity and phase within 0.02, not exact state/action or counterfactual trajectories.",
        "Pair bootstrap is by query world; repeated partners remain dependent, so its interval is descriptive.",
        "This measures geometry and nuisance dependence; it does not establish causal control benefit.",
    ]
    (args.output / "cluster_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    make_plots(args.output, data, metadata, pair_distances, result)
    print(json.dumps({key: value for key, value in result.items() if key != "nominal_per_motion"}, indent=2))


if __name__ == "__main__":
    main()
