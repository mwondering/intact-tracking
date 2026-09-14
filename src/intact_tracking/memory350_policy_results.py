"""Paired trajectory analysis with explicit failure censoring and motion bootstrap."""

import numpy as np


def assert_paired(reference, candidate):
    for key in ("protocol", "seed", "motion_files", "motion_ids", "start_frames", "horizons", "max_steps",
                "metric_names", "reward_contract", "memory_start", "physics_world_fingerprints",
                "actual_limb_masses_kg", "query_initial_state_sha256"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Unpaired tracking evaluations: {key}")
    for key in ("dr_profile", "original_events", "observation_corruption"):
        if reference["physics"][key] != candidate["physics"][key]:
            raise ValueError(f"Unpaired DR: {key}")
    # Parallel contact simulation is not bitwise deterministic. Pair the physical
    # world, query state and common frozen-policy warm-up protocol, while keeping
    # each actual warm trajectory digest in the evidence.
    for key in ("steps", "policy", "policy_sha256", "seed"):
        if reference["warmup"][key] != candidate["warmup"][key]:
            raise ValueError(f"Warm-up protocol differs: {key}")


def bootstrap(reference, candidate, *, seed=8102, repeats=2000):
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.shape != candidate.shape or reference.ndim != 1 or not np.isfinite(reference + candidate).all():
        raise ValueError("Expected finite paired per-motion values")
    rng = np.random.default_rng(seed)
    differences, reductions = [], []
    for start in range(0, repeats, 50):
        ids = rng.integers(len(reference), size=(min(50, repeats - start), len(reference)))
        a, b = reference[ids].mean(1), candidate[ids].mean(1)
        differences.extend(b - a)
        reductions.extend(100 * (a - b) / np.maximum(a, 1e-12))
    a, b = float(reference.mean()), float(candidate.mean())
    return {"reference": a, "candidate": b, "candidate_minus_reference": b - a,
            "difference_ci95": np.quantile(differences, [.025, .975]).tolist(),
            "reduction_percent": 100 * (a - b) / max(a, 1e-12),
            "reduction_percent_ci95": np.quantile(reductions, [.025, .975]).tolist()}


def compare(reference, candidate, reference_trace, candidate_trace, *, repeats=2000):
    assert_paired(reference, candidate)
    n = reference["episodes"]
    if n != len(reference["motion_files"]) or candidate["episodes"] != n:
        raise ValueError("This bootstrap expects one episode per distinct motion")
    lengths_a, lengths_b = np.asarray(reference["episode_lengths"]), np.asarray(candidate["episode_lengths"])
    common = np.minimum(lengths_a, lengths_b)
    if (common <= 0).any() or reference_trace.shape != candidate_trace.shape or reference_trace.shape[:2] != (n, reference["max_steps"]):
        raise ValueError("Invalid survival traces")
    means_a, means_b = np.asarray(reference["per_episode_metrics"]), np.asarray(candidate["per_episode_metrics"])
    output = {"paired_motions": n, "bootstrap_repeats": repeats,
              "bootstrap_unit": "motion; one fixed-start episode per motion",
              "training_seed_count": 1, "common_survival_steps_mean": float(common.mean())}
    output["failure_rate"] = bootstrap(reference["failed"], candidate["failed"], repeats=repeats)
    output["coverage"] = bootstrap(lengths_a / np.asarray(reference["horizons"]),
                                    lengths_b / np.asarray(candidate["horizons"]), repeats=repeats)
    for index, name in enumerate(reference["metric_names"]):
        output["truncated_" + name] = bootstrap(means_a[:, index], means_b[:, index], repeats=repeats)
        if index < 2:
            mask = np.arange(reference["max_steps"])[None] < common[:, None]
            a = (reference_trace[:, :, index] * mask).sum(1) / common
            b = (candidate_trace[:, :, index] * mask).sum(1) / common
            output["common_" + name] = bootstrap(a, b, repeats=repeats)
    return output
