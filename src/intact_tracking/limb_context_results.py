"""Matched comparisons; motion uncertainty is distinct from training-seed variability."""

from pathlib import Path

import numpy as np

PAIRED_FIELDS = ("protocol", "seed", "motion_ids", "start_frames", "horizons", "metric_names",
                 "max_steps", "motion_files", "physics_world_fingerprints", "reward_contract")
COLUMNS = ("reference_body", "reference_joint", "candidate_body", "candidate_joint",
           "reference_failure", "candidate_failure", "reference_coverage", "candidate_coverage",
           "reference_return", "candidate_return", "reference_common_body", "reference_common_joint",
           "candidate_common_body", "candidate_common_joint")


def paired_rows(reference, candidate, reference_trace, candidate_trace):
    for key in PAIRED_FIELDS:
        if reference[key] != candidate[key]:
            raise ValueError(f"Unmatched evaluation field {key}")
    from intact_tracking.limb_context_dr import LOAD_ONLY
    for key in ("dr_profile", "original_events", "observation_corruption"):
        default = LOAD_ONLY if key == "dr_profile" else None
        if reference.get("physics", {}).get(key, default) != candidate.get("physics", {}).get(key, default):
            raise ValueError(f"Unmatched evaluation DR field {key}")
    ids = np.asarray(reference["motion_ids"], dtype=int)
    rf, cf = np.asarray(reference["failed"], bool), np.asarray(candidate["failed"], bool)
    rn, cn = np.asarray(reference["episode_lengths"]), np.asarray(candidate["episode_lengths"])
    horizon = np.asarray(reference["horizons"])
    if (np.minimum(rn, cn) < 1).any():
        raise ValueError("Empty comparison episode")
    indices = [reference["metric_names"].index(key) for key in ("error_body_pos", "error_joint_pos")]
    r = np.asarray(reference["per_episode_metrics"])[:, indices]
    c = np.asarray(candidate["per_episode_metrics"])[:, indices]
    rt, ct = reference_trace["body_joint"], candidate_trace["body_joint"]
    if not np.array_equal(reference_trace["lengths"], rn) or not np.array_equal(candidate_trace["lengths"], cn):
        raise ValueError("Step trace and episode lengths disagree")
    if rt.shape != ct.shape or rt.shape != (len(ids), reference["max_steps"], 2):
        raise ValueError("Step trace shape differs")
    common = np.minimum(rn, cn)
    mask = np.arange(rt.shape[1])[None, :] < common[:, None]
    rcommon = (rt * mask[..., None]).sum(1, dtype=np.float64) / common[:, None]
    ccommon = (ct * mask[..., None]).sum(1, dtype=np.float64) / common[:, None]
    values = np.column_stack((r, c, rf, cf, rn / horizon, cn / horizon,
                              reference["episode_returns"], candidate["episode_returns"], rcommon, ccommon))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite paired result")
    sums = np.zeros((reference["motions"], len(COLUMNS)))
    np.add.at(sums, ids, values)
    counts = np.bincount(ids, minlength=len(sums))
    if not (counts > 0).all():
        raise ValueError("Missing motion episodes")
    return sums / counts[:, None], int((cf & ~rf).sum()), int((rf & ~cf).sum())


def summary_statistics(mean):
    return np.r_[mean[2:4] / np.maximum(mean[:2], 1e-12), mean[5] - mean[4],
                 mean[7] - mean[6], mean[9] - mean[8],
                 mean[12:14] / np.maximum(mean[10:12], 1e-12)]


def summarize(rows, new_failures, rescued, bootstrap_samples=2000):
    mean = rows.mean(0)
    rng = np.random.default_rng(7712)
    draws = np.stack([summary_statistics(rows[rng.integers(len(rows), size=len(rows))].mean(0))
                      for _ in range(bootstrap_samples)])
    lo, hi = np.quantile(draws, (.025, .975), axis=0)
    return {"motions": len(rows), "means": dict(zip(COLUMNS, mean.tolist(), strict=True)),
            "candidate_over_reference_body_joint": (mean[2:4] / np.maximum(mean[:2], 1e-12)).tolist(),
            "body_joint_ratio_ci95": np.stack((lo[:2], hi[:2]), -1).tolist(),
            "failure_rate_delta": float(mean[5] - mean[4]), "failure_rate_delta_ci95": [float(lo[2]), float(hi[2])],
            "coverage_delta": float(mean[7] - mean[6]), "coverage_delta_ci95": [float(lo[3]), float(hi[3])],
            "episode_return_delta": float(mean[9] - mean[8]), "episode_return_delta_ci95": [float(lo[4]), float(hi[4])],
            "common_prefix_body_joint_ratio": (mean[12:14] / np.maximum(mean[10:12], 1e-12)).tolist(),
            "common_prefix_body_joint_ratio_ci95": np.stack((lo[5:], hi[5:]), -1).tolist(),
            "new_failure_episodes": new_failures, "rescued_failure_episodes": rescued,
            "uncertainty": "paired motion bootstrap conditional on this training seed; not training-seed uncertainty",
            "scope": "new starts and loads on the training motion catalog; not unseen-motion generalization"}


def compare_files(reference_paths, candidate_paths, bootstrap_samples=2000):
    import json

    rows, added, rescued = [], 0, 0
    for rpath, cpath in zip(reference_paths, candidate_paths, strict=True):
        rpath, cpath = Path(rpath), Path(cpath)
        with np.load(rpath.with_suffix(".traces.npz")) as rtrace, np.load(cpath.with_suffix(".traces.npz")) as ctrace:
            row, new, fixed = paired_rows(json.loads(rpath.read_text()), json.loads(cpath.read_text()), rtrace, ctrace)
        rows.append(row)
        added += new
        rescued += fixed
    return summarize(np.concatenate(rows), added, rescued, bootstrap_samples)
