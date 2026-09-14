"""Compare matched evaluations with a paired motion-cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONFIRMATION_SEED_SETS = (
    (20001, 20002, 20003),
    (30001, 30002, 30003),
    (40001, 40002, 40003),
    (50001, 50002, 50003),
    (60001, 60002, 60003),
    (70001, 70002, 70003),
    (80001, 80002, 80003),
)


def compare(reference_paths, candidate_paths, margin=1.05, samples=10000):
    if len(reference_paths) != len(candidate_paths) or not reference_paths:
        raise ValueError("Supply equal nonempty lists of reference and candidate JSON files")
    records = []
    refs, candidates, ref_failures, candidate_failures = [], [], [], []
    metric_names = None
    motion_files = None
    experiment_identity = None
    all_reset_audits = True
    acceptance_conditions = True
    fixed_reward_candidate_eligible = True
    seen_seeds = set()
    for reference_path, candidate_path in zip(reference_paths, candidate_paths, strict=True):
        ref = json.loads(Path(reference_path).read_text())
        candidate = json.loads(Path(candidate_path).read_text())
        identity = (
            ref["protocol"],
            ref["checkpoint_sha256"],
            candidate["checkpoint_sha256"],
        )
        if experiment_identity is None:
            experiment_identity = identity
        if identity != experiment_identity:
            raise ValueError("Protocol or policy identity changed between evaluation seeds")
        if ref["seed"] in seen_seeds:
            raise ValueError("Duplicate evaluation seed")
        seen_seeds.add(ref["seed"])
        all_reset_audits &= all(
            item.get(key) is True
            for item in (ref, candidate)
            for key in (
                "reference_timeline_audited",
                "partial_reset_survivor_state_audited",
                "partial_reset_survivor_history_audited",
            )
        )
        acceptance_conditions &= (
            ref["physics"]["physics"] == "nominal"
            and candidate["physics"]["physics"] == "dr"
            and candidate.get("context_ablation") in (None, "normal")
            and candidate.get("teacher_feature_ablation") in (None, "normal")
            and candidate.get("teacher_remove_privilege") in (None, "normal")
            and not candidate.get("action_filter_strength", 0.0)
            and candidate.get("orientation_filter_weight", 1.0) == 1.0
            and not candidate.get("privileged_bias_compensation", 0.0)
            and not candidate.get("privileged_payload_gravity_compensation", 0.0)
        )
        fixed_reward_candidate_eligible &= candidate.get("training_reward_audit", {}).get("eligible_fixed_reward_candidate") is True
        for key in (
            "protocol",
            "seed",
            "motion_ids",
            "start_frames",
            "metric_names",
            "max_steps",
            "motion_files",
        ):
            if ref[key] != candidate[key]:
                raise ValueError(
                    f"Evaluation mismatch in {key}: {reference_path}, {candidate_path}"
                )
        if metric_names is None:
            metric_names, motion_files = ref["metric_names"], ref["motion_files"]
        if ref["metric_names"] != metric_names or ref["motion_files"] != motion_files:
            raise ValueError("Metric or motion schema changed between evaluation seeds")
        ids = np.asarray(ref["motion_ids"])
        r = np.asarray(ref["per_episode_metrics"])
        c = np.asarray(candidate["per_episode_metrics"])
        rf, cf = (
            np.asarray(ref["failed"], dtype=float),
            np.asarray(candidate["failed"], dtype=float),
        )
        for values, dest in (
            (r, refs),
            (c, candidates),
            (rf, ref_failures),
            (cf, candidate_failures),
        ):
            dest.append(np.asarray([values[ids == i].mean(axis=0) for i in range(ref["motions"])]))
        records.append(
            {
                "seed": ref["seed"],
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "ratios": {key: candidate["mean"][key] / ref["mean"][key] for key in metric_names},
                "failure_difference": candidate["failure_rate"] - ref["failure_rate"],
            }
        )
    refs, candidates = np.asarray(refs), np.asarray(candidates)
    ref_failures, candidate_failures = np.asarray(ref_failures), np.asarray(candidate_failures)
    rng = np.random.default_rng(9173)
    seed_ids = rng.integers(0, len(refs), size=(samples, len(refs)))
    # Keep all repeats for each sampled motion; treating them as independent
    # motions would overstate precision. Resample seeds as a second cluster axis.
    motion_ids = rng.integers(0, refs.shape[1], size=(samples, 1, refs.shape[1]))
    sampled_ref = refs[seed_ids[:, :, None], motion_ids].mean(axis=(1, 2))
    sampled_candidate = candidates[seed_ids[:, :, None], motion_ids].mean(axis=(1, 2))
    ratios = sampled_candidate / sampled_ref
    ci = np.quantile(ratios, (0.025, 0.975), axis=0)
    summaries = {}
    for i, name in enumerate(metric_names):
        summaries[name] = {
            "reference": float(refs[:, :, i].mean()),
            "candidate": float(candidates[:, :, i].mean()),
            "ratio": float(candidates[:, :, i].mean() / refs[:, :, i].mean()),
            "paired_motion_seed_bootstrap_95pct": ci[:, i].tolist(),
        }
    failure_diff = float(candidate_failures.mean() - ref_failures.mean())
    sampled_failure_diff = (
        candidate_failures[seed_ids[:, :, None], motion_ids]
        - ref_failures[seed_ids[:, :, None], motion_ids]
    ).mean(axis=(1, 2))
    primary = ("error_body_pos", "error_joint_pos")
    valid_protocol = (
        ref["protocol"] == "balanced_fixed_starts_v2_isolated_resets" and all_reset_audits
    )
    numerical_point_pass = (
        all(summaries[k]["ratio"] <= margin for k in primary) and failure_diff <= 0.0
    )
    eligible = valid_protocol and acceptance_conditions and fixed_reward_candidate_eligible
    point_pass = numerical_point_pass and eligible
    interval_pass = eligible and all(
        summaries[k]["paired_motion_seed_bootstrap_95pct"][1] <= margin for k in primary
    )
    failure_interval = np.quantile(sampled_failure_diff, (0.025, 0.975)).tolist()
    return {
        "reference_files": [str(p) for p in reference_paths],
        "candidate_files": [str(p) for p in candidate_paths],
        "seeds": records,
        "metrics": summaries,
        "reference_failure_rate": float(ref_failures.mean()),
        "candidate_failure_rate": float(candidate_failures.mean()),
        "failure_difference": failure_diff,
        "failure_difference_paired_bootstrap_95pct": failure_interval,
        "failure_upper_95pct_within_margin": eligible and failure_interval[1] <= 0.0,
        "allowed_observed_failure_increase": 0.0,
        "observed_failure_nonincrease": failure_diff <= 0.0,
        "fixed_reward_candidate_eligible": fixed_reward_candidate_eligible,
        "margin": margin,
        "evaluation_protocol": ref["protocol"],
        "valid_acceptance_protocol": valid_protocol,
        "matched_single_policy_identity": True,
        "acceptance_conditions": acceptance_conditions,
        "numerical_point_estimate_pass": numerical_point_pass,
        "point_estimate_pass": point_pass,
        "primary_upper_95pct_within_margin": interval_pass,
        "all_metrics_point_estimate_pass": eligible
        and all(v["ratio"] <= margin for v in summaries.values())
        and failure_diff <= 0.0,
        "all_metrics_upper_95pct_within_margin": eligible
        and all(v["paired_motion_seed_bootstrap_95pct"][1] <= margin for v in summaries.values())
        and failure_interval[1] <= 0.0,
        "confirmation_seeds_used": all(
            x["seed"] in {seed for cohort in CONFIRMATION_SEED_SETS for seed in cohort}
            for x in records
        ),
        "confirmation_seed_set_complete": tuple(sorted(x["seed"] for x in records))
        in CONFIRMATION_SEED_SETS,
        "warning": "Primary metrics alone do not establish whole-tracking equivalence. Also check same-physics frozen-tracker failures; finite samples cannot guarantee zero added risk."
        if eligible
        else "INVALID FOR ACCEPTANCE: missing v2 reset audits, fixed-reward lineage, or a diagnostic/ablation condition.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--margin", type=float, default=1.05)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.margin)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
