"""Audited fixed-motion comparisons; never claim across-motion generalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate, *, same_physics=True, samples=4000):
    a, b = [json.loads(Path(path).read_text()) for path in (reference, candidate)]
    matched = ("protocol", "seed", "motion_ids", "start_frames", "metric_names", "max_steps", "motion_files")
    if same_physics:
        matched += ("physics_world_fingerprints",)
        if a["physics"]["physics"] != b["physics"]["physics"]:
            raise ValueError("Expected the same testing physics")
    for key in matched:
        if a[key] != b[key]:
            raise ValueError(f"Unmatched {key}: {reference}, {candidate}")
    for record in (a, b):
        if record["motions"] != 1 or not all(record.get(key) for key in (
            "reference_timeline_audited", "partial_reset_survivor_state_audited",
            "partial_reset_survivor_history_audited",
        )):
            raise ValueError("Missing single-motion/reset audit")
    if not b["training_reward_audit"]["eligible_fixed_reward_candidate"]:
        raise ValueError("Candidate does not have the original reward contract")
    ea, eb = [np.asarray(x["per_episode_metrics"]) for x in (a, b)]
    fa, fb = [np.asarray(x["failed"], dtype=bool) for x in (a, b)]
    rng = np.random.default_rng(13091)
    index = rng.integers(len(ea), size=(samples, len(ea)))
    ci = np.quantile(eb[index].mean(1) / ea[index].mean(1), (0.025, 0.975), axis=0)
    names = a["metric_names"]
    return {
        "reference": str(reference), "candidate": str(candidate),
        "same_physics": same_physics, "episodes": len(ea),
        "scope": "Conditional on this ONE motion and this pair of trained checkpoints; not training-seed or across-motion uncertainty",
        "metrics": {name: {"reference": float(ea[:, j].mean()), "candidate": float(eb[:, j].mean()),
                           "ratio": float(eb[:, j].mean() / ea[:, j].mean()),
                           "conditional_world_start_bootstrap_95pct": ci[:, j].tolist()}
                    for j, name in enumerate(names)},
        "failures": {"reference": int(fa.sum()), "candidate": int(fb.sum()),
                     "new": int((fb & ~fa).sum()), "recovered": int((fa & ~fb).sum()),
                     "new_start_frames": np.asarray(a["start_frames"])[fb & ~fa].tolist()},
    }


def summarize(root, seed=121):
    results = {}
    for suffix in ("", "_ac"):
        for iteration in (100, 200, 300):
            true = root / f"dr_true{suffix}_seed{seed}_{iteration}_test_dr.json"
            zero = root / f"dr_zero{suffix}_seed{seed}_{iteration}_test_dr.json"
            nominal = root / f"nominal_zero{suffix}_seed{seed}_{iteration}_test_nominal.json"
            for label, reference, candidate, same in (
                ("preview_benefit", zero, true, True),
                ("nominal_gap", nominal, true, False),
                ("frozen_tracker_gain", root / "tracker_test_dr.json", true, True),
                ("fixed_nominal_gap", root / "fixed_nominal_test_nominal.json", true, False),
            ):
                if reference.exists() and candidate.exists():
                    results[f"{label}{suffix}_{iteration}"] = compare(reference, candidate, same_physics=same)
    for iteration in (100, 200, 300):
        for mode in ("true", "zero"):
            a = root / f"dr_{mode}_seed{seed}_{iteration}_test_dr.json"
            b = root / f"dr_{mode}_ac_seed{seed}_{iteration}_test_dr.json"
            if a.exists() and b.exists():
                results[f"critic_addition_{mode}_{iteration}"] = compare(a, b)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=121)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.train_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, item in result.items():
        print(name, {key: round(item["metrics"][key]["ratio"], 5) for key in ("error_body_pos", "error_joint_pos")}, item["failures"])


if __name__ == "__main__":
    main()
