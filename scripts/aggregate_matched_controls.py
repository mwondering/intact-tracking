"""Aggregate independent paired training seeds, not pseudoreplicated eval seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(root, training_seeds=(76, 77, 78), samples=10000):
    evaluation_seeds = (92001, 92002, 92003)
    if len(set(training_seeds)) != len(training_seeds) or len(training_seeds) < 2:
        raise ValueError("At least two distinct completed independent training seeds required")
    experiments = []
    for seed in training_seeds:
        audit = json.loads((root.parents[1] / f"matched_seed{seed}_initial_audit.json").read_text())
        if not audit["all_initial_tensors_bitwise_equal"] or not audit["agent_configuration_equal"]:
            raise ValueError("Missing exact pre-update actor/critic audit")
        experiments.append(json.loads((root / f"training_seed{seed}_comparison.json").read_text()))
    output = {"training_seeds": list(training_seeds), "evaluation_seeds": list(evaluation_seeds),
              "independent_training_pairs": len(training_seeds),
              "scope": "Only subsequent training DR differs; common DR-pretrained frozen tracker, original rewards and equal budget. Not from-scratch nominal-only training. DR payload is 1–3kg, nominal payload0kg outside that training support.",
              "uncertainty": "Paired bootstrap resamples training pairs, evaluation seeds within training pair, and shared motion clusters. Three training pairs still provide limited training-level precision.",
              "conditions": {}}
    for physics in ("nominal", "dr"):
        nominal_values, dr_values, nominal_failures, dr_failures = [], [], [], []
        metrics = None
        for seed in training_seeds:
            refs, candidates, rf, cf = [], [], [], []
            identities = None
            for evaluation_seed in evaluation_seeds:
                paths = [root / f"train{seed}_{train}_test{physics}_seed{evaluation_seed}.json" for train in ("nominal", "dr")]
                a, b = [json.loads(p.read_text()) for p in paths]
                for key in ("protocol", "seed", "motion_ids", "start_frames", "max_steps", "metric_names", "motion_files", "physics_world_fingerprints"):
                    if a[key] != b[key]:
                        raise ValueError(f"Unmatched {key} in {paths}")
                if not a["physics_world_fingerprints"]:
                    raise ValueError("Physical world audit missing")
                for record in (a, b):
                    if record["physics"]["physics"] != physics or record["protocol"] != "balanced_fixed_starts_v2_isolated_resets":
                        raise ValueError("Unexpected evaluation physics/protocol")
                    if not all(record.get(key) is True for key in ("reference_timeline_audited", "partial_reset_survivor_state_audited", "partial_reset_survivor_history_audited")):
                        raise ValueError("Reset audit missing")
                ids = (a["checkpoint_sha256"], b["checkpoint_sha256"])
                if identities is not None and identities != ids:
                    raise ValueError("Policy checkpoint changed within a training seed")
                identities = ids
                if metrics is not None and metrics != a["metric_names"]:
                    raise ValueError("Metric schema changed")
                metrics = a["metric_names"]
                motion_ids = np.asarray(a["motion_ids"])
                for record, values, failures in ((a, refs, rf), (b, candidates, cf)):
                    errors = np.asarray(record["per_episode_metrics"])
                    failed = np.asarray(record["failed"], dtype=float)
                    values.append([errors[motion_ids == motion].mean(0) for motion in range(a["motions"])])
                    failures.append([failed[motion_ids == motion].mean() for motion in range(a["motions"])])
            nominal_values.append(refs)
            dr_values.append(candidates)
            nominal_failures.append(rf)
            dr_failures.append(cf)
        a, b = np.asarray(nominal_values), np.asarray(dr_values)
        fa, fb = np.asarray(nominal_failures), np.asarray(dr_failures)
        rng = np.random.default_rng(19317)
        train_index = rng.integers(a.shape[0], size=(samples, a.shape[0], 1, 1))
        eval_index = rng.integers(a.shape[1], size=(samples, a.shape[0], a.shape[1], 1))
        motion_index = rng.integers(a.shape[2], size=(samples, 1, 1, a.shape[2]))
        # Chunk to avoid allocating >2GB for a10k x3 x3 x42 x10 tensor.
        boot_ratios, boot_failures = [], []
        for start in range(0, samples, 250):
            sl = slice(start, start + 250)
            index = (train_index[sl], eval_index[sl], motion_index[sl])
            boot_ratios.append(b[index].mean((1, 2, 3)) / a[index].mean((1, 2, 3)))
            boot_failures.append((fb[index] - fa[index]).mean((1, 2, 3)))
        ci = np.quantile(np.concatenate(boot_ratios), (0.025, 0.975), axis=0)
        condition = {"metrics": {}, "nominal_trained_failure_rate": float(fa.mean()),
                     "dr_trained_failure_rate": float(fb.mean()),
                     "failure_difference_95pct": np.quantile(np.concatenate(boot_failures), (0.025, 0.975)).tolist(),
                     "per_training_pair": []}
        for j, metric in enumerate(metrics):
            condition["metrics"][metric] = {"nominal_trained": float(a[..., j].mean()),
                                            "dr_trained": float(b[..., j].mean()),
                                            "dr_over_nominal": float(b[..., j].mean() / a[..., j].mean()),
                                            "paired_training_eval_motion_bootstrap_95pct": ci[:, j].tolist()}
        for seed, experiment in zip(training_seeds, experiments, strict=True):
            condition["per_training_pair"].append({
                "training_seed": seed,
                "ratios": {k: v["ratio"] for k, v in experiment[physics]["statistics"]["metrics"].items()},
                "failure_counts": {k: experiment[physics]["paired_failures"][k] for k in ("reference", "candidate", "new_failures", "recovered_failures", "episodes")},
            })
        output["conditions"][physics] = condition
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/adaptation_goal/eval_v2/matched_controls_920xx"))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(76, 77, 78))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root, tuple(args.training_seeds))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
