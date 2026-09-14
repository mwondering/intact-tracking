"""Aggregate matched actor-input ablations with independent training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def crossed_summary(reference, candidate, ref_failures, candidate_failures, samples=10000):
    """Resample training pairs and SHARED evaluation-world/motion axes."""
    a, b, fa, fb = map(np.asarray, (reference, candidate, ref_failures, candidate_failures))
    if a.shape != b.shape or a.ndim != 4 or a.shape[0] < 2 or fa.shape != a.shape[:3] or fb.shape != fa.shape:
        raise ValueError("Expected >=2 matched training pairs with eval/motion/metric axes")
    rng = np.random.default_rng(97131)
    ti = rng.integers(a.shape[0], size=(samples, a.shape[0], 1, 1))
    ei = rng.integers(a.shape[1], size=(samples, 1, a.shape[1], 1))
    mi = rng.integers(a.shape[2], size=(samples, 1, 1, a.shape[2]))
    ratios, failures = [], []
    for start in range(0, samples, 250):
        index = tuple(value[start:start + 250] for value in (ti, ei, mi))
        ratios.append(b[index].mean((1, 2, 3)) / a[index].mean((1, 2, 3)))
        failures.append((fb[index] - fa[index]).mean((1, 2, 3)))
    return {"reference_means": a.mean((0, 1, 2)).tolist(), "candidate_means": b.mean((0, 1, 2)).tolist(),
            "ratios": (b.mean((0, 1, 2)) / a.mean((0, 1, 2))).tolist(),
            "ratio_intervals": np.quantile(np.concatenate(ratios), (0.025, 0.975), axis=0).T.tolist(),
            "reference_failure_rate": float(fa.mean()), "candidate_failure_rate": float(fb.mean()),
            "failure_difference_interval": np.quantile(np.concatenate(failures), (0.025, 0.975)).tolist()}


def aggregate(roots):
    reference, candidate, ref_failures, candidate_failures = [], [], [], []
    seeds, per_pair = [], []
    shared_worlds, shared_schema, shared_arguments = {}, None, None
    for root in roots:
        comparison = json.loads((root / "comparison.json").read_text())
        audit = comparison["actor_input_only_training_audit"]
        if not audit or not comparison["same_physics_worlds_audited"] or audit["checkpoint_iteration"] != 500:
            raise ValueError("Expected audited fixed500 actor-input comparison")
        path = Path(comparison["plan"]["identities"]["candidate"]["path"])
        arguments = json.loads((path.parent / "run_config.json").read_text())["arguments"]
        seed = arguments["seed"]
        if seed in seeds:
            raise ValueError("Repeated training seed is not an independent replication")
        seeds.append(seed)
        settings = {key: value for key, value in arguments.items()
                    if key not in ("seed", "training_rng_seed", "output_dir", "iterations")}
        for key, default in {"actor_physics_input": "normal", "latent_low_rank": False,
                             "physics_critic": False, "frozen_nominal_prior": None, "shared_low_rank": 0,
                             "frozen_nominal_prior_deployable_base": False,
                             "add_teacher_refinement": False, "refinement_scale": 1.0}.items():
            settings.setdefault(key, default)
        if shared_arguments is not None and settings != shared_arguments:
            raise ValueError("Training replications changed architecture or optimization settings")
        shared_arguments = settings
        role_values, role_failures = [], []
        for role in ("reference", "candidate"):
            values, failures = [], []
            for evaluation_seed in comparison["plan"]["evaluation_seeds"]:
                record = json.loads((root / f"{role}_seed{evaluation_seed}.json").read_text())
                identity = {key: record[key] for key in (
                    "protocol", "motion_ids", "start_frames", "max_steps", "metric_names",
                    "motion_files", "physics_world_fingerprints",
                )}
                if evaluation_seed in shared_worlds and shared_worlds[evaluation_seed] != identity:
                    raise ValueError("Evaluation worlds/segments changed across roles or training seeds")
                shared_worlds[evaluation_seed] = identity
                schema = (comparison["plan"]["evaluation_seeds"], record["metric_names"])
                if shared_schema is not None and schema != shared_schema:
                    raise ValueError("Evaluation seed/metric schema changed")
                shared_schema = schema
                if record["checkpoint_sha256"] != comparison["plan"]["identities"][role]["sha256"]:
                    raise ValueError("Frozen checkpoint identity changed")
                ids = np.asarray(record["motion_ids"])
                errors, failed = np.asarray(record["per_episode_metrics"]), np.asarray(record["failed"], dtype=float)
                values.append([errors[ids == motion].mean(0) for motion in range(record["motions"])])
                failures.append([failed[ids == motion].mean() for motion in range(record["motions"])])
            role_values.append(values)
            role_failures.append(failures)
        reference.append(role_values[0])
        candidate.append(role_values[1])
        ref_failures.append(role_failures[0])
        candidate_failures.append(role_failures[1])
        per_pair.append({"training_seed": seed, "comparison": str(root / "comparison.json"),
                         "metrics": comparison["statistics"]["metrics"],
                         "paired_failures": comparison["paired_failures"]})
    result = crossed_summary(reference, candidate, ref_failures, candidate_failures)
    return {"training_seeds": seeds, "independent_training_pairs": len(seeds),
            "evaluation_seeds": shared_schema[0], "metric_names": shared_schema[1],
            "statistics": result, "per_training_pair": per_pair,
            "scope": "Only actor real physics input differs within each matched pair; controller, critic, original reward, initialization and fixed500 update budget are matched. Different planned run lengths do not change the fixed learning-rate updates being compared.",
            "uncertainty": "Crossed paired bootstrap resamples training pairs, shared evaluation-world seeds and shared motion clusters. Only two training pairs give very limited training-population precision; intervals must not be presented as a broad generalization guarantee.",
            "nominal_goal_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.roots)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
