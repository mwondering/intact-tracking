"""Strict development-only scoreboard for newly audited original-reward runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_adaptation_evals import compare

MATCHED = ("protocol", "seed", "motion_ids", "start_frames", "max_steps", "motion_files", "metric_names")


def summarize(root: Path):
    nominal_path = root / "reference_existing_nominal_seed10001.json"
    frozen_path = root / "reference_frozen_dr_seed10001.json"
    nominal = json.loads(nominal_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if nominal["checkpoint_sha256"] != "800da8c40016bba3263e685e9694e51e82b83a5e1c3df3e2e80543bd47ad8f1f":
        raise ValueError("The fixed nominal target checkpoint changed")
    if frozen["checkpoint_sha256"] != "fd7bd90d5552e573bbbce1417e9b415c64bb487a76b683ba3c20503b5ec77635":
        raise ValueError("The original frozen tracker checkpoint changed")
    records = []
    for path in sorted(root.glob("fixed_reward_*/eval_*_seed10001.json")):
        candidate = json.loads(path.read_text())
        if not candidate.get("training_reward_audit", {}).get("eligible_fixed_reward_candidate"):
            raise ValueError(f"Unaudited candidate in clean scoreboard: {path}")
        if any(candidate[key] != ref[key] for ref in (nominal, frozen) for key in MATCHED):
            raise ValueError(f"Evaluation protocol or segment mismatch: {path}")
        student = candidate["actor_input_contract"] == "deployable groups only"
        margin = 1.02 if student else 1.05
        statistics = compare([nominal_path], [path], margin=margin, samples=1000)
        failed = candidate["failed"]
        frozen_failed = frozen["failed"]
        same_physics_audited = candidate.get("physics_world_fingerprints") == frozen.get("physics_world_fingerprints") and bool(frozen.get("physics_world_fingerprints"))
        no_increase = sum(failed) <= sum(frozen_failed)
        ratios = {key: value["ratio"] for key, value in statistics["metrics"].items()}
        worst = max(ratios, key=ratios.get)
        records.append({
            "run": path.parent.name, "evaluation": str(path), "checkpoint": candidate["checkpoint"],
            "checkpoint_sha256": candidate["checkpoint_sha256"], "student": student,
            "ratios": ratios, "worst_metric": worst, "worst_ratio": ratios[worst],
            "failed": sum(failed), "nominal_failed": sum(nominal["failed"]),
            "frozen_dr_failed": sum(frozen_failed),
            "new_failures_vs_frozen_dr": sum(a and not b for a, b in zip(failed, frozen_failed, strict=True)),
            "recovered_failures_vs_frozen_dr": sum(b and not a for a, b in zip(failed, frozen_failed, strict=True)),
            "same_physics_frozen_fingerprints_audited": same_physics_audited,
            "observed_failure_nonincrease_both_references": no_increase and statistics["observed_failure_nonincrease"],
            "all_metrics_point_pass_nominal": statistics["all_metrics_point_estimate_pass"],
            "development_point_screen_pass": statistics["all_metrics_point_estimate_pass"] and no_increase and same_physics_audited,
            "not_final_confirmation": True,
        })
    records.sort(key=lambda row: (not row["observed_failure_nonincrease_both_references"], row["worst_ratio"]))
    return {"contract": "Development screen only; unchanged reward, all10 errors, no observed failure increase against nominal target and same-physics frozen tracker. Fresh multi-seed confirmation and student input/export audit still required.", "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/adaptation_goal/eval_v2/fixed_reward"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["records"]:
        print(f"{row['run']} {Path(row['checkpoint']).name}: worst={row['worst_ratio']:.5f} {row['worst_metric']}; failures={row['failed']}/{row['nominal_failed']}/{row['frozen_dr_failed']} (candidate/nominal/frozenDR); screen={row['development_point_screen_pass']}")


if __name__ == "__main__":
    main()
