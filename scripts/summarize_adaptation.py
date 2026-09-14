"""Rank development candidates without hiding non-primary tracking regressions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def summarize(root):
    baseline = json.loads((root / "existing_nominal_seed10001.json").read_text())
    records = []
    for path in root.glob("*/eval_*_seed10001.json"):
        if not re.fullmatch(r"eval_(\d+|final)_seed10001.json", path.name):
            continue
        data = json.loads(path.read_text())
        if any(
            data.get(key) != baseline.get(key)
            for key in (
                "protocol",
                "seed",
                "motion_ids",
                "start_frames",
                "max_steps",
                "motion_files",
            )
        ):
            continue
        if data["physics"]["physics"] != "dr":
            continue
        if (
            data.get("teacher_feature_ablation") not in (None, "normal")
            or data.get("context_ablation") not in (None, "normal")
            or data.get("privileged_bias_compensation", 0.0)
            or data.get("action_filter_strength", 0.0)
            or data.get("orientation_filter_weight", 1.0) != 1.0
        ):
            continue
        ratios = {key: value / baseline["mean"][key] for key, value in data["mean"].items()}
        student = data.get("actor_input_contract") == "deployable groups only"
        margin = 1.02 if student else 1.05
        failure_difference = data["failure_rate"] - baseline["failure_rate"]
        primary = max(ratios[key] for key in ("error_body_pos", "error_joint_pos"))
        worst = max(ratios, key=ratios.get)
        records.append(
            {
                "run": path.parent.name,
                "evaluation": str(path),
                "checkpoint": data["checkpoint"],
                "student": student,
                "body_m": data["mean"]["error_body_pos"],
                "joint_l2_rad": data["mean"]["error_joint_pos"],
                "anchor_m": data["mean"]["error_anchor_pos"],
                "failure_rate": data["failure_rate"],
                "failure_difference": failure_difference,
                "max_primary_ratio": primary,
                "worst_metric": worst,
                "max_all_metric_ratio": ratios[worst],
                "ratios": ratios,
                "primary_point_pass": primary <= margin and failure_difference <= 0.01,
                "all_metrics_point_pass": ratios[worst] <= margin and failure_difference <= 0.01,
            }
        )
    records.sort(key=lambda x: (x["failure_difference"] > 0.01, x["max_all_metric_ratio"]))
    return {
        "baseline": str(root / "existing_nominal_seed10001.json"),
        "scope": "development only; not final confirmation",
        "candidates": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/adaptation_goal/eval_v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--students-only", action="store_true")
    parser.add_argument("--per-run", action="store_true")
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    rows = result["candidates"]
    if args.students_only:
        rows = [row for row in rows if row["student"]]
    if args.per_run:
        seen = set()
        unique = []
        for row in rows:
            if row["run"] not in seen:
                seen.add(row["run"])
                unique.append(row)
        rows = unique
    for row in rows[:12]:
        print(
            f"{Path(row['evaluation']).parent.name:40} {Path(row['evaluation']).name:25} body={row['body_m']:.5f} joint={row['joint_l2_rad']:.4f} max_ratio={row['max_all_metric_ratio']:.3f} ({row['worst_metric']}) failure={row['failure_rate']:.3%}"
        )


if __name__ == "__main__":
    main()
