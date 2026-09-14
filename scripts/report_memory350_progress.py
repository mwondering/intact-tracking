"""Report matched periodic checkpoints after both policy evaluations finish."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from intact_tracking.memory350_policy_checkpoint_eval import digest, validate_result, load_protocol, write_json
from intact_tracking.memory350_policy_results import compare

ROOT = Path(__file__).resolve().parents[1]


def completed(root, fusion):
    path = root / "ppo" / f"{fusion}_121/endpoint_eval_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {row["completed_updates"]: row for row in rows if row.get("training_state_preserved")}


def report(root, update):
    protocol, files = load_protocol(root / "protocols/periodic.json")
    result = {"completed_updates": update, "motions": len(files), "cases": {}, "unix_time": time.time(),
              "scope": "interim periodic subset; final conclusions use the predeclared 4096-motion tests"}
    for case in protocol["cases"]:
        paths = [root / "ppo" / f"{f}_121/endpoint_eval/update_{update:06d}/{case}.json"
                 for f in ("baseline", "film")]
        rows = [json.loads(path.read_text()) for path in paths]
        traces = []
        for fusion, path, row in zip(("baseline", "film"), paths, rows):
            checkpoint = root / "ppo" / f"{fusion}_121/checkpoint_update_{update:06d}.pt"
            validate_result(row, protocol, files, digest(checkpoint), update, case)
            with np.load(path.with_suffix(".traces.npz")) as saved:
                if saved["lengths"].tolist() != row["episode_lengths"]:
                    raise ValueError("Periodic trace lengths changed")
                traces.append(saved["body_joint"].copy())
        result["cases"][case] = compare(*rows, *traces, repeats=2000)
    directory = root / "progress_comparisons"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"update_{update:06d}.json", result)
    lines = [f"# Memory350 PPO 第 {update} 轮配对测试", "",
             f"两组同为 {update} completed PPO updates，{len(files)} 个固定 motion/start/physics。"
             "下表为周期测试子集，最终结论使用第 5000 轮预先固定的 4096-motion 评估。", "",
             "| 场景 | Body baseline → latent (m) | 共同有效时段 body 降幅（95% CI） | Joint 降幅 | 失败率 baseline → latent | 覆盖率 baseline → latent |",
             "|---|---|---|---:|---|---|"]
    brief = {}
    for case, metrics in result["cases"].items():
        body, joint = metrics["common_error_body_pos"], metrics["common_error_joint_pos"]
        raw, fail, coverage = metrics["truncated_error_body_pos"], metrics["failure_rate"], metrics["coverage"]
        lo, hi = body["reduction_percent_ci95"]
        lines.append(f"| {case} | {raw['reference']:.5f} → {raw['candidate']:.5f} | "
                     f"{body['reduction_percent']:+.2f}% [{lo:+.2f}, {hi:+.2f}] | {joint['reduction_percent']:+.2f}% | "
                     f"{fail['reference']:.2%} → {fail['candidate']:.2%} | "
                     f"{coverage['reference']:.2%} → {coverage['candidate']:.2%} |")
        brief[case] = {"common_body_reduction_pct": body["reduction_percent"],
                       "common_joint_reduction_pct": joint["reduction_percent"],
                       "failure_delta_percentage_points": 100 * fail["candidate_minus_reference"]}
    lines += ["", "降幅为正代表 latent 误差较小；为负代表较大。共同有效时段使用两组都仍在追踪的步数，"
              "并单独报告提前失败和有效时长。区间按 motion 配对 bootstrap，不包含训练 seed 波动。", ""]
    temporary = ROOT / "docs/memory350_ppo_progress.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(ROOT / "docs/memory350_ppo_progress.md")
    print(json.dumps({"event": "paired_periodic_comparison", "completed_updates": update, "metrics": brief},
                     allow_nan=False), flush=True)


def run(root, watch):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Report outputs must stay inside the project")
    while True:
        common = sorted(set(completed(root, "baseline")) & set(completed(root, "film")))
        for update in common:
            if update > 0 and not (root / f"progress_comparisons/update_{update:06d}.json").exists():
                report(root, update)
        if not watch or (common and common[-1] >= 5000):
            return
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    run(args.run_root, args.watch)
