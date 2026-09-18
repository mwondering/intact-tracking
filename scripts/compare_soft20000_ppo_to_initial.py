"""Compare completed PPO evaluations with each arm's paired initial policy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from intact_tracking.memory350_policy_results import compare


def read_evaluation(path, expected_update):
    row = json.loads(path.read_text())
    if row["completed_training_updates"] != expected_update:
        raise ValueError(f"Unexpected checkpoint update: {path}")
    trace_path = path.with_suffix(".traces.npz")
    with np.load(trace_path) as saved:
        if saved["lengths"].tolist() != row["episode_lengths"]:
            raise ValueError(f"Trace lengths do not match: {path}")
        traces = saved["all_metrics"].copy()
    with trace_path.open("rb") as handle:
        trace_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    evidence = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_sha256": trace_sha,
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    return row, traces, evidence


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--warm-bounded", action="store_true")
    args = parser.parse_args()
    if args.update <= 0:
        raise ValueError("Expected a trained checkpoint update greater than zero")
    root = args.run_root.resolve()
    paired = root / ("paired_results_warm_bounded_v2" if args.warm_bounded else "paired_results") / f"update_{args.update:06d}.json"
    report = json.loads(paired.read_text())
    if report["completed_updates"] != args.update:
        raise ValueError("The matched evaluation has not completed")
    cases = list(report["comparisons"])
    output = {
        "reference_update": 0,
        "candidate_update": args.update,
        "training_seed_count": 1,
        "training_seed": report["training_seed"],
        "primary_cases": report["primary_cases"],
        "scope": report["scope"],
        "uncertainty": report["uncertainty"],
        "matched_report_sha256": hashlib.sha256(paired.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparisons": {},
        "inputs": {},
    }
    metrics = ("common_error_body_pos", "common_error_joint_pos",
               "common_error_body_pos_global", "common_error_anchor_pos_global")
    lines = [f"# u{args.update} 相对各自 u0 初始策略", "",
             "同一组、同一场景的 u0 与训练后策略配对比较；误差百分比为负表示改善。"
             "每一对使用自己的共同有效时段，因此不能直接由另一张组间比较表相除得到。", "",
             "| 组 | 场景 | 局部 body | joint | 全局 body 位置 | 全局 anchor 位置 | 初始 / 当前失败率 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ("baseline", "concat"):
        output["comparisons"][arm] = {}
        output["inputs"][arm] = {}
        for case in cases:
            name = f"{arm}_121_warm_bounded" if args.warm_bounded else f"{arm}_121"
            parent = root / "ppo" / name / "endpoint_eval"
            initial, initial_traces, initial_evidence = read_evaluation(
                parent / "update_000000" / f"{case}.json", 0)
            current, current_traces, current_evidence = read_evaluation(
                parent / f"update_{args.update:06d}" / f"{case}.json", args.update)
            expected = report["inputs"][case][arm]
            if any(current_evidence[key] != expected[key] for key in ("sha256", "checkpoint_sha256")):
                raise ValueError("Current evaluation differs from the completed matched result")
            result = compare(initial, current, initial_traces, current_traces)
            output["comparisons"][arm][case] = result
            output["inputs"][arm][case] = {"initial": initial_evidence, "current": current_evidence}
            values = " | ".join(f"{-result[key]['reduction_percent']:+.2f}%" for key in metrics)
            failure = result["failure_rate"]
            label = "latent" if arm == "concat" else arm
            lines.append(f"| {label} | {case} | {values} | "
                         f"{100*failure['reference']:.2f}% / {100*failure['candidate']:.2f}% |")
            del initial_traces, current_traces
    lines += ["", "完整的原始指标、95% motion 配对 bootstrap 区间、失败率与覆盖率见同名 JSON。"
              "这是单个训练种子的结果，区间不包含训练种子间波动。", "",
              ("只考虑warm。上限负载为双手各2.5kg、双小腿各4kg；0kg保留背景DR。" if args.warm_bounded else
               "主要测试是 mixture cold/warm。0kg 保留背景 DR；4kg 手部负载超过训练上限 2.5kg。"), ""]
    directory = root / "analysis"
    if args.warm_bounded:
        directory = directory / "warm_bounded_v2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"vs_initial_u{args.update:06d}.json"
    write_json(path, output)
    path.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"update": args.update, "report": str(path.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
