"""Expand existing paired results into complete tracking tables and CSV."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/limb_context_20260910_memory350_ppo"
COMMON = ("common_error_body_pos", "common_error_joint_pos")


def run(root, update=None, final=False, control=False, supplemental=None):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Reports must remain inside the project")
    if final:
        source = root / "final_results.json"
        data = json.loads(source.read_text())
        if data["training_updates_per_arm"] != 5000:
            raise ValueError("Final results require 5000 completed updates per arm")
        comparisons = data["comparisons"]
        stem, title = "final_005000", "最终 5000 轮完整 tracking 指标"
    elif control:
        source = root / "artifacts/frozen_tracker_completion_audit/baseline_vs_frozen.json"
        data = json.loads(source.read_text())
        if data["training_updates"] != 5000:
            raise ValueError("Control comparison requires the completed 5000-update baseline")
        comparisons = {"baseline_vs_frozen_tracker": data["cases"]}
        stem = "control_baseline_005000"
        title = "冻结 tracker 对照（baseline 5000 轮；latent 最终结果待完成）"
    elif supplemental is not None:
        source = Path(supplemental).resolve()
        if not source.is_relative_to(root / "supplemental"):
            raise ValueError("Supplemental results must be inside this experiment's supplemental directory")
        data = json.loads(source.read_text())
        update = data["completed_updates"]
        comparisons = {"film_vs_baseline": data["cases"]}
        stem = f"supplemental_{source.parent.name}"
        title = f"第 {update} 轮补充 tracking 指标"
    else:
        source = (root / "progress_comparisons" / f"update_{update:06d}.json" if update is not None
                  else sorted((root / "progress_comparisons").glob("update_*.json"))[-1])
        data = json.loads(source.read_text())
        update = data["completed_updates"]
        comparisons = {"film_vs_baseline": data["cases"]}
        stem, title = f"update_{update:06d}", f"第 {update} 轮完整 tracking 指标（周期测试）"
    lines = [f"# Memory350 {title}", "",
             "film 为 Memory350 latent 策略；baseline 为无 latent residual 策略；frozen_tracker 为冻结 tracker。"
             "每个 comparison 的 `_vs_` 前为候选，后为对照。", "",
             "误差降幅为正表示候选误差更低；区间为按 motion 配对 bootstrap 的 95% 区间，"
             "不包含训练 seed 波动。body/joint 主指标使用两组共同有效时段；其余误差按各自有效跟踪步数统计，"
             "可能受到提前失败影响，需与失败率和覆盖率一起解释。", "",
             f"数据来源：`{source.relative_to(ROOT)}`。", ""]
    csv_rows = []
    for comparison, cases in comparisons.items():
        lines += [f"## {comparison}", ""]
        for case, metrics in cases.items():
            lines += [f"### {case}（{metrics['paired_motions']} motions）", "",
                      "| 指标 | 对照均值 | 候选均值 | 误差降幅（95% CI） | 统计时段 |",
                      "|---|---:|---:|---|---|"]
            order = list(COMMON) + sorted(key for key in metrics if key.startswith("truncated_error_"))
            for metric in order + ["failure_rate", "coverage"]:
                value = metrics[metric]
                error = metric.startswith(("common_error_", "truncated_error_"))
                aggregation = "common_survival" if metric in COMMON else ("own_survival" if error else "motion")
                row = {"comparison": comparison, "case": case, "metric": metric,
                       "aggregation": aggregation, "paired_motions": metrics["paired_motions"],
                       "reference": value["reference"], "candidate": value["candidate"],
                       "candidate_minus_reference": value["candidate_minus_reference"],
                       "difference_ci95_low": value["difference_ci95"][0],
                       "difference_ci95_high": value["difference_ci95"][1],
                       "error_reduction_percent": value["reduction_percent"] if error else "",
                       "error_reduction_percent_ci95_low": value["reduction_percent_ci95"][0] if error else "",
                       "error_reduction_percent_ci95_high": value["reduction_percent_ci95"][1] if error else ""}
                csv_rows.append(row)
                if error and metric not in ("truncated_error_body_pos", "truncated_error_joint_pos"):
                    label = metric.removeprefix("common_").removeprefix("truncated_")
                    low, high = value["reduction_percent_ci95"]
                    period = "共同有效时段" if metric in COMMON else "各自有效步数"
                    lines.append(f"| {label} | {value['reference']:.5f} | {value['candidate']:.5f} | "
                                 f"{value['reduction_percent']:+.2f}% [{low:+.2f}, {high:+.2f}] | {period} |")
            for metric, label in (("failure_rate", "失败率"), ("coverage", "覆盖率")):
                value = metrics[metric]
                low, high = (100 * number for number in value["difference_ci95"])
                lines += ["", f"{label}：对照 {value['reference']:.2%} → 候选 {value['candidate']:.2%}；"
                          f"候选减对照 {100 * value['candidate_minus_reference']:+.2f} 个百分点"
                          f"（95% CI {low:+.2f} 至 {high:+.2f}）。"]
            lines.append("")
    output = root / "artifacts/tracking_details"
    output.mkdir(parents=True, exist_ok=True)
    target = output / stem
    target.with_suffix(".md").write_text("\n".join(lines))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    target.with_suffix(".csv").write_text(stream.getvalue())
    provenance = {"created_at": time.time(), "source": str(source.relative_to(ROOT)),
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "rows": len(csv_rows), "comparisons": list(comparisons), "final": final,
                  "control_only": control, "supplemental": supplemental is not None}
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"markdown": str(target.with_suffix('.md')), "csv": str(target.with_suffix('.csv')),
                      "rows": len(csv_rows), "source": str(source)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", type=int)
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--control", action="store_true")
    mode.add_argument("--supplemental", type=Path)
    args = parser.parse_args()
    run(args.run_root, args.update, args.final, args.control, args.supplemental)
