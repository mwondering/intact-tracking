"""Export matched PPO results; read-only with respect to training and evaluation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = (
    ("mixture_cold", "Mixture, cold", "#0072B2", "--"),
    ("mixture_warm", "Mixture, warm", "#D55E00", "-"),
)
DIAGNOSTIC = (
    ("all_0_cold", "0 kg, cold", "#0072B2", "--"),
    ("all_0_warm", "0 kg, warm", "#0072B2", "-"),
    ("all_4_cold", "4 kg, cold", "#D55E00", "--"),
    ("all_4_warm", "4 kg, warm", "#D55E00", "-"),
)
WARM = (
    ("mixture_warm", "Mixture, warm", "#0072B2", "-"),
    ("all_0_warm", "0 kg, warm", "#009E73", "-"),
    ("all_4_warm", "4 kg, warm", "#D55E00", "-"),
)
WARM_BOUNDED = (
    ("mixture_warm", "Mixture, warm", "#0072B2", "-"),
    ("all_0_warm", "0 kg, warm", "#009E73", "-"),
    ("upper_train_warm", "Hands 2.5 / shins 4 kg, warm", "#D55E00", "-"),
)
PANELS = (
    ("common_error_body_pos", "Local body position error", "error"),
    ("common_error_joint_pos", "Joint position error", "error"),
    ("common_error_body_pos_global", "Global body position error", "error"),
    ("common_error_anchor_pos_global", "Global anchor position error", "error"),
    ("failure_rate", "Failure rate", "rate"),
    ("coverage", "Coverage", "rate"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change(result, mode):
    if mode == "error":
        low, high = result["reduction_percent_ci95"]
        return -result["reduction_percent"], -high, -low
    low, high = result["difference_ci95"]
    return 100 * result["candidate_minus_reference"], 100 * low, 100 * high


def export_figure(rows, cases, stem, title):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    updates = [row["completed_updates"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), sharex=True)
    for axis, (metric, label, mode) in zip(axes.flat, PANELS):
        for case, legend, color, style in cases:
            values = [change(row["comparisons"][case][metric], mode) for row in rows]
            mean, low, high = zip(*values)
            axis.plot(updates, mean, color=color, linestyle=style, marker="o",
                      markersize=3, linewidth=1.7, label=legend)
            axis.fill_between(updates, low, high, color=color, alpha=.10)
        axis.axhline(0, color="#666666", linewidth=.8)
        axis.grid(alpha=.18)
        axis.set_title(label, fontsize=11)
        direction = "Higher favors latent" if metric == "coverage" else "Lower favors latent"
        unit = "Error change (%)" if mode == "error" else "Change (percentage points)"
        axis.set_ylabel(f"{unit}\n{direction}", fontsize=9)
        axis.set_xlim(0, max(25, updates[-1] * 1.02))
        if len(updates) <= 8:
            axis.set_xticks(updates)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.tick_params(labelsize=9)
        for edge in ("top", "right"):
            axis.spines[edge].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Completed PPO updates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .948),
               ncol=len(cases), frameon=False)
    fig.suptitle(f"{title} | matched updates through u{updates[-1]}", y=.987, fontsize=14)
    fig.text(.5, .047, "Error panels use the common surviving steps of each paired episode.",
             ha="center", fontsize=9)
    fig.text(.5, .030, "512 paired motions; pointwise 95% motion-bootstrap intervals; one training seed.",
             ha="center", fontsize=9)
    detail = ("Uniform motion sampling throughout; frozen soft encoder u20000; 4 x 8192 environments per arm."
              if cases == PRIMARY else
              "0 kg retains background DR; 4 kg hand loads exceed the 2.5 kg training maximum.")
    if cases == WARM_BOUNDED:
        detail = "Upper loads: 2.5 kg per hand and 4 kg per shin; 0 kg retains background DR."
    fig.text(.5, .013, detail, ha="center", fontsize=9)
    fig.subplots_adjust(top=.885, bottom=.12, left=.09, right=.985, hspace=.32, wspace=.30)
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--through-update", type=int)
    parser.add_argument("--warm-only", action="store_true",
                        help="Export only warm comparisons; leave evaluation and training unchanged.")
    parser.add_argument("--warm-bounded", action="store_true",
                        help="Use the corrected warm protocol with hand/shin upper masses 2.5/4 kg.")
    args = parser.parse_args()
    if args.warm_bounded:
        args.warm_only = True
    root = args.run_root.resolve()
    paired_directory = "paired_results_warm_bounded_v2" if args.warm_bounded else "paired_results"
    files = sorted((root / paired_directory).glob("update_*.json"))
    if args.through_update is not None:
        files = [path for path in files if int(path.stem.split("_")[-1]) <= args.through_update]
    if not files:
        raise ValueError("No completed paired results")
    rows = [json.loads(path.read_text()) for path in files]
    updates = [row["completed_updates"] for row in rows]
    if updates != sorted(set(updates)):
        raise ValueError("Duplicate or unordered matched updates")
    if args.through_update is not None and updates[-1] != args.through_update:
        raise ValueError("The requested matched update has not completed")
    warm_cases = WARM_BOUNDED if args.warm_bounded else WARM
    expected_cases = {case[0] for case in (WARM_BOUNDED if args.warm_bounded else PRIMARY + DIAGNOSTIC)}
    for row in rows:
        if set(row["comparisons"]) != expected_cases or row["training_seed_count"] != 1:
            raise ValueError("Unexpected evaluation cases or seed count")
        if any(case["paired_motions"] != 512 for case in row["comparisons"].values()):
            raise ValueError("Unexpected paired motion count")
    out = root / "analysis"
    if args.warm_bounded:
        out = out / "warm_bounded_v2"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    suffix = f"u{updates[-1]:06d}"
    figures = (("warm", warm_cases, "Soft u20000 latent vs baseline: warm"),) if args.warm_only else (
        ("primary", PRIMARY, "Soft u20000 latent vs baseline: mixture DR"),
        ("diagnostic", DIAGNOSTIC, "Soft u20000 latent vs baseline: load diagnostics"))
    artifacts = []
    for name, cases, title in figures:
        stem = out / f"{name}_{suffix}"
        export_figure(rows, cases, stem, title)
        artifacts.extend(stem.with_suffix(extension) for extension in (".png", ".pdf"))
    selected_cases = {case[0] for case in warm_cases} if args.warm_only else expected_cases
    report_suffix = f"warm_{suffix}" if args.warm_only else suffix
    table = out / f"matched_metrics_{report_suffix}.csv"
    columns = ["completed_updates", "case", "metric", "baseline", "latent", "delta_native",
               "delta_ci95_low", "delta_ci95_high", "change", "change_ci95_low", "change_ci95_high", "change_unit"]
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for case, metrics in row["comparisons"].items():
                if case not in selected_cases:
                    continue
                for metric, result in metrics.items():
                    if not isinstance(result, dict) or "candidate_minus_reference" not in result:
                        continue
                    mode = "rate" if metric in ("coverage", "failure_rate") else "error"
                    value, low, high = change(result, mode)
                    writer.writerow(dict(zip(columns, [row["completed_updates"], case, metric,
                        result["reference"], result["candidate"], result["candidate_minus_reference"],
                        *result["difference_ci95"], value, low, high,
                        "percentage_points" if mode == "rate" else "percent"])))
    focus = "（仅 warm）" if args.warm_only else ""
    lines = [f"# 共同 u{updates[-1]} PPO 评测{focus}", "",
             "误差变化为 latent 相对 baseline 的百分比，负数更好；使用共同有效时段。", "",
             "| 场景 | 局部 body | joint | 全局 body 位置 | 全局 anchor 位置 | baseline / latent 失败率 | baseline / latent 覆盖率 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case, metrics in rows[-1]["comparisons"].items():
        if case not in selected_cases:
            continue
        errors = " | ".join(f"{change(metrics[metric], 'error')[0]:+.2f}%" for metric, _, _ in PANELS[:4])
        rates = " | ".join(f"{metrics[k]['reference']*100:.2f}% / {metrics[k]['candidate']*100:.2f}%"
                           for k in ("failure_rate", "coverage"))
        lines.append(f"| {case} | {errors} | {rates} |")
    lines += ["", "置信区间按 motion 配对重抽样计算，只反映这批 motion 的不确定性，不代表训练种子间方差。"
              "评测使用训练目录内的 motion 和留出的物理种子及起点；单个训练种子 121。", "",
              ("主要测试是 mixture warm；本报告只考虑 warm。" if args.warm_only else
               "主要测试是 mixture cold/warm。") +
              ("0kg 保留背景 DR；上限负载为双手各2.5kg、双小腿各4kg，总计13kg。旧的四处各4kg结果不参与本报告。"
               if args.warm_bounded else
               "0kg 仍保留背景 DR；4kg 手部负载超过训练上限 2.5kg，属于辅助分布外诊断。"), ""]
    for name, _, title in figures:
        lines += [f"![{title}]({name}_{suffix}.png)", ""]
    lines += [f"[全部所选场景指标与区间 CSV]({table.name})", ""]
    summary = out / f"summary_{report_suffix}.md"
    summary.write_text("\n".join(lines))
    artifacts.extend((table, summary))
    manifest = {"last_matched_update": updates[-1], "updates": updates,
                "inputs": {str(path.relative_to(root)): sha(path) for path in files},
                "script_sha256": sha(Path(__file__)), "motion_count": 512, "training_seed_count": 1,
                "summary": str(summary), "error_change_negative_favors": "latent",
                "focus": "warm_bounded_v2" if args.warm_bounded else "warm" if args.warm_only else "all_cases",
                "artifacts": {path.name: sha(path) for path in artifacts}}
    (out / f"manifest_{report_suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    latest = out / ("latest_warm.json" if args.warm_only else "latest.json")
    tmp = latest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(latest)
    print(json.dumps({"updates": updates, "summary": str(summary), "csv": str(table)}))


if __name__ == "__main__":
    main()
