"""Summarize matched Short50/Memory350 checkpoints without touching either trainer."""

import argparse
import csv
import io
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
SHORT = ROOT / "runs/limb_context_20260910_short50"
REFERENCE = ROOT / "runs/limb_context_20260909_memory350/stage1_8192"


def rows(path):
    result = {}
    if not path.exists():
        return result
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue  # A live trainer may still be writing the last row.
            value = json.loads(line)
            result[value["update"]] = value
    return result


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def report():
    reference = rows(REFERENCE / "metrics.jsonl")
    short = rows(SHORT / "stage1_8192/metrics.jsonl")
    matched = []
    for update in sorted(reference.keys() & short.keys()):
        memory_nmse = reference[update]["fixed_probe"]["dr_five_step_nmse"]
        short_nmse = short[update]["fixed_probe"]["dr_five_step_nmse"]
        matched.append({"update": update,
                        "optimizer_steps": short[update]["optimizer_steps"],
                        "memory350_nmse": memory_nmse, "short50_nmse": short_nmse,
                        "short50_to_memory350_ratio": short_nmse / memory_nmse})
    paired = []
    for path in sorted((SHORT / "comparison").glob("update_*/result.json")):
        value = json.loads(path.read_text())
        if value["update"] > 1:
            paired.append(value)
    progress = json.loads((SHORT / "stage1_8192/progress.json").read_text())
    recent = matched[-10:]
    monitoring = {"recent_validation_count": len(recent)}
    if recent:
        monitoring.update(
            recent_update_range=[recent[0]["update"], recent[-1]["update"]],
            recent_memory_lower_error_count=sum(r["memory350_nmse"] < r["short50_nmse"] for r in recent),
            recent_memory_relative_short_percent=(
                sum(r["memory350_nmse"] for r in recent) / sum(r["short50_nmse"] for r in recent) - 1) * 100,
        )
    if paired:
        monitoring["latest_paired_update"] = paired[-1]["update"]
        monitoring["memory_relative_short_percent_by_group"] = {
            name: (1 / group["short50_to_memory350_error_ratio"] - 1) * 100
            for name, group in paired[-1]["groups"].items()
        }
        lo, hi = paired[-1]["groups"]["no_long_in_reference"]["paired_world_bootstrap_ratio_ci95"]
        monitoring["no_long_memory_relative_short_percent_ci95"] = [(1 / hi - 1) * 100, (1 / lo - 1) * 100]
        monitoring["no_long_regression_ci_excludes_equal"] = hi < 1
    result = {"timestamp": time.time(), "short50_progress": progress,
              "predeclared_primary_update": 7500,
              "primary_evaluation_ready": any(r["update"] == 7500 for r in paired),
              "matched_curve": matched, "monitoring": monitoring,
              "paired_results": [{"update": r["update"], "groups": r["groups"],
                                  "rank_mean_nmse": r["rank_mean_nmse"]} for r in paired]}
    destination = SHORT / "comparison"
    atomic_text(destination / "summary.json", json.dumps(result, indent=2) + "\n")
    if matched:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(matched[0]))
        writer.writeheader()
        writer.writerows(matched)
        atomic_text(destination / "matched_curve.csv", output.getvalue())
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
        for name, color in [("Memory350", "#1976b5"), ("Short50", "#d95f02")]:
            axes[0].plot([r["update"] for r in matched],
                         [r[name.lower() + "_nmse"] for r in matched],
                         label=name, color=color, linewidth=1.7)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("5-step NMSE (rank mean; lower is better)")
        axes[0].legend()
        axes[0].set_title("Matched training: same DR, normalization and held-out windows")
        axes[1].plot([r["update"] for r in matched],
                     [r["short50_to_memory350_ratio"] for r in matched], color="#333333")
        axes[1].axhline(1, linestyle="--", color="#999999")
        axes[1].set_ylabel("Short50 / Memory350")
        axes[1].set_xlabel("Training updates (4 optimizer steps each)")
        for ax in axes:
            ax.grid(alpha=.18)
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"matched_curve.{extension}", dpi=180)
        plt.close(fig)
        if paired:
            fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
            series = {"all": ("All windows", "#333333"),
                      "short_incomplete_with_long": ("Short < 50, long available", "#1976b5"),
                      "short_full_with_long": ("Short = 50, long available", "#248f55"),
                      "no_long_in_reference": ("Long memory empty", "#d95f02")}
            for name, (label, color) in series.items():
                ax.plot([r["update"] for r in paired],
                        [(1 / r["groups"][name]["short50_to_memory350_error_ratio"] - 1) * 100 for r in paired],
                        marker="o", markersize=4, label=label, color=color)
            ax.axhline(0, linestyle="--", color="#999999")
            ax.set_xlabel("Training updates")
            ax.set_ylabel("Memory350 error change vs Short50 (%)")
            ax.set_title("Group monitoring: positive = Memory350 worse")
            ax.grid(alpha=.18)
            ax.legend(fontsize=9)
            for extension in ("png", "pdf"):
                fig.savefig(destination / f"memory_group_trends.{extension}", dpi=180)
            plt.close(fig)
    lines = ["# Short50 与 Memory350 的预测对照结果", "",
             "本页由 `scripts/report_short50_comparison.py` 根据已落盘结果更新。",
             f"Short50 当前完成 {progress['completed_updates']} updates；预先确定的重点结论点为 7500 updates。", "",
             "两组从头训练、相同 DR/数据/环境数/优化设置，测试使用完全相同的固定窗口和归一化。",
             "仅短期模型移除了长期模块；整体参数量随之减少。配置和核验见 [实验设计](short50_matched_control.md)。", "",
             "## 持续监测", ""]
    if recent:
        lines += [f"最近 {len(recent)} 个验证点（{recent[0]['update']}–{recent[-1]['update']} updates），"
                  f"Memory350 在 {monitoring['recent_memory_lower_error_count']} 个点误差较低；"
                  f"这几个点的平均误差相对 Short50 为 {monitoring['recent_memory_relative_short_percent']:+.2f}%。",
                  "相邻 checkpoint 高度相关，此处只描述趋势，不当作独立重复实验。", ""]
    if paired:
        latest = paired[-1]
        lines += [f"最近一次配对评估为 {latest['update']} updates。以下百分比采用 **Memory350 相比 Short50** 的方向：负值更好，正值更差。", "",
                  "| 分组 | Memory350 误差变化 |", "|---|---:|"]
        for name, title in [("all", "全部"), ("short_incomplete_with_long", "短期不足 50 步且已有长期历史"),
                            ("short_under10_with_long", "短期不足 10 步且已有长期历史"),
                            ("short_full_with_long", "短期完整 50 步且已有长期历史"),
                            ("no_long_in_reference", "长期历史为空")]:
            lines.append(f"| {title} | {monitoring['memory_relative_short_percent_by_group'][name]:+.2f}% |")
        lo, hi = monitoring["no_long_memory_relative_short_percent_ci95"]
        lines += ["", f"长期为空分组的误差变化 95% 配对区间为 [{lo:+.2f}%, {hi:+.2f}%]。",
                  "长期为空时，Memory350 的 memory token 本来就被 mask；该组退化反映无长期输入时的性能，不能解释成某段旧历史直接干扰了预测。",
                  "这些分组是已有固定样本，不能据此估计长期部署时各状态的出现比例。", "",
                  "![按长期历史可用性监测](../runs/limb_context_20260910_short50/comparison/memory_group_trends.png)", ""]
    lines += ["## 相同训练轮次", "",
             "五步 NMSE 按 rank 平均，与训练日志口径一致。误差比大于 1 表示 Short50 误差更高。", "",
             "| Update | Memory350 | Short50 | Short50 / Memory350 |",
             "|---:|---:|---:|---:|"]
    milestones = {100, 500, 1000, 3000, 5000, 7500}
    if matched:
        milestones.add(matched[-1]["update"])
    for r in matched:
        if r["update"] in milestones:
            lines.append(f"| {r['update']} | {r['memory350_nmse']:.6f} | {r['short50_nmse']:.6f} | {r['short50_to_memory350_ratio']:.4f} |")
    lines += ["", "![同轮次训练曲线](../runs/limb_context_20260910_short50/comparison/matched_curve.png)", "",
              "## 配对 checkpoint 评估", "",
              "下表合并所有 rank 的误差，分母为同一组样本的无变化基线误差；与上表先按 rank 归一化再平均的口径略有不同。", "",
              "| Update | 分组 | 窗口数 | Memory350 NMSE | Short50 NMSE | 误差比（95% 区间） |",
              "|---:|---|---:|---:|---:|---:|"]
    group_names = {"all": "全部", "short_incomplete_with_long": "短期不足 50 步且已有长期历史",
                   "short_under10_with_long": "短期不足 10 步且已有长期历史",
                   "short_full_with_long": "短期完整 50 步且已有长期历史",
                   "no_long_in_reference": "长期历史为空"}
    for r in paired:
        for name, title in group_names.items():
            g = r["groups"][name]
            lo, hi = g["paired_world_bootstrap_ratio_ci95"]
            lines.append(f"| {r['update']} | {title} | {g['samples']} | {g['memory350_nmse']:.6f} | {g['short50_nmse']:.6f} | {g['short50_to_memory350_error_ratio']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "区间由验证 world 配对 bootstrap 得到，不包含训练种子之间的不确定性。",
              "当前验证窗口来自固定的启动采集样本，并非完整 motion 数据集的逐条泛化评估。",
              "预测误差结论不等同于阶段二 PPO 性能结论。", ""]
    primary = next((r for r in paired if r["update"] == 7500), None)
    if primary is None:
        lines += ["7500 updates 的对照尚未完成；早期曲线不能用于宣布收敛后的优劣。", ""]
    else:
        g = primary["groups"]["all"]
        ratio = g["short50_to_memory350_error_ratio"]
        lo, hi = g["paired_world_bootstrap_ratio_ci95"]
        if lo > 1:
            conclusion = "这次匹配训练在固定验证集上支持长期 memory 降低预测误差。"
        elif hi < 1:
            conclusion = "这次匹配训练没有支持长期 memory 改善整体预测；仅短期版本的整体误差更低。"
        else:
            conclusion = "这次匹配训练尚不能区分两种版本的整体预测误差。"
        lines += [f"7500 updates 时，移除长期模块后的整体误差变化为 {(ratio-1)*100:+.2f}%。{conclusion}",
                  "这是单次训练对照，结论仍受随机种子、训练轨迹数值差异及移除模块后的容量变化限制。", ""]
    atomic_text(ROOT / "docs/short50_prediction_results.md", "\n".join(lines))
    print(json.dumps({"short50_update": progress["completed_updates"],
                      "latest_matched": matched[-1] if matched else None,
                      "paired_updates": [r["update"] for r in paired],
                      "monitoring": monitoring,
                      "primary_evaluation_ready": result["primary_evaluation_ready"]}), flush=True)
    return result["primary_evaluation_ready"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true",
                        help="Refresh on completed validation/evaluation changes until update 7500 is evaluated")
    args = parser.parse_args()
    last_signature = None
    while True:
        files = [SHORT / "stage1_8192/metrics.jsonl",
                 *(SHORT / "comparison").glob("update_*/result.json")]
        signature = tuple((str(p), p.stat().st_mtime_ns) for p in sorted(files) if p.exists())
        if signature != last_signature:
            try:
                finished = report()
            except (OSError, json.JSONDecodeError) as error:
                if not args.watch:
                    raise
                # Existing progress/result writers can briefly expose a partial
                # JSON file. Retry without changing the running trainers.
                print(f"Report will retry after file read: {error}", flush=True)
            else:
                last_signature = signature
                if finished or not args.watch:
                    break
        time.sleep(10)
