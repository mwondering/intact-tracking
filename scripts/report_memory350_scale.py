"""Write the live matched-budget prediction comparison without modifying old runs."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def history(path):
    if not path.exists():
        return {}
    return {row["update"]: row for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n") and line.strip() for row in [json.loads(line)]}


def run(root):
    root = root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Report must remain inside the current project")
    contract = json.loads((root / "launch_contract.json").read_text())
    reference = Path(contract["reference"])
    original = history(reference / "metrics.jsonl")
    scaled = history(root / "stage1_8192/metrics.jsonl")
    common = sorted(set(original) & set(scaled))
    nominal50 = contract.get('nominal_a_fraction') == .5
    primary_group = 'dr' if nominal50 else 'all'
    lines = ["# Memory350 encoder 参数扩容：预测误差对照", "",
        "原版 encoder 1,035,328 参数；扩容版 2,026,688（1.9575 倍）。",
        "仅将三段 Transformer 深度由 1/2/2 改为 2/4/4；predictor 保持 19,059,798 参数，latent 保持 64 维。",
        "两版从头训练，使用相同 predictor/公共层初始权重、训练配置、归一化和固定验证样本。", "",
        "主要比较预先固定在 22700 updates；7500 及其他同轮 checkpoint 为阶段性结果。",
        "按相同优化步数比较，扩容版计算量增加；这是一个训练种子的容量实验。", ""]
    if nominal50:
        lines += ['两版 A 环境均为 50% clean nominal + 50% tracker DR/四肢负载。',
                  '主表比较 DR 验证误差；nominal 单独列在分组结果中。原版 GPU 0–3，2x GPU 4–7。', '']
    if not common:
        lines += ["尚无扩容版训练后的验证结果。资源排队状态见 state.json。", ""]
    else:
        update = common[-1]
        a = original[update]["fixed_probe"]["dr_five_step_nmse"]
        b = scaled[update]["fixed_probe"]["dr_five_step_nmse"]
        lines += [f"最新相同训练轮次：**{update}**。训练日志口径（四 rank NMSE 平均）：",
            f"原版 **{a:.8f}** → 扩容版 **{b:.8f}**，误差变化 **{100 * (b / a - 1):+.2f}%**。", "",
            "负的误差变化表示扩容版更好。尚未到 22700 轮时，不据中期方向宣称最终收敛优势。", ""]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for name, rows in (("Original: 1.04M context", original), ("Encoder2x: 2.03M context", scaled)):
            xs = [value for value in sorted(rows) if value <= update]
            axis.plot(xs, [rows[value]["fixed_probe"]["dr_five_step_nmse"] for value in xs], label=name)
        axis.set(xlabel="Completed training updates", ylabel="Held-out five-step NMSE", yscale="log")
        axis.grid(alpha=.2); axis.legend(); figure.tight_layout()
        figure.savefig(root / "prediction_comparison.png", dpi=160)
        figure.savefig(root / "prediction_comparison.pdf")
        plt.close(figure)
        lines += ["![固定验证曲线](prediction_comparison.png)", ""]
    lines += ["## 逐窗口配对评估", "",
        "同一批 2048 个固定验证窗口；先合并窗口计算 NMSE，和训练日志的 rank 平均口径略有不同。",
        "区间通过按物理 world 聚类的配对 bootstrap 计算，仅反映验证样本不确定性。", "",
        "| Update | 原版五步 NMSE | 扩容版五步 NMSE | 误差降低 | 扩容/原版误差比 95% CI |",
        "|---|---:|---:|---:|---|"]
    comparisons = []
    for path in sorted((root / "comparison").glob("update_*/result.json")):
        result = json.loads(path.read_text())
        comparisons.append(result)
        row = result['groups'][primary_group]
        low, high = row["paired_world_bootstrap_ratio_ci95"]
        lines.append(f'| {result["update"]} | {row["original_five_step_nmse"]:.8f} | '
                     f'{row["encoder2x_five_step_nmse"]:.8f} | {row["error_reduction_percent"]:+.2f}% | '
                     f'[{low:.4f}, {high:.4f}] |')
    if comparisons:
        latest = comparisons[-1]
        lines += ["", f'最新配对 checkpoint：{latest["update"]}。分组如下：', "",
            "| 分组 | 窗口数 | 原版 NMSE | 扩容版 NMSE | 误差降低 |",
            "|---|---:|---:|---:|---:|"]
        for group, row in latest["groups"].items():
            lines.append(f'| {group} | {row["samples"]} | {row["original_five_step_nmse"]:.8f} | '
                         f'{row["encoder2x_five_step_nmse"]:.8f} | {row["error_reduction_percent"]:+.2f}% |')
        primary = next((value for value in comparisons if value["update"] == 22700), None)
        if primary:
            row = primary['groups'][primary_group]
            low, high = row["paired_world_bootstrap_ratio_ci95"]
            if high < 1:
                conclusion = "在本次相同训练预算和验证协议下，扩容版降低了五步预测误差。"
            elif low > 1:
                conclusion = "在本次相同训练预算和验证协议下，扩容版的五步预测误差更高。"
            else:
                conclusion = "本次误差比区间包含 1，尚不能确认扩容带来了预测误差改善。"
            lines += ["", "**22700 轮预定主要比较：** " + conclusion,
                "该结果不证明更大模型已经达到其能力上限，也不代表跨训练种子的稳定结论。"]
    wandb = root / "stage1_8192/wandb_run.json"
    if wandb.exists():
        url = json.loads(wandb.read_text())["url"]
        lines += ["", f"[扩容版训练 W&B]({url})"]
    lines += ["", "原始逐窗口数据和分组区间见 comparison/update_*/paired_samples.npz、result.json。", ""]
    (root / "report.md").write_text("\n".join(lines))
    (root / "comparison_summary.json").write_text(json.dumps({
        "latest_common_training_update": common[-1] if common else None,
        "primary_comparison_update": 22700, "paired_evaluations": comparisons,
    }, indent=2, allow_nan=False) + "\n")
    print(str(root / "report.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    run(parser.parse_args().run_root)
