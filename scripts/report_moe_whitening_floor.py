"""Summarize the controlled whitening-floor experiment and paired uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ablate_moe_whitening_floor import FLOORS, KMEANS_SEEDS, TEMPERATURE_MODES, read_npz, save_json, sha


def summarize(root, tag, draws):
    results = json.loads((root/tag/"results.json").read_text())
    saved_targets = read_npz(root/tag/"targets.npz")
    targets = saved_targets["targets"].astype(float)
    worlds = saved_targets["worlds"]
    assert np.array_equal(worlds, np.repeat(np.arange(1024), 32))
    world_targets = targets.reshape(1024, 32, 8).mean(1)
    bootstrap_variance = draws@np.square(world_targets)/1024-np.square(draws@world_targets/1024)
    aggregates, bootstrap, bootstrap_load, checks = [], {}, {}, {}
    for mode in TEMPERATURE_MODES:
        for floor in sorted({row["floor"] for row in results["rows"]}, reverse=True):
            rows = [row for row in results["rows"] if row["temperature_mode"] == mode and row["floor"] == floor]
            rows.sort(key=lambda row: row["kmeans_seed"])
            assert [row["kmeans_seed"] for row in rows] == KMEANS_SEEDS
            errors = []
            for row in rows:
                prediction = read_npz(root/tag/f"{row['name']}.npz")["prediction"].astype(float)
                mse = np.square(prediction-targets)
                actual_r2 = 1-mse.mean(0)/targets.var(0)
                expected = np.asarray(row["metrics"]["r2"])[[0, 1, 2, 3, 5, 6, 7, 8]]
                delta = float(np.abs(actual_r2-expected).max())
                assert delta < 1e-6
                checks[row["name"]] = delta
                errors.append(mse.reshape(1024, 32, 8).mean(1))
            # Average losses over the fixed3 KMeans initializations, then resample
            # complete worlds. This interval does not cover all initialization uncertainty.
            world_error = np.mean(errors, axis=0)
            samples_by_coordinate = 1-(draws@world_error/1024)/bootstrap_variance
            samples = samples_by_coordinate.mean(-1)
            bootstrap[mode, floor] = samples
            bootstrap_load[mode, floor] = samples_by_coordinate[:, :4].mean(-1)
            values = [row["metrics"]["strong8_r2"] for row in rows]
            load_values = [row["metrics"]["load_r2"] for row in rows]
            coordinate = np.mean([row["metrics"]["r2"] for row in rows], axis=0)
            record = {"trace": tag, "floor": floor, "floor_percent": 100*floor,
                      "temperature_mode": mode, "mean8_R2": float(np.mean(values)),
                      "min_R2_across_kmeans_seeds": min(values), "max_R2_across_kmeans_seeds": max(values),
                      "per_seed_R2": dict(zip(map(str, KMEANS_SEEDS), values)),
                      "mean_load_R2": float(np.mean(load_values)),
                      "per_seed_load_R2": dict(zip(map(str, KMEANS_SEEDS), load_values)),
                      "per_coordinate_R2": coordinate.tolist(),
                      "R2_world_bootstrap_95pct": np.quantile(samples, [.025, .975]).tolist(),
                      "mean_maximum_expert_weight": float(np.mean([row["gate"]["mean_maximum_weight"] for row in rows])),
                      "mean_decoder_prototype_span": float(np.mean([row["amplification"]["strong8_mean_prototype_span_in_DR_ranges"] for row in rows]))}
            if "stability" in rows[0]:
                record["mean_gate_TV_per_step"] = float(np.mean([row["stability"]["mean_gate_TV"] for row in rows]))
                record["mean_decoded_DR_change_RMS_per_step"] = float(np.mean([row["stability"]["mean_decoded_DR_change_RMS_in_full_ranges"] for row in rows]))
            aggregates.append(record)
    for row in aggregates:
        base = next(x for x in aggregates if x["temperature_mode"] == row["temperature_mode"] and x["floor"] == .01)
        delta = bootstrap[row["temperature_mode"], row["floor"]]-bootstrap[row["temperature_mode"], .01]
        row["R2_difference_from_1pct"] = row["mean8_R2"]-base["mean8_R2"]
        row["difference_world_bootstrap_95pct"] = np.quantile(delta, [.025, .975]).tolist()
        row["per_seed_R2_difference"] = {seed: row["per_seed_R2"][seed]-base["per_seed_R2"][seed] for seed in map(str, KMEANS_SEEDS)}
        load_delta = bootstrap_load[row["temperature_mode"], row["floor"]]-bootstrap_load[row["temperature_mode"], .01]
        row["load_R2_difference_from_1pct"] = row["mean_load_R2"]-base["mean_load_R2"]
        row["load_difference_world_bootstrap_95pct"] = np.quantile(load_delta, [.025, .975]).tolist()
        row["per_seed_load_R2_difference"] = {seed: row["per_seed_load_R2"][seed]-base["per_seed_load_R2"][seed] for seed in map(str, KMEANS_SEEDS)}
    return {"aggregates": aggregates, "independent_recalculation_errors": checks,
            "bootstrap_replicates": len(draws), "resampling_unit": "whole DR world;32 times remain together",
            "uncertainty_scope": "Conditional on the frozen source policy and3 specified KMeans seeds; these are not3 PPO training seeds."}


def plot(root, summary, spectral):
    rows = summary["heldout30403"]["aggregates"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for mode, label, color in [("fixed_absolute", "Fixed temperature = 18.4013", "#2378a8"),
                               ("fixed_relative", "Fixed relative temperature multiplier = 3", "#c07835")]:
        selected = sorted([x for x in rows if x["temperature_mode"] == mode], key=lambda x: x["floor"], reverse=True)
        x = np.asarray([row["floor_percent"] for row in selected])
        y = np.asarray([row["mean8_R2"] for row in selected])
        axes[0, 0].plot(x, y, marker="o", markersize=4, label=label, color=color)
        axes[0, 0].fill_between(x, [row["min_R2_across_kmeans_seeds"] for row in selected],
                               [row["max_R2_across_kmeans_seeds"] for row in selected], color=color, alpha=.13)
    axes[0, 0].set_title("Held-out seed30403: mean R² of8 DR coordinates")
    axes[0, 0].set_ylabel("Mean R²; shade = range across3 KMeans seeds")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fixed = sorted([row for row in rows if row["temperature_mode"] == "fixed_absolute"], key=lambda x: x["floor"], reverse=True)
    x = [row["floor_percent"] for row in fixed]
    for index, label in enumerate(["Left hand", "Right hand", "Left shin", "Right shin"]):
        axes[0, 1].plot(x, [row["per_coordinate_R2"][index] for row in fixed], marker="o", markersize=3, label=label)
    axes[0, 1].set_title("Limb load R² | fixed numerical temperature")
    axes[0, 1].set_ylabel("Mean over3 KMeans initializations")
    axes[0, 1].legend(fontsize=8)
    near = [row for row in fixed if row["floor"] >= .003]
    for seed in map(str, KMEANS_SEEDS):
        axes[1, 0].plot([row["floor_percent"] for row in near], [row["per_seed_R2"][seed] for row in near],
                       marker="o", label=f"KMeans seed {seed}", alpha=.7)
    axes[1, 0].plot([row["floor_percent"] for row in near], [row["mean8_R2"] for row in near],
                   color="black", linewidth=2, label="Three-seed mean")
    axes[1, 0].set_title("Near1%: small floor effects vs center initialization")
    axes[1, 0].set_ylabel("Mean8 DR R² on held-out seed30403")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([100*row["floor"] for row in spectral],
                   [100*row["transformed_within_world_variance_fraction"] for row in spectral], marker="o", color="#a34b65")
    axes[1, 1].set_title("Share of whitened variance occurring within one DR world")
    axes[1, 1].set_ylabel("Percent | fit worlds, before EMA")
    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("Variance floor as percent of largest eigenvalue (lower →)")
        axis.grid(alpha=.2)
    figure.suptitle("Whitening floor ablation |16 continuous experts | EMA2s | frozen payload-trained context")
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(root/("whitening_floor_comparison"+suffix), dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_findings(root, summary, spectral):
    protocol = json.loads((root/"protocol.json").read_text())
    selection = json.loads((root/"selection.json").read_text())
    online = json.loads((root/"online_audit.json").read_text())
    def row(tag, floor, mode="fixed_absolute"):
        return next(x for x in summary[tag]["aggregates"] if x["floor"] == floor and x["temperature_mode"] == mode)
    base, lower = row("heldout30403", .01), row("heldout30403", .008)
    runtime_base, runtime_lower = row("runtime50hz30404", .01), row("runtime50hz30404", .008)
    lines = [
        "**白化方差下限实验，2026-09-15**",
        "",
        "小幅降至 0.8% 能略微改善四肢负载读出；主要八项 DR 的总体分数基本持平。继续降低通常恶化，0.2% 及以下出现明显下降。没有发现通过大幅降低下限获得更强 DR 路由的证据。",
        "",
        "此处的百分数均相对于最大协方差特征值：1% 对应代码参数 0.01，0.8% 对应 0.008，0.01% 对应 0.0001。R² 是从 router 输出预测 DR 参数的回归分数，不能解读成环境分类准确率或 PPO tracking 收益。",
        "",
        f"固定 16 个完整环境中心、连续混合全部 16 个 head、EMA 2 秒。主比较把温度数值固定在 {protocol['fixed_absolute_temperature']:.8f}；辅助比较保持温度为各模型中心最近邻平方距离中位数的 3 倍，以控制白化改变距离尺度的影响。每个下限用同一 KMeans 配置重新拟合中心，重复种子 731、1731、2731，各自 3 次重启，最多 60 次迭代。",
        "",
        "context 和原 policy 均冻结，复用实际交互轨迹，未训练新 PPO、未改生产 router。仍使用带四肢负载训练的 nominal50 Memory350 扩容 context、response10/predictor5、update15000，以及原共享 MoE update1000。当前交互的手部负载上限 2.5 kg、胫部上限 4 kg，完整 129827 条 motion 目录、uniform 采样。",
        "",
        f"context SHA256：`{protocol['context_sha256']}`。policy SHA256：`{protocol['policy_checkpoint_sha256']}`。",
        "",
        "seed30402 的 768 个完整 world 用于估计白化统计、拟合 KMeans 和评估用线性读出器，其余 256 个用于选择读出器正则化和更低下限候选。每个 world 使用相同的 32 个记忆充分、policy step 至少 500 的时刻。按三次 KMeans 初始化的平均验证分数，从更低下限中预选出 0.8%；它在验证集的总体分数也未超过 1% 基线。候选先封存，再评估独立 world/交互种子30403和30404。它们是已保存的旧轨迹，不是本次新采集的轨迹，没有进入本次拟合和选参。",
        "",
        "白化矩阵和 KMeans 不使用数值 DR 标签；真实 DR 只用于读出评估及验证选参。下面是 seed30403 的三次 KMeans 初始化均值，所以 1% 基线与此前单次报告的 0.882 略有不同。八项包括四肢负载、三轴 torso COM 和足底摩擦；全部 67 项参数也保留在原始结果中。",
        "",
        "| 方差下限 | 八项平均 R²，固定温度 | 八项平均 R²，固定温度比例 | 四肢平均 R²，固定温度 |",
        "|---|---:|---:|---:|",
    ]
    for floor in FLOORS:
        a, b = row("heldout30403", floor), row("heldout30403", floor, "fixed_relative")
        lines.append(f"| {floor*100:g}% | {a['mean8_R2']:.4f} | {b['mean8_R2']:.4f} | {a['mean_load_R2']:.4f} |")
    lines.extend([
        "",
        f"固定数值温度时，0.8% 相对 1% 的八项平均 R² 增益为 {lower['R2_difference_from_1pct']:+.6f}，95% world-bootstrap 区间为 [{lower['difference_world_bootstrap_95pct'][0]:+.6f}, {lower['difference_world_bootstrap_95pct'][1]:+.6f}]。seed30404 的对应增益为 {runtime_lower['R2_difference_from_1pct']:+.6f}，区间为 [{runtime_lower['difference_world_bootstrap_95pct'][0]:+.6f}, {runtime_lower['difference_world_bootstrap_95pct'][1]:+.6f}]。两个区间都包含零，总体指标没有显示清楚的改善。",
        "",
        f"四肢负载分项略有不同：seed30403 从 {base['mean_load_R2']:.4f} 提升到 {lower['mean_load_R2']:.4f}，差值区间为 [{lower['load_difference_world_bootstrap_95pct'][0]:+.6f}, {lower['load_difference_world_bootstrap_95pct'][1]:+.6f}]；seed30404 从 {runtime_base['mean_load_R2']:.4f} 提升到 {runtime_lower['mean_load_R2']:.4f}，差值区间为 [{runtime_lower['load_difference_world_bootstrap_95pct'][0]:+.6f}, {runtime_lower['load_difference_world_bootstrap_95pct'][1]:+.6f}]。这说明负载读出有小幅收益，但不能据此推导实际动作控制收益。",
        "",
        "| seed30403 分项 R² | 左手 | 右手 | 左胫 | 右胫 | COM z |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, data in [("1%", base), ("0.8%", lower)]:
        lines.append("| "+label+" | "+" | ".join(f"{data['per_coordinate_R2'][i]:.4f}" for i in [0,1,2,3,7])+" |")
    lines.extend([
        "",
        "COM z 等分项下降抵消了负载改善，不能把总体持平理解成每个因素都没有变化。置信区间采用 2000 次配对 world-bootstrap，完整重采样 world，每个 world 的 32 个时刻保持在一起，并对固定的三次 KMeans 初始化平均损失。它没有覆盖重新训练 PPO 的不确定性，也不能代表所有 KMeans 初始化；初始化之间的原始差异另存于 summary.json。",
        "",
        "![参数扫描图](whitening_floor_comparison.png)",
        "",
        "50 Hz 复核在 seed30404、1024 个 world、1500 个 policy step 上逐步回放。按预先封存的规则，仅检查 1%、0.8% 以及 0.01% 的强白化参照，均重复三次 KMeans 初始化。",
        "",
        "| 方差下限 | 八项平均 R² | 四肢平均 R² | 每 20ms 平均权重 TV | 每 20ms 解码 DR 变化 RMS |",
        "|---|---:|---:|---:|---:|",
    ])
    for floor in selection["runtime50hz_floors"]:
        data = row("runtime50hz30404", floor)
        lines.append(f"| {100*floor:g}% | {data['mean8_R2']:.4f} | {data['mean_load_R2']:.4f} | {data['mean_gate_TV_per_step']:.7f} | {data['mean_decoded_DR_change_RMS_per_step']:.7f} |")
    lines.extend([
        "",
        "权重 TV 定义为 0.5×各权重变化绝对值之和；解码 DR RMS 用各参数完整范围归一化，可辅助识别信息藏在微小权重差异里的情形。0.8% 相比 1% 的稳定性基本接近。种子731的三个主比较模型还运行了实际 PyTorch 有状态 router，与 NumPy 回放最大权重差异均小于 6e-7。",
        "",
        "另外用当前每 24 步更新一次中心、更新率 0.01、平均权重 TV 限制 0.02 的设置，对 1% 和 0.8% 做了在线中心更新回放。两者从种子731的初始中心出发，各完成62次更新，旧 rollout gate 保持不变，router 没有可训练参数。读出器保持更新前的参数：",
        "",
        "| 在线更新后的下限 | 八项 R² | 四肢负载 R² |",
        "|---|---:|---:|",
    ])
    for data in online["rows"]:
        lines.append(f"| {data['floor']*100:g}% | {data['metrics']['strong8_r2']:.4f} | {data['metrics']['load_r2']:.4f} |")
    lines.extend([
        "",
        "在线更新后仍保留“负载略好、总体接近”的相对趋势。两个模型的绝对分数均略降，其中包含中心变化后与冻结读出器的失配；这不等于证明信息不可逆丢失。该检查只有30秒固定原policy轨迹，不能代替长期 PPO 与中心共同变化的验证。",
        "",
        "更低下限会提高同一 DR world 内部变化在距离里的占比。对拟合样本、在 EMA 之前做方差分解，该占比随下限变化如下：",
        "",
        "| 下限 | 同一 world 内部变化占白化后总方差 |",
        "|---|---:|",
    ])
    for floor in [.01, .008, .001, .0001, .000001]:
        data = next(x for x in spectral if x["floor"] == floor)
        lines.append(f"| {100*floor:g}% | {100*data['transformed_within_world_variance_fraction']:.2f}% |")
    lines.extend([
        "",
        "这些内部变化包括 motion、交互历史和测量等因素，并不能全部称为无用噪声。不过，它们在当前16中心度量中获得更大权重，与 DR 读出退化同时出现。白化变换本身保持64维、在精确算术下可逆；本次分数下降发生在其后的有限中心路由及读出环节，不能解释成原 latent 里的负载信息凭空消失。",
        "",
        "如果主要关注四肢负载，可保留 floor=0.008 作为小幅调整候选；若关注八项总体效果，0.01 与 0.008 目前基本相当。不支持继续大幅降低下限，也未声称连续区间内所有可能下限都已被穷尽。生产训练配置没有修改。",
        "",
        "记录包含 protocol.json、selection_seal.json、完整 comparison.csv、summary.json、逐候选读出预测、50Hz PyTorch 对照、online_audit.json，以及 verification.json / execution_manifest.json。评估脚本为 scripts/ablate_moe_whitening_floor.py，在线检查为 scripts/audit_moe_whitening_floor_online.py，报告生成脚本为 scripts/report_moe_whitening_floor.py。",
        "",
        "复算现有已封存模型的评估使用：",
        "",
        "```bash",
        f"OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/ablate_moe_whitening_floor.py --output {root} --stage evaluate",
        f"OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/audit_moe_whitening_floor_online.py --root {root}",
        f"OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/report_moe_whitening_floor.py --root {root}",
        "```",
        "",
        "若要从头拟合，使用 --stage fit 并指定新的输出目录，避免覆盖已经封存的选择记录。",
    ])
    content = "\n".join(lines)+"\n"
    (root/"FINDINGS.md").write_text(content)
    doc = Path("docs/moe_whitening_floor_ablation_20260915.md")
    doc.write_text(content.replace("](whitening_floor_comparison.png)", f"](../{root}/whitening_floor_comparison.png)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    seal = json.loads((root/"selection_seal.json").read_text())
    for name in ("selection", "fit_results", "protocol"):
        assert sha(root/f"{name}.json") == seal[f"{name}_sha256"]
    draws = np.random.default_rng(8119).multinomial(1024, np.full(1024, 1/1024), size=2000).astype(float)
    summary = {tag: summarize(root, tag, draws) for tag in ("heldout30403", "runtime50hz30404")}
    save_json(root/"summary.json", summary)
    spectral = json.loads((root/"spectral_floor_effects.json").read_text())
    plot(root, summary, spectral)
    if (root/"online_audit.json").exists():
        write_findings(root, summary, spectral)
    csv_rows = []
    for tag, result in summary.items():
        for row in result["aggregates"]:
            flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for index, label in enumerate(["left_hand", "right_hand", "left_shin", "right_shin"]):
                flat[label+"_R2"] = row["per_coordinate_R2"][index]
            flat["paired_difference_CI_low"], flat["paired_difference_CI_high"] = row["difference_world_bootstrap_95pct"]
            csv_rows.append(flat)
    keys = list(dict.fromkeys(key for row in csv_rows for key in row))
    with (root/"comparison.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    diagnostic = {tag: [row for row in report["aggregates"] if row["floor"] in (.01, .008, .0001)
                        and row["temperature_mode"] == "fixed_absolute"] for tag, report in summary.items()}
    print(json.dumps({tag: [{"floor": row["floor"], "R2": row["mean8_R2"],
                             "difference_CI": row["difference_world_bootstrap_95pct"],
                             "load_R2": row["mean_load_R2"], "load_difference_CI": row["load_difference_world_bootstrap_95pct"],
                             "per_seed_difference": row["per_seed_R2_difference"]} for row in rows]
                      for tag, rows in diagnostic.items()}, indent=2))


if __name__ == "__main__":
    main()
