"""Summarize the expert DR proximity audit and draw physical parameter ranges."""

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


IDS = [0, 1, 2, 3, 5, 6, 7, 8]
LABELS = ["Left hand (kg)", "Right hand (kg)", "Left shin (kg)", "Right shin (kg)",
          "COM x (cm)", "COM y (cm)", "COM z (cm)", "Foot friction"]
CHINESE = ["左手负载 / kg", "右手负载 / kg", "左胫负载 / kg", "右胫负载 / kg",
           "COM x / cm", "COM y / cm", "COM z / cm", "足底摩擦系数"]
SCALES = np.asarray([1, 1, 1, 1, 100, 100, 100, 1.])


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.tolist())+"\n")


def select(rows, **conditions):
    return [r for r in rows if all(r[k] == v for k, v in conditions.items())]


def aggregate(rows):
    result = {"replicates": len(rows), "trace_seeds": sorted(set(r["trace_seed"] for r in rows)),
              "kmeans_seeds": sorted(set(r["kmeans_seed"] for r in rows))}
    for key in ["random_pair_rms_physical", "same_expert_pair_rms_physical",
                "same_expert_to_random_pair_rms_ratio", "between_expert_variance_fraction"]:
        a = np.asarray([r["proximity"][key] for r in rows])
        result[key] = {"mean": a.mean(0), "min": a.min(0), "max": a.max(0)}
    for key in ["group_pair_rms_ratio", "group_mean_between_expert_variance_fraction"]:
        result[key] = {k: float(np.mean([r["proximity"][key][k] for r in rows]))
                       for k in rows[0]["proximity"][key]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--floor-root", type=Path, default=Path("runs/limb_context_20260915_whitening_floor_ablation"))
    args = parser.parse_args()
    result = json.loads((args.root/"results.json").read_text())
    rows = result["rows"]
    assert len(rows) == 120
    assert json.loads((args.root/"verification.json").read_text())["passed"]
    # Match the exact earlier gate statistics for both whitening floors,
    # all three center initializations and both trajectory seeds.
    reproduced = []
    for seed, tag in [(30403, "heldout30403"), (30404, "runtime50hz30404")]:
        old = json.loads((args.floor_root/tag/"results.json").read_text())["rows"]
        for floor in [.01, .008]:
            for km in [731, 1731, 2731]:
                a = select(rows, trace_seed=seed, floor=floor, kmeans_seed=km, EMA_seconds=2.,
                           mode="soft", temperature=18.401254177093506)[0]
                b = next(r for r in old if r["floor"] == floor and r["kmeans_seed"] == km
                         and r["temperature_mode"] == "fixed_absolute")
                for key in ["mean_weight_by_expert", "effective_experts_per_point", "mean_maximum_weight"]:
                    np.testing.assert_allclose(a["gate"][key], b["gate"][key], atol=1e-7, rtol=0.)
                reproduced.append(a["name"])
    summary = {
        "aggregation": "Mean over two existing independent trajectory seeds and three fixed center initializations. Min/max spans these six conditions; it is not a confidence interval.",
        "primary_floor": .008, "primary_EMA_seconds": 2.,
        "hard": {}, "soft": {},
    }
    for floor in [.01, .008]:
        for tau in [0., 2.]:
            chosen = select(rows, floor=floor, EMA_seconds=tau, mode="hard")
            summary["hard"][f"floor{floor:g}_ema{tau:g}"] = aggregate(chosen)
        summary["soft"][f"floor{floor:g}"] = []
        for t in sorted(set(r["temperature"] for r in select(rows, floor=floor, EMA_seconds=2., mode="soft")), reverse=True):
            summary["soft"][f"floor{floor:g}"].append({
                "temperature": t,
                **aggregate(select(rows, floor=floor, EMA_seconds=2., mode="soft", temperature=t)),
            })
    save_json(args.root/"summary.json", summary)
    with (args.root/"comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trace_seed", "floor", "kmeans_seed", "EMA_seconds", "mode", "temperature", "parameter",
                         "random_pair_RMS", "same_expert_pair_RMS", "same_to_random_RMS_ratio", "between_expert_variance_fraction"])
        for r in rows:
            p = r["proximity"]
            for j, name in enumerate(p["per_parameter_names"]):
                writer.writerow([r[k] for k in ["trace_seed", "floor", "kmeans_seed", "EMA_seconds", "mode", "temperature"]]+
                                [name, p["random_pair_rms_physical"][j], p["same_expert_pair_rms_physical"][j],
                                 p["same_expert_to_random_pair_rms_ratio"][j], p["between_expert_variance_fraction"][j]])
    canonical = select(rows, trace_seed=30403, floor=.008, kmeans_seed=731, EMA_seconds=2., mode="hard")[0]
    schema = json.loads((args.root/"seed30403/dr_schema.json").read_text())
    with np.load(args.root/"seed30403/worlds.npz") as z:
        raw = z["raw_dr"]
    expert = canonical["proximity"]["per_expert"]
    means = np.asarray([r["mean_physical"] for r in expert])[:, IDS]*SCALES
    bounds = np.asarray([r["p10_p90_physical"] for r in expert])[:, IDS]*SCALES[None, :, None]
    global_bounds = np.quantile(raw[:, IDS], [.1, .9], axis=0)*SCALES
    lower = np.asarray(schema["lower"])[IDS]*SCALES
    upper = np.asarray(schema["upper"])[IDS]*SCALES
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 14), layout="constrained")
    for j, ax in enumerate(axes.flat):
        ax.axvspan(*global_bounds[:, j], color="#e5e7eb", label="Global P10-P90")
        ax.hlines(np.arange(16), bounds[:, j, 0], bounds[:, j, 1], color="#2463a3", lw=2.3, label="Expert P10-P90")
        ax.scatter(means[:, j], np.arange(16), color="#b73d2c", s=15, zorder=3, label="Expert mean")
        ax.set(xlim=(lower[j], upper[j]), ylim=(15.6, -.6), yticks=np.arange(16),
               xlabel=LABELS[j], ylabel="Nearest-center expert", title=LABELS[j])
        ax.grid(axis="x", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("DR ranges within each expert: wide leg-load overlap remains\n"
                 "Floor 0.8%, EMA 2 s; held-out seed 30403, center seed 731; 1,024 worlds x 32 samples",
                 fontsize=14)
    fig.savefig(args.root/"expert_dr_ranges.png", dpi=165)
    fig.savefig(args.root/"expert_dr_ranges.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5.4), layout="constrained")
    hard = summary["hard"]["floor0.008_ema2"]
    soft = summary["soft"]["floor0.008"]
    comparisons = [(soft[0], "Soft, T=18.40", "#a7afb8"),
                   (soft[-1], "Soft, T=0.61", "#df9a3e"),
                   (hard, "Nearest center", "#2463a3")]
    x = np.arange(8)
    for i, (data, label, color) in enumerate(comparisons):
        stats = data["same_expert_to_random_pair_rms_ratio"]
        mean, lo, hi = [np.asarray(stats[k])[IDS] for k in ["mean", "min", "max"]]
        ax.bar(x+(i-1)*.25, mean, width=.23, label=label, color=color,
               yerr=np.stack([mean-lo, hi-mean]), capsize=2, error_kw={"lw": .8})
    ax.axhline(1, color="#b73d2c", ls="--", lw=1.2, label="Random distinct-world pairs")
    ax.set(xticks=x, xticklabels=["L hand", "R hand", "L shin", "R shin", "COM x", "COM y", "COM z", "Friction"],
           ylim=(0, 1.08), ylabel="Within-expert / random pair RMS (lower is closer)",
           title="Per-expert DR exposure: strong soft mixing restores nearly the full global spread\n"
                 "Floor 0.8%, EMA 2 s; mean and min-max of 2 trajectory seeds x 3 center initializations")
    ax.legend(ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=.18)
    fig.savefig(args.root/"expert_dr_proximity.png", dpi=170)
    fig.savefig(args.root/"expert_dr_proximity.pdf")
    plt.close(fig)

    lines = [
        "**白化路由下的专家簇内 DR 接近程度，2026-09-15**", "",
        "同一最近中心内的 DR 有部分相似性，但不能视为固定或窄范围 DR 的环境专家。摩擦、COM x/y 和手部负载有一定分组，腿部负载与 COM z 仍很宽。高温度连续混合让每个专家接收到的 DR 范围进一步接近全局。", "",
        "复用两组各1024个 world 的原策略实际交互轨迹（seed30403/30404），每个world取32个长期记忆充分、policy step≥500的观测；每个world的DR保持不变。中心/白化统计来自seed30402，本次不拟合、更新中心或训练PPO。检查白化下限1%和0.8%、三个中心初始化731/1731/2731、EMA 0/2秒及四个绝对温度。", "",
        "context 仍为带四肢负载训练的 nominal50 Memory350 扩容 response10/predictor5 update15000，SHA256 `db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac`；原交互policy为共享MoE update1000，SHA256 `5239793a01de2c95045a0bb39365a2d539360fd5e0fa5a6df115b2acc835c0d1`。这些是新白化候选在旧真实轨迹上的离线路由结果，不是新MoE训练后的控制结果。", "",
        "主要比较按每次观测的最近中心归属。固定latent、中心和EMA时，正温度不会改变argmax归属；已对全部候选逐样本验证。另按完整软权重计算每个专家接收到的DR分布，这只反映路由系数，不等同于实际PPO梯度大小。", "",
        "配对方法：按专家的全局权重选择专家，再按该专家接收每个world的权重选择两个不同world，计算真实DR参数差的RMS。排除同一world与自身配对，避免同一DR的32次重复观测人为缩小差异。随机参照从全体world均匀取两个不同world。各维单独保留物理单位；跨维聚合前除以各参数完整采样范围。", "",
        "下表为floor=0.008、EMA=2秒的最近中心分组，两条轨迹×三次中心初始化的均值。比例1表示与随机配对一样分散，比例0表示同专家内该参数完全一致。", "",
        "| DR参数 | 随机两环境差异RMS | 同专家两环境差异RMS | 同专家/随机 |",
        "|---|---:|---:|---:|",
    ]
    random = np.asarray(hard["random_pair_rms_physical"]["mean"])[IDS]*SCALES
    same = np.asarray(hard["same_expert_pair_rms_physical"]["mean"])[IDS]*SCALES
    ratios = np.asarray(hard["same_expert_to_random_pair_rms_ratio"]["mean"])[IDS]
    for j, label in enumerate(CHINESE):
        lines.append(f"| {label} | {random[j]:.3f} | {same[j]:.3f} | {ratios[j]:.1%} |")
    lines.extend(["", "两组轨迹、三个中心初始化得出相同的方向性结论；图中误差线是六个条件的最小值和最大值，不是置信区间。1%下限的结果相近，降低到0.8%没有让所有因素形成紧凑簇。", "",
                  "簇间均值差异解释的测试集DR方差比例，八项依次为："+
                  "、".join(f"{v:.1%}" for v in np.asarray(hard["between_expert_variance_fraction"]["mean"])[IDS])+"。这是一项描述性方差分解，簇均值在当前评测集上统计，不是此前跨world验证的连续权重线性读出R²。完整world标签置换100次的参照也保留；真实分组虽优于随机标签，但不代表簇内参数已足够接近。", "",
                  "torso mass、armature与encoder bias的同专家/随机RMS比例均接近1。这些因素目前没有呈现有用的紧凑分组，全部67维结果保存在comparison.csv和results.json。", "",
                  "![每簇物理参数区间](../runs/limb_context_20260915_expert_dr_proximity/expert_dr_ranges.png)", "",
                  "上图画出一个固定初始化下每簇的加权P10–P90区间，灰色背景为全局P10–P90。专家编号只在该中心初始化内有意义，未跨初始化对齐编号。", "",
                  "| 温度/路由 | 每专家接收的八项DR差异 / 随机 | 四肢负载差异 / 随机 |",
                  "|---|---:|---:|"])
    for r in soft:
        lines.append(f"| T={r['temperature']:.2f} | {r['group_pair_rms_ratio']['primary8']:.1%} | {r['group_pair_rms_ratio']['limb_load']:.1%} |")
    lines.append(f"| 只按最近中心归属 | {hard['group_pair_rms_ratio']['primary8']:.1%} | {hard['group_pair_rms_ratio']['limb_load']:.1%} |")
    lines.extend(["", "![不同温度的每专家DR差异](../runs/limb_context_20260915_expert_dr_proximity/expert_dr_proximity.png)", "",
                  "降低温度能减少软混合带来的额外环境混杂，但不会重划最近中心的16个簇。当前16个簇已经包含较宽的腿部负载范围，因此不能仅靠降低温度把每簇变成固定DR专家。此前约0.88的R²读取的是全部16个连续权重，不能推导单一专家编号已经确定所有8项DR。", "",
                  "同DR在不同motion/历史下可能分给多个专家，本报告按每次实际离线路由计入对应簇，而不是人为把一个world强制固定到它最常去的专家。完整32次观测归属频率保存在membership文件；当前这项报告主要判断簇内DR接近程度，不将其解读成已验证的控制器适配性。", "",
                  "验证：120项配置完成；12项现有高温度gate统计逐项复现；所有温度argmax一致；每项方差分解恒等式通过；8个详细配置各选3个专家，以真实world×world配对矩阵独立核对RMS公式；原有目标DR与world顺序复现。", "",
                  "运行命令：", "", "```bash",
                  "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/audit_moe_expert_dr_proximity.py --output runs/limb_context_20260915_expert_dr_proximity",
                  "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/report_moe_expert_dr_proximity.py --root runs/limb_context_20260915_expert_dr_proximity",
                  "```", "", "已完成的审计目录不会被分析脚本覆盖；重新计算使用新的输出目录。"])
    report = "\n".join(lines)+"\n"
    Path("docs/moe_expert_dr_proximity_20260915.md").write_text(report)
    (args.root/"FINDINGS.md").write_text(report.replace("../runs/limb_context_20260915_expert_dr_proximity/", ""))
    save_json(args.root/"report_verification.json", {
        "passed": True, "existing_gate_results_reproduced": reproduced,
        "results_sha256": hashlib.sha256((args.root/"results.json").read_bytes()).hexdigest(),
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print(json.dumps({"completed": True, "hard_primary8_ratio": hard["group_pair_rms_ratio"]["primary8"],
                      "hard_load_ratio": hard["group_pair_rms_ratio"]["limb_load"],
                      "physical_random": random.tolist(), "physical_same_expert": same.tolist(),
                      "ratios": ratios.tolist(), "gate_records_reproduced": len(reproduced)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
