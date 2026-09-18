"""Parameter-only audit of independent 0.5*delta(0) + 0.5*Uniform per limb."""

import csv
import json
from math import comb
from pathlib import Path

import numpy as np


SOURCE = Path("runs/limb_context_20260916_dr16384/analysis/selected_worlds.npz")
OUTPUT = Path("runs/limb_context_20260917_per_limb_half_zero_parameters")
SEED = 20260917


def main():
    with np.load(SOURCE) as data:
        selected = ~data["nominal"]
        original = data["physics"][selected].astype(np.float64)
        worlds, names = data["world"][selected], data["names"]
    assert original.shape == (16384, 38)
    # Independent Bernoulli per world AND limb, independent of the saved U draws.
    zero = np.random.default_rng(SEED).random((len(worlds), 4)) < 0.5
    changed = original.copy()
    changed[:, 34:38] = np.where(zero, 0, original[:, 34:38])
    np.testing.assert_array_equal(changed[:, :34], original[:, :34])
    np.testing.assert_array_equal(changed[:, 34:38][~zero], original[:, 34:38][~zero])
    np.testing.assert_array_equal(changed[:, 34:38] == 0, zero)
    assert np.any(zero.any(1) & ~zero.all(1))
    edges = np.linspace(0, 13, 9)
    summaries = {}
    for label, parameters in (("original", original), ("per_limb_half_zero", changed)):
        loads = parameters[:, 34:38]
        total = loads.sum(1)
        histogram = np.histogram(total, edges)[0]
        assert histogram.sum() == len(worlds)
        summaries[label] = {
            "worlds": len(worlds), "mean_total_kg": float(total.mean()),
            "mean_per_limb_kg": loads.mean(0).tolist(),
            "per_limb_zero_fraction": (loads == 0).mean(0).tolist(),
            "active_limb_counts_0_to_4": np.bincount((loads > 0).sum(1), minlength=5).tolist(),
            "zero_load_worlds": int((total == 0).sum()),
            "positive_below_2kg_worlds": int(((total > 0) & (total < 2)).sum()),
            "below_2kg_worlds": int((total < 2).sum()),
            "above_8kg_worlds": int((total > 8).sum()),
            "bin_counts": histogram.tolist(),
            "bin_probabilities": (histogram / len(worlds)).tolist(),
        }
    results = {
        "source": str(SOURCE), "seed": SEED,
        "protocol": "For each world and each limb independently, set payload to zero with "
                    "probability 0.5, otherwise retain the saved independent uniform draw. "
                    "Keep original maxima: hands 2.5 kg each, shins 4 kg each. Background DR unchanged.",
        "theory": {"all_limbs_zero_probability": 0.5 ** 4,
                   "active_limb_probabilities_0_to_4": [comb(4, k) / 16 for k in range(5)],
                   "mean_total_load_kg": (2.5 + 2.5 + 4 + 4) / 4},
        "histogram": "Eight equal-width total-load bins, NOT latent or full-DR distance bins.",
        "edges_kg": edges.tolist(), "results": summaries,
        "mask_pairwise_correlations": np.corrcoef(zero.T).tolist(),
        "checks": {"background_parameters_unchanged": True, "unmasked_loads_unchanged": True,
                   "independent_mask_shape_16384_by_4": True, "no_simulation_or_network": True},
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "parameters.npz", world=worlds, names=names,
                        original=original, per_limb_half_zero=changed, zero_payload_mask=zero)
    lines = [
        "# 每条肢体独立：50% 零负载、50% 原均匀负载", "",
        "这是用户要求的采样规则。前一份按环境整体清零四肢的实验不适用于本方案。",
        "在 16384 组原随机参数上，对每条肢体独立抽一次 Bernoulli(0.5)，决定是否将该项负载置零。",
        "保留原上限：双手各 2.5 kg、双小腿各 4 kg；其他 DR 参数逐项保持不变。",
        "仅统计参数，没有运行仿真、交互、encoder 或训练。", "",
        "| 四肢总负载 / kg | 原采样 | 每条肢体独立 50% 零负载 |",
        "|---|---:|---:|",
    ]
    with (OUTPUT / "load_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lower_kg", "upper_kg", "original_count", "per_limb_half_zero_count",
                         "original_probability", "per_limb_half_zero_probability"])
        for k in range(8):
            old, new = (summaries[name]["bin_counts"][k] for name in ("original", "per_limb_half_zero"))
            writer.writerow([edges[k], edges[k + 1], old, new, old / len(worlds), new / len(worlds)])
            lines.append(f"| {edges[k]:.3f}–{edges[k+1]:.3f} | {old/len(worlds):.2%} | {new/len(worlds):.2%} |")
    new = summaries["per_limb_half_zero"]
    lines += ["", "以上为总负载分档，不是原来的 latent 距离类别。", "",
              f'平均总负载：6.5134 → {new["mean_total_kg"]:.4f} kg。',
              f'四肢全零：{new["zero_load_worlds"]} / 16384 = {new["zero_load_worlds"]/16384:.2%}；理论概率为 1/16 = 6.25%。',
              f'0 < 总负载 < 2 kg：98 → {new["positive_below_2kg_worlds"]} 个。',
              f'总负载 > 8 kg：3707 → {new["above_8kg_worlds"]} 个。', "",
              "该规则显著增加连续轻载组合，保留单肢、双肢、三肢和四肢负载情况；四肢全零不会占一半。",
              "重载环境明显减少，总负载分布向低负载移动，尚非八档等概率。没有估计新的 latent 档位占比。", "",
              "| 有负载的肢体数 | 实际环境数 | 实际占比 | 理论占比 |",
              "|---|---:|---:|---:|"]
    for k, count in enumerate(new["active_limb_counts_0_to_4"]):
        lines.append(f"| {k} | {count} | {count/16384:.2%} | {comb(4,k)/16:.2%} |")
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
