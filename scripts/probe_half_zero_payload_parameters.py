"""Compare saved random DR parameters after zeroing all loads in half the worlds.

Parameter statistics only: no simulator, encoder, GPU interaction, or training.
"""

import csv
import json
from pathlib import Path

import numpy as np


SOURCE = Path("runs/limb_context_20260916_dr16384/analysis/selected_worlds.npz")
OUTPUT = Path("runs/limb_context_20260917_half_zero_payload_parameters")
SEED = 20260917


def main():
    with np.load(SOURCE) as data:
        keep = ~data["nominal"]
        physics = data["physics"][keep].astype(np.float64)
        worlds, names = data["world"][keep], data["names"]
    assert physics.shape == (16384, 38)
    zero = np.zeros(len(physics), dtype=bool)
    zero[np.random.default_rng(SEED).choice(len(physics), len(physics) // 2, replace=False)] = True
    changed = physics.copy()
    changed[zero, 34:38] = 0
    np.testing.assert_array_equal(changed[:, :34], physics[:, :34])
    np.testing.assert_array_equal(changed[~zero], physics[~zero])
    assert (changed[:, 34:38].sum(1) == 0).sum() == 8192
    edges = np.linspace(0, 13, 9)
    summaries = {}
    for label, parameters in (("original", physics), ("half_zero", changed)):
        loads = parameters[:, 34:38]
        total = loads.sum(1)
        counts = np.histogram(total, edges)[0]
        assert counts.sum() == len(parameters)
        summaries[label] = {
            "worlds": len(parameters), "mean_total_kg": float(total.mean()),
            "mean_per_limb_kg": loads.mean(0).tolist(),
            "zero_load_worlds": int((total == 0).sum()),
            "positive_below_2kg_worlds": int(((total > 0) & (total < 2)).sum()),
            "below_2kg_worlds": int((total < 2).sum()),
            "above_8kg_worlds": int((total > 8).sum()),
            "bin_counts": counts.tolist(), "bin_probabilities": (counts / len(parameters)).tolist(),
        }
    output = {
        "source": str(SOURCE), "seed": SEED,
        "protocol": "Select exactly half the DR worlds uniformly without replacement, "
                    "independently of their parameters; set all four limb payloads to zero. "
                    "Retain all background DR and remaining worlds' loads exactly.",
        "histogram": "Eight equal-width bins of total limb load in [0,13] kg; NOT latent-distance bins.",
        "edges_kg": edges.tolist(), "results": summaries,
        "checks": {"background_DR_unchanged": True, "random_load_half_unchanged": True,
                   "exactly_8192_zero_load_worlds": True, "no_simulation_or_network": True},
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "parameters.npz", world=worlds, names=names,
                        original=physics, half_zero=changed, zero_payload=zero)
    lines = [
        "# 一半无负载、一半随机负载：仅环境参数分布", "",
        "复用 16384 个随机 DR 参数组，随机选 8192 个环境将四肢附加负载全部设为零。",
        "其余环境保留原随机负载；所有环境的背景 DR 保持原值。没有运行仿真、交互、网络或训练。",
        "这里的一半是按环境分配，不是每条肢体独立有一半概率为零。", "",
        "| 四肢总负载 / kg | 原采样 | 一半无负载 |",
        "|---|---:|---:|",
    ]
    with (OUTPUT / "load_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lower_kg", "upper_kg", "original_count", "half_zero_count",
                         "original_probability", "half_zero_probability"])
        for k in range(8):
            old, new = (summaries[name]["bin_counts"][k] for name in ("original", "half_zero"))
            writer.writerow([edges[k], edges[k + 1], old, new, old / 16384, new / 16384])
            lines.append(f"| {edges[k]:.3f}–{edges[k+1]:.3f} | {old/16384:.2%} | {new/16384:.2%} |")
    lines += ["", "上表只是总负载等宽分档，不是原来的 8 个 latent 距离类别。",
              "无负载占比从 0% 变为 50%；平均总负载从 6.5134 kg 降到 3.2566 kg。",
              "轻载覆盖的增加主要来自精确的零负载：0 < 总负载 < 2 kg 的环境反而从 98 个变成 51 个。",
              "这能解决无负载环境缺失，但不会均匀铺开不同负载大小：分布变为一半零负载加一半原钟形分布。",
              "背景 DR 未变，所以无负载环境不等于完整 nominal 环境；未测量新的 latent 类别比例。", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
