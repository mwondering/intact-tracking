"""Compare load separation using environment latent centers over matched 50s records."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path("runs/limb_context_20260917_per_limb_half_zero_50s")
OLD = Path("runs/limb_context_20260917_response10_environment_bins")
OUT = ROOT / "analysis"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(x):
    x = np.asarray(x)
    if not len(x):
        return {k: None for k in ("mean", "min", "p10", "p50", "p90", "max")}
    return dict(zip(("mean", "min", "p10", "p50", "p90", "max"),
                    map(float, (x.mean(), x.min(), *np.quantile(x, [.1, .5, .9]), x.max()))))


def aggregate(z, valid, reference):
    unit = z.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    unit[~valid] = np.nan
    centers = np.nanmean(unit, axis=1)
    return dict(centroid=centers, centroid_radius=np.linalg.norm(centers - reference, axis=-1),
                mean_radius=np.nanmean(np.linalg.norm(unit - reference, axis=-1), axis=1),
                full_query_count=valid.sum(axis=1))


def separation(score, physics, edges):
    n_classes = len(edges) - 1
    total = physics[:, 34:38].sum(axis=1)
    labels = np.searchsorted(edges[1:-1], score, side="right")
    light, heavy = total < 2, total > 8
    rows = []
    for k in range(n_classes):
        keep = labels == k
        row = dict(class_id=k, lower=float(edges[k]), upper=float(edges[k+1]),
                   worlds=int(keep.sum()), fraction=float(keep.mean()))
        for name, values in {
            "radius": score, "total_load_kg": total, "left_hand_kg": physics[:, 34],
            "right_hand_kg": physics[:, 35], "left_shin_kg": physics[:, 36],
            "right_shin_kg": physics[:, 37], "com_norm_cm": np.linalg.norm(physics[:, :3], axis=1)*100,
            "friction": physics[:, 4], "torso_abs_delta_kg": abs(physics[:, 3])/.12790995401552643,
        }.items():
            row.update({name + "_" + stat: value for stat, value in stats(values[keep]).items()})
        for name, mask in (("zero_load", total == 0), ("light_below_2kg", light), ("heavy_above_8kg", heavy)):
            row[name + "_worlds"] = int((keep & mask).sum())
            row[name + "_fraction"] = float(mask[keep].mean()) if keep.any() else None
        rows.append(row)
    groups = []
    for name, keep in (("zero", total == 0), ("positive_below_2", (total > 0) & light),
                       ("2_to_4", (total >= 2) & (total < 4)), ("4_to_6", (total >= 4) & (total < 6)),
                       ("6_to_8", (total >= 6) & (total <= 8)), ("above_8", heavy)):
        groups.append(dict(load_group=name, worlds=int(keep.sum()), radius=stats(score[keep]),
                           class_counts=np.bincount(labels[keep], minlength=n_classes).tolist()))
    order_probability = float(mannwhitneyu(score[heavy], score[light], alternative="greater").statistic
                              / (int(heavy.sum()) * int(light.sum())))
    hands = physics[:, 34:36].sum(axis=1)
    shins = physics[:, 36:38].sum(axis=1)
    patterns = []
    for name, mask in (("all_zero", total == 0), ("hands_only", (hands > 0) & (shins == 0)),
                       ("shins_only", (hands == 0) & (shins > 0)),
                       ("hands_and_shins", (hands > 0) & (shins > 0))):
        patterns.append(dict(pattern=name, worlds=int(mask.sum()), radius=stats(score[mask]),
                             total_load_kg=stats(total[mask]),
                             class_counts=np.bincount(labels[mask], minlength=n_classes).tolist()))
    # Keep load-vector information distinct from a single radial score.
    class_mean = np.array([total[labels == k].mean() if (labels == k).any() else 0 for k in range(n_classes)])
    return dict(class_counts=np.bincount(labels, minlength=n_classes).tolist(), bins=rows, load_groups=groups,
        load_patterns=patterns,
        light_worlds=int(light.sum()), heavy_worlds=int(heavy.sum()),
        light_class_probabilities=(np.bincount(labels[light], minlength=n_classes)/light.sum()).tolist(),
        heavy_class_probabilities=(np.bincount(labels[heavy], minlength=n_classes)/heavy.sum()).tolist(),
        heavy_farther_than_light_probability=order_probability,
        total_load_spearman=float(spearmanr(total, score).statistic),
        per_limb_spearman=[float(spearmanr(physics[:, i], score).statistic) for i in range(34, 38)],
        total_load_variance_explained_by_class=float(1 - np.var(total - class_mean[labels])/np.var(total)),
        underflow=int((score < edges[0]).sum()), overflow=int((score > edges[-1]).sum())), labels


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(OLD / "environment_assignments.npz") as saved:
        reference, edges = saved["nominal_center"], saved["edges_16"][::2]
    parts = {"old": [], "new": []}
    manifests = []
    for s in range(8):
        folder = ROOT / f"shard_{s:02d}"
        meta = json.loads((folder / "metadata.json").read_text())
        assert meta["complete"] and meta["steps"] == 2500 and meta["seconds_per_world"] == 50
        for filename, key in (("physics.npz", "physics_sha256"), ("queries.npz", "queries_sha256"),
                              ("environment_assignments.npz", "assignments_sha256")):
            assert digest(folder / filename) == meta[key]
        with np.load(folder / "physics.npz") as physical, np.load(folder / "queries.npz") as queries:
            np.testing.assert_array_equal(physical["world"], queries["world"])
            np.testing.assert_array_equal(queries["step"], np.arange(100, 2501, 100))
            new = {k: physical[k] for k in ("world", "physics", "nominal")}
            new.update(aggregate(queries["z"], queries["valid"], reference))
            new["motion_count"] = np.array([len(np.unique(row[v])) for row, v in zip(queries["motion"], queries["valid"], strict=True)])
            with np.load(folder / "environment_assignments.npz") as saved:
                for k in ("centroid", "centroid_radius", "mean_radius", "full_query_count"):
                    np.testing.assert_array_equal(new[k], saved[k])
        with np.load(OLD / f"full_shard_{s:02d}.npz") as saved:
            old = {k: saved[k] for k in ("world", "physics", "nominal")}
            old.update(aggregate(saved["z"][:, :25], saved["valid"][:, :25], reference))
            old["motion_count"] = np.array([len(np.unique(row[v])) for row, v in zip(saved["motion"][:, :25], saved["valid"][:, :25], strict=True)])
        for key in ("world", "nominal"):
            np.testing.assert_array_equal(new[key], old[key])
        np.testing.assert_array_equal(new["physics"][:, :34], old["physics"][:, :34])
        for name, data in (("old", old), ("new", new)):
            parts[name].append(data)
        manifests.append(meta)
    data = {name: {k: np.concatenate([p[k] for p in rows]) for k in rows[0]} for name, rows in parts.items()}
    with np.load("runs/limb_context_20260917_per_limb_half_zero_parameters/parameters.npz") as saved:
        keep = ~data["new"]["nominal"]
        np.testing.assert_array_equal(saved["world"], data["new"]["world"][keep])
        np.testing.assert_array_equal(saved["per_limb_half_zero"], data["new"]["physics"][keep])
    assert int(keep.sum()) == len(np.unique(data["new"]["world"][keep])) == 16384
    with np.load("runs/limb_context_20260917_response10_radial_bins/radial_router_and_assignments.npz") as saved:
        nominal_test = np.isin(data["new"]["world"], saved["nominal_test_worlds"])
    report = dict(protocol=dict(worlds=16384, seconds_per_world=50, checkpoint=manifests[0]["checkpoint"],
        checkpoint_sha256=manifests[0]["checkpoint_sha256"],
        primary="Euclidean distance of each environment's mean unit latent to the fixed nominal reference; no renormalization of its center.",
        secondary="Mean of window-to-nominal distances, as in previous analysis.",
        baseline="Previous full interaction archives truncated to their first 2500 steps (50s). Identical background parameters, frozen policy and initial RNG seeds. Subsequent motion/reset paths may differ with dynamics.",
        edges="Previous 16 equal-width distance bins merged by adjacent pairs, unchanged for old/new and both aggregation definitions.",
        sampling="Each of four limb loads independently 50% zero, 50% original uniform. Hands max2.5kg, shins max4kg. Other DR unchanged.",
        limitations="Observational separation under sampled background DR and motions, not an isolated-load causal sweep or expert-policy performance test."),
        edges=edges.tolist(), results={})
    bank = {"nominal_center": reference, "edges": edges}
    for name, d in data.items():
        dr = ~d["nominal"]
        record = dict(full_queries=int(d["full_query_count"][dr].sum()),
                      queries_per_world=stats(d["full_query_count"][dr]),
                      motions_per_world=stats(d["motion_count"][dr]),
                      total_load_kg=stats(d["physics"][dr, 34:38].sum(axis=1)))
        for method in ("centroid_radius", "mean_radius"):
            result, labels = separation(d[method][dr], d["physics"][dr], edges)
            result["nominal_test_c0_fraction"] = float((d[method][nominal_test] < edges[1]).mean())
            record[method] = result
            bank[f"{name}_{method}_class"] = labels
            csv_write(OUT / f"{name}_{method}_bins.csv", result["bins"])
        record["aggregation_class_agreement"] = float((bank[f"{name}_centroid_radius_class"] == bank[f"{name}_mean_radius_class"]).mean())
        report["results"][name] = record
        for key, value in d.items():
            bank[f"{name}_{key}"] = value[dr]
    report["shards"] = [{k: m[k] for k in ("shard", "elapsed_seconds", "full_queries", "minimum_queries", "maximum_queries", "physics_sha256", "queries_sha256")} for m in manifests]
    np.savez_compressed(OUT / "environment_comparison.npz", **bank)
    write_json(OUT / "summary.json", report)
    write_report(report)
    plot(bank)
    print(json.dumps({name: {method: {k: result[k] for k in ("class_counts", "heavy_farther_than_light_probability", "total_load_spearman", "nominal_test_c0_fraction")}
                      for method, result in r.items() if method in ("centroid_radius", "mean_radius")}
                      for name, r in report["results"].items()}, indent=2))


def write_report(report):
    lines = ["# 每肢独立半零负载：16384 个环境各交互 50 秒", "",
        "每条肢体独立以 50% 概率负载为零，否则沿用原均匀分布：手各 0–2.5 kg，小腿各 0–4 kg。",
        "使用 mjwarp GPU、冻结原 tracker、response10/u15000；未训练任何网络。原背景 DR、噪声、力脉冲保留。",
        "每环境 2500 控制步（dt=0.02s）；每 100 步提取 latent，仅使用 short50 和 long30 都完整的查询。",
        "环境中心 = 全程有效查询的 unit(latent) 均值；中心距离 = ||环境中心 − 固定 nominal 中心||₂。中心不再归一化。",
        "每环境只归一类。固定 8 个等宽距离档，沿用旧边界；旧采样也截到前 50 秒。另报旧口径的窗口距离均值。", "",
        "| 距离档 | 旧采样数量 / 占比 | 新采样数量 / 占比 | 新平均总负载 kg | 新负载 P10–P90 kg | 新最大总负载 kg | 新轻载 <2kg 占比 | 新重载 >8kg 占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    old, new = (report["results"][k]["centroid_radius"] for k in ("old", "new"))
    for a, b in zip(old["bins"], new["bins"], strict=True):
        lines.append(f'| C{b["class_id"]} | {a["worlds"]} / {a["fraction"]:.2%} | {b["worlds"]} / {b["fraction"]:.2%} | {b["total_load_kg_mean"]:.2f} | {b["total_load_kg_p10"]:.2f}–{b["total_load_kg_p90"]:.2f} | {b["total_load_kg_max"]:.2f} | {b["light_below_2kg_fraction"]:.2%} | {b["heavy_above_8kg_fraction"]:.2%} |')
    lines += ["", "轻重载排序：随机取一个总负载 <2 kg 的环境和一个 >8 kg 的环境，重载中心更远的概率（平局算一半）：",
              f'旧采样 {old["heavy_farther_than_light_probability"]:.2%}；新采样 {new["heavy_farther_than_light_probability"]:.2%}。',
              "这里统计的是环境对，不是独立重复试验数；背景 DR 仍然变化。", "",
              "| 实际总负载组 | 新环境数 | 中心距离中位数 | 中心距离 P10–P90 | C0→C7 数量 |",
              "|---|---:|---:|---:|---|"]
    for g in new["load_groups"]:
        r = g["radius"]
        lines.append(f'| {g["load_group"]} | {g["worlds"]} | {r["p50"]:.3f} | {r["p10"]:.3f}–{r["p90"]:.3f} | {g["class_counts"]} |')
    lines += ["", "| 负载位置 | 环境数 | 平均总负载 kg | 中心距离中位数 | C0→C7 数量 |",
              "|---|---:|---:|---:|---|"]
    for group in new["load_patterns"]:
        lines.append(f'| {group["pattern"]} | {group["worlds"]} | {group["total_load_kg"]["mean"]:.2f} | {group["radius"]["p50"]:.3f} | {group["class_counts"]} |')
    lines += ["", "| 汇总方式 | 旧 C0→C7 数量 | 新 C0→C7 数量 | 新轻重载排序正确率 |",
              "|---|---|---|---:|"]
    for method in ("centroid_radius", "mean_radius"):
        a, b = (report["results"][name][method] for name in ("old", "new"))
        lines.append(f'| {method} | {a["class_counts"]} | {b["class_counts"]} | {b["heavy_farther_than_light_probability"]:.2%} |')
    lines += ["", "各肢体均值、P10/P90、最大值以及 COM、摩擦、躯干质量统计保存在 *_bins.csv。",
              "背景参数与上一轮采样表逐项相同，只有负载的独立置零掩码改变。物理参数全程固定；episode 和 motion 可以变化。",
              "50 秒指控制交互总时长，包含 episode 重置后的轨迹；不代表每环境覆盖全部动作或共享同一动作序列。",
              "额外运行的 2048 个 nominal 对照不混入 16384 个 DR 环境；旧参考中心保持固定，只用留出的 nominal 对照检查距离。",
              "这些结果评估采样分布和 latent 的负载分流，不代表已经验证专家策略收益。", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def plot(bank):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    colors = {"old": "#7f8c8d", "new": "#087e8b"}
    edges = bank["edges"]
    for offset, name in ((-.18, "old"), (.18, "new")):
        labels = bank[f"{name}_centroid_radius_class"]
        axes[0].bar(np.arange(8)+offset, np.bincount(labels, minlength=8)/len(labels)*100,
                    width=.36, label=name, color=colors[name])
    axes[0].axhline(12.5, ls="--", lw=1, color="black", label="12.5% target")
    axes[0].set(xticks=np.arange(8), xlabel="Fixed distance bin (C0 near → C7 far)", ylabel="Environments (%)")
    axes[0].legend(frameon=False)
    total = bank["new_physics"][:, 34:38].sum(axis=1)
    axes[1].hexbin(bank["new_centroid_radius"], total, gridsize=45, mincnt=1, bins="log", cmap="viridis")
    axes[1].set(xlabel="Environment latent-center distance to nominal", ylabel="Total limb load (kg)")
    for boundary in edges[1:-1]:
        axes[1].axvline(boundary, color="grey", lw=.5, alpha=.4)
    for label, mask, color in (("Zero load", total == 0, "#1a9850"),
                                ("0 < load < 2 kg", (total > 0) & (total < 2), "#4575b4"),
                                ("Load > 8 kg", total > 8, "#d73027")):
        ordered = np.sort(bank["new_centroid_radius"][mask])
        axes[2].plot(ordered, np.arange(1, len(ordered)+1)/len(ordered), label=label, color=color)
    axes[2].set(xlabel="Environment latent-center distance to nominal", ylabel="Cumulative fraction", ylim=(0, 1))
    axes[2].legend(frameon=False)
    fig.suptitle("Independent half-zero limb loads · 16,384 DR worlds · 50 s/world · response10/u15000")
    fig.savefig(OUT / "load_separation.png", dpi=180)
    fig.savefig(OUT / "load_separation.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
