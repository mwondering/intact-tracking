"""Compare three DR samplers using fixed reference and environment latent centers."""

import json
from pathlib import Path

import numpy as np

from analyze_per_limb_half_zero_50s import aggregate, csv_write, digest, separation, stats, write_json


ROOT = Path("runs/limb_context_20260917_all_fixed_dr_half_nominal_50s")
PREVIOUS = Path("runs/limb_context_20260917_per_limb_half_zero_50s/analysis/environment_comparison.npz")
PARAMETERS = Path("runs/limb_context_20260917_all_fixed_dr_half_nominal/parameter_bank.npz")
OUT = ROOT / "analysis"
LABELS = {"old": "原均匀采样", "payload_only": "仅负载混合", "new": "全部固定 DR 混合"}


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(PREVIOUS) as saved:
        bank = {k: saved[k] for k in ("nominal_center", "edges")}
        data = {}
        for name, prefix in (("old", "old"), ("payload_only", "new")):
            data[name] = {k.removeprefix(prefix + "_"): saved[k] for k in saved.files
                          if k.startswith(prefix + "_") and not k.endswith("_class")}
    center, edges = bank["nominal_center"], bank["edges"]
    manifests, parts = [], []
    for shard in range(8):
        folder = ROOT / f"shard_{shard:02d}"
        meta = json.loads((folder / "metadata.json").read_text())
        assert meta["complete"] and meta["steps"] == 2500 and meta["seconds_per_world"] == 50
        assert meta["sampling_mode"] == "all-fixed-half-nominal"
        assert meta["rollout_config"]["dr_nominal_probability"] == .5
        assert meta["parameters_sha256"] == digest(PARAMETERS)
        assert meta["physics_unchanged"] and meta["encoder_bias_unchanged"]
        for filename, key in (("physics.npz", "physics_sha256"), ("queries.npz", "queries_sha256"),
                              ("environment_assignments.npz", "assignments_sha256")):
            assert digest(folder / filename) == meta[key]
        with np.load(folder / "physics.npz") as physics, np.load(folder / "queries.npz") as queries:
            np.testing.assert_array_equal(physics["world"], queries["world"])
            np.testing.assert_array_equal(queries["step"], np.arange(100, 2501, 100))
            d = {k: physics[k] for k in ("world", "physics", "nominal")}
            d.update(aggregate(queries["z"], queries["valid"], center))
            d["motion_count"] = np.array([len(np.unique(row[v])) for row, v in
                                          zip(queries["motion"], queries["valid"], strict=True)])
        with np.load(folder / "environment_assignments.npz") as saved:
            for key in ("centroid", "centroid_radius", "mean_radius", "full_query_count"):
                np.testing.assert_array_equal(saved[key], d[key])
            np.testing.assert_array_equal(saved["nominal_center"], center)
            np.testing.assert_array_equal(saved["edges"], edges)
        parts.append(d)
        manifests.append(meta)
    all_new = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    dr = ~all_new["nominal"]
    assert dr.sum() == len(np.unique(all_new["world"][dr])) == 16384
    data["new"] = {k: v[dr] for k, v in all_new.items()}
    for name in data:
        np.testing.assert_array_equal(data[name]["world"], data["new"]["world"])
    with np.load(PARAMETERS) as saved:
        np.testing.assert_array_equal(data["new"]["world"], saved["world"])
        np.testing.assert_array_equal(data["new"]["physics"], saved["mixed"])
        np.testing.assert_array_equal(data["old"]["physics"], saved["original"])
    with np.load("runs/limb_context_20260917_response10_radial_bins/radial_router_and_assignments.npz") as saved:
        nominal_test = np.isin(all_new["world"], saved["nominal_test_worlds"])
    assert nominal_test.sum() == 1024
    previous_meta = json.loads((PREVIOUS.parent / "summary.json").read_text())
    assert all(m["checkpoint_sha256"] == previous_meta["protocol"]["checkpoint_sha256"] for m in manifests)
    report = dict(protocol={
        "worlds_per_sampler": 16384, "seconds_per_world": 50,
        "checkpoint": manifests[0]["checkpoint"], "checkpoint_sha256": manifests[0]["checkpoint_sha256"],
        "primary": "Mean of mature unit latents per environment, then Euclidean distance to the fixed original nominal center; no re-normalization of environment means.",
        "secondary": "Mean distance of mature window unit latents to the same nominal center.",
        "edges": "Original 8 equal-width distance bins retained for all three samplers.",
        "sampling": "Each fixed scalar DR parameter independently 50% nominal, otherwise original uniform draw; original force pulses and observation noise.",
        "comparability": "Same 50s duration, frozen tracker/encoder, seeds and original continuous draws. The two mixture schemes use different nominal masks; motion/reset paths can diverge with dynamics.",
        "interpretation": "Coverage and physical-parameter ordering under the sampled motions; no encoder training or expert-policy performance evaluation.",
    }, edges=edges.tolist(), results={})
    for name, d in data.items():
        result = dict(full_queries=int(d["full_query_count"].sum()),
                      queries_per_world=stats(d["full_query_count"]),
                      motions_per_world=stats(d["motion_count"]),
                      total_load_kg=stats(d["physics"][:, 34:38].sum(axis=1)))
        for method in ("centroid_radius", "mean_radius"):
            measured, labels = separation(d[method], d["physics"], edges)
            counts = np.array(measured["class_counts"])
            probabilities = counts / counts.sum()
            measured.update(
                near_c0_c3_fraction=float(probabilities[:4].sum()),
                c5_c6_fraction=float(probabilities[5:7].sum()),
                largest_class_fraction=float(probabilities.max()),
                total_variation_from_uniform=float(abs(probabilities - 1/8).sum()/2),
                effective_bin_count=float(np.exp(-sum(p*np.log(p) for p in probabilities if p))),
                overall_distance=stats(d[method]),
            )
            result[method] = measured
            bank[f"{name}_{method}_class"] = labels
            csv_write(OUT / f"{name}_{method}_bins.csv", measured["bins"])
        report["results"][name] = result
        bank.update({f"{name}_{k}": v for k, v in d.items()})
    report["nominal_test"] = {
        "worlds": int(nominal_test.sum()),
        "center_distance": stats(all_new["centroid_radius"][nominal_test]),
        "c0_fraction": float((all_new["centroid_radius"][nominal_test] < edges[1]).mean()),
    }
    report["shards"] = [{k: m[k] for k in ("shard", "elapsed_seconds", "full_queries", "minimum_queries", "maximum_queries", "physics_sha256", "queries_sha256")} for m in manifests]
    np.savez_compressed(OUT / "environment_comparison.npz", **bank)
    write_json(OUT / "summary.json", report)
    write_report(report)
    plot(bank)
    print(json.dumps({name: {k: result["centroid_radius"][k] for k in
                            ("class_counts", "near_c0_c3_fraction", "largest_class_fraction", "c5_c6_fraction", "heavy_farther_than_light_probability")}
                      for name, result in report["results"].items()}, indent=2))


def write_report(report):
    lines = ["# 全部固定 DR 各自混入 50% nominal：全程 latent 距离", "",
        "每种采样各 16,384 个 DR 环境，mjwarp GPU 交互 50 秒，response10/u15000 和原 tracker 均冻结。",
        "中心是每环境全程有效 unit latent 的均值，距离是该中心到原固定 nominal 中心的欧氏距离。",
        "原采样截取到相同的前 50 秒；八档边界、nominal 参考中心均沿用之前实验，每环境只归一类。", "",
        "| 档位 | 距离范围 | 原均匀采样 | 仅负载混合 | 全部 DR 混合 | 新平均总负载 kg | 新最大总负载 kg | 新 COM 平均偏移 cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    results = report["results"]
    for i in range(8):
        bins = [results[name]["centroid_radius"]["bins"][i] for name in LABELS]
        row = bins[-1]
        amounts = " | ".join(f'{b["worlds"]:,} / {b["fraction"]:.2%}' for b in bins)
        lines.append(f'| C{i} | {row["lower"]:.3f}–{row["upper"]:.3f} | {amounts} | {row["total_load_kg_mean"]:.2f} | {row["total_load_kg_max"]:.2f} | {row["com_norm_cm_mean"]:.2f} |')
    lines += ["", "| 汇总 | 原均匀采样 | 仅负载混合 | 全部 DR 混合 |", "|---|---:|---:|---:|"]
    for title, key in (("近端 C0–C3 占比", "near_c0_c3_fraction"),
                       ("C5+C6 占比", "c5_c6_fraction"), ("最大单档占比", "largest_class_fraction"),
                       ("随机重载 >8kg 比随机轻载 <2kg 更远的概率", "heavy_farther_than_light_probability")):
        values = " | ".join(f'{results[name]["centroid_radius"][key]:.2%}' for name in LABELS)
        lines.append(f"| {title} | {values} |")
    lines += ["", "| 全部 DR 混合下的负载组 | 环境数 | 中心距离中位数 | C0→C7 数量 |", "|---|---:|---:|---|"]
    for row in results["new"]["centroid_radius"]["load_groups"]:
        lines.append(f'| {row["load_group"]} | {row["worlds"]} | {row["radius"]["p50"]:.3f} | {row["class_counts"]} |')
    lines += ["", "额外 nominal 对照不计入上述 DR 数量。留出 1,024 个 nominal 对照的 C0 占比为 "
              f'{report["nominal_test"]["c0_fraction"]:.2%}。', "",
              "各环境可因 episode 重置更换 motion；50 秒不代表覆盖全部动作。三组初始种子与原连续参数抽样相同，",
              "两种混合方案的 nominal 掩码不同，后续 motion/reset 路径也可因动力学不同而分叉，不能视为逐环境的单因素因果干预。",
              "没有重训 encoder；距离覆盖更均衡并不等于 latent 表征本身更有辨识力，也不代表专家策略收益已验证。",
              "完整每肢负载、COM、摩擦、质量变化及窗口距离均值对照见 summary.json 和 *_bins.csv。", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def plot(bank):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    names = ("old", "payload_only", "new")
    captions = ("Original uniform", "Payload-only mixture", "All fixed DR mixture")
    colors = ("#8794a1", "#d79632", "#158f85")
    for offset, name, caption, color in zip((-.25, 0, .25), names, captions, colors, strict=True):
        counts = np.bincount(bank[f"{name}_centroid_radius_class"], minlength=8)
        axes[0].bar(np.arange(8)+offset, counts/counts.sum()*100, width=.24, label=caption, color=color)
        axes[1].hist(bank[f"{name}_centroid_radius"], bins=np.linspace(0, bank["edges"][-1], 65),
                     histtype="step", density=True, lw=1.8, color=color, label=caption)
    axes[0].axhline(12.5, ls="--", color="black", lw=1, label="Equal-bin target: 12.5%")
    axes[0].set(xticks=np.arange(8), xticklabels=[f"C{i}" for i in range(8)],
                xlabel="Fixed distance bin: near to far", ylabel="Environments (%)")
    axes[1].set(xlabel="Environment latent-center distance to nominal", ylabel="Probability density")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "latent_distance_comparison.png", dpi=180)
    fig.savefig(OUT / "latent_distance_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
