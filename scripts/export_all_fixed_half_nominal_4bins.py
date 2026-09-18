"""Merge adjacent latent-distance bins for the existing 16384-world 50s audit."""

import json
from pathlib import Path

import numpy as np

from analyze_per_limb_half_zero_50s import csv_write, digest, separation, write_json


ROOT = Path("runs/limb_context_20260917_all_fixed_dr_half_nominal_50s")
SOURCE = ROOT / "analysis/environment_comparison.npz"
PARAMETERS = Path("runs/limb_context_20260917_all_fixed_dr_half_nominal/parameter_bank.npz")
OUT = ROOT / "analysis4"


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(SOURCE) as saved:
        bank = {k: saved[k] for k in saved.files}
    edges = bank["edges"][::2]
    previous = json.loads((ROOT / "analysis/summary.json").read_text())
    report = {"source": str(SOURCE), "source_sha256": digest(SOURCE),
              "protocol": {**previous["protocol"], "edges": "Four equal-width bins, merging adjacent pairs of the original eight bins.",
                           "new_simulation_or_training": False},
              "edges": edges.tolist(), "results": {}}
    exported = {"nominal_center": bank["nominal_center"], "edges": edges,
                "world": bank["new_world"], "physics": bank["new_physics"],
                "centroid": bank["new_centroid"],
                "centroid_radius": bank["new_centroid_radius"],
                "full_query_count": bank["new_full_query_count"]}
    with np.load(PARAMETERS) as saved:
        np.testing.assert_array_equal(exported["world"], saved["world"])
        np.testing.assert_array_equal(exported["physics"], saved["mixed"])
        exported.update(names=saved["names"], encoder_bias=saved["encoder_bias"])
    for name in ("old", "payload_only", "new"):
        report["results"][name] = {}
        for method in ("centroid_radius", "mean_radius"):
            result, labels = separation(bank[f"{name}_{method}"], bank[f"{name}_physics"], edges)
            np.testing.assert_array_equal(labels, bank[f"{name}_{method}_class"] // 2)
            expected_counts = np.array(previous["results"][name][method]["class_counts"]).reshape(4, 2).sum(1)
            np.testing.assert_array_equal(result["class_counts"], expected_counts)
            assert result["underflow"] == result["overflow"] == 0
            report["results"][name][method] = result
            csv_write(OUT / f"{name}_{method}_bins.csv", result["bins"])
            if name == "new":
                exported[method + "_class"] = labels
    assert len(np.unique(exported["world"])) == 16384
    labels = exported["centroid_radius_class"]
    csv_write(OUT / "world_assignments.csv", [
        dict(world=int(world), class_id=int(label), distance=float(radius),
             total_load_kg=float(physics[34:38].sum()))
        for world, label, radius, physics in zip(exported["world"], labels,
            exported["centroid_radius"], exported["physics"], strict=True)])
    np.savez_compressed(OUT / "environment_assignments.npz", **exported)
    write_json(OUT / "summary.json", report)
    write_json(OUT / "verification.json", {
        "all_16384_worlds_accounted_for": True,
        "physics_and_bias_from_verified_parameter_bank": True,
        "all_three_samplers_both_distance_methods_match_pairwise_bin_merge": True,
        "counts_match_previous_8bin_report": True,
        "no_distances_outside_original_range": True,
        "assignments_sha256": digest(OUT / "environment_assignments.npz"),
    })
    lines = ["# 全部固定 DR 混合采样：四档环境划分", "",
             "使用已完成的 16,384 个环境各 50 秒交互。response10/u15000、全程环境中心和 nominal 参考均保持不变。",
             "将原 C0+C1、C2+C3、C4+C5、C6+C7 分别合并成新 C0、C1、C2、C3。仍按距离等宽分档，每环境一类。", "",
             "| 新类别 | latent 距离 | 环境数 | 占比 | 平均总负载 kg | 总负载 P10–P90 kg | 最大总负载 kg |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in report["results"]["new"]["centroid_radius"]["bins"]:
        lines.append(f'| C{r["class_id"]} | {r["lower"]:.3f}–{r["upper"]:.3f} | {r["worlds"]:,} | {r["fraction"]:.2%} | {r["total_load_kg_mean"]:.2f} | {r["total_load_kg_p10"]:.2f}–{r["total_load_kg_p90"]:.2f} | {r["total_load_kg_max"]:.2f} |')
    lines += ["", "| 新类别 | 左手均值 kg | 右手均值 kg | 左小腿均值 kg | 右小腿均值 kg | COM 平均偏移 cm | 摩擦均值 | 躯干质量平均绝对变化 kg |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["results"]["new"]["centroid_radius"]["bins"]:
        values = [r[k] for k in ("left_hand_kg_mean", "right_hand_kg_mean", "left_shin_kg_mean", "right_shin_kg_mean", "com_norm_cm_mean", "friction_mean", "torso_abs_delta_kg_mean")]
        lines.append(f'| C{r["class_id"]} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ["", "| 采样方式 | 新 C0 数量 | 新 C1 数量 | 新 C2 数量 | 新 C3 数量 |", "|---|---:|---:|---:|---:|"]
    for name, title in (("old", "原均匀采样"), ("payload_only", "仅负载混合"), ("new", "全部固定 DR 混合")):
        lines.append(f"| {title} | " + " | ".join(f'{n:,}' for n in report["results"][name]["centroid_radius"]["class_counts"]) + " |")
    lines += ["", "四类减少了稀疏小类，但仍非等频；合并不会提升原 latent 的辨识力，各类负载范围仍有重叠。",
              "environment_assignments.npz 保存环境 ID、参数、encoder bias、中心、距离和四类标签；world_assignments.csv 是可直接查看的逐环境分类表。",
              "所有环境和两种距离汇总口径均验证新标签等于原八档标签整除二，所有数量与原报告逐对相加一致。", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
