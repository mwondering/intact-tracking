"""Export eight equally sampled bins of full-interaction distance to nominal.

This is an offline parameter bank and sampler, not a simulator/training adapter.
Run from the repository root. No new interaction or network training is needed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path("runs/limb_context_20260917_response10_environment_bins")
PHYSICS = Path("runs/limb_context_20260916_dr16384/analysis")
OUTPUT = Path("runs/limb_context_20260917_response10_balanced8")
CLASS_COUNT = 8


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sample_rows(labels, size, rng, *, balanced_batch=True):
    """Return bank row indices; choose a class, then a whole row within that class.

    balanced_batch=True gives floor(size/8) slots to every class, with remainder
    slots assigned to distinct random classes. Otherwise class draws are IID.
    Within a class, rows are always sampled uniformly with replacement.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Expected a one-dimensional integer class array")
    if size < 0 or (labels < 0).any() or (labels >= CLASS_COUNT).any():
        raise ValueError("Invalid sample size or class ID")
    members = [np.flatnonzero(labels == k) for k in range(CLASS_COUNT)]
    if any(len(rows) == 0 for rows in members):
        raise ValueError("All eight classes must have parameter rows")
    if balanced_batch:
        classes = np.concatenate((
            np.tile(np.arange(CLASS_COUNT), size // CLASS_COUNT),
            rng.choice(CLASS_COUNT, size % CLASS_COUNT, replace=False),
        ))
        rng.shuffle(classes)
    else:
        classes = rng.integers(CLASS_COUNT, size=size)
    result = np.empty(size, dtype=np.int64)
    for k, rows in enumerate(members):
        selected = classes == k
        result[selected] = rng.choice(rows, int(selected.sum()), replace=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    assignment_path = SOURCE / "environment_assignments.npz"
    manifest = json.loads((SOURCE / "encoding_manifest.json").read_text())
    schema = json.loads((PHYSICS / "collection_audit.json").read_text())["schema"]
    with np.load(assignment_path) as saved:
        source = {key: saved[key] for key in (
            "world", "nominal", "physics", "full_query_count", "mean_radius",
            "nominal_center", "labels_16", "edges_16",
        )}
    with np.load(PHYSICS / "selected_worlds.npz") as saved:
        for key in ("world", "nominal", "physics"):
            np.testing.assert_array_equal(source[key], saved[key])
        np.testing.assert_array_equal(saved["names"], schema["names"])
    nominal_physics = source["physics"][source["nominal"]]
    np.testing.assert_array_equal(
        nominal_physics, np.broadcast_to(nominal_physics[0], nominal_physics.shape)
    )
    rows = np.flatnonzero(~source["nominal"])
    edges = source["edges_16"][::2]
    score = source["mean_radius"][rows]
    labels = np.searchsorted(edges[1:-1], score, side="right")
    np.testing.assert_array_equal(labels, source["labels_16"][rows] // 2)
    assert np.isfinite(score).all() and ((score >= edges[0]) & (score <= edges[-1])).all()
    assert len(rows) == len(np.unique(source["world"][rows])) == 16384
    counts = np.bincount(labels, minlength=CLASS_COUNT)
    if (counts == 0).any():
        raise ValueError("Cannot balance empty distance bins")
    probability = 1.0 / (CLASS_COUNT * counts[labels])
    np.testing.assert_allclose(
        np.bincount(labels, weights=probability), np.full(CLASS_COUNT, 1 / CLASS_COUNT)
    )

    args.output.mkdir(parents=True, exist_ok=False)
    bank_path = args.output / "parameter_bank.npz"
    np.savez_compressed(
        bank_path, source_row=rows, world=source["world"][rows],
        physics=source["physics"][rows], names=np.asarray(schema["names"]),
        mean_radius=score, class_id=labels, draw_probability=probability,
        full_query_count=source["full_query_count"][rows],
    )
    config = {
        "version": "response10_full_interaction_radial_balanced8_v1",
        "source": str(assignment_path), "source_sha256": digest(assignment_path),
        "checkpoint": manifest["checkpoint_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "encoding_manifest": str(SOURCE / "encoding_manifest.json"),
        "encoding_manifest_sha256": digest(SOURCE / "encoding_manifest.json"),
        "script_sha256": digest(__file__),
        "bank": bank_path.name, "bank_sha256": digest(bank_path),
        "distance": "mean_t(||unit(z_t) - fixed_nominal_center||_2)",
        "interaction": "All mature queries in each saved 3200-step / 64-second interaction; "
                       "23-29 queries per world, with sampled motions and episode resets.",
        "nominal_center": source["nominal_center"].tolist(),
        "boundaries": "Merge each adjacent pair of the previous 16 equal-width bins; "
                      "left-inclusive, right-exclusive except the final upper edge. "
                      "Do not silently assign future out-of-range scores.",
        "edges": edges.tolist(), "class_count": CLASS_COUNT,
        "class_counts": counts.tolist(), "class_probabilities": [1 / CLASS_COUNT] * CLASS_COUNT,
        "sampling": "Choose a class uniformly, then choose a complete saved parameter row "
                    "uniformly with replacement within it. Never sample coordinates independently.",
        "balanced_batch": "Give each class floor(num_envs/8) slots; randomly distribute the "
                          "remainder to distinct classes and shuffle all slots.",
        "nominal_control_worlds": "The 2048 nominal controls are excluded from the DR bank. "
                                  "A separate nominal training quota is not specified here.",
        "physics_names": schema["names"],
        "physics_lower": schema["lower"], "physics_upper": schema["upper"],
        "physics_nominal": nominal_physics[0].tolist(),
        "physics_units": ["m"] * 3 + ["relative_mass_delta", "friction_coefficient"]
                         + ["armature_scale"] * 29 + ["kg"] * 4,
        "physics_excluded": schema["excluded"],
        "scope": "Offline bank and sampler only; not wired into legacy --dr-bank. "
                 "No new simulation, encoder training, PPO training, or running-task changes.",
        "limitation": "Balancing probabilities does not add unique DR settings. Class membership "
                      "is calibrated on the saved finite motion interactions, not established "
                      "for every motion or a newly trained policy. New parameter combinations "
                      "need full-interaction calibration before being added.",
    }
    save_json(args.output / "sampler.json", config)
    summary = []
    for k, count in enumerate(counts):
        loads = source["physics"][rows[labels == k], 34:38].sum(axis=1)
        summary.append({
            "class_id": k, "distance_lower": float(edges[k]),
            "distance_upper": float(edges[k + 1]), "parameter_rows": int(count),
            "original_probability": float(count / len(rows)),
            "target_probability": 1 / CLASS_COUNT,
            "mean_total_limb_load_kg": float(loads.mean()),
            "min_total_limb_load_kg": float(loads.min()),
            "max_total_limb_load_kg": float(loads.max()),
        })
    with (args.output / "classes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    checks = {"seed": args.seed, "batches": {}}
    for size in (8192, 16384):
        sampled = sample_rows(labels, size, np.random.default_rng(args.seed + size))
        sampled_counts = np.bincount(labels[sampled], minlength=CLASS_COUNT)
        np.testing.assert_array_equal(sampled_counts, np.full(CLASS_COUNT, size // CLASS_COUNT))
        np.savez_compressed(
            args.output / f"sample_{size}.npz", bank_row=sampled,
            world=source["world"][rows[sampled]], class_id=labels[sampled],
            physics=source["physics"][rows[sampled]],
        )
        checks["batches"][str(size)] = {
            "slots_per_class": sampled_counts.tolist(),
            "distinct_parameter_rows_per_class": [
                len(np.unique(sampled[labels[sampled] == k])) for k in range(CLASS_COUNT)
            ],
        }
    iid = sample_rows(labels, 800000, np.random.default_rng(args.seed), balanced_batch=False)
    checks["iid_sample_count"] = len(iid)
    checks["iid_class_probabilities"] = (np.bincount(labels[iid], minlength=8) / len(iid)).tolist()
    checks["analytical_class_probabilities"] = np.bincount(labels, weights=probability).tolist()
    save_json(args.output / "sampling_check.json", checks)
    report = [
        "# 按全程 latent 距离分成 8 档，均衡抽取 DR 环境", "",
        "目标：8 个固定、等宽的距离档，每档采样概率 12.5%。先选档，再在档内随机抽取整组参数。",
        "沿用 response10/u15000 和固定 nominal 中心；每环境取已有 64 秒完整交互中所有充分记忆查询的平均距离。",
        "将此前 16 档每两档合并；没有按分位数移动边界。2048 个 nominal 对照不进入 DR 参数库。", "",
        "| 类别 | 距离范围 | 独立参数组 | 原采样占比 | 均衡后占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        report.append(
            f'| C{item["class_id"]} | {item["distance_lower"]:.4f}–{item["distance_upper"]:.4f}'
            f' | {item["parameter_rows"]:,} | {item["original_probability"]:.4%} | 12.5% |'
        )
    report += [
        "", "采样每条参数行的概率为 `1 / (8 × 该档参数行数)`；参数及类别保持成组对应。",
        "8192 个采样槽位各档恰好 1024 个，16384 个槽位各档恰好 2048 个；结果见 sample_*.npz。",
        "这些是离线抽样结果，不是新运行的物理环境。随机余数分配也支持不能被 8 整除的批次。", "",
        "近端前四档只有 2、3、6、11 组参数。均衡采样可以马上实现，但会重复使用这些组合；",
        "后续补库应优先覆盖这些档，新增参数必须完成交互并重新归档，不能直接把档内参数极值当独立均匀采样范围。",
        "同样距离不意味着同一 DR 变化方向；类别仅表达相对 nominal 的该项距离。",
        "当前类别由有限的已采样 motion 记录标定，尚未证明换 motion 或 policy 后仍完全不变。", "",
        "文件：sampler.json 保存边界、概率和来源；parameter_bank.npz 保存 16384 组原始参数及类号；",
        "classes.csv 保存各档数量和负载范围；sampling_check.json 保存抽样核验。",
        "采样函数：`sample_rows(bank['class_id'], num_envs, rng)` 返回参数库行索引。", "",
        "此表未接入旧版 --dr-bank，未启动训练或修改运行任务。", "",
        "复现：`.venv/bin/python scripts/export_response10_balanced8.py --output <新目录>`", "",
    ]
    (args.output / "README.md").write_text("\n".join(report))
    print(json.dumps({"output": str(args.output), "counts": counts.tolist(), **checks}, indent=2))


if __name__ == "__main__":
    main()
