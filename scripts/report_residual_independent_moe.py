"""Four-way comparison: baseline, shared-encoder MoE, independent MoE, specialists."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_policy_results import assert_paired, bootstrap
from report_residual_uniform_moe import recovery

ARMS = ("baseline", "shared_moe", "independent_moe", "specialist")


def common_motion_means(rows, traces):
    reference = rows["baseline"]
    shape = (reference["episodes"], reference["max_steps"], len(reference["metric_names"]))
    for arm in ARMS:
        assert_paired(reference, rows[arm])
        if traces[arm].shape != shape:
            raise ValueError("Full metric trace shape differs")
    common = np.min([rows[arm]["episode_lengths"] for arm in ARMS], axis=0)
    if (common <= 0).any():
        raise ValueError("No common surviving window")
    mask = np.arange(shape[1])[None, :, None] < common[:, None, None]
    means = {arm: np.where(mask, traces[arm], 0).sum(axis=1, dtype=np.float64) / common[:, None] for arm in ARMS}
    if any(not np.isfinite(value).all() for value in means.values()):
        raise ValueError("Nonfinite shared-window metric")
    return means, common


def metric_comparison(means, index):
    result = {arm: float(means[arm][:, index].mean()) for arm in ARMS}
    candidate = means["independent_moe"][:, index]
    for arm in ("baseline", "shared_moe", "specialist"):
        result["independent_vs_" + arm] = bootstrap(means[arm][:, index], candidate)
    result["expert_gap_recovered"] = recovery(means["baseline"][:, index], candidate, means["specialist"][:, index])
    return result


def report(root):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    previous = Path(plan["shared_moe_run"])
    checkpoint = root / "moe/checkpoint_update_001000.pt"
    sha = file_sha256(checkpoint)
    shared_sha = file_sha256(previous / "moe/checkpoint_update_001000.pt")
    if shared_sha != plan["shared_moe_checkpoint_sha256"]:
        raise ValueError("Shared MoE reference checkpoint changed")
    result = {"passed": True, "completed_updates": 1000, "ppo_transitions_per_policy": 196608000,
              "training_seed_count": 1, "independent_checkpoint_sha256": sha,
              "shared_checkpoint_sha256": shared_sha,
              "window": "Four policies share each identical trial's minimum survival window",
              "bootstrap": "2000 paired samples of 512 motions, after equal-weight averaging over eight DRs",
              "modes": {}}
    table = []
    for mode in ("cold", "warm"):
        all_means, failures, reports = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}, []
        for entry in plan["specialists"]:
            i = entry["id"]
            paths = {
                "baseline": Path(entry["cold_evaluations"]["baseline"]) if mode == "cold" else previous / "evaluation" / mode / f"baseline_dr_{i:02d}.json",
                "specialist": Path(entry["cold_evaluations"]["specialist"]) if mode == "cold" else previous / "evaluation" / mode / f"specialist_dr_{i:02d}.json",
                "shared_moe": previous / "evaluation" / mode / f"moe_dr_{i:02d}.json",
                "independent_moe": root / "evaluation" / mode / f"moe_dr_{i:02d}.json",
            }
            expected = {"baseline": plan["baseline"]["sha256"], "specialist": entry["checkpoint_sha256"],
                        "shared_moe": shared_sha, "independent_moe": sha}
            rows = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
            traces = {}
            for arm, row in rows.items():
                assert row["checkpoint_sha256"] == expected[arm]
                assert row["completed_training_updates"] == 1000 and row["episodes"] == 512
                assert row["policy_precision"] == "fp32" and row["residual_output_bounded"] is False
                assert row["memory_start"] == mode and row["warmup"]["steps"] == (0 if mode == "cold" else 500)
                fixed = row["physics"]["runtime_audit"]["fixed_dr"]
                assert fixed["id"] == i and fixed["bank_sha256"] == file_sha256(root / "protocols/dr_bank.json")
                for key in ("reference_timeline_audited", "partial_reset_survivor_state_audited", "partial_reset_survivor_history_audited"):
                    assert row[key]
                if "moe" in arm:
                    assert row["context_sha256"] == plan["selected_context"]["sha256"]
                with np.load(paths[arm].with_suffix(".traces.npz")) as trace:
                    assert trace["metric_names"].tolist() == row["metric_names"]
                    np.testing.assert_array_equal(trace["lengths"], row["episode_lengths"])
                    traces[arm] = trace["all_metrics"]
                    recovered = traces[arm].sum(axis=1, dtype=np.float64) / np.asarray(row["episode_lengths"])[:, None]
                    np.testing.assert_allclose(recovered, row["per_episode_metrics"], atol=1e-7, rtol=1e-6)
            means, common = common_motion_means(rows, traces)
            names = rows["baseline"]["metric_names"]
            metrics = {name: metric_comparison(means, j) for j, name in enumerate(names)}
            for name, value in metrics.items():
                table.append({"memory_start": mode, "dr": i, "profile": entry["profile"], "metric": name,
                              **{arm: value[arm] for arm in ARMS},
                              "independent_reduction_vs_shared_percent": value["independent_vs_shared_moe"]["reduction_percent"],
                              "independent_reduction_vs_baseline_percent": value["independent_vs_baseline"]["reduction_percent"]})
            for arm in ARMS:
                all_means[arm].append(means[arm])
                failures[arm].append(rows[arm]["failed"])
            reports.append({"id": i, "profile": entry["profile"], "metrics": metrics,
                            "failure_counts": {arm: int(np.sum(rows[arm]["failed"])) for arm in ARMS},
                            "common_survival_steps_mean": float(common.mean()),
                            "evaluation_files": {arm: str(path) for arm, path in paths.items()}})
        averaged = {arm: np.mean(all_means[arm], axis=0) for arm in ARMS}
        result["modes"][mode] = {
            "rows": reports, "overall": {name: metric_comparison(averaged, j) for j, name in enumerate(names)},
            "failure_counts": {arm: int(np.sum(failures[arm])) for arm in ARMS}, "episodes_per_arm": 4096,
            "failure_independent_vs_shared": bootstrap(np.mean(failures["shared_moe"], axis=0), np.mean(failures["independent_moe"], axis=0)),
        }
    (root / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# 1000 轮：独立 encoder MoE 与共享 encoder MoE", "",
             "新版本的16个 expert 各有独立 actor encoder/head/Gaussian std 和 critic encoder/head。Actor/critic 不共享参数；冻结 tracker/context、非梯度 K-means 路由和 critic 运行归一化统计沿用原设置。", "",
             "两种MoE及MLP均从零训练8卡×1024环境、1000轮、196608000个PPO transition；每个环境专家具有相同单policy样本预算，八个专家合计八倍。Uniform motion、物理参数、不限幅 residual、奖励、termination、学习率和更新规则相同。独立版本参数更多，此对照同时改变共享程度与总容量。", "",
             "四方复用相同512条motion、完整DR、起点和seed；每个trial统一取四方共同存活窗口，同时报告失败数。Cold从空记忆开始；warm先执行同一冻结tracker的500步交互。", ""]
    for mode in ("cold", "warm"):
        section = result["modes"][mode]
        lines += [f"## {mode}", "", "| 指标 | MLP | 共享 encoder MoE | 独立 encoder MoE | 环境专家 | 独立相对共享改善 |", "|---|---:|---:|---:|---:|---:|"]
        for key, label, scale in (("error_body_pos_global", "全局 body pos (cm)", 100),
                                  ("error_anchor_pos_global", "全局 root pos (cm)", 100),
                                  ("error_body_pos", "对齐 body pos (cm)", 100),
                                  ("error_joint_pos", "joint pos (rad)", 1),
                                  ("error_anchor_rot_global", "全局 root rot (rad)", 1)):
            m = section["overall"][key]
            values = " | ".join(f"{scale * m[arm]:.4f}" for arm in ARMS)
            lines.append(f"| {label} | {values} | {m['independent_vs_shared_moe']['reduction_percent']:+.2f}% |")
        m = section["overall"]["error_body_pos_global"]
        lines += ["", f"全局body位置相对共享版改善95%区间：{m['independent_vs_shared_moe']['reduction_percent_ci95']}。", "",
                  f"独立版弥补MLP→环境专家差距：{m['expert_gap_recovered']['percent']}%。", "",
                  f"失败数（各4096条episode）：{section['failure_counts']}。", ""]
    lines += ["一个训练seed；motion配对区间不代表训练seed不确定性。完整20项指标、逐DR结果、来源和区间见comparison.json/csv。", ""]
    (root / "README.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    report(parser.parse_args().run_root)
