"""Three-way fixed-DR comparison with a shared survival window for all policies."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_policy_results import assert_paired, bootstrap

ARMS = ("baseline", "moe", "specialist")


def common_motion_means(rows, traces):
    """Match physical trials and censor all three arms at the same first endpoint."""
    for arm in ARMS[1:]:
        assert_paired(rows["baseline"], rows[arm])
    reference = rows["baseline"]
    shape = (reference["episodes"], reference["max_steps"], len(reference["metric_names"]))
    if any(traces[arm].shape != shape for arm in ARMS):
        raise ValueError("Full metric trace shape differs")
    common = np.min([rows[arm]["episode_lengths"] for arm in ARMS], axis=0)
    if (common <= 0).any():
        raise ValueError("No common surviving window")
    mask = np.arange(shape[1])[None, :, None] < common[:, None, None]
    means = {arm: np.where(mask, traces[arm], 0).sum(axis=1, dtype=np.float64) / common[:, None] for arm in ARMS}
    if any(not np.isfinite(value).all() for value in means.values()):
        raise ValueError("Nonfinite shared-window metric")
    return means, common


def recovery(baseline, moe, expert, repeats=2000):
    """Fraction of the baseline-to-expert error gap closed by the shared MoE."""
    baseline, moe, expert = [np.asarray(a, dtype=float) for a in (baseline, moe, expert)]
    denominator = float((baseline - expert).mean())
    if denominator <= 0:
        return {"percent": None, "reason": "Expert has no mean advantage over baseline"}
    rng = np.random.default_rng(8102)
    sampled = []
    invalid = 0
    for start in range(0, repeats, 50):
        ids = rng.integers(len(baseline), size=(min(50, repeats - start), len(baseline)))
        den = (baseline[ids] - expert[ids]).mean(axis=1)
        valid = den > 0
        invalid += int((~valid).sum())
        sampled.extend(100 * (baseline[ids] - moe[ids]).mean(axis=1)[valid] / den[valid])
    return {"percent": float(100 * (baseline - moe).mean() / denominator),
            "ci95": np.quantile(sampled, [.025, .975]).tolist() if not invalid else None,
            "bootstrap_nonpositive_expert_advantage_samples": invalid,
            "definition": "100*(baseline_error-moe_error)/(baseline_error-expert_error); 0=baseline, 100=expert"}


def report(root):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    moe_sha = file_sha256(root / "moe/checkpoint_update_001000.pt")
    result = {"passed": True, "completed_updates": 1000, "ppo_transitions_per_policy": 196608000,
              "training_seed_count": 1, "moe_checkpoint_sha256": moe_sha, "context": plan["selected_context"],
              "window": "All three policies share the minimum survival window for each identical motion/DR trial",
              "bootstrap": "Equal-weight eight-DR mean per motion; 2000 paired samples over 512 shared motion IDs",
              "modes": {}}
    csv_rows = []
    for mode in ("cold", "warm"):
        reports, all_means, failures = [], {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
        for entry in plan["specialists"]:
            paths = {arm: (Path(entry["cold_evaluations"][arm]) if mode == "cold" and arm != "moe" else
                           root / "evaluation" / mode / f"{arm}_dr_{entry['id']:02d}.json") for arm in ARMS}
            rows = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
            assert rows["baseline"]["checkpoint_sha256"] == plan["baseline"]["sha256"]
            assert rows["moe"]["checkpoint_sha256"] == moe_sha
            assert rows["specialist"]["checkpoint_sha256"] == entry["checkpoint_sha256"]
            assert rows["moe"]["context_sha256"] == plan["selected_context"]["sha256"]
            traces = {}
            for arm, row in rows.items():
                assert row["completed_training_updates"] == 1000 and row["episodes"] == 512
                assert row["policy_precision"] == "fp32" and row["residual_output_bounded"] is False
                assert row["memory_start"] == mode and row["warmup"]["steps"] == (0 if mode == "cold" else 500)
                fixed = row["physics"]["runtime_audit"]["fixed_dr"]
                assert fixed["id"] == entry["id"] and fixed["bank_sha256"] == file_sha256(root / "protocols/dr_bank.json")
                assert row["reference_timeline_audited"] and row["partial_reset_survivor_state_audited"]
                assert row["partial_reset_survivor_history_audited"]
                with np.load(paths[arm].with_suffix(".traces.npz")) as trace:
                    assert trace["metric_names"].tolist() == row["metric_names"]
                    np.testing.assert_array_equal(trace["lengths"], row["episode_lengths"])
                    traces[arm] = trace["all_metrics"]
                    recovered = traces[arm].sum(axis=1, dtype=np.float64) / np.asarray(row["episode_lengths"])[:, None]
                    np.testing.assert_allclose(recovered, row["per_episode_metrics"], atol=1e-7, rtol=1e-6)
            means, common = common_motion_means(rows, traces)
            names = rows["baseline"]["metric_names"]
            metrics = {}
            for i, name in enumerate(names):
                metrics[name] = {**{arm: float(means[arm][:, i].mean()) for arm in ARMS},
                                 "moe_vs_baseline": bootstrap(means["baseline"][:, i], means["moe"][:, i]),
                                 "moe_vs_specialist": bootstrap(means["specialist"][:, i], means["moe"][:, i])}
                csv_rows.append({"memory_start": mode, "dr": entry["id"], "profile": entry["profile"], "metric": name,
                                 **{arm: metrics[name][arm] for arm in ARMS},
                                 "moe_reduction_vs_baseline_percent": metrics[name]["moe_vs_baseline"]["reduction_percent"],
                                 "moe_reduction_vs_specialist_percent": metrics[name]["moe_vs_specialist"]["reduction_percent"]})
            for arm in ARMS:
                all_means[arm].append(means[arm])
                failures[arm].append(rows[arm]["failed"])
            reports.append({"id": entry["id"], "profile": entry["profile"], "metrics": metrics,
                            "evaluation_files": {arm: str(path) for arm, path in paths.items()},
                            "common_survival_steps_mean": float(common.mean()),
                            "failure_counts": {arm: sum(rows[arm]["failed"]) for arm in ARMS},
                            "coverage": {arm: rows[arm]["coverage_fraction"] for arm in ARMS},
                            "moe_context": {key: rows["moe"][key] for key in ("initial_context", "final_context", "warmup")}})
        averaged = {arm: np.mean(all_means[arm], axis=0) for arm in ARMS}
        overall = {}
        for i, name in enumerate(names):
            overall[name] = {**{arm: float(averaged[arm][:, i].mean()) for arm in ARMS},
                             "moe_vs_baseline": bootstrap(averaged["baseline"][:, i], averaged["moe"][:, i]),
                             "moe_vs_specialist": bootstrap(averaged["specialist"][:, i], averaged["moe"][:, i]),
                             "expert_gap_recovered": recovery(averaged["baseline"][:, i], averaged["moe"][:, i], averaged["specialist"][:, i])}
        result["modes"][mode] = {"rows": reports, "overall": overall,
            "failure_counts": {arm: int(np.sum(failures[arm])) for arm in ARMS}, "episodes_per_arm": 4096,
            "failure_moe_vs_baseline": bootstrap(np.mean(failures["baseline"], axis=0), np.mean(failures["moe"], axis=0)),
            "failure_moe_vs_specialist": bootstrap(np.mean(failures["specialist"], axis=0), np.mean(failures["moe"], axis=0))}
    (root / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# Uniform DR：共享 MLP、在线 K-means16 MoE 与八个独立专家", "",
             "各 policy 均比较第 1000 轮、196608000 个 PPO transition。MLP 和 MoE 都是 8 卡 × 1024 环境；单个专家是 1 卡 × 8192 环境。MoE actor/critic 各有独立 encoder 和 16 个独立 head；冻结 latent 只负责硬路由，中心随训练更新。", "",
             "原 residual 不限幅、wrist pitch/yaw ±10 Nm、手部 [0,2.5] kg / 小腿 [0,4] kg、uniform motion、奖励和 termination 均与对照一致。", "",
             "每组复用相同 512 条 motion、起点、随机种子和完整静态 DR。下表对三个 policy 使用同一共同存活窗口；因此与之前只比较两方的数值可能略有不同。Cold 从空记忆开始，warm 先用同一冻结 tracker 交互 500 步再 reset 到测试起点；baseline/专家也执行相同 warm 流程。", ""]
    for mode, label in (("cold", "空记忆启动"), ("warm", "500 步冻结 tracker 记忆预热")):
        section = result["modes"][mode]
        lines += [f"## {label}", "", "| DR | 全局 body pos：MLP / MoE / 专家 (cm) | MoE 相对 MLP 改善 | MoE 相对专家改善 |", "|---|---:|---:|---:|"]
        for row in section["rows"]:
            m = row["metrics"]["error_body_pos_global"]
            lines.append(f"| B{row['id']:02d} {row['profile']} | {100*m['baseline']:.3f} / {100*m['moe']:.3f} / {100*m['specialist']:.3f} | {m['moe_vs_baseline']['reduction_percent']:+.2f}% | {m['moe_vs_specialist']['reduction_percent']:+.2f}% |")
        lines += ["", "| 汇总指标 | MLP | MoE | 专家 | MoE 相对 MLP 改善 |", "|---|---:|---:|---:|---:|"]
        for name, label, scale in (("error_body_pos_global", "全局 body pos (cm)", 100),
                                   ("error_anchor_pos_global", "全局 pelvis pos (cm)", 100),
                                   ("error_body_rot_global", "全局 body rot (rad)", 1),
                                   ("error_anchor_rot_global", "全局 pelvis rot (rad)", 1),
                                   ("error_body_pos", "对齐 body pos (cm)", 100),
                                   ("error_joint_pos", "joint pos (rad)", 1),
                                   ("error_body_lin_vel_global", "全局 body lin vel (m/s)", 1),
                                   ("error_body_ang_vel_global", "全局 body ang vel (rad/s)", 1)):
            m = section["overall"][name]
            lines.append(f"| {label} | {scale*m['baseline']:.4f} | {scale*m['moe']:.4f} | {scale*m['specialist']:.4f} | {m['moe_vs_baseline']['reduction_percent']:+.2f}% |")
        m = section["overall"]["error_body_pos_global"]
        gap = m["expert_gap_recovered"]
        lines += ["", f"全局 body 位置：MoE 相对 MLP 改善的配对 95% 区间 {m['moe_vs_baseline']['reduction_percent_ci95']}；MoE 相对专家改善的区间 {m['moe_vs_specialist']['reduction_percent_ci95']}。", "",
                  f"MoE 弥补的 MLP→专家差距：{gap['percent']}%；0% 为 MLP，100% 为专家。", "",
                  f"失败数（每组总共 4096 条 episode）：{section['failure_counts']}。", ""]
    lines += ["同一训练 seed；八个专家合计使用八倍训练样本。MoE 比单头 MLP 参数更多，该结果同时包含容量和路由的影响。区间反映 motion 抽样，不反映训练 seed 的不确定性。", "",
              "完整 20 项指标、置信区间、逐 DR 结果和评测来源见 comparison.json / comparison.csv。所有 policy 和中心在评测时保持冻结。", ""]
    (root / "README.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    report(parser.parse_args().run_root)
