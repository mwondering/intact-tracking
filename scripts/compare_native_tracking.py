"""Compare matched native-DR evaluation files, including global tracking traces."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit_query_forces(reference, candidate, required_steps):
    """Verify actual forces over every scored pair's common time interval."""
    force_fields = {"query_force_steps", "query_force_sha256", "query_force_prefix_sha256"}
    if ({k: v for k, v in reference.items() if k not in force_fields}
            != {k: v for k, v in candidate.items() if k not in force_fields}):
        raise ValueError("Unmatched non-force evaluation diagnostics")
    lengths = [row["query_force_steps"] for row in (reference, candidate)]
    common_steps = min(lengths)
    if required_steps < 1 or common_steps < required_steps:
        raise ValueError("Force audit does not cover the common scored trajectory")
    for row, length in zip((reference, candidate), lengths, strict=True):
        prefixes = row.get("query_force_prefix_sha256")
        if prefixes is not None and (len(prefixes) != length or prefixes[-1] != row["query_force_sha256"]):
            raise ValueError("Incomplete force prefix audit")
    if lengths[0] == lengths[1]:
        # Preserve compatibility with completed evaluations predating prefix
        # hashes; their full hashes already prove an equally long trajectory.
        matched = reference["query_force_sha256"] == candidate["query_force_sha256"]
    else:
        if any("query_force_prefix_sha256" not in row for row in (reference, candidate)):
            raise ValueError("Unequal query lengths require force prefix hashes; rerun both evaluations")
        matched = (reference["query_force_prefix_sha256"][common_steps - 1]
                   == candidate["query_force_prefix_sha256"][common_steps - 1])
    if not matched:
        raise ValueError("Unmatched forces in the common query prefix")
    return common_steps, lengths[0] == lengths[1]


def compare(directory, bootstrap_samples=4000):
    directory = Path(directory).resolve()
    protocol = json.loads((directory / "protocol.json").read_text())
    reference = json.loads((directory / "frozen_tracker.json").read_text())
    candidate = json.loads((directory / "residual.json").read_text())
    paired = (
        "protocol", "seed", "motion_files", "motion_ids", "start_frames", "horizons",
        "metric_names", "max_steps", "physics_world_fingerprints", "is_nominal",
        "reward_contract", "query_initial_state_sha256", "global_metric_contract",
    )
    for key in paired:
        if reference[key] != candidate[key]:
            raise ValueError(f"Unmatched comparison field: {key}")
    if not reference.get("frozen_tracker_weights_verified"):
        raise ValueError("Frozen baseline weights were not verified against the original tracker")
    if reference["fusion"] != "frozen":
        raise ValueError("This comparison requires a tracker-only baseline, without latent input")
    for key in ("steps", "policy", "policy_sha256", "seed"):
        if reference["warmup"][key] != candidate["warmup"][key]:
            raise ValueError(f"Unmatched warmup configuration: {key}")
    # The baseline never reads context memory. The query is reset with matched
    # state, physical/controller parameters and force trajectory (audited above).
    # Separate GPU simulations need not yield bit-identical warmup trajectories;
    # retain that observation instead of claiming the histories were identical.
    warmup_identical = reference["warmup"] == candidate["warmup"]
    if (reference["checkpoint_sha256"] != protocol["tracker_sha256"]
            or candidate["checkpoint_sha256"] != protocol["checkpoint_sha256"]
            or candidate["completed_training_updates"] != protocol["completed_updates"]
            or candidate["context_sha256"] != protocol["context_sha256"]):
        raise ValueError("Evaluated checkpoint identities differ from the selected protocol")
    for row in (reference, candidate):
        if row["memory_start"] != "warm" or row["extra_payload"]:
            raise ValueError("Use the declared warm/native-DR/no-extra-payload comparison")
        if not all(row[k] for k in ("reference_timeline_audited", "partial_reset_survivor_state_audited",
                                    "partial_reset_survivor_history_audited")):
            raise ValueError("Missing survivor/timeline audit")
    rn = np.asarray(reference["episode_lengths"])
    cn = np.asarray(candidate["episode_lengths"])
    common = np.minimum(rn, cn)
    if (common < 1).any():
        raise ValueError("Empty paired episode")
    force_steps, full_forces_identical = audit_query_forces(
        reference["evaluation_diagnostics"], candidate["evaluation_diagnostics"], int(common.max()))
    traces = []
    for role, counts in (("frozen_tracker", rn), ("residual", cn)):
        with np.load(directory / f"{role}.traces.npz") as saved:
            np.testing.assert_array_equal(saved["lengths"], counts)
            np.testing.assert_array_equal(saved["metric_names"], reference["metric_names"])
            value = saved["all_metrics"]
            mask = np.arange(value.shape[1])[None, :] < common[:, None]
            traces.append((value * mask[..., None]).sum(1, dtype=np.float64) / common[:, None])
    rcommon, ccommon = traces
    rraw = np.asarray(reference["per_episode_metrics"])
    craw = np.asarray(candidate["per_episode_metrics"])
    rf, cf = np.asarray(reference["failed"], bool), np.asarray(candidate["failed"], bool)
    nominal = np.asarray(reference["is_nominal"], bool)
    ids = np.asarray(reference["motion_ids"])
    horizons = np.asarray(reference["horizons"])
    groups = {}
    for name, mask in (("all", np.ones(len(ids), bool)), ("dr_only", ~nominal), ("nominal_only", nominal)):
        if not mask.any():
            continue
        unique, index = np.unique(ids[mask], return_inverse=True)
        count = np.bincount(index, minlength=len(unique))
        m = rcommon.shape[1]
        aggregates = np.zeros((len(unique), 2 * m + 1))
        np.add.at(aggregates, index, np.column_stack((rcommon[mask], ccommon[mask],
                                                     cf[mask].astype(float) - rf[mask])))
        rng = np.random.default_rng(7712)
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=bootstrap_samples)
        draws = weights @ aggregates
        ratio = 100 * (draws[:, m:2*m] / np.maximum(draws[:, :m], 1e-12) - 1)
        ci = np.quantile(ratio, (.025, .975), axis=0)
        fail_ci = np.quantile(100 * draws[:, -1] / (weights @ count), (.025, .975))
        metric_rows = {}
        for i, metric in enumerate(reference["metric_names"]):
            before, after = float(rcommon[mask, i].mean()), float(ccommon[mask, i].mean())
            raw_before, raw_after = float(rraw[mask, i].mean()), float(craw[mask, i].mean())
            metric_rows[metric] = {
                "frozen": before, "residual": after, "change_percent": 100 * (after / before - 1),
                "change_percent_ci95": ci[:, i].tolist(),
                "own_episode_frozen": raw_before, "own_episode_residual": raw_after,
                "own_episode_change_percent": 100 * (raw_after / raw_before - 1),
            }
        groups[name] = {
            "episodes": int(mask.sum()), "motions": len(unique), "metrics": metric_rows,
            "frozen_failures": int(rf[mask].sum()), "residual_failures": int(cf[mask].sum()),
            "new_failure_episodes": int((~rf & cf & mask).sum()),
            "rescued_failure_episodes": int((rf & ~cf & mask).sum()),
            "failure_rate_change_percentage_points": 100 * float(cf[mask].mean() - rf[mask].mean()),
            "failure_rate_change_percentage_points_ci95": fail_ci.tolist(),
            "frozen_coverage": float((rn[mask] / horizons[mask]).mean()),
            "residual_coverage": float((cn[mask] / horizons[mask]).mean()),
            "mean_common_control_steps": float(common[mask].mean()),
        }
    report = {
        "checkpoint": protocol["checkpoint"], "completed_updates": protocol["completed_updates"],
        "context_checkpoint": protocol["context_checkpoint"], "tracker_checkpoint": protocol["tracker_checkpoint"],
        "pairing_audit_passed": True, "paired_fields": list(paired), "warmup_identical": warmup_identical,
        "warmup_configuration_identical": True,
        "warmup_trajectory_sha256": {"frozen": reference["warmup"]["trajectory_sample_sha256"],
                                     "residual": candidate["warmup"]["trajectory_sample_sha256"]},
        "warmup_comparison_scope": "The frozen baseline does not consume latent history; query state, physical/controller parameters and forces must match exactly",
        "force_trajectory_identical": full_forces_identical,
        "common_force_prefix_identical": True, "force_audited_common_steps": force_steps,
        "bootstrap_samples": bootstrap_samples,
        "metric_convention": "Equal weight per episode; paired common valid prefix for each policy pair; lower errors are better",
        "uncertainty": "Paired motion-cluster bootstrap, conditional on this training checkpoint and evaluation sample",
        "scope": protocol["scope"], "groups": groups,
        "initial_context": candidate["initial_context"], "final_context": candidate["final_context"],
    }
    (directory / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    labels = {
        "error_anchor_pos_global": "全局 root 位置 (m)",
        "error_anchor_xy_global": "全局 root XY (m)",
        "error_anchor_rot_global": "全局 root 姿态 (rad)",
        "error_body_pos_global": "全局 body 位置 (m)",
        "error_body_pos": "局部 body 位置 (m)",
        "error_joint_pos": "关节角整体 L2 误差 (rad)",
    }
    lines = [f"# Residual PPO u{protocol['completed_updates']} 与冻结 tracker", "",
             f"checkpoint: `{protocol['checkpoint']}`", "",
             f"{protocol['motions']} 条随机 motion × {protocol['repeats']} 个环境，共 {len(ids)} 对；原生 DR，{int(nominal.sum())} 个 nominal，无额外负载。",
             f"两组按相同策略与种子由冻结 tracker 预热 {protocol['warmup_steps']} 控制步，再评测最多 {protocol['steps']} 步。正式评测的环境参数、初始状态一致，共同仿真时段的逐步推力已校验一致（{force_steps} 步，覆盖全部配对计分时段）。",
             "误差按每对轨迹共同有效时段计算，再对 episode 等权平均；同时列出失败数。负的变化百分比表示改善。", ""]
    if not warmup_identical:
        lines.extend(["预热轨迹未逐位一致：相同首步动作之后的仿真状态已经不同。冻结 tracker 对照不读取 latent 历史；正式评测前两组均重置到完全相同的 qpos/qvel，并核对物理/控制器参数和逐步推力一致。本结果不声称跨 GPU 仿真轨迹逐位可复现。", ""])
    for name, label in (("all", "完整混合环境"), ("dr_only", "仅 DR 环境"), ("nominal_only", "仅 nominal 环境")):
        row = groups[name]
        lines.extend([f"## {label}：{row['episodes']} 个 episode", "",
                      "| 指标 | 冻结 tracker | Residual PPO | 变化 | 变化的 95% 区间 |",
                      "|---|---:|---:|---:|---:|"])
        for metric, title in labels.items():
            x = row["metrics"][metric]
            lo, hi = x["change_percent_ci95"]
            lines.append(f"| {title} | {x['frozen']:.5f} | {x['residual']:.5f} | {x['change_percent']:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] |")
        lines.extend(["", f"失败：{row['frozen_failures']} → {row['residual_failures']}；"
                      f"新增失败 {row['new_failure_episodes']}，避免失败 {row['rescued_failure_episodes']}。",
                      f"覆盖率：{row['frozen_coverage']:.2%} → {row['residual_coverage']:.2%}。", ""])
    lines.extend(["此结果针对训练数据目录中的抽样 motion 和新 DR/起点；不是未见 motion 泛化测试。",
                  "置信区间来自按 motion 分组的配对 bootstrap，不包含不同训练随机种子的变异。", ""])
    (directory / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"completed_updates": report["completed_updates"], "pairing_audit_passed": True,
                      "groups": groups}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    compare(args.directory, args.bootstrap_samples)
