"""Quantify differences between expert action means on exactly shared inputs."""

import argparse
import json
import math
from pathlib import Path

import torch


def rms(value):
    return float(value.square().mean().sqrt())


def pairwise_summary(actions):
    """Average squared difference over every distinct expert pair and joint."""
    count = actions.shape[1]
    variance = actions.var(dim=1, correction=0)
    pair_variance = (2 * count / (count - 1)) * variance
    per_observation = pair_variance.mean(-1).sqrt()
    return {"pairwise_joint_rms": float(pair_variance.mean().sqrt()),
            "per_observation_pairwise_rms_quantiles_10_50_90":
                per_observation.quantile(torch.tensor([.1, .5, .9])).tolist(),
            "per_joint_pairwise_rms": pair_variance.mean(0).sqrt().tolist()}


def summarize(data):
    tensors = {key: torch.cat([s[key] for s in data["snapshots"]]) for key in
               ("worlds", "step", "expert_residual_means", "ids", "weights", "actual_residual_mean", "action_std", "joint_target_scale")}
    means = tensors["expert_residual_means"]
    ids, weights = tensors["ids"], tensors["weights"]
    mixed = tensors["actual_residual_mean"]
    selected = means.gather(1, ids[..., None].expand(-1, -1, means.shape[-1]))
    torch.testing.assert_close((selected * weights[..., None]).sum(1), mixed, rtol=2e-4, atol=2e-5)
    scale = tensors["joint_target_scale"]
    joint_target = means * scale[:, None]
    selected_target = selected * scale[:, None]
    p = torch.triu_indices(5, 5, offset=1)
    cosine = torch.nn.functional.cosine_similarity(selected[:, p[0]], selected[:, p[1]], dim=-1)
    cosine_valid = (selected[:, p[0]].norm(dim=-1) > 1e-8) & (selected[:, p[1]].norm(dim=-1) > 1e-8)
    weighted_variance = (weights[..., None] * (selected - mixed[:, None]).square()).sum(1)
    expected_norm = (weights * selected.norm(dim=-1)).sum(1)
    cancel_ratio = mixed.norm(dim=-1) / expected_norm.clamp_min(1e-12)
    ordered = means.permute(1, 0, 2).reshape(means.shape[1], -1)
    gram = (ordered @ ordered.T) / ordered.shape[1]
    d2 = (gram.diag()[:, None] + gram.diag()[None] - 2 * gram).clamp_min(0)
    distances = d2.sqrt()
    grid = data["grid_kg"]
    normalized_grid = grid / grid.max(0).values
    load_distances = torch.cdist(normalized_grid, normalized_grid)
    pairs = torch.triu_indices(len(grid), len(grid), offset=1)
    load_d = load_distances[pairs[0], pairs[1]]
    action_d = distances[pairs[0], pairs[1]]
    corr = torch.corrcoef(torch.stack((load_d, action_d)))[0, 1]
    neighboring = torch.isclose(load_d, torch.tensor(1 / 3), rtol=1e-5, atol=1e-6)
    far = load_d >= load_d.quantile(.8)
    random = torch.Generator().manual_seed(910616)
    perm = torch.randperm(len(means), generator=random)
    alternate = (means.gather(1, ids[perm, :, None].expand(-1, -1, 29)) * weights[perm, :, None]).sum(1)
    anchor = data["anchor_ids"][tensors["worlds"]]
    grid_mask = anchor >= 0
    oracle = means[torch.arange(len(means))[grid_mask], anchor[grid_mask]]
    all_stats, top_stats = pairwise_summary(means), pairwise_summary(selected)
    result = {"observations": len(means), "unique_worlds": int(tensors["worlds"].unique().numel()),
              "snapshot_steps": tensors["step"].unique().tolist(),
              "all_256": all_stats, "routed_top5": top_stats,
              "all_256_target_radians": pairwise_summary(joint_target),
              "routed_top5_target_radians": pairwise_summary(selected_target),
              "top1_vs_top2_joint_rms": rms(selected[:, 0] - selected[:, 1]),
              "top1_vs_mixture_joint_rms": rms(selected[:, 0] - mixed),
              "actual_mixture_residual_rms": rms(mixed),
              "selected_expert_residual_rms": rms(selected),
              "gaussian_exploration_std_rms": rms(tensors["action_std"]),
              "weighted_expert_spread_joint_rms": float(weighted_variance.mean().sqrt()),
              "selected_residual_cosine_mean": float(cosine[cosine_valid].mean()),
              "selected_residual_cosine_quantiles_10_50_90": cosine[cosine_valid].quantile(torch.tensor([.1, .5, .9])).tolist(),
              "fraction_selected_pairs_cosine_negative": float((cosine[cosine_valid] < 0).float().mean()),
              "weighted_mixture_norm_over_weighted_expert_norm_mean": float(cancel_ratio.mean()),
              "route_from_another_observation_action_change_joint_rms": rms(alternate - mixed),
              "actual_route_vs_true_grid_expert_joint_rms": rms(oracle - mixed[grid_mask]),
              "load_distance_vs_action_distance_pearson": float(corr),
              "adjacent_load_expert_pair_rms_mean": float(action_d[neighboring].mean()),
              "far_load_expert_pair_rms_mean": float(action_d[far].mean()),
              "joint_names": data["joint_names"],
              "dense_sparse_max_absolute_error": max(s["dense_sparse_max_absolute_error"] for s in data["snapshots"])}
    result["top5_difference_over_actual_residual"] = top_stats["pairwise_joint_rms"] / max(result["actual_mixture_residual_rms"], 1e-12)
    result["top5_difference_over_exploration_std"] = top_stats["pairwise_joint_rms"] / max(result["gaussian_exploration_std_rms"], 1e-12)
    masses = data["actual_masses_kg"][tensors["worlds"]]
    result["groups"] = {}
    for name, mask in (("grid", grid_mask), ("continuous", ~grid_mask),
                       ("both_hands_zero", masses[:, :2].abs().sum(1) < 1e-6)):
        if mask.any():
            result["groups"][name] = {"observations": int(mask.sum()),
                "top5": pairwise_summary(selected[mask]), "residual_rms": rms(mixed[mask])}
    per_joint = torch.tensor(result["routed_top5_target_radians"]["per_joint_pairwise_rms"])
    result["largest_top5_joint_differences"] = [
        {"name": data["joint_names"][i], "raw_action_rms": top_stats["per_joint_pairwise_rms"][i],
         "target_radians_rms": float(per_joint[i]), "target_degrees_rms": float(per_joint[i]) * 180 / math.pi}
        for i in per_joint.argsort(descending=True)[:8].tolist()]
    return result, distances, load_d, action_d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results, plotting = {}, {}
    provenance = json.loads((args.directory / "provenance.json").read_text())
    for mode in ("mean", "sample"):
        data = torch.load(args.directory / f"runtime_{mode}.expert_actions.pt", map_location="cpu", weights_only=False)
        if data["completed_updates"] != provenance["completed_updates"] or data["checkpoint"] != provenance["checkpoint"]:
            raise ValueError("Expert means do not come from the audited formal checkpoint")
        result, distances, load_distances, action_distances = summarize(data)
        results[mode] = result
        plotting[mode] = (distances, load_distances, action_distances)
    report = {"checkpoint": provenance["checkpoint"], "completed_updates": provenance["completed_updates"],
              "cases": results,
              "definition": "For each observation, hold all inputs fixed; compare deterministic expert residual means. Pairwise joint RMS pools squared differences over observations, distinct expert pairs and 29 joints.",
              "interpretation": "Different expert means establish output diversity, not control benefit. All-256 comparisons include experts that would not normally receive an observation. Top-5 reflects currently selected experts.",
              "target_angle_units": "Raw action difference times the actual action scale, before actuator dynamics; radians. This is not a measured torque or executed movement."}
    (args.directory / "expert_action_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, mode in zip(axes, ("mean", "sample"), strict=True):
        matrix = plotting[mode][0]
        im = ax.imshow(matrix.numpy(), cmap="magma", interpolation="nearest", vmin=0)
        ax.set_title(f"u{provenance['completed_updates']} / {mode} trajectories")
        ax.set_xlabel("Expert ID"); ax.set_ylabel("Expert ID")
        fig.colorbar(im, ax=ax, label="Same-obs action RMS difference")
    fig.savefig(args.directory / "expert_action_distances.png", dpi=170)
    plt.close(fig)
    lines = ["# 同 obs 下不同 expert 的动作差异", "", f"正式 checkpoint u{provenance['completed_updates']}。", "",
             "所有比较使用同一份 obs、同一 tracker action，并比较动作均值。",
             "冻结 tracker 项在相减时消去，因此这里的 residual 差值也等于最终 action 均值的差值。", "",
             "| 轨迹来源 | 观测数 | 全部256两两RMS差 | 路由Top5两两RMS差 | 实际residual RMS | 探索std RMS | Top5目标角RMS差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, r in results.items():
        degrees = r['routed_top5_target_radians']['pairwise_joint_rms'] * 180 / math.pi
        lines.append(f"| {mode} | {r['observations']} | {r['all_256']['pairwise_joint_rms']:.5f} | "
                     f"{r['routed_top5']['pairwise_joint_rms']:.5f} | {r['actual_mixture_residual_rms']:.5f} | "
                     f"{r['gaussian_exploration_std_rms']:.5f} | {degrees:.3f}° |")
    lines.extend(["", "两两 RMS 差：先求同一观测下所有不同专家对的 29 维动作差平方，跨观测、专家对、关节取平均再开方。",
                  "全部 256 的结果包含不常接收该观测的专家；判断实际分工应优先看被路由的 Top-5。", "",
                  "| 轨迹来源 | Top5残差方向平均cosine | 方向夹角>90°比例 | 相邻负载专家动作差 | 远负载专家动作差 | 负载距离与动作差相关系数 | 更换观测来源的路由后输出差 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for mode, r in results.items():
        lines.append(f"| {mode} | {r['selected_residual_cosine_mean']:.4f} | "
                     f"{r['fraction_selected_pairs_cosine_negative'] * 100:.2f}% | "
                     f"{r['adjacent_load_expert_pair_rms_mean']:.4f} | {r['far_load_expert_pair_rms_mean']:.4f} | "
                     f"{r['load_distance_vs_action_distance_pearson']:.4f} | {r['route_from_another_observation_action_change_joint_rms']:.4f} |")
    lines.extend(["", "相邻负载：仅一个肢体相差一档；远负载：归一化负载距离最高20%的专家对。",
                  "更换路由仅在同 obs 上离线重算混合动作，并未施加给仿真。较弱的全局距离相关性不能单独证明专家无用。", "",
                  "动作不同并不证明动作差异有益；这里没有替换专家实际执行后的回报对照。", "",
                  "![专家动作差异矩阵](expert_action_distances.png)", ""])
    outlier_path = args.directory / "expert_action_outlier_check.json"
    if outlier_path.exists():
        outlier = json.loads(outlier_path.read_text())
        lines.extend(["排除输出RMS最大的 expert 30 后，未路由该专家的样本中，Top5差异RMS仍为 "
                      f"mean {outlier['mean']['top5_without_largest_expert_pair_rms']:.4f} / "
                      f"sample {outlier['sample']['top5_without_largest_expert_pair_rms']:.4f}。详见 expert_action_outlier_check.json。", ""])
    (args.directory / "expert_actions.md").write_text("\n".join(lines))
    print(json.dumps({mode: {k: r[k] for k in ("observations", "actual_mixture_residual_rms",
        "top5_difference_over_actual_residual", "top5_difference_over_exploration_std",
        "selected_residual_cosine_mean", "load_distance_vs_action_distance_pearson")} for mode, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
