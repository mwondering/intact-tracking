"""Compare a formal load-MoE routing audit with the earlier u2 smoke audit."""

import argparse
import json
from pathlib import Path

import torch

from report_payload_expert_mapping import compact


def read(path):
    return json.loads(path.read_text())


def trace_breakdown(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    trace = data["traces"]["full_grid"]
    ids, weights = trace["ids"], trace["weights"]
    worlds = trace["worlds"]
    masses = data["actual_masses_kg"][worlds]
    labels = data["anchor_ids"][worlds]
    correct = ids == labels[:, None]
    hand_zero = masses[:, :2].abs().sum(1) < 1e-6
    groups = {}
    for name, mask in (("both_hands_zero", hand_zero), ("at_least_one_hand_loaded", ~hand_zero)):
        groups[name] = {"samples": int(mask.sum()),
                        "top1": float(correct[mask, 0].float().mean()),
                        "top5": float(correct[mask].any(1).float().mean()),
                        "true_expert_weight": float((correct[mask] * weights[mask]).sum(1).mean())}
    maximum = weights.max(1).values
    high = maximum >= .9
    confidence = {"max_weight_mean": float(maximum.mean()),
                  "max_weight_mean_correct": float(maximum[correct[:, 0]].mean()),
                  "max_weight_mean_wrong": float(maximum[~correct[:, 0]].mean()),
                  "top1_correct_given_max_weight_ge_0p9": float(correct[high, 0].float().mean()),
                  "all_samples_wrong_and_max_weight_ge_0p9": float((high & ~correct[:, 0]).float().mean())}
    return {"load_groups": groups, "confidence": confidence}


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.run.resolve()
    output = root / f"expert_mapping_u{args.update}"
    provenance = read(output / "provenance.json")
    if provenance["completed_updates"] != args.update:
        raise ValueError("Unexpected audited update")
    old = root / "expert_mapping_u2"
    cases = {
        "u2_mean": read(old / "runtime_mean.json"),
        "u2_sample": read(old / "runtime_mapping.json"),
        "formal_mean": read(output / "runtime_mean.json"),
        "formal_sample": read(output / "runtime_sample.json"),
    }
    traces = {
        "u2_mean": old / "runtime_mean.traffic.pt",
        "u2_sample": old / "runtime_mapping.traffic.pt",
        "formal_mean": output / "runtime_mean.traffic.pt",
        "formal_sample": output / "runtime_sample.traffic.pt",
    }
    summary = {"provenance": provenance, "cases": {}, "change_percentage_points": {},
               "definitions": {
                   "exact_grid_top1": "The main expert matches all four actual load levels; 256-way identification",
                   "exact_grid_top5": "The true load prototype occurs in the selected five experts",
                   "full_history": "50 valid recent steps and 30 complete long chunks; sample steps 500-790 every ten steps",
                   "continuous_nearest": "Nearest load grid point after normalizing by [2.5,2.5,4,4]; not a control-optimal expert label",
                   "comparison_scope": "Same evaluation protocol and seed; u2 is an earlier one-motion smoke model, not a checkpoint on the formal training chain; trajectories can diverge",
                   "statistical_scope": "One diagnostic seed; time samples are correlated; no claim of convergence or policy return superiority"}}
    for name, case in cases.items():
        if name.startswith("formal_") and (case["completed_updates"] != args.update or
                                           Path(case["checkpoint"]) != Path(provenance["checkpoint"])):
            raise ValueError("Runtime audit does not belong to the verified formal checkpoint")
        summary["cases"][name] = {"completed_updates": case["completed_updates"],
            "full_grid": compact(case["mapping"]["full_grid"]),
            "full_continuous": compact(case["mapping"]["full_continuous"]),
            "cold_grid": compact(case["mapping"]["cold_grid"]),
            "gate_sharpness": case["groups"],
            "full_history_top1_switch_fraction": case["full_history_top1_switch_fraction"],
            **trace_breakdown(traces[name])}
    for mode in ("mean", "sample"):
        summary["change_percentage_points"][mode] = {key: 100 * (
            summary["cases"][f"formal_{mode}"]["full_grid"][key] -
            summary["cases"][f"u2_{mode}"]["full_grid"][key]) for key in
            ("exact_grid_top1", "exact_grid_top5", "true_expert_mean_weight", "light_heavy_all_four_accuracy")}
    state = torch.load(provenance["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    std = state["actor_state_dict"]["distribution.std_param"]
    summary["formal_action_std"] = {"mean": float(std.mean()), "min": float(std.min()), "max": float(std.max())}
    del state
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, (name, case) in zip(axes.flat, cases.items(), strict=True):
        confusion = torch.tensor(case["mapping"]["full_grid"]["top1_confusion_counts"])
        fractions = confusion / confusion.sum(1, keepdim=True).clamp_min(1)
        im = ax.imshow(fractions.numpy(), vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        label = (f"Formal u{args.update} {name.removeprefix('formal_')}" if name.startswith("formal_")
                 else f"Smoke u2 {name.removeprefix('u2_')}")
        ax.set_title(f"{label}: exact top-1 {pct(case['mapping']['full_grid']['exact_grid_top1'])}")
        ax.set_xlabel("Selected main expert ID")
        ax.set_ylabel("True load environment ID")
    fig.colorbar(im, ax=list(axes.flat), label="Fraction within each true environment")
    fig.savefig(output / "environment_to_expert.png", dpi=160)
    plt.close(fig)

    lines = [f"# 正式 latent MoE u{args.update} 路由检查", "",
             f"实际 checkpoint：`{provenance['checkpoint']}`，内置 completed_updates={args.update}。",
             f"SHA256：`{provenance['checkpoint_sha256']}`。encoder、中心和温度与原标定一致；训练未中断。", "",
             "1024 worlds：512 个离散负载（256 类各 2 个），512 个连续负载；512-motion 清单，seed 910616，800 步。",
             "下表只统计记忆完整的离散负载样本。Top-1 要求主 expert 与真实四肢档位全部相同。", "",
             "| 模型 / 动作 | Top-1 | Top-5 | 正确专家平均权重 | 主导专家正确的类别 | 样本数 |",
             "|---|---:|---:|---:|---:|---:|"]
    labels = {"u2_mean": "启动测试 u2 / 动作均值", "u2_sample": "启动测试 u2 / 探索动作",
              "formal_mean": f"正式 u{args.update} / 动作均值", "formal_sample": f"正式 u{args.update} / 探索动作"}
    for name, data in summary["cases"].items():
        g = data["full_grid"]
        lines.append(f"| {labels[name]} | {pct(g['exact_grid_top1'])} | {pct(g['exact_grid_top5'])} | "
                     f"{pct(g['true_expert_mean_weight'])} | {g['classes_correct_modal_expert']}/256 | {g['samples']} |")
    lines.extend(["", "| 正式模型，各肢体档位正确率 | 左手 | 右手 | 左小腿 | 右小腿 |", "|---|---:|---:|---:|---:|"])
    for mode in ("mean", "sample"):
        g = summary["cases"][f"formal_{mode}"]["full_grid"]
        lines.append("| " + mode + " | " + " | ".join(map(pct, g["per_limb_top1_exact_level"])) + " |")
    lines.extend(["", "| 正式模型，负载子集 | Top-1 | Top-5 |", "|---|---:|---:|"])
    for mode in ("mean", "sample"):
        for name, g in summary["cases"][f"formal_{mode}"]["load_groups"].items():
            group_label = "双手均零负载" if name == "both_hands_zero" else "至少一手有负载"
            lines.append(f"| {mode} / {group_label} | {pct(g['top1'])} | {pct(g['top5'])} |")
    lines.extend(["", f"正式 checkpoint Gaussian std 均值：{summary['formal_action_std']['mean']:.6f}。", "",
                  "最大路由权重衡量尖锐程度，不等于选对专家。完整连续负载指标、置信度与正确率、冷记忆数据见 summary.json。", "",
                  "![环境与主专家对应矩阵](environment_to_expert.png)", "",
                  "范围：u2 来自启动测试，非正式训练的中间快照。两者评测协议与起始随机种子相同，但控制轨迹可分歧。",
                  "这是单 seed 的路由诊断，时刻样本不是独立实验；不能据此宣称控制性能优于 baseline 或训练已收敛。", ""])
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "change_percentage_points": summary["change_percentage_points"],
                      "formal_action_std": summary["formal_action_std"]}, indent=2))


if __name__ == "__main__":
    main()
