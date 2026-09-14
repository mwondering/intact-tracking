"""Create local research figures and a Chinese report from completed experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_limb_context_experiment import ppo_directory, experiment_layout, training_termination_profile


def run(root):
    layout = experiment_layout(root)
    result = json.loads((root / "eval/comparison.json").read_text())
    audit = json.loads((root / "independent_training_audit.json").read_text())
    if not result["complete"]:
        raise ValueError("Final evaluation is incomplete")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    colors = {"baseline": "#555555", "film": "#176db5", "concat": "#b66b19", "constant": "#679942"}
    for fusion, seeds in (("baseline", (121, 122, 123)), ("film", (121, 122, 123)),
                          ("concat", (121,)), ("constant", (121,))):
        for index, seed in enumerate(seeds):
            if not (ppo_directory(root) / f"{fusion}_{seed}/metrics.jsonl").exists():
                continue
            history = [json.loads(line) for line in (ppo_directory(root) / f"{fusion}_{seed}/metrics.jsonl").read_text().splitlines()]
            x = np.asarray([row["completed_updates"] for row in history])
            for axis, key in zip(axes[:2], ("mean_reward", "mean_episode_length")):
                y = np.asarray([row[key] if row[key] is not None else np.nan for row in history])
                smoothed = np.asarray([np.nanmean(y[max(0, k-49):k+1]) for k in range(len(y))])
                axis.plot(x, smoothed, color=colors[fusion], alpha=.7,
                          label=fusion if index == 0 else None, linewidth=1)
            if fusion == "film":
                axes[2].plot(x, [row["loss"].get("context_full_fraction", np.nan) for row in history],
                             label=f"film seed {seed}", linewidth=1)
    for axis, title in zip(axes, ("Training episode return (50-update mean)", "Training episode length", "Full 100-frame context fraction")):
        axis.set(title=title, xlabel="Completed PPO updates")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(figures / "ppo_learning.png", dpi=180)
    plt.close(fig)
    stage1 = json.loads((root / "stage1/history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = [row["update"] for row in stage1]
    for key in ("one_step_nmse", "dr_five_step_nmse"):
        axes[0].plot(x, [row["fixed_probe"][key] for row in stage1], label=key)
    for key in ("latent_response_correlation", "latent_positive_cosine"):
        axes[1].plot(x, [row["fixed_probe"][key] for row in stage1], label=key)
    for axis in axes:
        axis.set_xlabel("Stage 1 outer updates")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    axes[0].set_title("Independent broad-context prediction")
    axes[1].set_title("Independent full-context representation")
    fig.savefig(figures / "stage1_validation.png", dpi=180)
    plt.close(fig)
    comparisons = result["comparisons"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for seed in (121, 122, 123):
        rows = [comparisons[f"all_{mass}/film_{seed}_vs_baseline_{seed}"] for mass in range(5)]
        for index in range(2):
            axes[index].plot(range(5), [r["candidate_over_reference_body_joint"][index] for r in rows],
                             marker="o", label=str(seed))
        axes[2].plot(range(5), [100*r["failure_rate_delta"] for r in rows], marker="o", label=str(seed))
    for index, axis in enumerate(axes):
        axis.axhline(1 if index < 2 else 0, color="gray", linestyle="--", linewidth=1)
        axis.set_xlabel("Added mass at each of the four limbs (kg)")
        axis.grid(alpha=.2)
        axis.legend(title="Training seed", fontsize=8)
    axes[0].set_title("Body error: FiLM / baseline")
    axes[1].set_title("Joint error: FiLM / baseline")
    axes[2].set_title("Failure rate: FiLM - baseline (pp)")
    fig.savefig(figures / "load_comparison.png", dpi=180)
    plt.close(fig)

    primary = result["primary_across_training_seeds"]
    resumes = audit["stage1"].get("resume_history", [])
    resume_note = ("- 第一阶段续训记录：" + "；".join(
        f"update {r['completed_update']}，预算 {r['previous_update_limit']}→{r['update_limit'] if r['update_limit'] is not None else '不限'}，"
        "保留模型/Adam/归一化/固定验证样本，重建模拟器及 replay"
        + ("，从原学习率连续延长余弦衰减" if r.get("schedule_extension") else "")
        for r in resumes) + "。" if resumes else "- 第一阶段无断点续训。")
    supported = primary["all_three_seeds_improve_both_tracking_metrics"] and primary["mean_failure_rate_does_not_increase"]
    conclusion = ("均匀负载主评估中，FiLM 的三个训练种子均降低了 body/joint 误差，平均失败率没有增加；本协议下观察到一致的性能收益。"
                  if supported else "均匀负载主评估未同时满足“三个种子均降低 body/joint 误差且平均失败率不增加”；本轮不能宣称稳定的整体性能提升。")
    lines = ["# 四肢负载下 context latent 与 residual PPO：最终结果", "", conclusion, "",
             "## 协议与完成情况", "",
             "- 全部 129,827 条 NPZ motion 参与训练。四个部位分别独立 U(0,4 kg)，固定于 environment 启动；其余 DR 和观测噪声关闭。",
             f"- 腕部与小腿中部负载采用质量/COM/惯量合成；保留原 reward 和 0.25 residual 上限。训练 termination 配置：{training_termination_profile(root)}；评估保留原完整 termination。",
             f"- 第一阶段完成 {audit['stage1']['completed_updates']} outer updates / {audit['stage1']['optimizer_steps']} optimizer steps；选中 update {audit['stage1']['selected_update']}。",
             f"- 第一阶段不设步数上限，不按平台判据自动停止；由用户确认收敛后放行第二阶段：{audit['stage1']['user_confirmed_converged']}。自动平台诊断仅供参考：{audit['stage1'].get('automatic_plateau_detected', False)}。",
             resume_note,
             f"- baseline/FiLM 分别为 seeds 121/122/123；额外对照：{layout['controls']}。每组 {layout['gpus_per_policy']} GPU、每 GPU 8192 environments（全局 {layout['gpus_per_policy'] * 8192}），均完成至少 5000 完整 updates。",
             ("- motion sampling：每组前 1000 updates 为 uniform，从准确的第 1000 轮 checkpoint 恢复模型、Adam、归一化，随后至少 4000 updates 为 adaptive；四肢负载始终独立 U(0,4)，failure rewind 关闭。"
              if (root / "sampling_curriculum.json").exists() else "- motion sampling 全程 uniform，failure rewind 关闭。"),
             "- 64 维 encoder 与其归一化被冻结；PPO 不执行 predictor。actor/critic 原信息分别为 1645/6330 维。残差 actor 和 critic 主干均从头训练；配对 seed 使用相同随机初始权重。",
             "- 底层 tracker 及其特征处理冻结。残差 actor 输出层零初始化、动作标准差统一从 0.25 开始；critic 不加载旧权重或统计，归一化从当前 DR 的首批观测开始累计。旧预训练 critic 的 PPO 不参与比较。",
             "- FiLM 分别调节 actor/critic 的隐藏层。", "",
             "## 主比较：全目录 IID 新起点与新负载", "",
             "每 motion 一个 episode，最多 500 steps。误差为 episode 内均值再等权汇总；同时保留失败、覆盖率和共同存活时间段的结果。", "",
             "| seed | baseline body (m) | FiLM body (m) | body 变化 | joint 变化 | 失败率变化 (百分点) | return 变化 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for seed in (121, 122, 123):
        row = comparisons[f"iid_full/film_{seed}_vs_baseline_{seed}"]
        ratio = row["candidate_over_reference_body_joint"]
        means = row["means"]
        lines.append(f"| {seed} | {means['reference_body']:.5f} | {means['candidate_body']:.5f} | {100*(ratio[0]-1):+.2f}% | {100*(ratio[1]-1):+.2f}% | {100*row['failure_rate_delta']:+.3f} | {row['episode_return_delta']:+.3f} |")
    lines += ["", "| seed | 共同存活段 body 变化 | 共同存活段 joint 变化 | 新失败 | 救回 | 覆盖率变化 (百分点) |",
              "|---|---:|---:|---:|---:|---:|"]
    for seed in (121, 122, 123):
        row = comparisons[f"iid_full/film_{seed}_vs_baseline_{seed}"]
        ratio = row["common_prefix_body_joint_ratio"]
        lines.append(f"| {seed} | {100*(ratio[0]-1):+.2f}% | {100*(ratio[1]-1):+.2f}% | {row['new_failure_episodes']} | {row['rescued_failure_episodes']} | {100*row['coverage_delta']:+.3f} |")
    lines += ["", "## 固定负载与不对称布局", "",
              "全 0/1/2/3/4 kg 使用同一组 4096 条分层采样 motion 与相同起点；不对称组合使用同一组 1024 条。表中是三个配对训练种子的平均相对变化。", "",
              "| 负载条件 | body 变化 | joint 变化 | 失败率变化 (百分点) |", "|---|---:|---:|---:|"]
    for case in result["protocol"]["cases"]:
        name = case["name"]
        if name in ("iid_full", "latent_intervention"):
            continue
        rows = [comparisons[f"{name}/film_{seed}_vs_baseline_{seed}"] for seed in (121, 122, 123)]
        ratio = np.mean([r["candidate_over_reference_body_joint"] for r in rows], 0)
        delta = np.mean([r["failure_rate_delta"] for r in rows])
        lines.append(f"| {name}: {case['fixed_masses']} | {100*(ratio[0]-1):+.2f}% | {100*(ratio[1]-1):+.2f}% | {100*delta:+.3f} |")
    if layout["controls"]:
        lines += ["", "## 融合方式与常量分支对照（单训练种子）", "",
                  "| 比较 | body 变化 | joint 变化 | 失败率变化 (百分点) |", "|---|---:|---:|---:|"]
    for key in ("concat_121_vs_baseline_121", "film_121_vs_concat_121", "film_121_vs_constant_121"):
        if f"iid_full/{key}" not in comparisons:
            continue
        row = comparisons["iid_full/" + key]
        ratio = row["candidate_over_reference_body_joint"]
        lines.append(f"| {key} | {100*(ratio[0]-1):+.2f}% | {100*(ratio[1]-1):+.2f}% | {100*row['failure_rate_delta']:+.3f} |")
    lines += ["", "## Latent 闭环干预", "",
              "对三个 FiLM 种子分别比较正确、全零和跨负载交换 latent。交换仅在 motion/phase 相同、两边仍存活且历史均满 100 帧时执行。", "",
              "| seed | 干预相对正确 latent | body 变化 | joint 变化 | 失败率变化 (百分点) |", "|---|---|---:|---:|---:|"]
    for seed in (121, 122, 123):
        for mode in ("zero", "paired-swap"):
            row = comparisons[f"latent_intervention/film_{seed}/{mode}_vs_correct"]
            ratio = row["candidate_over_reference_body_joint"]
            lines.append(f"| {seed} | {mode} | {100*(ratio[0]-1):+.2f}% | {100*(ratio[1]-1):+.2f}% | {100*row['failure_rate_delta']:+.3f} |")
    gap = result.get("gap_vs_original_termination_batch")
    if gap:
        lines += ["", "## 取消训练终止前后的 latent 优势", "",
                  "相同 seed 121、5000 updates、motions/starts 和原评估终止标准。每组训练环境数由 16384 增至 32768，因此差距变化同时包含每轮样本量变化。", "",
                  "| 负载 | 旧 FiLM/B body 比 | 新 FiLM/B body 比 | body 相对优势增加 (百分点) | joint 相对优势增加 (百分点) |",
                  "|---|---:|---:|---:|---:|"]
        for case, row in gap["cases"].items():
            increase = row["increase_in_relative_body_joint_advantage_pp"]
            lines.append(f"| {case} | {row['old_body_joint_ratios'][0]:.4f} | {row['new_body_joint_ratios'][0]:.4f} | {increase[0]:+.2f} | {increase[1]:+.2f} |")
    lines += ["", "## 解释范围", "",
              "- 主结论针对增加预训练 latent 的整体性能，不拆分 actor/critic 贡献。", 
              "- 全零 latent 属于输入干预；跨负载交换虽匹配 motion/phase/历史长度，实际状态仍可能不同。这些诊断不能单独证明 latent 只编码负载。",
              "- 训练与评估使用同一 motion 目录；评估是新负载、新起点，不能据此宣称未见 motion 泛化。",
              "- JSON 中的 95% 区间由 motion 配对 bootstrap 得到，不能当作训练种子不确定性；训练种子只有三个。", "",
              "## 产物", "", f"运行目录：`{root}`。",
              "完整指标与区间：`eval/comparison.json`；逐 episode 及逐步误差：`eval/episodes/`；独立训练审计：`independent_training_audit.json`。", "",
              f"![PPO 学习曲线]({figures / 'ppo_learning.png'})", "",
              f"![第一阶段验证]({figures / 'stage1_validation.png'})", "",
              f"![固定负载结果]({figures / 'load_comparison.png'})", ""]
    report = root / "report.md"
    report.write_text("\n".join(lines))
    name = "limb_context_no_ee_results.md" if training_termination_profile(root) == "no_ee_body_pos" else "limb_context_results.md"
    (Path(__file__).resolve().parents[1] / "docs" / name).write_text("\n".join(lines))
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    run(parser.parse_args().run_root.resolve())
