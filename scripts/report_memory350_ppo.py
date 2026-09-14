"""Summarize completed, paired 5000-update tracking tests and publish W&B metrics."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from intact_tracking.memory350_policy_results import compare
from intact_tracking.memory350_policy_checkpoint_eval import audit_saved_evaluations, write_json
from intact_tracking.wandb_logger import WandbLogger

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def run(root):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Report must stay inside the project")
    cases = read(root / "protocols/final.json")["cases"]
    rows, traces, audits = {}, {}, {}
    for fusion in ("baseline", "film", "frozen_tracker"):
        update = 0 if fusion == "frozen_tracker" else 5000
        directory = root / "final" / fusion / "endpoint_eval" / f"update_{update:06d}"
        rows[fusion], traces[fusion] = {}, {}
        for case in cases:
            target = directory / f"{case}.json"
            rows[fusion][case] = read(target)
            with np.load(target.with_suffix(".traces.npz")) as saved:
                traces[fusion][case] = saved["body_joint"].copy()
                if saved["lengths"].tolist() != rows[fusion][case]["episode_lengths"]:
                    raise ValueError("Trace and reported lengths disagree")
        if fusion != "frozen_tracker":
            training = root / "ppo" / f"{fusion}_121"
            completion = read(training / "completion.json")
            if not completion["complete"] or completion["completed_updates"] != 5000:
                raise ValueError("Both arms must complete the matched 5000-update budget")
            audits[fusion] = audit_saved_evaluations(training, 5000, root / "protocols/periodic.json", 0)
    result = {"created_at": time.time(), "training_updates_per_arm": 5000,
              "training_seed": 121, "training_seed_count": 1, "periodic_audits": audits,
              "primary_cases": ["uniform_warm", "uniform_cold"],
              "scope": "Training dataset motions, unseen fixed physics seeds and starts; no unseen-motion generalization claim",
              "uncertainty": "Paired motion bootstrap; does not estimate variability across training seeds",
              "comparisons": {}}
    for reference, candidate in (("baseline", "film"), ("frozen_tracker", "baseline"), ("frozen_tracker", "film")):
        name = f"{candidate}_vs_{reference}"
        result["comparisons"][name] = {}
        for case in cases:
            result["comparisons"][name][case] = compare(rows[reference][case], rows[candidate][case],
                                                        traces[reference][case], traces[candidate][case])
    paired = result["comparisons"]["film_vs_baseline"]
    statements = []
    for case, label in (("uniform_warm", "跨 reset、已有长期记忆"), ("uniform_cold", "冷启动、初始无记忆")):
        row = paired[case]
        body, joint, fail = (row[k] for k in ("common_error_body_pos", "common_error_joint_pos", "failure_rate"))
        statements.append(f"{label}：共同有效时段 body/joint 误差相对 baseline 分别降低 "
                          f"{body['reduction_percent']:.2f}% / {joint['reduction_percent']:.2f}%；"
                          f"失败率变化 {100 * fail['candidate_minus_reference']:+.2f} 个百分点。")
    strong = all(paired[case]["common_error_body_pos"]["reduction_percent_ci95"][0] > 0
                 and paired[case]["common_error_joint_pos"]["reduction_percent_ci95"][0] > 0
                 and paired[case]["failure_rate"]["difference_ci95"][1] <= 0 for case in result["primary_cases"])
    result["strong_improvement_in_both_primary_cases"] = strong
    result["interpretation"] = statements + [
        "两个主要场景均支持 tracking 改善，且失败率未上升。" if strong else
        "尚不满足两个主要场景均改善且失败率未上升的严格判据；需分别解释误差、失败率与置信区间。",
        "这是单个配对训练 seed 的结论，bootstrap 区间只反映测试动作的不确定性。"]
    write_json(root / "final_results.json", result)
    lines = ["# Memory350 residual PPO 对比结果", "", *[line + "\n" for line in result["interpretation"]],
             "两组均完成 5000 PPO updates；每组 2 卡、每卡 8192 环境，全 129827 条 motion。"
             "冻结 tracker 原始 DR + 双手及小腿中部各独立 U(0,4 kg)，1000-step episode，"
             "训练关闭 ee_body_pos termination；评估沿用冻结 tracker 原始失败条件。", "",
             "每项评估含 4096 个固定配对 motion/start/physics。warm 场景先用同一个冻结 tracker "
             "按相同 seed 收集 500 步，再 reset 到测试起点。接触仿真不保证逐位确定性，实际 warm 轨迹摘要均保留。", "",
             "| 场景 | Baseline body m | Latent body m | 共同有效时段 body 降幅（95% CI） | Joint 降幅 | 失败率 baseline → latent | 覆盖率 baseline → latent |",
             "|---|---:|---:|---|---:|---|---|"]
    for case in cases:
        row, a, b = paired[case], rows["baseline"][case], rows["film"][case]
        body, joint = row["common_error_body_pos"], row["common_error_joint_pos"]
        lo, hi = body["reduction_percent_ci95"]
        lines.append(f"| {case} | {a['mean']['error_body_pos']:.5f} | {b['mean']['error_body_pos']:.5f} | "
                     f"{body['reduction_percent']:.2f}% [{lo:.2f}, {hi:.2f}] | {joint['reduction_percent']:.2f}% | "
                     f"{a['failure_rate']:.2%} → {b['failure_rate']:.2%} | "
                     f"{a['coverage_fraction']:.2%} → {b['coverage_fraction']:.2%} |")
    lines += ["", "body 均值按实际存活步数统计，可能受到提前失败影响；主要误差比较使用两组共同有效时段，"
              "并同时报告失败率与追踪覆盖率。固定 0/2/4 kg 为分布切片，uniform 为训练负载分布。", "",
              "完整逐动作、逐步结果与冻结 tracker 对照见本实验目录的 final_results.json 和 final/。", ""]
    (ROOT / "docs/memory350_ppo_results_20260910.md").write_text("\n".join(lines))
    logger = WandbLogger(enabled=True, is_main=True, project="intact-preview-v2",
                         entity="2486344338-zhejiang-university", group="memory350-trackerdr-residual-20260910",
                         name="paired_5000_tracking_summary", output_dir=root, config={
                             "updates": 5000, "training_seed": 121, "final_protocol": read(root / "protocols/final.json")})
    try:
        payload = {}
        for pair, comparisons in result["comparisons"].items():
            for case, metrics in comparisons.items():
                for metric, values in metrics.items():
                    if isinstance(values, dict):
                        for name, value in values.items():
                            if isinstance(value, (float, int)):
                                payload[f"final/{pair}/{case}/{metric}/{name}"] = value
        logger.log(payload, step=5000)
        write_json(root / "final_wandb.json", {"url": logger.url, "id": logger.id})
    finally:
        logger.finish()
    print(json.dumps({"event": "paired_memory350_result", "interpretation": result["interpretation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    run(parser.parse_args().run_root)
