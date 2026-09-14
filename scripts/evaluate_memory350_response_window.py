"""Same cached cross-motion readout and five-step prediction for the response-window arms."""

import json
from pathlib import Path

import evaluate_memory350_weak_pairs as base


def validate_checkpoints(states, comparison_kind):
    a, b = states["baseline"], states["memory350"]
    for key in ("model_config", "loss_config", "normalization", "tracker", "nominal_a_fraction"):
        if a[key] != b[key]:
            raise ValueError(f"Response-window comparison changed {key}")
    if a["nominal_a_fraction"] != .5:
        raise ValueError("Both arms must use nominal50")
    if a["update"] not in (11000, 15000) or b["update"] not in (12000, 13000, 14000, 15000):
        raise ValueError("Expected u11000 reference or the two u15000 endpoints")
    for state in states.values():
        if state["optimizer_steps"] != 4 * state["update"]:
            raise ValueError("Wrong optimizer step budget")
        if state.get("supervision_horizons", {"predictor": 5})["predictor"] != 5:
            raise ValueError("Predictor supervision must still have five steps")
        if (state["scheduler"]["T_max"] != 32000
                or state["scheduler"]["last_epoch"] != state["optimizer_steps"]
                or any(group["lr"] != 1e-5 for group in state["optimizer"]["param_groups"])):
            raise ValueError("Original optimizer schedule must remain unchanged")


def report(summary, output):
    paths = {key: Path(value["path"]) for key, value in summary["checkpoints"].items()}
    horizons = {}
    for name, path in paths.items():
        cfg = json.loads((path.parent / "run_config.json").read_text())
        horizons[name] = cfg["arguments"].get("response_label_horizon", 5)
    summary["response_label_horizons"] = horizons
    summary["predictor_horizon"] = 5
    summary["selection_metric"] = "memory_training / disjoint / top1_accuracy"
    rows = ["# Memory350 响应窗口对比", "",
            f"参考：response{horizons['baseline']} / u{summary['reference_update']}；"
            f"候选：response{horizons['memory350']} / u{summary['update']}。两组 predictor 均监督 5 步。", "",
            "|条件|参考 Top1|候选 Top1|参考 Top5|候选 Top5|", "|---|---:|---:|---:|---:|"]
    for profile, label in (("common", "普通 DR"), ("memory_training", "DR＋负载")):
        readout = summary["readout"][profile]["disjoint"]
        a, b = [readout["models"][key] for key in ("baseline", "memory350")]
        rows.append(f"|{label}（{readout['worlds']} 个中心）|{a['top1_accuracy']:.2%}|{b['top1_accuracy']:.2%}|"
                    f"{a['top5_accuracy']:.2%}|{b['top5_accuracy']:.2%}|")
    rows += ["", "主指标为 DR＋负载的严格跨 motion、历史不重叠识别率。"
             "使用同一缓存轨迹、中心/测试划分和 5 步预测验证。", "",
             "10 步组在 u11000 恢复；5 步组连续训练至 u12000 后恢复。"
             "本实验包含这个恢复时点差异，不能视为完全配对的单因素因果检验。", ""]
    (output / "README.md").write_text("\n".join(rows))


def main():
    base.validate_checkpoints = validate_checkpoints
    base.report = report
    base.main()


if __name__ == "__main__":
    main()
