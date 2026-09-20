"""Measure warm reset effects from saved online PPO trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def auc(negative, positive):
    n, p = len(negative), len(positive)
    if not n or not p:
        return None
    ranks = rankdata(np.r_[negative, positive])
    return float((ranks[n:].sum() - p * (p + 1) / 2) / (n * p))


def distribution(x):
    return {"count": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95))} if len(x) else {"count": 0}


def run(directory):
    directory = Path(directory)
    report = {"source": str(directory), "checkpoint_updates": 2501, "encoder_updates": 7179,
              "scope": "Existing fixed-checkpoint online trajectories, long memory full; no new simulation",
              "sampling_stride": 5, "arms": {},
              "limitation": "Natural reset changes state, motion and five-frame policy input too; these comparisons do not isolate a causal short-history effect on control."}
    for mode in ("sampled", "mean"):
        with np.load(directory / f"{mode}.npz") as saved:
            data = {key: saved[key] for key in saved.files}
        short, long = data["short_count"], data["long_count"]
        z = data["latent"].astype(np.float64)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        full = (short == 50) & (long == 30)
        center = (z * full[..., None]).sum(0) / full.sum(0)[:, None]
        center /= np.linalg.norm(center, axis=-1, keepdims=True)
        own_distance = np.linalg.norm(z - center[None], axis=-1)
        groups = {}
        for title, lower, upper in (("short0", 0, 0), ("short1_4", 1, 4), ("short5_9", 5, 9),
                                    ("short10_24", 10, 24), ("short25_49", 25, 49), ("short50", 50, 50)):
            selected = (short >= lower) & (short <= upper) & (long == 30)
            group = {}
            for label, nominal in (("nominal", data["is_nominal"]), ("dr", ~data["is_nominal"])):
                mask = selected & nominal[None]
                group[label] = {
                    "anchor_distance": distribution(data["radius"][mask]),
                    "own_world_full_center_distance": distribution(own_distance[mask]),
                    "residual_rms": float(np.sqrt(np.square(data["residual"][mask].astype(np.float64)).mean())),
                    "raw_latent_norm_mean": float(np.linalg.norm(data["latent"][mask], axis=-1).mean()),
                    "worlds": int(mask.any(0).sum())}
            neg = data["radius"][selected & data["is_nominal"][None]]
            pos = data["radius"][selected & ~data["is_nominal"][None]]
            group["auroc"] = auc(neg, pos)
            group["balanced_accuracy_at_0_4"] = float(((neg < .4).mean() + (pos >= .4).mean()) / 2)
            groups[title] = group
        # Pair exactly the same environment and uninterrupted new episode at
        # reset (short=0) and 50 steps later (10 saved sample intervals).
        lag = 10
        valid = ((short[:-lag] == 0) & (long[:-lag] == 30) & full[lag:]
                 & (data["episode_id"][:-lag] == data["episode_id"][lag:])
                 & (data["motion_id"][:-lag] == data["motion_id"][lag:]))
        paired = {}
        distance = np.linalg.norm(z[lag:] - z[:-lag], axis=-1)
        for label, nominal in (("nominal", data["is_nominal"]), ("dr", ~data["is_nominal"])):
            mask = valid & nominal[None]
            paired[label] = {"unit_latent_displacement": distribution(distance[mask]),
                             "reset_anchor_distance": distribution(data["radius"][:-lag][mask]),
                             "after50_anchor_distance": distribution(data["radius"][lag:][mask])}
        report["arms"][mode] = {"groups": groups, "same_episode_zero_to_50": paired}
    (directory / "reset_latents.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Warm reset 后的 latent", "",
             "使用 PPO u2501 / encoder u7179 的既有在线轨迹。仅统计长期 30 块完整的样本；short0 指短期历史恰好为 0。", ""]
    for mode, arm in report["arms"].items():
        lines += [f"## {mode}", "", "| 短期历史 | nominal 帧数 | DR 帧数 | nominal 锚点距离中位数 | DR 锚点距离中位数 | AUROC | nominal residual RMS | DR residual RMS |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label, row in arm["groups"].items():
            n, d = row["nominal"], row["dr"]
            lines.append(f"| {label} | {n['anchor_distance']['count']} | {d['anchor_distance']['count']} | {n['anchor_distance']['median']:.5f} | {d['anchor_distance']['median']:.5f} | {row['auroc']:.6f} | {n['residual_rms']:.5f} | {d['residual_rms']:.5f} |")
        lines += ["", "同环境、同一新 episode 的 reset 帧与 50 步后配对："]
        for label, row in arm["same_episode_zero_to_50"].items():
            d = row["unit_latent_displacement"]
            lines += [f"- {label}: {d['count']} 对，单位 latent 位移中位数 {d.get('median', float('nan')):.5f}，P95 {d.get('p95', float('nan')):.5f}。"]
        lines += [""]
    lines += ["短期历史清空时，旧短期中完整的 10 步块先归档到长期记忆；长期表示会更新，并非原样冻结。", "",
              "自然 reset 同时改变机器人状态、motion 和 policy 五帧队列，因此不能把 residual 幅度变化单独归因于 encoder 短期历史。此处也没有测量 tracking 的因果收益或损失。", ""]
    (directory / "RESET_LATENTS.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    run(parser.parse_args().directory)
