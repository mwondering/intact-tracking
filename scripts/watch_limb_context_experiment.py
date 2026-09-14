"""Observe matched checkpoint trends and experiment health without controlling training."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path

from report_limb_context_endpoints import ROOT, completed_results, write_state
from status_limb_context_experiment import last_records


def matched_results(results):
    groups = {}
    for job, row in results:
        fusion, _, seed = job["name"].rpartition("_")
        if fusion not in ("baseline", "film") or row["completed_updates"] <= 0:
            continue
        key = (seed, row["completed_updates"], row["protocol_sha256"])
        group = groups.setdefault(key, {})
        if fusion in group:
            raise ValueError(f"Duplicate completed checkpoint for {job['name']}")
        group[fusion] = row
    output = []
    for (seed, update, protocol_sha), group in groups.items():
        if set(group) != {"baseline", "film"}:
            continue
        baseline, film = group["baseline"], group["film"]
        if baseline["metrics"]["motions"] != film["metrics"]["motions"]:
            raise ValueError("Matched checkpoints have different evaluation sizes")
        row = {"seed": int(seed), "completed_updates": update, "protocol_sha256": protocol_sha,
               "unix_time": max(baseline["unix_time"], film["unix_time"]),
               "motions": baseline["metrics"]["motions"], "cases": {},
               "checkpoints": {name: {"path": value["checkpoint"], "sha256": value["checkpoint_sha256"]}
                               for name, value in group.items()}}
        for case in ("all_0", "all_4"):
            b, f = baseline["metrics"][case], film["metrics"][case]
            row["cases"][case] = {
                "baseline": b, "film": f,
                "body_error_change_percent": 100 * (f["error_body_pos"] / max(b["error_body_pos"], 1e-12) - 1),
                "joint_error_change_percent": 100 * (f["error_joint_pos"] / max(b["error_joint_pos"], 1e-12) - 1),
                "failure_change_pp": 100 * (f["failure_rate"] - b["failure_rate"]),
                "coverage_change_pp": 100 * (f["coverage_fraction"] - b["coverage_fraction"]),
                "return_change": f["mean_episode_return"] - b["mean_episode_return"],
            }
        output.append(row)
    return sorted(output, key=lambda row: (row["unix_time"], row["seed"]))


def comparison_id(row):
    hashes = "/".join(row["checkpoints"][name]["sha256"] for name in ("baseline", "film"))
    return f"{row['seed']}/{row['completed_updates']}/{row['protocol_sha256']}/{hashes}"


def format_comparison(row, previous=None):
    lines = ["", f"[同轮对比] seed {row['seed']} | PPO update {row['completed_updates']} | 每端 {row['motions']} motions",
             "变化均为 FiLM 相对 baseline；误差和失败率变化为负值表示降低。"]
    for case, label in (("all_0", "0 kg"), ("all_4", "4 kg")):
        value = row["cases"][case]
        b, f = value["baseline"], value["film"]
        lines += [
            f"每处 {label}: body {b['error_body_pos'] * 100:.3f} -> {f['error_body_pos'] * 100:.3f} cm"
            f" ({value['body_error_change_percent']:+.2f}%)；joint {value['joint_error_change_percent']:+.2f}%",
            f"  失败率 {b['failure_rate']:.3%} -> {f['failure_rate']:.3%} ({value['failure_change_pp']:+.3f} pp)；"
            f"覆盖率变化 {value['coverage_change_pp']:+.3f} pp；平均回报变化 {value['return_change']:+.4f}",
        ]
        if previous is not None:
            old = previous["cases"][case]
            lines.append(f"  上一同轮对比 update {previous['completed_updates']}: "
                         f"body 变化 {old['body_error_change_percent']:+.2f}%，失败率变化 {old['failure_change_pp']:+.3f} pp。")
    lines += ["以上是训练过程观察；整体性能结论仍需至少 5000 updates 及跨 seed 的最终配对评估。", ""]
    return "\n".join(lines)


def nonfinite(value):
    if isinstance(value, dict):
        return any(nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(nonfinite(item) for item in value)
    return isinstance(value, float) and not math.isfinite(value)


def observe_health(root, jobs, now):
    state = json.loads((root / "state.json").read_text())
    alerts, training = {}, {}
    if any(job["status"] == "running" for job in jobs) and now - state["updated_at"] > 180:
        alerts["scheduler_stale"] = "训练调度器超过 3 分钟没有更新心跳。"
    for job in jobs:
        name = job["name"]
        directory = Path(job["output"])
        records = last_records(directory / "metrics.jsonl", count=20)
        last = records[-1] if records else {}
        training[name] = {"status": job["status"], "completed_updates": last.get("completed_updates"),
                          "gpus": job.get("gpus", [])}
        if job["status"] == "failed":
            alerts[f"failed/{name}"] = f"{name} 已失败，退出码 {job.get('exit_code')}；需检查训练日志。"
        if job["status"] != "running":
            continue
        if any(nonfinite(record) for record in records):
            alerts[f"nonfinite/{name}"] = f"{name} 最近的训练指标包含 NaN 或 Inf。"
        activity = max(last.get("unix_time", 0), job.get("started_at", now))
        update = last.get("completed_updates", 0)
        expected = (update // 100 + 1) * 100
        evaluation = directory / "endpoint_eval" / f"update_{expected:06d}"
        if (evaluation / "processes.json").exists():
            training[name]["evaluating_checkpoint_update"] = expected
            activity = max([activity, *(p.stat().st_mtime for p in evaluation.glob("*.log"))])
        if now - activity > 2100:
            alerts[f"stalled/{name}"] = f"{name} 超过 35 分钟没有训练或端点评估进展。"
    remote = root / ".wandb_sync/state.json"
    if remote.exists():
        wandb = json.loads(remote.read_text())
        if any(job["status"] == "running" for job in jobs) and now - wandb.get("updated_at", 0) > 180:
            alerts["wandb_stale"] = "W&B 同步器超过 3 分钟没有更新心跳。"
        for name, row in wandb.get("runs", {}).items():
            if row.get("last_error"):
                alerts[f"wandb/{name}"] = f"W&B run {name} 上报出现错误，详见 logs/wandb_sync.log。"
    return {"updated_at": now, "training": training, "alerts": alerts}


def watch(root):
    directory = root / ".experiment_monitor"
    directory.mkdir(exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        seen, active = set(state.get("reported_pairs", [])), state.get("active_alerts", {})
        write_state(directory / "process.json", {
            "pid": os.getpid(), "started_at": time.time(),
            "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
        })
        while True:
            try:
                results, jobs = completed_results(root)
                pairs = matched_results(results)
                previous = {}
                for row in pairs:
                    key = comparison_id(row)
                    group = (row["seed"], row["protocol_sha256"])
                    if key not in seen:
                        print(format_comparison(row, previous.get(group)), flush=True)
                        with (directory / "paired_metrics.jsonl").open("a") as handle:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                        seen.add(key)
                    previous[group] = row
                health = observe_health(root, jobs, time.time())
                health["latest_paired_updates"] = {str(seed): row["completed_updates"] for (seed, _), row in previous.items()}
                write_state(directory / "health.json", health)
                alerts = health["alerts"]
                for key in sorted(set(alerts) - set(active)):
                    print(f"[运行异常] {alerts[key]}", flush=True)
                for key in sorted(set(active) - set(alerts)):
                    print(f"[异常已解除] {active[key]}", flush=True)
                active = alerts
                write_state(path, {"reported_pairs": sorted(seen), "active_alerts": active, "updated_at": time.time()})
                final = root / "eval/comparison.json"
                if final.exists():
                    result = json.loads(final.read_text())
                    if result.get("complete"):
                        print(f"[最终评估完成] {final}", flush=True)
                        print(json.dumps(result["primary_across_training_seeds"], ensure_ascii=False), flush=True)
                        break
            except (OSError, ValueError, KeyError) as error:
                # A read-only monitor must keep observing after transient read
                # failures and must never restart or otherwise modify PPO.
                key = f"monitor/{type(error).__name__}"
                if key not in active:
                    print(f"[监测异常] {type(error).__name__}: {error}", flush=True)
                active[key] = "监测数据读取异常"
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Experiment directory must remain inside this project")
    if args.follow:
        watch(root)
    else:
        results, jobs = completed_results(root)
        pairs = matched_results(results)
        for row in pairs:
            print(format_comparison(row), flush=True)
        print(json.dumps(observe_health(root, jobs, time.time()), ensure_ascii=False), flush=True)
