"""Print completed endpoint tests; optionally follow them without restarting PPO."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def completed_results(root):
    state = json.loads((root / "state.json").read_text())
    results = []
    jobs = [job for job in state["jobs"]
            if job["phase"] == "ppo" and not job["name"].startswith("smoke_")]
    for job in jobs:
        directory = Path(job["output"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Training output must remain inside this experiment")
        history = directory / "endpoint_eval_metrics.jsonl"
        if not history.exists():
            continue
        for line in history.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue  # A writer may still be committing its newest record.
            row = json.loads(line)
            if row.get("training_state_preserved"):
                results.append((job, row))
    return sorted(results, key=lambda item: item[1]["unix_time"]), jobs


def report_id(job, row):
    return f"{job['name']}/{row['completed_updates']}/{row['checkpoint_sha256']}/{row['protocol_sha256']}"


def format_report(job, row):
    metrics = row["metrics"]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(row["unix_time"]))
    lines = [
        "",
        f"[端点测试完成] {job['name']} | PPO update {row['completed_updates']} | {stamp}",
        f"每个端点 {metrics['motions']} motions；两端使用相同 motions 和起点。",
        "每个部位负载   body_pos(m)   joint_pos(rad)   失败率     覆盖率     平均回报",
    ]
    for case, label in (("all_0", "0 kg"), ("all_4", "4 kg")):
        value = metrics[case]
        lines.append(
            f"{label:>12}   {value['error_body_pos']:11.6f}   {value['error_joint_pos']:14.6f}"
            f"   {value['failure_rate']:7.3%}   {value['coverage_fraction']:7.3%}"
            f"   {value['mean_episode_return']:10.4f}"
        )
    lines += [
        "训练侧 RNG、环境步数、critic 归一化统计保持不变：通过。",
        f"checkpoint: {row['checkpoint']}",
        "不同训练轮数的结果不直接比较；最终结论使用相同步数及跨 seed 的配对评估。",
        "",
    ]
    return "\n".join(lines)


def write_state(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def follow(root):
    directory = root / ".endpoint_reports"
    directory.mkdir(exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cursor_path = directory / "state.json"
        previous = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
        seen = set(previous.get("reported", []))
        write_state(directory / "process.json", {
            "pid": os.getpid(), "started_at": time.time(),
            "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
        })
        while True:
            results, jobs = completed_results(root)
            for job, row in results:
                key = report_id(job, row)
                if key in seen:
                    continue
                message = format_report(job, row)
                # The scheduler also opens these logs in append mode. Emit one
                # complete block so it is readable alongside the PPO output.
                log = root / "logs" / Path(job["output"]).parent.name / f"{job['name']}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as handle:
                    handle.write(message + "\n")
                print(message, flush=True)
                seen.add(key)
                write_state(cursor_path, {"reported": sorted(seen), "updated_at": time.time()})
            if jobs and all(job["status"] in ("complete", "failed") for job in jobs):
                break
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--follow", action="store_true", help="Append each new test to terminal and training logs")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Experiment directory must remain inside the project")
    if args.follow:
        follow(root)
    else:
        latest = {}
        for job, row in completed_results(root)[0]:
            latest[job["name"]] = (job, row)
        for job, row in latest.values():
            print(format_report(job, row), flush=True)
