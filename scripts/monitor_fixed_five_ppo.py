"""Observe a five-PPO run; persist progress, recent metrics and health alerts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time


def read_json(path):
    # Shared storage can briefly hide a path while its atomic update lands.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt < 2:
                time.sleep(.05)
    return {}


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def snapshot(root, launch):
    train = root / "train"
    progress = read_json(train / "progress.json")
    if not progress:
        matches = re.findall(r"motion_load progress loaded=(\d+)/(\d+)",
                             (root / "train.log").read_text(errors="replace"))
        if matches:
            progress = {"phase": "motion_load", "loaded_motions": int(matches[-1][0]),
                        "total_motions": int(matches[-1][1])}
    state = read_json(train / "state.json")
    pid = launch["pid"]
    proc = Path(f"/proc/{pid}")
    try:
        alive = ("fixed_five_ppo_train" in (proc / "cmdline").read_text()
                 and (proc / "stat").read_text().split(")", 1)[1].split()[0] != "Z")
    except (FileNotFoundError, ProcessLookupError):
        alive = False
    rows = []
    metrics = train / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A writer can be in the middle of appending the last line.
    recent = rows[-20:]
    report = {"observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "pid": pid, "process_alive": alive, "progress": progress, "state": state,
              "recent_window_updates": len(recent), "alerts": []}
    report["rank_progress"] = {p.parent.name: read_json(p) for p in sorted(train.glob("rank_*/progress.json"))}
    if recent:
        durations = [r["collect_seconds"] + r["learn_seconds"] for r in recent]
        median = statistics.median(durations)
        report.update(median_seconds_per_update=median,
                      remaining_training_seconds_estimate=max(0, launch["target_updates"] - recent[-1]["completed_updates"]) * median)
        means = {}
        for role in recent[-1]["roles"]:
            valid = [r["roles"][role] for r in recent if r["roles"][role]["worlds"]]
            if valid:
                means[role] = {k: statistics.mean(r[k] for r in valid) for k in
                               ("mean_step_reward", "value", "surrogate", "entropy", "raw_advantage_std")}
                means[role]["worlds"] = valid[-1]["worlds"]
        report["recent_role_means"] = means
        experts = [v for k, v in means.items() if k.startswith("expert_")]
        total = sum(r["worlds"] for r in experts)
        report["recent_expert_weighted_step_reward"] = sum(r["mean_step_reward"]*r["worlds"] for r in experts) / total
        report["recent_baseline_step_reward"] = means["baseline"]["mean_step_reward"]
        roles = recent[-1]["roles"]
        if roles["baseline"]["transitions"] != sum(v.get("transitions", 0) for k, v in roles.items() if k.startswith("expert_")):
            report["alerts"].append("Unequal total transition budgets between arms")
        for role, value in roles.items():
            if value["worlds"] == 0:
                report["alerts"].append(f"Empty class: {role}")
            elif value["updates"] != recent[-1]["completed_updates"]:
                report["alerts"].append(f"PPO update count mismatch: {role}")
    if not finite(progress) or not finite(recent):
        report["alerts"].append("Nonfinite logged training metric")
    if not alive and not state.get("complete"):
        report["alerts"].append("Training process exited before target completion")
        report["last_log_lines"] = (root / "train.log").read_text(errors="replace").splitlines()[-25:]
    try:
        gpu_ids = launch["gpu"] if isinstance(launch["gpu"], list) else [launch["gpu"]]
        gpu = subprocess.run(["nvidia-smi", "--id=" + ",".join(str(i) for i in gpu_ids),
                              "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        report["gpus"] = [dict(zip(("index", "memory_used_mib", "memory_total_mib", "utilization_percent"),
                                  [int(v.strip()) for v in line.split(",")])) for line in gpu.stdout.strip().splitlines()]
        report["gpu"] = report["gpus"][0]
    except (subprocess.SubprocessError, ValueError):
        report["gpu_query_unavailable"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--interval", type=float, default=60.)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    root = args.run_root.resolve()
    launch = read_json(root / "launch.json")
    last_key, changed = None, time.monotonic()
    while True:
        report = snapshot(root, launch)
        progress = report["progress"]
        key = (progress.get("phase"), progress.get("step"), progress.get("completed_updates"),
               progress.get("loaded_motions"))
        if key != last_key:
            last_key, changed = key, time.monotonic()
        report["seconds_since_progress_changed"] = time.monotonic() - changed
        if report["process_alive"] and report["seconds_since_progress_changed"] > 1200:
            report["alerts"].append("No progress change for more than 20 minutes")
        raw = json.dumps(json_safe(report), ensure_ascii=False, allow_nan=False)
        tmp = root / "monitor.tmp"
        tmp.write_text(raw + "\n")
        tmp.replace(root / "monitor.json")
        with (root / "monitor_history.jsonl").open("a") as stream:
            stream.write(raw + "\n")
        print(json.dumps({"observed_utc": report["observed_utc"], "phase": progress.get("phase", "initializing"),
                          "updates": progress.get("completed_updates", 0), "alerts": report["alerts"]}), flush=True)
        if not report["process_alive"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
