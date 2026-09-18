"""Upload every locally recorded five-PPO update without restarting training."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import uuid

import wandb

from monitor_fixed_five_ppo import read_json, finite


def update_metrics(row):
    result = {"trainer/update": row["completed_updates"],
              "train/time/collect_seconds": row["collect_seconds"],
              "train/time/learn_seconds": row["learn_seconds"]}
    for role, metrics in row["roles"].items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                result[f"train/{role}/{key}"] = value
    experts = [value for role, value in row["roles"].items() if role.startswith("expert_")]
    total = sum(value["worlds"] for value in experts)
    result["train/experts_combined/mean_step_reward"] = sum(value["worlds"]*value["mean_step_reward"] for value in experts) / total
    result["train/experts_combined/transitions"] = sum(value.get("transitions", 0) for value in experts)
    result["train/experts_minus_baseline/mean_step_reward"] = result["train/experts_combined/mean_step_reward"] - row["roles"]["baseline"]["mean_step_reward"]
    if not finite(result):
        raise ValueError("Nonfinite PPO metrics; consult the local monitor alert")
    return result


def save_status(root, state):
    tmp = root / "wandb_status.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(root / "wandb_status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--project", default="intact-preview-v2")
    parser.add_argument("--entity", default="2486344338-zhejiang-university")
    parser.add_argument("--interval", type=float, default=30.)
    args = parser.parse_args()
    key_file = Path(__file__).resolve().parents[1] / ".runtime/limb_context/wandb_api_key"
    if "WANDB_API_KEY" not in os.environ and key_file.exists():
        os.environ["WANDB_API_KEY"] = key_file.read_text().strip()
    root = args.run_root.resolve()
    launch = read_json(root / "launch.json")
    status = read_json(root / "wandb_status.json")
    if not status:
        status = {"id": uuid.uuid4().hex[:8], "last_uploaded_update": 0,
                  "metadata_uploaded": False, "status": "initializing"}
        save_status(root, status)
    run = wandb.init(entity=args.entity, project=args.project, id=status["id"], resume="allow",
                     name=f"fixed-four-experts-plus-baseline-full-{launch.get('world_size', 1)}x4096", group=root.name,
                     job_type="five-independent-ppo", dir=str(root), mode="online",
                     config={**launch, "warmup_seconds": 100, "fixed_environment_assignment": True,
                             "independent_ppo_count": 5, "actor_critic_use_latent": False,
                             "logging": "all local updates via background logger; train metrics use trainer/update"},
                     settings=wandb.Settings(x_disable_stats=True, console="off"))
    run.define_metric("trainer/update")
    run.define_metric("train/*", step_metric="trainer/update")
    run.define_metric("monitor/elapsed_seconds")
    run.define_metric("monitor/*", step_metric="monitor/elapsed_seconds")
    status.update(url=run.url, project=args.project, entity=args.entity, status="running")
    save_status(root, status)
    print(run.url, flush=True)
    started = datetime.fromisoformat(launch["started_utc"].replace("Z", "+00:00")).timestamp()
    seen_monitor = None
    while True:
        metadata_path = root / "train" / "run_config.json"
        if metadata_path.exists() and not status["metadata_uploaded"]:
            metadata = read_json(metadata_path)
            keys = ("training_configuration", "context_sha256", "tracker_sha256", "physics_sha256",
                    "independence", "warmup_seconds", "source_sha256")
            run.config.update({**{k: metadata[k] for k in keys},
                               "dataset_motion_count": metadata.get("global_motion_count", len(metadata["motion_files"])),
                               "global_world_counts": metadata.get("global_world_counts")}, allow_val_change=True)
            status["metadata_uploaded"] = True
        path = root / "train" / "metrics.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["completed_updates"] > status["last_uploaded_update"]:
                    run.log(update_metrics(row))
                    status["last_uploaded_update"] = row["completed_updates"]
        monitor = read_json(root / "monitor.json")
        if monitor and monitor.get("observed_utc") != seen_monitor:
            values = {"monitor/elapsed_seconds": time.time() - started,
                      "monitor/process_alive": int(monitor["process_alive"]),
                      "monitor/alert_count": len(monitor["alerts"])}
            for group in ("gpu", "progress"):
                values.update({f"monitor/{group}/{k}": v for k, v in monitor.get(group, {}).items()
                               if isinstance(v, (int, float))})
            for gpu in monitor.get("gpus", []):
                values.update({f"monitor/gpu{gpu['index']}/{k}": v for k, v in gpu.items() if k != "index"})
            run.log(values)
            run.summary["phase"] = monitor.get("progress", {}).get("phase", "initializing")
            run.summary["latest_alerts"] = monitor["alerts"]
            seen_monitor = monitor["observed_utc"]
        status["last_poll_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_status(root, status)
        if monitor and not monitor["process_alive"]:
            complete = monitor["state"].get("complete", False)
            run.summary["training_complete"] = complete
            run.summary["completed_updates"] = status["last_uploaded_update"]
            run.finish(exit_code=0 if complete else 1)
            status["status"] = "complete" if complete else "training_exited"
            save_status(root, status)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
