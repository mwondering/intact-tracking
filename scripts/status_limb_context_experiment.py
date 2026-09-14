"""Read-only progress and throughput for the local limb-context experiment."""

import argparse
import json
import time
from pathlib import Path


def last_records(path, count=20):
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 262144))
        lines = handle.read().decode(errors="replace").splitlines()
    result = []
    for line in lines[-count:]:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def status(root):
    state = json.loads((root / "state.json").read_text())
    result = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
              "scheduler_heartbeat_age_s": round(time.time() - state["updated_at"]),
              "training": {}, "failed": []}
    for job in state["jobs"]:
        if job["status"] == "failed":
            result["failed"].append(job["name"])
        if job["name"].startswith("smoke") or job["phase"] == "audit":
            continue
        row = {"status": job["status"]}
        records = last_records(Path(job["output"]) / "metrics.jsonl")
        if records:
            last = records[-1]
            row["updates"] = last.get("completed_updates", last.get("update"))
            if job["phase"] == "ppo":
                seconds = sum(r["collect_seconds"] + r["learn_seconds"] for r in records) / len(records)
                endpoint_rows = last_records(Path(job["output"]) / "endpoint_eval_metrics.jsonl", count=5)
                regular = [r for r in endpoint_rows if r["completed_updates"] > 0 and r["completed_updates"] % 100 == 0]
                if regular:
                    evaluation_seconds = sum(r["wall_seconds"] for r in regular) / len(regular)
                    row["ppo_seconds_per_update"] = round(seconds, 3)
                    row["evaluation_seconds_per_100_updates"] = round(evaluation_seconds, 2)
                    seconds += evaluation_seconds / 100
                if endpoint_rows:
                    row["latest_endpoint_checkpoint_update"] = endpoint_rows[-1]["completed_updates"]
                row.update(reward=round(last["mean_reward"], 3) if last["mean_reward"] is not None else None,
                           seconds_per_update=round(seconds, 3),
                           eta_hours=round(max(0, 5000 - row["updates"]) * seconds / 3600, 2))
                for key in ("context_full_fraction", "residual_saturation_fraction"):
                    if key in last["loss"]:
                        row[key] = round(last["loss"][key], 4)
            else:
                target = int(job["command"][job["command"].index("--updates") + 1])
                manual = "--until-user-stop" in job["command"]
                row["target_updates"] = None if manual else target
                row["stopping_mode"] = "until_user_stop" if manual else "automatic_budget_or_plateau"
                row["validation_nmse"] = round(last["fixed_probe"]["dr_five_step_nmse"], 5)
                row["latent_response_correlation"] = round(last["fixed_probe"]["latent_response_correlation"], 4)
                if len(records) > 1:
                    seconds = (last["unix_time"] - records[0]["unix_time"]) / (last["update"] - records[0]["update"])
                    row.update(seconds_per_update=round(seconds, 3))
                    if not manual:
                        row["cap_eta_hours"] = round(max(0, target - row["updates"]) * seconds / 3600, 2)
                completion = job.get("completion", {})
                if job["status"] == "complete" and "completed_updates" in completion:
                    row["validation_update"] = last["update"]
                    row["updates"] = completion["completed_updates"]
        result["training"][job["name"]] = row
    evaluation = root / "eval/state.json"
    if evaluation.exists():
        jobs = json.loads(evaluation.read_text())["jobs"]
        result["evaluation"] = {"complete": sum(j["status"] == "complete" for j in jobs), "total": len(jobs)}
        result["failed"] += [j["name"] for j in jobs if j["status"] == "failed"]
    result["final_results_exist"] = (root / "eval/comparison.json").exists()
    wandb_state = root / ".wandb_sync/state.json"
    if wandb_state.exists():
        remote = json.loads(wandb_state.read_text())
        result["wandb"] = {
            "destination": f"{remote['entity']}/{remote['project']}",
            "stage1_destination": f"{remote['entity']}/{remote.get('stage1_project', remote['project'])}",
            "runs_with_urls": sum(bool(r.get("url")) for r in remote["runs"].values()),
            "errors": sum(bool(r.get("last_error")) for r in remote["runs"].values()),
            "heartbeat_age_s": round(time.time() - remote.get("updated_at", 0)),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    while True:
        row = status(args.run_root)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not args.follow or row["failed"] or row["final_results_exist"]:
            break
        time.sleep(55)
