"""Backfill and continuously mirror every experiment training job to W&B."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import wandb

from run_limb_context_experiment import ROOT, jobs, write_state, job_log_path, experiment_layout

ENTITY = "2486344338-zhejiang-university"
PROJECT = "intact-preview-v2"
STAGE1_PROJECT = "intact-forward-predictor"


def job_destination(job, args):
    stage = "stage1" if job["phase"] == "encoder" else "stage2"
    project = args.stage1_project if stage == "stage1" else args.project
    scale = f"-{experiment_layout(args.root)['gpus_per_policy']}gpu8192-scratch" if stage == "stage2" else ""
    group = f"{args.root.name}-{stage}{scale}" + ("-smoke" if job["name"].startswith("smoke_") else "")
    return project, group


def scalar_metrics(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = value
    return result


def completed_update(record):
    return int(record.get("completed_updates", record.get("update", -1)))


class Tail:
    def __init__(self, job, metadata, state, args):
        self.job, self.directory, self.state = job, Path(job["output"]), state
        self.metadata = metadata
        self.root = args.root
        self.saved_files = set()
        self.offset = 0
        state.setdefault("id", "limb-" + hashlib.sha256(str(self.directory).encode()).hexdigest()[:12])
        smoke = job["name"].startswith("smoke_")
        project, group = job_destination(job, args)
        if state.get("project", project) != project:
            raise ValueError("Move the existing run before changing its destination")
        self.run = wandb.init(
            entity=args.entity, project=project, id=state["id"],
            name=f"{args.root.name}/{job['name']}",
            group=group,
            job_type="stage1" if job["phase"] == "encoder" else "ppo",
            tags=["limb-context", "nominal-supervision", "independent-uniform-0-4kg",
                  "smoke" if smoke else "full-dataset", metadata.get("fusion", "encoder")],
            config={**metadata, "logging_method": "live JSONL mirror, original update axis",
                    "experiment_job": job["name"]},
            resume="allow", reinit="create_new", mode="online", dir=str(args.state_dir),
            save_code=False,
            settings=wandb.Settings(console="off", disable_code=True, disable_git=True,
                                    x_disable_stats=True, x_disable_meta=True, init_timeout=45),
        )
        # The server's resumed step is authoritative if a previous uploader died
        # after recording a local cursor but before finishing its network upload.
        self.last_update = int(self.run.step) - 1
        self.run.define_metric("update")
        self.run.define_metric("*", step_metric="update")
        self.run.define_metric("endpoint_eval/checkpoint_update")
        self.run.define_metric("endpoint_eval/*", step_metric="endpoint_eval/checkpoint_update")
        state.update(url=self.run.url, project=project, group=group,
                     last_update=self.last_update, finished=False)
        print(json.dumps({"job": job["name"], "url": self.run.url}), flush=True)

    def read(self, hardware=None):
        path = self.directory / "metrics.jsonl"
        if not path.exists():
            return 0
        if path.stat().st_size < self.offset:
            self.offset = 0
        count = 0
        with path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    self.offset = start
                    break
                record = json.loads(line)
                update = completed_update(record)
                if update > self.last_update:
                    payload = scalar_metrics(record)
                    payload["update"] = update
                    # Historical backfill has no historical GPU telemetry.
                    if hardware and record.get("unix_time", 0) >= time.time() - 20:
                        payload.update(hardware)
                    # An explicit step otherwise leaves the final point pending
                    # until the next log call (three minutes later for stage 1).
                    self.run.log(payload, step=update, commit=True)
                    self.last_update = update
                    count += 1
                self.offset = handle.tell()
        self.state["last_update"] = self.last_update
        if count:
            self.run.summary.update({"sync/last_update": self.last_update,
                                     "sync/last_upload_unix": time.time()})
        return count

    def sync_log_files(self):
        """Mirror the worker's own console into this run without mixing jobs."""
        log = job_log_path(self.root, self.job)
        for path in (self.directory / "run_config.json", self.directory / "metrics.jsonl",
                     self.directory / "completion.json", self.directory / "endpoint_eval_metrics.jsonl", log):
            if path.exists() and path not in self.saved_files:
                self.run.save(str(path), base_path=str(path.parent), policy="live", glob=False)
                self.saved_files.add(path)
        if not log.exists():
            return
        offset = self.state.get("console_offset", 0)
        if offset > log.stat().st_size:
            offset = 0
        with log.open("rb") as handle:
            handle.seek(offset)
            lines = []
            while handle.tell() - offset < 262144:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    handle.seek(start)
                    break
                lines.append(line)
            if lines:
                # This SDK callback publishes output to the selected run. Global
                # stdout capture would incorrectly mix the concurrent workers.
                self.run._console_callback("stdout", b"".join(lines).decode(errors="replace"))
                self.state["console_offset"] = handle.tell()


def gpu_stats():
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in out.splitlines():
        index, memory, utilization, power = [s.strip() for s in line.split(",")]
        values = {"memory_mib": memory, "utilization_percent": utilization, "power_watts": power}
        result[int(index)] = {k: float(v) for k, v in values.items() if v != "[N/A]"}
    return result


def main(args):
    args.root = args.root.resolve()
    if not args.root.is_relative_to(ROOT):
        raise ValueError("All synchronization files must remain within the project")
    args.state_dir = args.root / ".wandb_sync"
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "sidecar.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.state_dir / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if state and (state["entity"], state["project"]) != (args.entity, args.project):
        raise ValueError("The W&B destination differs from existing synchronization state")
    state.update(entity=args.entity, project=args.project, stage1_project=args.stage1_project, pid=os.getpid())
    state.setdefault("runs", {})
    queue = sorted((j for j in jobs(args.root) if j["phase"] in ("encoder", "ppo") and not j.get("reused")),
                   key=lambda j: j["name"].startswith("smoke_"))
    tails, stopping = {}, False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopping:
            live = json.loads((args.root / "state.json").read_text())
            live_jobs = {j["name"]: j for j in live["jobs"]}
            hardware = gpu_stats()
            for job in queue:
                name = job["name"]
                entry = state["runs"].setdefault(name, {})
                if entry.get("finished"):
                    continue
                config = Path(job["output"]) / "run_config.json"
                if not config.exists():
                    continue
                try:
                    metadata = json.loads(config.read_text())
                    if name not in tails:
                        tails[name] = Tail(job, metadata, entry, args)
                    entry.pop("last_error", None)
                    tail = tails[name]
                    if tail.metadata != metadata:
                        tail.run.config.update(metadata, allow_val_change=True)
                        tail.metadata = metadata
                    tail.sync_log_files()
                    active = live_jobs.get(name, {})
                    gpu = {f"gpu/{i}/{key}": value for i in active.get("gpus", [])
                           for key, value in hardware.get(i, {}).items()}
                    count = tail.read(gpu if active.get("status") == "running" else None)
                    manual_stop = metadata["arguments"].get("until_user_stop", False)
                    tail.run.summary.update({"training/status": active.get("status", "unknown"),
                                             "training/unbounded": manual_stop,
                                             "training/target_updates": None if manual_stop else metadata["arguments"].get("updates", metadata["arguments"].get("iterations")),
                                             "sync/heartbeat_unix": time.time()})
                    if active.get("status") == "running":
                        tail.run.summary.update({f"live/{k}": v for k, v in gpu.items()})
                    completion = Path(job["output"]) / "completion.json"
                    if completion.exists():
                        done = json.loads(completion.read_text())
                        finished = (done.get("complete", False) if job["phase"] == "ppo" else
                                    (active.get("status") == "complete" if manual_stop else
                                     (done.get("converged") or done.get("hit_cap")) and not done.get("stopped")))
                        if finished:
                            tail.run.summary.update({f"completion/{k}": v for k, v in done.items()})
                            tail.run.summary["training/status"] = "complete"
                            tail.run.finish()
                            entry["finished"] = True
                            del tails[name]
                    if count:
                        print(json.dumps({"job": name, "rows": count,
                                          "last_update": entry["last_update"]}), flush=True)
                except Exception as error:
                    entry["last_error"] = f"{type(error).__name__}: {error}"
                    print(json.dumps({"job": name, "error": entry["last_error"]}), flush=True)
                    if "permission denied" in str(error).lower():
                        state["updated_at"] = time.time()
                        write_state(path, state)
                        raise RuntimeError("W&B rejected write access to the selected project") from error
                state["updated_at"] = time.time()
                write_state(path, state)
            state["updated_at"] = time.time()
            write_state(path, state)
            if args.once or all(state["runs"].get(j["name"], {}).get("finished") for j in queue):
                break
            time.sleep(10)
    finally:
        for tail in tails.values():
            tail.read()
            tail.sync_log_files()
            tail.run.finish()
        write_state(path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--stage1-project", default=STAGE1_PROJECT)
    parser.add_argument("--once", action="store_true")
    main(parser.parse_args())
