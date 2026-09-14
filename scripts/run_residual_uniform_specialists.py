"""Eight independent, unlimited uniform/unbounded residual specialists."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, DATASET, process_environment
from run_fixed_dr_specialists import last_record as _last_record
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.residual_uniform_protocol import VERSION, physics_contract


def read_optional_json(path, previous=None):
    """Keep the last complete value while another process replaces its status file."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return previous


def last_record(path):
    try:
        return _last_record(path)
    except FileNotFoundError:
        return None


def evaluation_history(path):
    try:
        lines = path.read_text().split("\n")[:-1]
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line]


class AttachedProcess:
    """Recheck the recorded worker identity before polling or sending a signal."""

    def __init__(self, record):
        self.pid = record["pid"]
        self.start_ticks = str(record["process_start_ticks"])
        self.command = [os.fsencode(v) for v in record["command"]]
        if not self._matches():
            raise ValueError("Existing worker identity does not match its recorded command/start time")

    def _matches(self):
        try:
            proc = Path(f"/proc/{self.pid}")
            fields = proc.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
            command = proc.joinpath("cmdline").read_bytes().split(b"\0")[:-1]
            return fields[0] not in {"Z", "X"} and fields[19] == self.start_ticks and command == self.command
        except (FileNotFoundError, ProcessLookupError):
            return False

    def poll(self):
        return None if self._matches() else 1

    def terminate(self):
        if self.poll() is None:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def training_command(root, profile):
    name = f"B{profile['id']:02d}_{profile['name']}"
    return [PYTHON, "-B", "-u", "-m", "intact_tracking.cli.residual_uniform_train",
            "--fusion", "baseline", "--output-dir", str(root / "ppo" / name),
            "--training-ranks", "1", "--num-envs", "8192", "--seed", "121",
            "--motion-path", DATASET, "--until-user-stop", "--save-interval", "100",
            "--motion-sampling", "uniform", "--training-terminations", "original", "--policy-precision", "fp32",
            "--dr-bank", str(root / "protocols/dr_bank.json"), "--specialist-id", str(profile["id"]),
            "--eval-motion-manifest", str(root / "protocols/evaluation_motions.txt"),
            "--eval-interval", "1000", "--eval-steps", "1000", "--eval-seed", "20001",
            "--wandb-group", root.name, "--wandb-name", root.name + "-" + name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--attach-existing", action="store_true", help="Restore monitoring of the recorded live workers without launching training")
    args = parser.parse_args()
    root = args.run_root.resolve()
    lock = (root / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((root / "READY.json").read_text())
    if not ready["passed"]:
        raise ValueError("Complete the new-protocol preflight first")
    for path, expected in ready["pinned_files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Validated source changed: {path}")
    bank = json.loads((root / "protocols/dr_bank.json").read_text())
    if bank["residual_physics_contract"] != physics_contract():
        raise ValueError("Unexpected physics protocol")
    previous_state = None
    if args.attach_existing:
        previous_state = json.loads((root / "state.json").read_text())
        if previous_state["version"] != VERSION or set(previous_state["arms"]) != {str(p["id"]) for p in bank["profiles"]}:
            raise ValueError("Existing run does not match this specialist protocol")
    elif (root / "ppo").exists():
        raise FileExistsError("Do not overwrite an existing formal training run")
    else:
        (root / "ppo").mkdir()
    jobs, stopping = {}, False
    handles = []

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        for job in jobs.values():
            if job["child"].poll() is None:
                job["child"].terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for profile in bank["profiles"]:
        i = profile["id"]
        name = f"B{i:02d}_{profile['name']}"
        if previous_state is not None:
            old = previous_state["arms"][str(i)]
            if old["process"]["command"] != training_command(root, profile) or old["gpu"] != i:
                raise ValueError(f"B{i}: existing worker has an unexpected command or GPU")
            jobs[str(i)] = {key: old[key] for key in ("gpu", "profile", "output", "process")}
            jobs[str(i)].update(child=AttachedProcess(old["process"]), phase="training")
            continue
        env = evaluation_environment(i, process_environment())
        env["WANDB_API_KEY"] = (ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip()
        env["WANDB_RUN_ID"] = "uniform-specialist-" + hashlib.sha256(f"{root}/{i}".encode()).hexdigest()[:12]
        env["WANDB_RESUME"] = "allow"
        command = training_command(root, profile)
        handle = (root / "ppo" / f"B{i:02d}.log").open("a")
        handles.append(handle)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        jobs[str(i)] = {"child": child, "gpu": i, "profile": profile["name"], "phase": "training",
                       "output": str(root / "ppo" / name), "process": {"pid": child.pid, "command": command,
                       "physical_gpus": str(i), "started_at": time.time(),
                       "process_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split()[21]}}
    while True:
        if (root / "STOP").exists() and not stopping:
            stop(None, None)
        arms, issues, histories = {}, [], []
        for key, job in jobs.items():
            code = job["child"].poll()
            if code is not None:
                job["phase"] = "stopped" if stopping else "failed"
                if not stopping:
                    issues.append(f"B{key}: process exited {code}")
            directory = Path(job["output"])
            row = {k: v for k, v in job.items() if k != "child"}
            for filename, field in (("progress.json", "progress"), ("evaluation_status.json", "evaluation_status"),
                                    ("wandb_run.json", "wandb")):
                path = directory / filename
                row[field] = read_optional_json(path, job.get(field))
                job[field] = row[field]
            metrics = last_record(directory / "metrics.jsonl")
            if metrics:
                row.update(global_transitions=metrics["global_transitions"],
                           seconds_per_update=metrics["collect_seconds"] + metrics["learn_seconds"], metric_time=metrics["unix_time"])
                if not all(math.isfinite(float(v)) for v in metrics["loss"].values()):
                    issues.append(f"B{key}: nonfinite metrics")
                if metrics["motion_sampling"]["adaptive_enabled"] or metrics["loss"].get("residual_output_bounded") != 0:
                    issues.append(f"B{key}: wrong sampler or output bound")
            evaluation = last_record(directory / "specialist_eval_metrics.jsonl")
            row["latest_evaluation_update"] = evaluation["completed_updates"] if evaluation else None
            arms[key] = row
            history = directory / "specialist_eval_metrics.jsonl"
            rows = evaluation_history(history)
            histories.append({r["completed_updates"]: r for r in rows})
        common = sorted(set.intersection(*(set(rows) for rows in histories)))
        status = "stop_requested" if stopping else "specialists_need_attention" if issues else "eight_specialists_training"
        state = {"status": status, "launcher_pid": os.getpid(), "heartbeat": time.time(), "version": VERSION,
                 "maximum_updates": None, "physics_contract": physics_contract(), "residual_output_bounded": False,
                 "evaluation_interval": 1000, "arms": arms, "issues": issues}
        write_json(root / "state.json", state)
        (root / "health").mkdir(exist_ok=True)
        write_json(root / "health/latest.json", state)
        write_json(root / "comparison.json", {"matched_specialist_updates": common,
                   "rows": [{"specialist_update": update, "by_dr": {str(i): h[update] for i, h in enumerate(histories)}} for update in common],
                   "shared_A": "Old A used bounded residual, 5 Nm wrists and adaptive sampling; a new shared run is required for a matched-training comparison"})
        write_json(ROOT / ".runtime/limb_context/current_ppo.json", {"run_root": str(root), "supervisor_pid": os.getpid(),
                   "status": status, "updated_at": time.time(), "version": VERSION,
                   "architecture": "8 independent fixed-DR residual MLP specialists; no latent; unbounded residual output",
                   "physics_contract": physics_contract(), "gpu_assignments": {k: v["profile"] for k, v in jobs.items()},
                   "periodic_updates": {"checkpoint": 100, "evaluation": 1000}, "health": str(root / "health/latest.json")})
        if all(job["child"].poll() is not None for job in jobs.values()):
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
