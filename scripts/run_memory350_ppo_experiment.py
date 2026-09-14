"""Persistent two-arm Memory350 PPO supervisor; all writes stay in this project."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "runs/limb_context_20260910_memory350_ppo"
PYTHON = str(ROOT / ".venv/bin/python")
GROUP = "memory350-trackerdr-residual-20260910"
HELPERS = runpy.run_path(str(ROOT / "scripts/run_limb_context_experiment.py"))


def read(path):
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(0.1)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"state": fields[0], "start_ticks": fields[19]}


class AttachedTrainingProcess:
    """Observe an existing launcher without relaunching or signalling it.

    A non-parent cannot retrieve its exit code. After it exits, require fresh
    5000-update completion and distributed agreement before final evaluation.
    """

    def __init__(self, root, fusion, job):
        if job["phase"] != "adaptive":
            raise RuntimeError("Live attachment requires the existing adaptive phase")
        self.pid = job["pid"]
        identity = process_identity(self.pid)
        if identity is None or identity["state"] == "Z":
            raise RuntimeError(f"Cannot attach to exited {fusion} launcher {self.pid}")
        directory = Path(f"/proc/{self.pid}")
        actual_command = directory.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        if actual_command != job["command"]:
            raise RuntimeError(f"{fusion} launcher command changed")
        variables = dict(item.split(b"=", 1) for item in directory.joinpath("environ").read_bytes().split(b"\0")
                         if b"=" in item)
        if variables.get(b"CUDA_VISIBLE_DEVICES", b"").decode() != ",".join(map(str, job["gpus"])):
            raise RuntimeError(f"{fusion} launcher GPU assignment changed")
        self.start_ticks = identity["start_ticks"]
        self.output = root / "ppo" / f"{fusion}_121"
        self.returncode = None
        self.exit_observed_at = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        identity = process_identity(self.pid)
        if identity and identity["start_ticks"] == self.start_ticks and identity["state"] != "Z":
            return None
        if self.exit_observed_at is None:
            self.exit_observed_at = time.monotonic()
        try:
            completion = read(self.output / "completion.json")
        except (FileNotFoundError, json.JSONDecodeError):
            completion = {}
        agreement = completion.get("distributed_parameter_agreement", {})
        ranks = agreement.get("ranks", [])
        valid_completion = (
            completion.get("complete") is True
            and completion.get("target_updates") == 5000
            and completion.get("completed_updates") == 5000
            and not completion.get("stopped")
            and not completion.get("planned_sampling_transition")
            and agreement.get("passed") is True
            and agreement.get("world_size") == 2
            and {row.get("rank") for row in ranks} == {0, 1}
            and all(row.get("completed_updates") == 5000 and row.get("finite") is True for row in ranks)
            and (self.output / "checkpoint_update_005000.pt").is_file()
        )
        if valid_completion:
            self.returncode = 0
        elif time.monotonic() - self.exit_observed_at >= 30:
            self.returncode = 1
        return self.returncode


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment(gpus, fusion=None):
    env = HELPERS["process_environment"]()
    for key in list(env):
        if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "_WANDB_SERVICE", "WANDB_SERVICE", "WANDB_RUN_ID", "WANDB_RESUME"} or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
               WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip())
    if fusion:
        env.update(WANDB_RUN_ID=f"m350ppo-{fusion}-121-20260910", WANDB_RESUME="allow")
    return env


def audit_inputs(root):
    protected = read(ROOT / ".runtime/memory350_ppo_v1/protected_sources.json")
    changed = [str(p) for p, sha in protected.items() if digest(ROOT / p) != sha]
    if changed:
        raise RuntimeError(f"Protected source files changed: {changed}")
    selected = read(root / "context_selection.json")
    if digest(selected["checkpoint"]) != selected["sha256"]:
        raise RuntimeError("Selected frozen encoder was changed")
    rows = [read(root / "smoke" / f"{fusion}_121/completion.json") for fusion in ("baseline", "film")]
    if any(not row["complete"] or row["completed_updates"] != 3 for row in rows):
        raise RuntimeError("Both 8192-env/rank smoke runs and exact-checkpoint resumes must pass")
    meta = [read(root / "smoke" / f"{fusion}_121/run_config.json") for fusion in ("baseline", "film")]
    for key in ("actor_common_trunk_sha256", "critic_common_trunk_sha256"):
        if meta[0]["input_audit"][key] != meta[1]["input_audit"][key]:
            raise RuntimeError(f"Smoke arms differ in initial {key}")
    numerical = read(root / "smoke_initialization_audit.json")
    if not numerical["passed"]:
        raise RuntimeError("Initial critic moment equality failed the numerical audit")
    for fusion in ("baseline", "film"):
        if numerical["checkpoint_sha256"][fusion] != digest(root / "smoke" / f"{fusion}_121/checkpoint_initial.pt"):
            raise RuntimeError("Initialization audit refers to a different smoke checkpoint")
    return selected


def command(root, fusion, resume=None):
    output = root / "ppo" / f"{fusion}_121"
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
           "-m", "intact_tracking.cli.memory350_policy_train", "--fusion", fusion,
           "--output-dir", str(output), "--motion-path", HELPERS["DATASET"],
           "--num-envs", "8192", "--iterations", "5000", "--seed", "121",
           "--episode-steps", "1000", "--motion-sampling", "adaptive", "--adaptive-after-update", "1000",
           "--training-terminations", "no_ee_body_pos", "--dr-profile", "tracker_dr_plus_limb_payload",
           "--endpoint-eval-protocol", str(root / "protocols/periodic.json"),
           "--wandb-group", GROUP, "--wandb-name", f"{fusion}_121_memory350_trackerdr"]
    if fusion == "film":
        cmd += ["--context-checkpoint", read(root / "context_selection.json")["checkpoint"]]
    if resume:
        cmd += ["--resume", str(resume)]
    return cmd


def launch(root, fusion, gpus, resume=None):
    phase = "adaptive" if resume else "uniform"
    log = root / "logs" / f"ppo_{fusion}_{phase}.log"
    cmd = command(root, fusion, resume)
    with log.open("a") as handle:
        process = subprocess.Popen(cmd, cwd=ROOT, env=environment(gpus, fusion),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return process, {"pid": process.pid, "gpus": gpus, "phase": phase, "command": cmd,
                     "log": str(log), "started_at": time.time()}


def evaluate_final(root, fusion, gpus):
    if fusion == "frozen_tracker":
        cmd = [PYTHON, "-u", str(ROOT / "scripts/evaluate_memory350_ppo.py"),
               "--run-root", str(root), "--fusion", fusion, "--gpus", *map(str, gpus)]
    else:
        checkpoint = root / "ppo" / f"{fusion}_121/checkpoint_update_005000.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Final numbered checkpoint is missing: {checkpoint}")
        cmd = [PYTHON, "-u", str(ROOT / "scripts/evaluate_memory350_ppo.py"),
               "--run-root", str(root), "--fusion", fusion, "--gpus", *map(str, gpus)]
    log = root / "logs" / f"final_{fusion}.log"
    with log.open("a") as stream:
        p = subprocess.Popen(cmd, cwd=ROOT, env=environment(gpus), stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
    return p, {"pid": p.pid, "gpus": gpus, "phase": "final_evaluation", "log": str(log), "started_at": time.time()}


def run(root, attach_running=False):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Experiment must remain inside the project")
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_inputs(root)
        state = {"phase": "training", "pid": os.getpid(), "started_at": time.time(), "jobs": {}, "group": GROUP}
        active = {}
        finished = set()
        assignments = {"baseline": [0, 1], "film": [2, 3]}
        try:
            if attach_running:
                previous = read(root / "supervisor_state.json")
                if previous["phase"] != "needs_inspection" or set(previous["jobs"]) != set(assignments):
                    raise RuntimeError("Live attachment requires an interrupted supervisor with both training jobs")
                for fusion, gpus in assignments.items():
                    job = previous["jobs"][fusion]
                    if job["gpus"] != gpus:
                        raise RuntimeError(f"Unexpected GPU assignment for {fusion}")
                    active[fusion] = AttachedTrainingProcess(root, fusion, job)
                    state["jobs"][fusion] = dict(job, attached_at=time.time(),
                                                start_ticks=active[fusion].start_ticks,
                                                exit_verification="live process identity plus fresh 5000-update completion")
                state["recovery"] = {"previous_pid": previous["pid"], "previous_error": previous.get("error"),
                                     "mode": "attach_existing_training", "attached_at": time.time()}
            else:
                for fusion, gpus in assignments.items():
                    output = root / "ppo" / f"{fusion}_121"
                    completion = read(output / "completion.json") if (output / "completion.json").exists() else None
                    if completion and completion["complete"]:
                        active[fusion], state["jobs"][fusion] = evaluate_final(root, fusion, gpus)
                    else:
                        resume = None
                        if completion and completion["planned_sampling_transition"]:
                            resume = output / "checkpoint_update_001000.pt"
                        elif output.exists() and any(output.iterdir()):
                            raise RuntimeError(f"Inspect interrupted output before restarting: {output}")
                        active[fusion], state["jobs"][fusion] = launch(root, fusion, gpus, resume)
            while active:
                for fusion, process in list(active.items()):
                    job = state["jobs"][fusion]
                    if process.poll() is None:
                        progress = root / "ppo" / f"{fusion}_121/progress.json"
                        try:
                            job["progress"] = read(progress)
                            job.pop("progress_read_error", None)
                        except (FileNotFoundError, json.JSONDecodeError) as error:
                            # Progress is advisory; its temporary unavailability
                            # must not stop supervision of a verified live job.
                            job["progress_read_error"] = {"error": str(error), "at": time.time()}
                        continue
                    if process.returncode != 0:
                        raise RuntimeError(f"{fusion} {job['phase']} failed, see {job['log']}")
                    if job["phase"] == "final_evaluation":
                        finished.add(fusion)
                        del active[fusion]
                        job.update(phase="complete", finished_at=time.time())
                        if fusion == "baseline" and "frozen_tracker" not in active and "frozen_tracker" not in finished:
                            active["frozen_tracker"], state["jobs"]["frozen_tracker"] = evaluate_final(root, "frozen_tracker", [0, 1])
                        continue
                    output = root / "ppo" / f"{fusion}_121"
                    completion = read(output / "completion.json")
                    if completion["planned_sampling_transition"] and completion["completed_updates"] == 1000:
                        audit_inputs(root)
                        active[fusion], state["jobs"][fusion] = launch(root, fusion, assignments[fusion],
                                                                     output / "checkpoint_update_001000.pt")
                    elif completion["complete"] and completion["completed_updates"] == 5000:
                        active[fusion], state["jobs"][fusion] = evaluate_final(root, fusion, assignments[fusion])
                    else:
                        raise RuntimeError(f"Unexpected training exit: {completion}")
                state["updated_at"] = time.time()
                write(root / "supervisor_state.json", state)
                if active:
                    time.sleep(15)
            if finished != {"baseline", "film", "frozen_tracker"}:
                raise RuntimeError(f"Missing final cases: {finished}")
            subprocess.run([PYTHON, "-u", str(ROOT / "scripts/report_memory350_ppo.py"),
                            "--run-root", str(root)], cwd=ROOT, env=environment([]), check=True)
            state.update(phase="complete", finished_at=time.time())
            write(root / "supervisor_state.json", state)
        except BaseException as error:
            # Preserve healthy jobs and checkpoints for inspection; never kill
            # unrelated processes, including the user's GPUs 4–7.
            state.update(phase="needs_inspection", error=str(error), updated_at=time.time())
            write(root / "supervisor_state.json", state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT))
    parser.add_argument("--attach-running", action="store_true")
    args = parser.parse_args()
    run(args.run_root, attach_running=args.attach_running)
