"""Launch only the new stage1 on GPU 0–3; retry at 4096 only after actual OOM."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from run_limb_context_experiment import ROOT, PYTHON, TRACKER, DATASET, SMOKE_MOTION, process_environment

ENTITY = "2486344338-zhejiang-university"
PROJECT = "intact-forward-predictor"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command_for(output, num_envs, smoke):
    command = [PYTHON, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
               "--max-restarts=0", "-m", "intact_tracking.cli.forward_memory_nominal_train",
               "--nominal-fraction", "0.5", "--checkpoint-file", TRACKER, "--output-dir", str(output), "--num-envs", str(num_envs),
               "--wandb", "--wandb-project", PROJECT, "--wandb-entity", ENTITY,
               "--wandb-group", output.parent.name + ("-stage1-smoke" if smoke else "-stage1"),
               "--wandb-name", output.parent.name + "-" + output.name]
    if smoke:
        command += ["--motion-file", SMOKE_MOTION, "--bounded-smoke", "--updates", "2"]
    else:
        command += ["--motion-path", DATASET, "--until-user-stop", "--updates", "8000"]
    return command


def baseline_unchanged():
    manifest = ROOT / ".runtime/memory350_v1/baseline_source_manifest.json"
    expected = json.loads(manifest.read_text())
    changed = [name for name, digest in expected.items() if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"Protected baseline sources changed: {changed}")
    return expected


def observed_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return [list(map(int, line.split(","))) for line in result.stdout.splitlines() if int(line.split(",")[0]) in (0, 1, 2, 3)]


def has_oom(log_path):
    body = log_path.read_text(errors="replace")
    return bool(re.search(r"CUDA out of memory|CUDA_ERROR_OUT_OF_MEMORY|torch\.OutOfMemoryError|Warp.*out of memory", body, re.I))


def existing_workers(root):
    """A lost launcher heartbeat must never start a second copy of live training."""
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = process.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if "intact_tracking.cli.forward_memory_nominal_train" not in argv or "--output-dir" not in argv:
            continue
        output = Path(argv[argv.index("--output-dir") + 1]).resolve()
        if output.is_relative_to(root):
            found.append(int(process.name))
    return found


def fresh_directory(root, kind, num_envs):
    base = root / f"{kind}_{num_envs}"
    if not base.exists():
        return base
    index = 2
    while (root / f"{kind}_{num_envs}_attempt{index}").exists():
        index += 1
    return root / f"{kind}_{num_envs}_attempt{index}"


def run_job(root, output, num_envs, smoke, history):
    baseline_unchanged()
    command = command_for(output, num_envs, smoke)
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", PYTHONUNBUFFERED="1",
               WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
               WANDB_RUN_ID="m350n50-" + hashlib.sha256(str(output).encode()).hexdigest()[:12],
               WANDB_RESUME="allow")
    log = root / (output.name + ".log")
    record = {"phase": "smoke" if smoke else "stage1", "output": str(output), "log": str(log),
              "command": command, "num_envs_per_rank": num_envs, "gpus": [0, 1, 2, 3],
              "gpu_memory_mib_before_launch": observed_gpus(), "started_at": time.time()}
    with log.open("w") as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record.update(torchrun_pid=child.pid, process_start_ticks=int(Path(f"/proc/{child.pid}/stat").read_text().split()[21]))
        history.append(record)
        if not smoke:
            link = root / "stage1"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(output.name, target_is_directory=True)
        write_json(root / "state.json", {"status": "running", "launcher_pid": os.getpid(), "job": record,
                                          "attempts": history, "heartbeat": time.time()})
        print(json.dumps({"event": "training_launched", **record}), flush=True)
        while child.poll() is None:
            write_json(root / "state.json", {"status": "running", "launcher_pid": os.getpid(), "job": record,
                                              "attempts": history, "heartbeat": time.time()})
            time.sleep(5)
        record.update(exit_code=child.returncode, finished_at=time.time())
    record["oom"] = child.returncode != 0 and has_oom(log)
    write_json(root / "state.json", {"status": "job_finished", "job": record, "attempts": history})
    print(json.dumps({"event": "training_process_finished", **record}), flush=True)
    if child.returncode == 0:
        completion = json.loads((output / "completion.json").read_text())
        if smoke and completion["completed_updates"] != 2:
            raise RuntimeError("Four-rank smoke did not complete both optimizer updates")
        config = json.loads((output / "run_config.json").read_text())
        ranks = config["dataset"]["runtime_audits_by_rank"]
        if (len(ranks) != 4 or {r["physical_gpu"] for r in ranks} != {"0", "1", "2", "3"}
                or any(r["num_envs"] != num_envs or r["episode_length_control_steps"] != 1000 for r in ranks)):
            raise RuntimeError("Runtime GPU/environment/episode contract changed")
    return child.returncode, record["oom"]


def run(root):
    root = root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("All experiment files must remain inside the current project")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live = existing_workers(root)
        if live:
            raise RuntimeError(f"Training processes are still live; observe them instead of restarting: {live}")
        protected = baseline_unchanged()
        write_json(root / "launch_contract.json", {
            "gpus": [0, 1, 2, 3], "environments_first": 8192, "environments_after_confirmed_oom": 4096,
            "dataset": DATASET, "nominal_a_fraction": 0.5, "initialization": "from scratch", "maximum_updates": None, "stage2_jobs": [],
            "wandb_project": PROJECT, "wandb_entity": ENTITY,
            "protected_baseline_sources": protected,
            "new_sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT / "src/intact_tracking/cli/forward_memory_nominal_train.py",
                                      *sorted((ROOT / "src/intact_tracking").glob("memory350_*.py"))]},
        })
        history = []
        if (root / "state.json").exists():
            history = json.loads((root / "state.json").read_text()).get("attempts", [])
        for count in (8192, 4096):
            smoke = fresh_directory(root, "smoke", count)
            code, oom = run_job(root, smoke, count, True, history)
            if code:
                if oom and count == 8192:
                    continue
                raise RuntimeError(f"Smoke failed; no automatic non-OOM restart. See {smoke.name}.log")
            output = fresh_directory(root, "stage1", count)
            code, oom = run_job(root, output, count, False, history)
            if code and oom and count == 8192:
                continue
            if code:
                raise RuntimeError(f"Formal training failed. See {output.name}.log")
            write_json(root / "state.json", {"status": "stopped", "attempts": history, "finished_at": time.time()})
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/limb_context_20260911_memory350_nominal50")
    args = parser.parse_args()
    run(args.run_root)
