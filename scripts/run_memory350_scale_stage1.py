"""Queue four-GPU encoder scaling, then train and compare fixed checkpoints."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, TRACKER, DATASET, SMOKE_MOTION, process_environment


REFERENCE = ROOT / "runs/limb_context_20260909_memory350/stage1_8192"
MILESTONES = (100, 500, 1000, 3000, 5000, 7500, 10000, 15000, 20000, 22700)
ENTITY = "2486344338-zhejiang-university"
PROJECT = "intact-forward-predictor"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def source_check():
    manifest = ROOT / ".runtime/memory350_encoder2x_v1/protected_sources.json"
    expected = json.loads(manifest.read_text())
    changed = [name for name, digest in expected.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"Protected reference sources changed: {changed}")
    return expected


def gpu_status(gpus):
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    cards = {uuid.strip(): {"index": int(index), "free_mib": int(free), "processes": []}
             for index, uuid, free in csv.reader(raw.splitlines()) if int(index) in gpus}
    if len(cards) != len(gpus):
        raise RuntimeError("Requested physical GPUs do not exist")
    raw = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    for row in csv.reader(raw.splitlines()):
        if len(row) == 3 and row[0].strip() in cards:
            cards[row[0].strip()]["processes"].append({"pid": int(row[1]), "executable": row[2].strip()})
    return sorted(cards.values(), key=lambda value: value["index"])


def available(cards):
    return all(not card["processes"] and card["free_mib"] >= 60000 for card in cards)


def command_for(output, smoke, reference=REFERENCE):
    command = [PYTHON, "-B", "-u", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
        "--max-restarts=0", "-m", "intact_tracking.cli.forward_memory_scale_train",
        "--checkpoint-file", TRACKER, "--output-dir", str(output), "--num-envs", "8192",
        "--context-depth", "4", "--chunk-depth", "2", "--memory-depth", "4",
        "--wandb", "--wandb-project", PROJECT, "--wandb-entity", ENTITY,
        "--wandb-group", output.parent.name + ("-stage1-smoke" if smoke else "-stage1"),
        "--wandb-name", "memory350_encoder2x_" + ("smoke" if smoke else "seed717")]
    if smoke:
        command += ["--motion-file", SMOKE_MOTION, "--bounded-smoke", "--updates", "2"]
    else:
        command += ["--motion-path", DATASET, "--until-user-stop", "--updates", "8000",
                    "--comparison-reference-dir", str(reference)]
    return command


def live_workers(root):
    result = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if "intact_tracking.cli.forward_memory_scale_train" in argv and "--output-dir" in argv:
                output = Path(argv[argv.index("--output-dir") + 1]).resolve()
                if output.is_relative_to(root):
                    result.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def update_report(root, env):
    with (root / "report.log").open("a") as log:
        result = subprocess.run([PYTHON, "-B", str(ROOT / "scripts/report_memory350_scale.py"),
            "--run-root", str(root)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def poll_evaluation(root, output, reference, env, gpu, active):
    if active is not None:
        child, handle, update = active
        if child.poll() is None:
            return active
        handle.close()
        write_json(root / "comparison" / f"update_{update:06d}" / "process_result.json",
                   {"update": update, "exit_code": child.returncode, "finished_at": time.time()})
        update_report(root, env)
        print(json.dumps({"event": "paired_evaluation_finished", "update": update,
                          "exit_code": child.returncode}), flush=True)
        result = read_json(root / "comparison" / f"update_{update:06d}" / "result.json")
        if result is not None:
            values = result["groups"]["all"]
            print(f'[配对预测测试 {update}] 原版 NMSE={values["original_five_step_nmse"]:.8f}, '
                  f'encoder2x={values["encoder2x_five_step_nmse"]:.8f}, '
                  f'误差降低={values["error_reduction_percent"]:+.2f}%, '
                  f'误差比95%CI={values["paired_world_bootstrap_ratio_ci95"]}', flush=True)
    for update in MILESTONES:
        folder = root / "comparison" / f"update_{update:06d}"
        if (folder / "process_result.json").exists():
            continue
        scaled = output / f"update_{update:06d}.pt"
        original = reference / f"update_{update:06d}.pt"
        if not scaled.exists() or not original.exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu),
                         WANDB_RUN_ID="m350scale-eval-" + hashlib.sha256(str(folder).encode()).hexdigest()[:12])
        command = [PYTHON, "-B", "-u", str(ROOT / "scripts/evaluate_memory350_scale.py"),
            "--reference-checkpoint", str(original), "--scaled-checkpoint", str(scaled),
            "--output", str(folder), "--wandb", "--group", root.name + "-paired-evaluation"]
        handle = (folder / "evaluation.log").open("w")
        child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write_json(folder / "process.json", {"pid": child.pid, "update": update,
                   "command": command, "physical_gpu": gpu, "started_at": time.time()})
        return child, handle, update
    return None


def run_job(root, output, smoke, reference, gpus):
    source_check()
    if (output / "run_config.json").exists():
        raise RuntimeError(f"Refusing to restart existing training directory: {output}")
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONUNBUFFERED="1",
        WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
        WANDB_RUN_ID="m350scale-" + hashlib.sha256(str(output).encode()).hexdigest()[:12], WANDB_RESUME="allow")
    command = command_for(output, smoke, reference)
    log_path = root / (output.name + ".log")
    with log_path.open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = {"torchrun_pid": child.pid,
        "process_start_ticks": int(Path(f"/proc/{child.pid}/stat").read_text().split()[21]),
        "command": command, "gpus": gpus, "num_envs_per_rank": 8192,
        "output": str(output), "log": str(log_path), "started_at": time.time()}
    write_json(root / ("smoke_process.json" if smoke else "training_process.json"), record)
    print(json.dumps({"event": "training_launched", **record}), flush=True)
    evaluation, last_report, signal_sent = None, None, False
    while child.poll() is None:
        progress = read_json(output / "progress.json")
        if not smoke:
            evaluation = poll_evaluation(root, output, reference, env, gpus[0], evaluation)
            metrics = output / "metrics.jsonl"
            stamp = metrics.stat().st_mtime_ns if metrics.exists() else None
            if stamp is not None and stamp != last_report:
                update_report(root, env)
                last_report = stamp
        write_json(root / "state.json", {"status": "smoke_running" if smoke else "training_running",
            "launcher_pid": os.getpid(), "job": record, "progress": progress, "heartbeat": time.time()})
        if (root / "STOP").exists() and not signal_sent:
            # This session contains only the child training job launched above.
            os.killpg(child.pid, signal.SIGTERM)
            signal_sent = True
        time.sleep(10)
    record.update(exit_code=child.returncode, finished_at=time.time())
    write_json(root / ("smoke_process_result.json" if smoke else "training_process_result.json"), record)
    while not smoke:
        evaluation = poll_evaluation(root, output, reference, env, gpus[0], evaluation)
        if evaluation is None:
            break
        time.sleep(10)
    completion = read_json(output / "completion.json")
    if child.returncode != 0:
        raise RuntimeError(f"Training exited with {child.returncode}; no automatic restart: {log_path}")
    if completion is None or (smoke and completion["completed_updates"] != 2):
        raise RuntimeError("Missing or incomplete successful-training record")
    config = read_json(output / "run_config.json")
    ranks = config["dataset"]["runtime_audits_by_rank"]
    if (len(ranks) != 4 or {int(rank["physical_gpu"]) for rank in ranks} != set(gpus)
            or any(rank["num_envs"] != 8192 for rank in ranks)):
        raise RuntimeError("Training runtime GPU/environment configuration differs")
    return completion


def run(args):
    root, reference = args.run_root.resolve(), args.reference.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("All output must stay inside the current project")
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(set(gpus)) != 4 or any(gpu not in (0, 1, 2, 3) for gpu in gpus):
        raise ValueError("This experiment is assigned physical GPUs 0-3")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live_workers(root):
            raise RuntimeError("Existing training workers must not be duplicated")
        contract = {"gpus": gpus, "num_envs_per_rank": 8192, "dataset": DATASET,
            "reference": str(reference), "context_parameters_original": 1035328,
            "context_parameters_scaled": 2026688, "predictor_parameters": 19059798,
            "attention_depths_original": [1, 2, 2], "attention_depths_scaled": [2, 4, 4],
            "maximum_updates": None, "cosine_horizon_updates": 8000,
            "paired_updates": list(MILESTONES), "primary_comparison_update": 22700,
            "protected_sources": source_check(), "stage2_jobs": [],
            "resource_policy": "wait for all assigned GPUs to be idle; never stop external jobs",
            "formal_command": command_for(root / "stage1_8192", False, reference),
            "wandb_entity": ENTITY, "wandb_project": PROJECT}
        write_json(root / "launch_contract.json", contract)
        stable = 0
        while stable < 2:
            if (root / "STOP").exists():
                write_json(root / "state.json", {"status": "cancelled_before_launch", "heartbeat": time.time()})
                return
            cards = gpu_status(gpus)
            stable = stable + 1 if available(cards) else 0
            write_json(root / "state.json", {"status": "waiting_for_gpus", "launcher_pid": os.getpid(),
                "gpus": cards, "required_free_mib_per_gpu": 60000, "heartbeat": time.time()})
            if stable < 2:
                time.sleep(30)
        smoke = root / "smoke_8192"
        if read_json(smoke / "completion.json") is None:
            run_job(root, smoke, True, reference, gpus)
        completion = run_job(root, root / "stage1_8192", False, reference, gpus)
        write_json(root / "state.json", {"status": "training_stopped", "completion": completion,
                                        "heartbeat": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/limb_context_20260911_memory350_encoder2x")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.run_root.resolve().is_relative_to(ROOT) and args.run_root.exists():
            write_json(args.run_root / "state.json", {"status": "failed", "error": str(error), "heartbeat": time.time()})
        raise
