#!/usr/bin/env python3
"""Export completed portable PPO snapshots on CPU without blocking training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

from embed_memory350_checkpoints import atomic_json, process_start
from intact_tracking.memory350_onnx_export import export_policy
from intact_tracking.rollout.mjlab_adapter import _sha256


def publish_latest(directory, exported):
    deploy = directory / "deploy"
    deploy.mkdir(exist_ok=True)
    for name in ("policy.onnx", "policy.json", "deploy_metadata.json", "policy_runtime.py"):
        link = directory / name
        target = f"deploy/latest/{name}"
        if os.path.lexists(link):
            if not link.is_symlink() or os.readlink(link) != target:
                raise FileExistsError(f"Deployment alias already belongs to another artifact: {link}")
        else:
            link.symlink_to(target)
    temporary = deploy / f".latest.{os.getpid()}"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(exported.name, target_is_directory=True)
    os.replace(temporary, deploy / "latest")


def launch_watcher(directory, status_directory, training_pid, training_start_ticks):
    directory, status_directory = Path(directory).resolve(), Path(status_directory).resolve()
    status_directory.mkdir(parents=True, exist_ok=True)
    record = status_directory / "launch.json"
    if record.exists():
        previous = json.loads(record.read_text())
        if process_start(previous["pid"]) == previous["process_start_ticks"]:
            if previous["training_pid"] != training_pid or previous["training_start_ticks"] != training_start_ticks:
                raise RuntimeError("Existing exporter belongs to a different training process")
            return previous
    root = Path(__file__).resolve().parents[1]
    command = [str(root / ".venv/bin/python"), "-B", "-u", str(Path(__file__).resolve()),
               "--checkpoint-directory", str(directory), "--status-directory", str(status_directory),
               "--training-pid", str(training_pid), "--training-start-ticks", str(training_start_ticks)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (status_directory / "watcher.log").open("ab") as stream:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"pid": child.pid, "process_start_ticks": process_start(child.pid), "started_at": time.time(),
              "training_pid": training_pid, "training_start_ticks": training_start_ticks,
              "command": command, "automatic_training_control": False}
    atomic_json(record, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--status-directory", required=True, type=Path)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--training-start-ticks", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("Polling interval must be positive")
    torch.set_num_threads(1)
    args.status_directory.mkdir(parents=True, exist_ok=True)
    with (args.status_directory / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {"pid": os.getpid(), "process_start_ticks": process_start(os.getpid()),
                    "started_at": time.time(), "automatic_training_control": False}
        atomic_json(args.status_directory / "process.json", identity)
        stopped, last_result, last_signature = False, None, None

        def stop(_signum, _frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not stopped:
            errors = []
            training_live = process_start(args.training_pid) == args.training_start_ticks
            paths = list(args.checkpoint_directory.glob("checkpoint_[0-9]*.pt"))
            if not training_live:
                paths += [p for name in ("checkpoint_final.pt", "checkpoint_interrupted.pt")
                          if (p := args.checkpoint_directory / name).exists()]
            if paths:
                if training_live:
                    checkpoint = max(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
                else:
                    checkpoint = max(paths, key=lambda p: int(torch.load(
                        p, map_location="cpu", weights_only=False)["completed_updates"]))
                stat = checkpoint.stat()
                signature = (str(checkpoint), stat.st_ino, stat.st_mtime_ns)
                if signature != last_signature:
                    try:
                        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        portable = "inference_bundle_version" in state
                        update = int(state["completed_updates"])
                        del state
                        if portable:
                            exported = args.checkpoint_directory / "deploy" / checkpoint.stem
                            atomic_json(args.status_directory / "status.json", {
                                **identity, "checked_at": time.time(), "status": "exporting",
                                "checkpoint": str(checkpoint), "training_live": training_live,
                                "last_result": last_result, "errors": []})
                            if not exported.exists():
                                export_policy(checkpoint, exported)
                            metadata = json.loads((exported / "policy.json").read_text())
                            if (metadata["checkpoint_sha256"] != _sha256(checkpoint)
                                    or metadata["onnx_sha256"] != _sha256(exported / "policy.onnx")
                                    or not metadata["validation"]["passed"]):
                                raise RuntimeError("Existing export does not match the checkpoint")
                            publish_latest(args.checkpoint_directory, exported)
                            last_result = {"checkpoint": str(checkpoint), "directory": str(exported),
                                           "completed_updates": update, "exported_at": time.time(),
                                           "validation_passed": True,
                                           "onnx_bytes": (exported / "policy.onnx").stat().st_size}
                            last_signature = signature
                            with (args.status_directory / "events.jsonl").open("a") as stream:
                                stream.write(json.dumps(last_result) + "\n")
                            print(json.dumps(last_result), flush=True)
                        elif not training_live:
                            errors.append({"checkpoint": str(checkpoint), "error": "Final checkpoint is not portable yet"})
                    except Exception as error:
                        errors.append({"checkpoint": str(checkpoint), "error": repr(error)})
                        print(json.dumps(errors[-1]), flush=True)
            atomic_json(args.status_directory / "status.json", {
                **identity, "checked_at": time.time(), "status": "error" if errors else "healthy",
                "training_live": training_live, "last_result": last_result, "errors": errors})
            if not training_live and not errors:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
