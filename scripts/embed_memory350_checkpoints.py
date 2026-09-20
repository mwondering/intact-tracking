#!/usr/bin/env python3
"""Embed frozen inference dependencies in new checkpoints of an existing trainer.

Future trainers save these dependencies directly. This observer bridges a
running process that already imported the old save implementation; it never
signals, restarts, or changes the trainer or its optimizer.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import torch

from intact_tracking.memory350_checkpoint import (
    dependencies_from_legacy_checkpoint, embed_checkpoint_file,
)


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def checkpoint_paths(directory, minimum):
    numbered = []
    for path in directory.glob("checkpoint_*.pt"):
        suffix = path.stem.removeprefix("checkpoint_")
        if suffix.isdigit() and int(suffix) >= minimum:
            numbered.append((int(suffix), path))
    return [p for _, p in sorted(numbered)] + [
        p for name in ("checkpoint_interrupted.pt", "checkpoint_final.pt")
        if (p := directory / name).is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--status-directory", required=True, type=Path)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--training-start-ticks", required=True, type=int)
    parser.add_argument("--start-iteration", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.start_iteration < 0:
        parser.error("Use a positive polling interval and nonnegative start iteration")
    torch.set_num_threads(1)
    args.status_directory.mkdir(parents=True, exist_ok=True)
    with (args.status_directory / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = {"pid": os.getpid(), "process_start_ticks": process_start(os.getpid()),
                   "started_at": time.time(), "arguments": {k: str(v) if isinstance(v, Path) else v
                                                            for k, v in vars(args).items()},
                   "automatic_training_control": False}
        atomic_json(args.status_directory / "process.json", process)
        stopped = False

        def stop(_signum, _frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        dependencies, seen, last_result = None, {}, None
        while not stopped:
            errors = []
            for path in checkpoint_paths(args.checkpoint_directory, args.start_iteration):
                try:
                    stat = path.stat()
                    identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if seen.get(str(path)) == identity:
                        continue
                    if dependencies is None:
                        state = torch.load(path, map_location="cpu", weights_only=False)
                        if "inference_bundle_version" not in state:
                            dependencies = dependencies_from_legacy_checkpoint(state)
                        del state
                    result = embed_checkpoint_file(path, dependencies)
                    result["embedded_at"] = time.time()
                    stat = path.stat()
                    seen[str(path)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    last_result = result
                    with (args.status_directory / "events.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    print(json.dumps(result), flush=True)
                except Exception as error:
                    errors.append({"path": str(path), "error": repr(error)})
                    print(json.dumps(errors[-1]), flush=True)
            training_live = process_start(args.training_pid) == args.training_start_ticks
            atomic_json(args.status_directory / "status.json", {
                "checked_at": time.time(), "status": "error" if errors else "healthy",
                "training_live": training_live, "last_result": last_result,
                "processed_files": len(seen), "errors": errors,
                "pid": os.getpid(), "process_start_ticks": process["process_start_ticks"],
            })
            if not training_live:
                if errors:
                    raise RuntimeError("Trainer exited with unpackaged checkpoints; see status.json")
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
