"""Keep the experiment uploader alive independently of the training scheduler."""

import argparse
import fcntl
import json
import os
import time

from run_limb_context_experiment import ROOT, ensure_wandb_sync, process_environment, write_state


def run(root):
    root = root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Experiment directory must remain inside the project")
    directory = root / ".wandb_sync"
    directory.mkdir(exist_ok=True)
    lock = (directory / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = process_environment()
    while True:
        state_file = directory / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            runs = state.get("runs", {})
            if runs and all(row.get("finished") for row in runs.values()):
                write_state(directory / "supervisor.json", {
                    "pid": os.getpid(), "updated_at": time.time(), "complete": True})
                return
        ensure_wandb_sync(root, env)
        write_state(directory / "supervisor.json", {
            "pid": os.getpid(), "updated_at": time.time(), "complete": False})
        time.sleep(20)


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    run(parser.parse_args().root)
