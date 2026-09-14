"""Evaluate completed controls on their released GPUs while latent PPO continues."""

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from run_limb_context_experiment import ROOT, PYTHON, process_environment, ppo_directory

CONTROLS = ("baseline_121", "baseline_122", "baseline_123", "constant_121")


def run(root):
    root = root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Run directory must remain inside this project")
    lock = (root / "control_eval_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        paths = [ppo_directory(root) / name / "completion.json" for name in CONTROLS]
        if all(p.exists() for p in paths):
            records = [json.loads(p.read_text()) for p in paths]
            if all(r["complete"] and r["completed_updates"] >= 5000 for r in records):
                break
        time.sleep(30)
    state = json.loads((root / "state.json").read_text())
    gpus = sorted({gpu for job in state["jobs"] if job["name"] in CONTROLS for gpu in job["gpus"]})
    command = [PYTHON, "-u", str(ROOT / "scripts/evaluate_limb_context_experiment.py"),
               "--run-root", str(root), "--policies", "frozen", *CONTROLS,
               "--gpus", *map(str, gpus)]
    print(json.dumps({"event": "completed_control_evaluation", "gpus": gpus,
                      "policies": ["frozen", *CONTROLS]}), flush=True)
    subprocess.run(command, cwd=ROOT, env=process_environment(), check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    run(parser.parse_args().run_root)
