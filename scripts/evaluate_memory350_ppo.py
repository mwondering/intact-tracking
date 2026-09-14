"""Run the predeclared final tracking matrix on a completed PPO arm or frozen tracker."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from intact_tracking.memory350_policy_checkpoint_eval import (
    ROOT, PYTHON, digest, evaluate_checkpoint, evaluation_environment, load_protocol, write_json,
)
from intact_tracking.limb_context_protocol import TRACKER, TRACKER_SHA256
from intact_tracking.memory350_policy_protocol import EVAL_VERSION


def run(root, fusion, gpus):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Evaluation outputs must remain in the project")
    protocol_path = root / "protocols/final.json"
    directory = root / "final" / fusion
    if fusion != "frozen_tracker":
        checkpoint = root / "ppo" / f"{fusion}_121/checkpoint_update_005000.pt"
        evaluate_checkpoint(checkpoint, directory, protocol_path, 5000, gpus)
        return
    protocol, files = load_protocol(protocol_path)
    output = directory / "endpoint_eval/update_000000"
    output.mkdir(parents=True, exist_ok=True)
    items = list(protocol["cases"].items())
    for start in range(0, len(items), 2):
        children, handles = [], []
        try:
            for gpu, (case, spec) in zip(gpus, items[start:start + 2]):
                target = output / f"{case}.json"
                if target.exists():
                    continue
                cmd = [PYTHON, "-u", "-m", "intact_tracking.cli.memory350_policy_eval",
                       "--tracker-checkpoint", TRACKER, "--dr-profile", "tracker_dr_plus_limb_payload",
                       "--motion-manifest", protocol["motion_manifest"], "--motion-path", protocol["motion_path"],
                       "--output", str(target), "--seed", str(protocol["seed"]), "--steps", str(protocol["steps"]),
                       "--memory-start", spec["memory_start"], "--warmup-steps", str(protocol["warmup_steps"])]
                if spec["masses"] is not None:
                    cmd += ["--fixed-masses", *map(str, spec["masses"])]
                handle = (output / f"{case}.log").open("a")
                handles.append(handle)
                children.append(subprocess.Popen(cmd, cwd=ROOT, env=evaluation_environment(gpu),
                                                 stdout=handle, stderr=subprocess.STDOUT))
            for child in children:
                if child.wait(timeout=1800) != 0:
                    raise RuntimeError(f"Frozen tracker evaluation failed; see {output}")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for handle in handles:
                handle.close()
    metrics = {}
    for case, spec in items:
        target = output / f"{case}.json"
        row = json.loads(target.read_text())
        if (row["protocol"] != EVAL_VERSION or row["checkpoint_sha256"] != TRACKER_SHA256
                or row["completed_training_updates"] is not None or row["motion_files"] != files
                or row["memory_start"] != spec["memory_start"]
                or row["arguments"]["fixed_masses"] != spec["masses"]
                or row["max_steps"] != protocol["steps"] or row["seed"] != protocol["seed"]
                or not target.with_suffix(".traces.npz").exists()):
            raise RuntimeError(f"Frozen evaluation contract mismatch: {case}")
        metrics[case] = {**row["mean"], "failure_rate": row["failure_rate"], "coverage_fraction": row["coverage_fraction"]}
    write_json(output / "summary.json", {"checkpoint": TRACKER, "checkpoint_sha256": TRACKER_SHA256,
               "protocol_sha256": digest(protocol_path), "metrics": metrics, "unix_time": time.time()})
    print(json.dumps({"event": "frozen_tracker_final_complete", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--fusion", choices=("baseline", "film", "frozen_tracker"), required=True)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or not set(args.gpus).issubset(range(4)):
        raise ValueError("Only GPUs 0–3 are authorized for this experiment")
    run(args.run_root, args.fusion, args.gpus)
