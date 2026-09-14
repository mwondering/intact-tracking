"""Sequential single-motion evaluations on an explicitly authorized GPU.

Checkpoint 300 is the predeclared primary endpoint. Earlier checkpoints are
learning-curve diagnostics, not an implicit best-checkpoint selection rule.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/simulator_preview_single_motion"
MOTION = "/data_zcy/wxy/motion_data_correct/motion_data_full/AMASS_LAFAN_Qingtong/lafan_qingtong/dance1_subject2.motion.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4), default=3)
    parser.add_argument("--seed", type=int, default=13001)
    parser.add_argument("--train-seed", type=int, default=121)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--actor-critic", action="store_true", help="Evaluate the matched shared-information critic runs")
    args = parser.parse_args()
    infix = "_ac" if args.actor_critic else ""
    jobs = []
    prior = ROOT / "runs/residual_policy_no_latent_nominal_v13_run2/checkpoint_1000.pt"
    for label, checkpoint in (("tracker", None), ("fixed_nominal", prior)):
        for physics in ("nominal", "dr"):
            jobs.append((label, checkpoint, physics, None))
    for iteration in (100, 200, 300):
        for training, mode in (("dr", "true"), ("dr", "zero"), ("nominal", "zero")):
            label = f"{training}_{mode}{infix}_seed{args.train_seed}_{iteration}"
            filename = "checkpoint_final.pt" if iteration == 300 else f"checkpoint_{iteration}.pt"
            checkpoint = RUNS / f"{training}_{mode}{infix}_seed{args.train_seed}" / filename
            jobs.append((label, checkpoint, training, None))
    for training, mode in (("dr", "true"), ("dr", "zero"), ("nominal", "zero")):
        label = f"{training}_{mode}{infix}_seed{args.train_seed}_300"
        checkpoint = RUNS / f"{training}_{mode}{infix}_seed{args.train_seed}/checkpoint_final.pt"
        jobs.append((label, checkpoint, "nominal" if training == "dr" else "dr", None))
    for override in ("zero", "shuffle"):
        label = f"dr_true{infix}_seed{args.train_seed}_300_override_{override}"
        checkpoint = RUNS / f"dr_true{infix}_seed{args.train_seed}/checkpoint_final.pt"
        jobs.append((label, checkpoint, "dr", override))
    failed = []
    for label, checkpoint, physics, override in jobs:
        output = RUNS / f"eval/seed{args.seed}/{label}_test_{physics}.json"
        if output.exists():
            continue
        deadline = time.monotonic() + 6 * 3600
        while checkpoint is not None and (not checkpoint.exists() or time.time() - checkpoint.stat().st_mtime < 10):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Checkpoint not available: {checkpoint}")
            time.sleep(15)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "intact_tracking.cli.adaptation_eval",
                   "--motion-file", MOTION, "--physics", physics,
                   "--seed", str(args.seed), "--repeats", str(args.repeats),
                   "--steps", str(args.steps), "--output", str(output)]
        if checkpoint is not None:
            command += ["--checkpoint", str(checkpoint)]
        if override:
            command += ["--simulator-preview-override", override]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        print(json.dumps({"event": "start", "label": label, "physics": physics}), flush=True)
        with output.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failed.append(str(output))
            print(json.dumps({"event": "failed", "label": label, "physics": physics}), flush=True)
        else:
            metrics = json.loads(output.read_text())
            print(json.dumps({"event": "complete", "label": label, "physics": physics,
                              "mean": metrics["mean"], "failure_rate": metrics["failure_rate"]}), flush=True)
    print(json.dumps({"event": "done", "failed": failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
