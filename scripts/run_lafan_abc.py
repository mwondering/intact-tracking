"""Train A/B/C for the fixed budget, then evaluate all endpoints and summarize."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from intact_tracking.tracker_finetune import LAFAN


def compare(rows, reference, endpoint):
    refs, candidates, rfails, cfails = [], [], [], []
    new_failures = rescued = 0
    for seed in (20001, 20002, 20003):
        r, c = rows[(reference, endpoint, seed)], rows[("C", endpoint, seed)]
        for key in ("protocol", "seed", "motion_files", "motion_ids", "start_frames", "metric_names",
                    "max_steps", "physics_world_fingerprints"):
            if r[key] != c[key]:
                raise ValueError(f"Unmatched endpoint evaluation: {key}")
        indices = [r["metric_names"].index(k) for k in ("error_body_pos", "error_joint_pos")]
        ids = np.asarray(r["motion_ids"])
        rv, cv = np.asarray(r["per_episode_metrics"])[:, indices], np.asarray(c["per_episode_metrics"])[:, indices]
        rf, cf = np.asarray(r["failed"], bool), np.asarray(c["failed"], bool)
        new_failures += int((cf & ~rf).sum())
        rescued += int((rf & ~cf).sum())
        for values, output in ((rv, refs), (cv, candidates), (rf, rfails), (cf, cfails)):
            output.append(np.array([values[ids == i].mean(0) for i in range(r["motions"])]))
    ref, cand = np.mean(refs, 0), np.mean(candidates, 0)
    rf, cf = np.mean(rfails, 0), np.mean(cfails, 0)
    rng = np.random.default_rng(39201)
    draws = []
    for _ in range(5000):
        ids = rng.integers(len(ref), size=len(ref))
        draws.append([*(cand[ids].mean(0) / ref[ids].mean(0)), cf[ids].mean() - rf[ids].mean()])
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return {"reference": reference, "endpoint": endpoint,
            "reference_body_joint": ref.mean(0).tolist(), "C_body_joint": cand.mean(0).tolist(),
            "C_over_reference": (cand.mean(0) / ref.mean(0)).tolist(),
            "ratio_ci95_low": low[:2].tolist(), "ratio_ci95_high": high[:2].tolist(),
            "reference_failure_rate": float(rf.mean()), "C_failure_rate": float(cf.mean()),
            "failure_delta_ci95": [float(low[2]), float(high[2])],
            "C_new_failure_episodes": new_failures, "C_rescued_failure_episodes": rescued,
            "both_errors_significantly_higher_this_training_seed": bool((low[:2] > 1).all()),
            "uncertainty": "paired bootstrap over motion, averaging three evaluation seeds; one training seed"}


def summarize(root):
    rows = {}
    table = []
    for arm in ("A", "B", "C", "frozen"):
        for endpoint in ("nominal", "hardest"):
            values = []
            for seed in (20001, 20002, 20003):
                row = json.loads((root / "eval" / f"{arm}_{endpoint}_{seed}.json").read_text())
                if arm != "frozen" and not row["training_reward_audit"]["eligible_fixed_reward_candidate"]:
                    raise ValueError("Candidate failed original reward audit")
                rows[(arm, endpoint, seed)] = row
                values.append(row)
            table.append({"policy": arm, "endpoint": endpoint,
                          "body_error": float(np.mean([v["mean"]["error_body_pos"] for v in values])),
                          "joint_error": float(np.mean([v["mean"]["error_joint_pos"] for v in values])),
                          "coverage_fraction": float(np.mean([v["coverage_fraction"] for v in values])),
                          "failures": sum(sum(v["failed"]) for v in values),
                          "episodes": sum(v["episodes"] for v in values)})
    result = {"table": table, "C_vs_A_nominal": compare(rows, "A", "nominal"),
              "C_vs_B_hardest": compare(rows, "B", "hardest"),
              "C_vs_frozen_nominal": compare(rows, "frozen", "nominal"),
              "C_vs_frozen_hardest": compare(rows, "frozen", "hardest"),
              "scope": "LaFAN in-distribution fine-tuning, not from-scratch nominal/DR training"}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/abc_lafan_20260907")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--a-gpus", default="4,5")
    parser.add_argument("--b-gpus", default="6,7")
    parser.add_argument("--c-gpus", default="2,3")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".orchestrator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes, logs = {}, []
        if not args.evaluate_only:
            for arm in ("A", "B", "C"):
                if (root / arm / "completion_audit.json").exists():
                    continue
                if (root / arm).exists():
                    raise ValueError(f"Incomplete existing {arm} run requires explicit recovery")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=getattr(args, arm.lower() + "_gpus"),
                           OMP_NUM_THREADS="1")
                log = (root / f"{arm}_train.log").open("w")
                logs.append(log)
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                           "-m", "intact_tracking.cli.tracker_finetune_train", "--condition", arm,
                           "--num-envs", "2048", "--iterations", str(args.iterations), "--seed", str(args.seed),
                           "--actor-lr", str(args.actor_lr), "--critic-lr", str(args.critic_lr),
                           "--output-dir", str(root / arm)]
                processes[arm] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(json.dumps({"started": arm, "pid": processes[arm].pid,
                                  "gpus": env["CUDA_VISIBLE_DEVICES"], "command": command}), flush=True)
            (root / "launch.json").write_text(json.dumps({"arguments": vars(args),
                "pids": {arm: p.pid for arm, p in processes.items()}}, indent=2) + "\n")
            while any(p.poll() is None for p in processes.values()):
                for arm, process in processes.items():
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f"Training {arm} failed ({process.returncode}); other jobs left intact")
                time.sleep(5)
            for arm, process in processes.items():
                if process.returncode != 0:
                    raise RuntimeError(f"Training {arm} failed")
            for log in logs:
                log.close()
        for arm in ("A", "B", "C"):
            audit = json.loads((root / arm / "completion_audit.json").read_text())
            if audit["completed_updates"] != args.iterations or audit["global_envs"] != 4096:
                raise ValueError("Training budget mismatch")
        print("All three training runs completed; starting paired endpoint evaluations", flush=True)
        (root / "eval").mkdir(exist_ok=True)
        tasks = [(arm, endpoint, seed) for arm in ("A", "B", "C", "frozen")
                 for endpoint in ("nominal", "hardest") for seed in (20001, 20002, 20003)]
        gpu_ids = (args.a_gpus + "," + args.b_gpus).split(",")

        def worker(gpu, jobs):
            for arm, endpoint, seed in jobs:
                output = root / "eval" / f"{arm}_{endpoint}_{seed}.json"
                if output.exists():
                    continue
                cmd = [sys.executable, "-m", "intact_tracking.cli.adaptation_eval",
                       "--motion-path", LAFAN, "--motion-manifest", str(root / "A/motions.txt"),
                       "--physics", endpoint, "--seed", str(seed), "--repeats", "8", "--steps", "500",
                       "--output", str(output)]
                if arm != "frozen":
                    cmd.extend(["--checkpoint", str(root / arm / "checkpoint_final.pt")])
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1"),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                print(json.dumps({"evaluated": arm, "endpoint": endpoint, "seed": seed}), flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            jobs = [pool.submit(worker, gpu, tasks[i::len(gpu_ids)]) for i, gpu in enumerate(gpu_ids)]
            for job in jobs:
                job.result()
        print(json.dumps(summarize(root), indent=2), flush=True)


if __name__ == "__main__":
    main()
