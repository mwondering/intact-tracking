"""Run equal-budget no-privilege residual controls with audited identical initialization."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/adaptation_goal"


def audit_initializations(paths):
    left, right = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for state_name in ("actor_state_dict", "critic_state_dict"):
        a, b = left[state_name], right[state_name]
        if a.keys() != b.keys():
            raise RuntimeError(f"Different initial {state_name} schema")
        different = [key for key in a if not torch.equal(a[key], b[key])]
        if different:
            raise RuntimeError(f"Different initial {state_name}: {different}")
    acfg, bcfg = [OmegaConf.to_container(r["cfg"].agent, resolve=True) for r in (left, right)]
    if acfg != bcfg:
        raise RuntimeError("Paired agent configurations differ")
    return dict(actor_tensors=len(left["actor_state_dict"]), critic_tensors=len(left["critic_state_dict"]),
                all_initial_tensors_bitwise_equal=True, agent_configuration_equal=True,
                initial_iteration=[left["iter"], right["iter"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[76, 77, 78])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--wait-audit-pid", type=int)
    args = parser.parse_args()
    if not set(args.gpus) <= {0, 1, 2, 3} or len(set(args.gpus)) != 2:
        parser.error("Two distinct authorized physical GPUs required")
    if args.wait_audit_pid:
        while True:
            try:
                command = Path(f"/proc/{args.wait_audit_pid}/cmdline").read_bytes()
            except FileNotFoundError:
                break
            if b"scripts/audit_adaptation_matrix.py" not in command:
                break
            time.sleep(10)
    for seed in args.seeds:
        names = [f"matched_{physics}_seed{seed}" for physics in ("nominal", "dr")]
        if any((RUNS / name).exists() for name in names):
            raise FileExistsError(f"Refusing to overwrite paired controls: {names}")
        children, handles = [], []
        try:
            for physics, name, gpu in zip(("nominal", "dr"), names, args.gpus, strict=True):
                command = [sys.executable, "-m", "intact_tracking.cli.adaptation_train",
                           "--physics", physics, "--no-privileged", "--seed", str(seed),
                           "--training-rng-seed", str(seed * 10000 + 1), "--save-initial",
                           "--num-envs", str(args.num_envs), "--iterations", str(args.iterations),
                           "--save-interval", "250", "--residual-scale", "1.0",
                           "--actor-lr", "0.00002", "--actor-lr-schedule", "fixed",
                           "--critic-lr", "0.0005", "--critic-warmup-updates", "50",
                           "--initial-action-std", "0.1", "--entropy-coef", "0.0002",
                           "--sampling-mode", "uniform", "--training-start-mode", "reference",
                           "--output-dir", str(RUNS / name)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                env.pop("PYTHONPATH", None)
                log = (RUNS / f"v2_{name}.log").open("w")
                handles.append(log)
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append(process)
                print(json.dumps(dict(event="training_started", name=name, gpu=gpu, pid=process.pid,
                                      command=command)), flush=True)
            paths = [RUNS / name / "checkpoint_initial.pt" for name in names]
            deadline = time.monotonic() + 300
            while not all(p.exists() and time.time() - p.stat().st_mtime > 3 for p in paths):
                if any(child.poll() is not None for child in children):
                    raise RuntimeError("Paired trainer exited before initial audit")
                if time.monotonic() > deadline:
                    raise TimeoutError("Paired initialization was not saved")
                time.sleep(3)
            report = audit_initializations(paths)
            report.update(seed=seed, names=names, created_utc=datetime.now(timezone.utc).isoformat(),
                          iterations=args.iterations, num_envs=args.num_envs,
                          contract="same pretrained frozen tracker, same residual architecture/observations/rewards/initialization/budget; only subsequent training physics differs; not training from scratch")
            (RUNS / f"matched_seed{seed}_initial_audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(event="initial_audit_passed", **report)), flush=True)
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("Paired trainer failed")
                time.sleep(10)
            if any(child.returncode for child in children):
                raise RuntimeError("Paired trainer failed")
            print(json.dumps(dict(event="pair_completed", seed=seed)), flush=True)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
