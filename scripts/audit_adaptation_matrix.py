"""Frozen cross-physics and all-path privilege-ablation evaluation matrix.

This is an audit/development cohort, not final stage-two confirmation.
No checkpoint selection occurs inside the matrix. Each GPU runs one evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKER = "/data_zcy/wxy/SP_Tracking/logs/rsl_rl/g1_tracking/2026-09-02_04-48-42_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_8gpu_12288env_motion_data_correct/checkpoint_72000.pt"
MODELS = {
    "frozen_tracker": TRACKER,
    "existing_nominal": "runs/residual_policy_no_latent_nominal_v13_run2/checkpoint_1000.pt",
    "matched_nominal1750": "runs/adaptation_goal/nominal_scale1_seed42/checkpoint_1750.pt",
    "matched_dr1750": "runs/adaptation_goal/dr_control_scale1_seed42/checkpoint_1750.pt",
    "clean_teacher500": "runs/adaptation_goal/oracle_clean_payload_seed58/checkpoint_500.pt",
    "student_rotation350": "runs/adaptation_goal/context_right_arm_rotation4_ppo_seed75/checkpoint_350.pt",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not set(args.gpus) <= {0, 1, 2, 3} or len(set(args.gpus)) != len(args.gpus):
        parser.error("Distinct physical GPUs 0–3 only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    # First finish the complete first seed, then repeat for independent starts.
    for seed in (91001, 91002, 91003):
        for physics in ("nominal", "dr"):
            for name in MODELS:
                jobs.append(dict(model=name, physics=physics, seed=seed, remove="normal"))
            for remove in ("height", "contact", "height_contact"):
                jobs.append(dict(model="clean_teacher500", physics=physics, seed=seed, remove=remove))
    identity = {name: dict(path=str(Path(path).resolve()), sha256=sha256(path)) for name, path in MODELS.items()}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), models=identity,
                    jobs=jobs, protocol="balanced_fixed_starts_v2_isolated_resets",
                    repeats=8, steps=500, cohort="audit/development, NOT final stage-two confirmation",
                    interpretation="matched1750 has one training seed and equal update budget; not a from-scratch-only-nominal comparison; ablation is test-time dependence, not retraining causal benefit",
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (
                        Path(__file__).resolve(), ROOT / "src/intact_tracking/adaptation_policy.py",
                        ROOT / "src/intact_tracking/cli/adaptation_eval.py")})
    manifest_path = output / "freeze.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("models", "jobs", "protocol", "source_sha256"):
            if old[key] != manifest[key]:
                raise RuntimeError(f"Frozen audit identity changed: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        failures = []
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return failures
            label = f"{job['model']}_{job['physics']}_{job['remove']}_seed{job['seed']}"
            destination = output / f"{label}.json"
            if destination.exists():
                result = json.loads(destination.read_text())
                assert result["checkpoint_sha256"] == identity[job["model"]]["sha256"]
                assert result["protocol"] == manifest["protocol"]
                continue
            command = [sys.executable, "-m", "intact_tracking.cli.adaptation_eval",
                       "--physics", job["physics"], "--seed", str(job["seed"]),
                       "--teacher-remove-privilege", job["remove"], "--output", str(destination)]
            if job["model"] != "frozen_tracker":
                command += ["--checkpoint", identity[job["model"]]["path"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            env.pop("PYTHONPATH", None)
            print(json.dumps(dict(event="started", gpu=gpu, label=label)), flush=True)
            with destination.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps(dict(event="finished", gpu=gpu, label=label, exit_code=result.returncode)), flush=True)
            if result.returncode:
                failures.append(label)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        failures = sum(pool.map(worker, args.gpus), [])
    print(json.dumps(dict(event="audit_complete", failures=failures)), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
