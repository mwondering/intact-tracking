"""Evaluate predeclared final paired controls on fresh, shared nominal/DR starts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from audit_adaptation_matrix import sha256
from compare_adaptation_evals import compare
from summarize_adaptation_audit import paired_failure_audit

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/adaptation_goal"


def final_pair_ready(models, controller_alive, now):
    paths = tuple(models.values())
    missing = [path for path in paths if not path.exists()]
    if missing and not controller_alive:
        raise RuntimeError(f"Controller exited with missing final checkpoints: {missing}")
    # A successful controller can exit during the final checkpoint's 15-second
    # settling interval. Existing but fresh files are not a training failure.
    return not missing and all(now - path.stat().st_mtime > 15 for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    output = RUNS / "eval_v2/matched_controls_920xx"
    output.mkdir(parents=True, exist_ok=True)
    for training_seed in (76, 77, 78):
        models = {physics: RUNS / f"matched_{physics}_seed{training_seed}/checkpoint_final.pt" for physics in ("nominal", "dr")}
        while True:
            try:
                command = Path(f"/proc/{args.controller_pid}/cmdline").read_bytes()
            except FileNotFoundError:
                command = b""
            if final_pair_ready(models, b"scripts/run_matched_dr_controls.py" in command, time.time()):
                break
            time.sleep(10)
        audit = json.loads((RUNS / f"matched_seed{training_seed}_initial_audit.json").read_text())
        assert audit["all_initial_tensors_bitwise_equal"] and audit["agent_configuration_equal"]
        identity = {physics: dict(path=str(path), sha256=sha256(path)) for physics, path in models.items()}
        frozen = output / f"training_seed{training_seed}_freeze.json"
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(), models=identity,
                      training_seed=training_seed, evaluation_seeds=[92001, 92002, 92003],
                      checkpoint_selection="predeclared final1000updates, no test-based checkpoint selection",
                      initial_audit=audit, cohort="controlled training comparison, not final stage-two confirmation")
        if frozen.exists():
            assert json.loads(frozen.read_text())["models"] == identity
        else:
            frozen.write_text(json.dumps(record, indent=2) + "\n")

        def evaluate_training_branch(train_physics, gpu, training_seed=training_seed, identity=identity, models=models):
            for test_physics in ("nominal", "dr"):
                for seed in (92001, 92002, 92003):
                    label = f"train{training_seed}_{train_physics}_test{test_physics}_seed{seed}"
                    path = output / f"{label}.json"
                    if path.exists():
                        assert json.loads(path.read_text())["checkpoint_sha256"] == identity[train_physics]["sha256"]
                        continue
                    command = [sys.executable, "-m", "intact_tracking.cli.adaptation_eval", "--checkpoint", str(models[train_physics]),
                               "--physics", test_physics, "--seed", str(seed), "--output", str(path)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                    env.pop("PYTHONPATH", None)
                    print(json.dumps(dict(event="evaluation_started", gpu=gpu, label=label)), flush=True)
                    with path.with_suffix(".log").open("w") as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    print(json.dumps(dict(event="evaluation_finished", label=label)), flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            workers = [pool.submit(evaluate_training_branch, "nominal", 1), pool.submit(evaluate_training_branch, "dr", 3)]
            for worker in workers:
                worker.result()
        report = {}
        for physics in ("nominal", "dr"):
            references = [output / f"train{training_seed}_nominal_test{physics}_seed{seed}.json" for seed in (92001, 92002, 92003)]
            candidates = [output / f"train{training_seed}_dr_test{physics}_seed{seed}.json" for seed in (92001, 92002, 92003)]
            report[physics] = dict(statistics=compare(references, candidates), paired_failures=paired_failure_audit(references, candidates))
        (output / f"training_seed{training_seed}_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(event="paired_comparison_complete", training_seed=training_seed)), flush=True)


if __name__ == "__main__":
    main()
