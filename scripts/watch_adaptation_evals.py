"""Evaluate stable new checkpoints while their explicitly identified trainers run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def trainer_alive(pid, run_name):
    path = Path(f"/proc/{pid}/cmdline")
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    recognized_trainer = any(
        module in command
        for module in (
            "intact_tracking.cli.adaptation_train",
            "intact_tracking.cli.adaptation_distill",
        )
    )
    return recognized_trainer and run_name in command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="RUN_NAME:PHYSICAL_GPU:TRAINER_PID"
    )
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    jobs = []
    for entry in args.run:
        name, gpu, pid = entry.split(":")
        if int(gpu) not in range(4):
            parser.error("Only physical GPUs 0–3 are authorized")
        run_dir = ROOT / "runs" / "adaptation_goal" / name
        metadata = json.loads((run_dir / "run_config.json").read_text())
        jobs.append(
            dict(
                name=name,
                gpu=gpu,
                pid=int(pid),
                path=run_dir,
                evaluation_path=args.evaluation_root.resolve() / name,
                physics=metadata["physics"]["physics"],
                process=None,
                log=None,
                attempted=set(),
                current=None,
            )
        )
    failed = []
    while True:
        pending = False
        for job in jobs:
            if job["process"] is not None:
                code = job["process"].poll()
                if code is None:
                    pending = True
                    continue
                job["log"].close()
                print(
                    json.dumps(
                        {
                            "event": "evaluation_finished",
                            "run": job["name"],
                            "checkpoint": job["current"],
                            "exit_code": code,
                        }
                    ),
                    flush=True,
                )
                if code:
                    failed.append((job["name"], job["current"], code))
                job["process"] = None
            paths = [
                path for path in job["path"].glob("checkpoint_*.pt")
                if path.stem == "checkpoint_final"
                or path.stem.removeprefix("checkpoint_").isdigit()
            ]
            paths.sort(
                key=lambda p: 10**12 if p.stem == "checkpoint_final" else int(p.stem.split("_")[-1])
            )
            for path in paths:
                label = path.stem.removeprefix("checkpoint_")
                if label != "final" and (int(label) == 0 or int(label) % args.interval):
                    continue
                output = job["evaluation_path"] / f"eval_{label}_seed10001.json"
                if output.exists() or str(path) in job["attempted"]:
                    continue
                pending = True
                if time.time() - path.stat().st_mtime < 15:
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                log = output.with_suffix(".log").open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"], PYTHONUNBUFFERED="1")
                env.pop("PYTHONPATH", None)
                command = [
                    sys.executable,
                    "-m",
                    "intact_tracking.cli.adaptation_eval",
                    "--physics",
                    job["physics"],
                    "--checkpoint",
                    str(path),
                    "--output",
                    str(output),
                ]
                job["process"] = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                job["log"], job["current"] = log, label
                job["attempted"].add(str(path))
                print(
                    json.dumps(
                        {
                            "event": "evaluation_started",
                            "run": job["name"],
                            "checkpoint": label,
                            "pid": job["process"].pid,
                        }
                    ),
                    flush=True,
                )
                break
            pending |= trainer_alive(job["pid"], job["name"])
        if not pending:
            break
        time.sleep(15)
    print(
        json.dumps(
            {
                "event": "watch_complete",
                "failed_evaluations": failed,
                "trainers_without_final_checkpoint": [
                    j["name"] for j in jobs if not (j["path"] / "checkpoint_final.pt").exists()
                ],
            }
        ),
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
