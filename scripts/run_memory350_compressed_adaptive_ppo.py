"""Train matched compressed PPO from scratch, with immediate adaptive sampling and EE termination."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import time

from run_limb_context_experiment import ROOT, process_environment
from resume_memory350_weak_pairs import child_start, sha256
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json
from run_memory350_compressed_ppo import ASSIGNMENTS, command_for, update_comparison


def scratch_command(root, fusion, context):
    command = command_for(root, fusion, context)
    command[command.index("--training-terminations") + 1] = "original"
    command[command.index("--adaptive-after-update") + 1] = "0"
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Training must stay inside the project")
    lock = (root / ".ppo_launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = read_json(root / "PPO_READY.json")
    plan = read_json(root / "ppo_launch_contract.json")
    if not ready or not ready["passed"] or not plan:
        raise ValueError("The paired PPO branch must pass preflight")
    for name, expected in {**ready["source_sha256"], **ready["artifact_sha256"]}.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Validated artifact changed: {name}")
    if (root / "ppo").exists():
        raise FileExistsError("Formal PPO already exists; do not restart from scratch")
    for fusion in ASSIGNMENTS:
        expected = scratch_command(root, fusion, plan["selected_context"]["checkpoint"])
        if plan["formal_commands"][fusion] != expected:
            raise ValueError("Launch command differs from the matched scratch protocol")
    cards = gpu_status(list(range(8)))
    if any(card["processes"] or card["free_mib"] < 60000 for card in cards):
        raise ValueError("Wait for the previous paired PPO and its evaluations to release all GPUs")
    write_json(root / "gpu_before_launch.json", cards)
    (root / "ppo").mkdir()
    children = {}
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    def stop_children():
        for child, _ in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for fusion, gpus in ASSIGNMENTS.items():
            output = root / "ppo" / f"{fusion}_121"
            env = process_environment()
            env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                       WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
                       WANDB_RESUME="allow",
                       WANDB_RUN_ID="m350compressed-" + hashlib.sha256(str(output).encode()).hexdigest()[:12])
            child, record = child_start(plan["formal_commands"][fusion], env,
                                        root / "ppo" / f"{fusion}_initial.log")
            record.update(fusion=fusion, phase="scratch_adaptive", maximum_updates=None)
            children[fusion] = child, record
            write_json(root / "ppo" / f"{fusion}_initial_process.json", record)
        while True:
            if stopping or (root / "STOP").exists() or (root / "STOP_PPO").exists():
                stop_children()
                write_json(root / "state.json", {"status": "ppo_stop_requested", "heartbeat": time.time()})
                return
            for fusion, (child, record) in children.items():
                if child.poll() is not None:
                    record.update(exit_code=child.returncode, finished_at=time.time())
                    write_json(root / "ppo" / f"{fusion}_initial_result.json", record)
                    raise RuntimeError(f"{fusion} PPO stopped unexpectedly; inspect completion and log")
            update_comparison(root)
            write_json(root / "state.json", {"status": "paired_ppo_training", "maximum_updates": None,
                "launcher_pid": os.getpid(), "heartbeat": time.time(),
                "training_terminations": "original", "adaptive_after_update": 0,
                "initialization": "scratch", "selected_context": plan["selected_context"],
                "arms": {fusion: {"job": record,
                    "progress": read_json(root / "ppo" / f"{fusion}_121/progress.json")}
                    for fusion, (_, record) in children.items()}})
            time.sleep(10)
    except BaseException as error:
        stop_children()
        write_json(root / "ppo_launcher_error.json", {"error": repr(error), "unix_time": time.time()})
        raise


if __name__ == "__main__":
    main()
