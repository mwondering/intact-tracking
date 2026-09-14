"""Supervise both response-window continuations, endpoint selection, then paired endless PPO."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, process_environment
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json
from resume_memory350_weak_pairs import child_start, sha256

ENTRY = "intact_tracking.cli.forward_memory_response_window_train"
PARENT_11000_SHA = "ba45c2fd63ec5589c3a944fb50ca9574bd957d278c0b4d21365cf554c634c07a"


def command_for(root, parent, horizon):
    command = list(read_json(parent / "launch_contract.json")["formal_command"])
    command[command.index("intact_tracking.cli.forward_memory_weak_margin_train")] = ENTRY
    start = 11000 if horizon == 10 else 12000
    options = {"output-dir": str(root / "stage1_8192"), "stop-after-updates": "15000",
               "resume": str(parent / f"stage1_8192/update_{start:06d}.pt"),
               "comparison-reference-dir": str(parent / "stage1_8192"),
               "response-label-horizon": str(horizon), "wandb-group": root.parent.name,
               "wandb-name": root.parent.name + f"-response{horizon}-predictor5"}
    for key, value in options.items():
        flag = "--" + key
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command += [flag, value]
    return command


def prepare_arm(root, parent, horizon):
    import torch
    from intact_tracking.cli.forward_memory_response_window_train import build_parser, reference_contract
    from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_arguments
    root.mkdir(exist_ok=True)
    output, source = root / "stage1_8192", parent / "stage1_8192"
    start = 11000 if horizon == 10 else 12000
    checkpoint = source / f"update_{start:06d}.pt"
    digest = sha256(checkpoint)
    if horizon == 10 and digest != PARENT_11000_SHA:
        raise ValueError("The experimental arm must start at the exact evaluated u11000 checkpoint")
    previous = read_json(source / "run_config.json")
    command = command_for(root, parent, horizon)
    args = build_parser().parse_args(command[command.index(ENTRY) + 1:])
    _validate_arguments(args)
    actual = deepcopy(previous)
    actual["arguments"] = vars(args)
    contract = reference_contract(source, actual)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert state["update"] == start and state["optimizer_steps"] == 4 * start
    assert state["scheduler"]["T_max"] == 32000 and state["scheduler"]["last_epoch"] == 4 * start
    assert state["optimizer"]["param_groups"][0]["lr"] == 1e-5
    assert all(int(value["step"]) == 4 * start for value in state["optimizer"]["state"].values())
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    files = sorted(source.glob("validation*rank_*.pt"))
    assert len(files) == 8
    validation = {}
    for path in files:
        shutil.copy2(path, output / path.name)
        validation[path.name] = sha256(path)
        assert validation[path.name] == sha256(output / path.name)
    shutil.copy2(source / "normalization.json", output / "normalization.json")
    write_json(output / "run_config.json", previous)
    write_json(output / "history.json", [row for row in read_json(source / "history.json") if row["update"] <= start])
    plan = {"passed": True, "from_update": start, "stop_after_updates": 15000,
            "response_label_horizon": horizon, "predictor_horizon": 5,
            "parent_checkpoint": str(checkpoint), "parent_sha256": digest,
            "validation_sha256": validation, "formal_command": command, "contract": contract,
            "physical_gpus": list(range(4)) if horizon == 10 else list(range(4, 8)),
            "cached_trajectories": read_json(parent / "launch_contract.json")["cached_trajectories"],
            "prepared_at": time.time()}
    write_json(root / "launch_contract.json", plan)
    return plan


def environment(root, gpus):
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONUNBUFFERED="1",
               WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
               WANDB_RESUME="allow", WANDB_RUN_ID="m350window-" + hashlib.sha256(str(root).encode()).hexdigest()[:12])
    return env


def start_evaluation(root, candidate, reference, cache, gpu):
    command = [PYTHON, "-B", "-u", str(ROOT / "scripts/evaluate_memory350_response_window.py"),
               "--candidate", str(candidate), "--reference", str(reference), "--cache", cache,
               "--output", str(root), "--comparison-kind", "tuning"]
    env = dict(process_environment(), CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH="/tmp/intact-tsne-deps", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    child, record = child_start(command, env, root.with_suffix(".log"))
    write_json(root.with_suffix(".process.json"), record)
    return child, record, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, parent = args.run_root.resolve(), args.parent_root.resolve()
    root.mkdir(exist_ok=True)
    lock = (root / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "state.json").exists():
        raise RuntimeError("This supervisor already ran; inspect state before resuming it")
    assert sha256(parent / "stage1_8192/update_011000.pt") == PARENT_11000_SHA
    review = read_json(parent / "comparison_011000/review_verification.json")
    assert review["passed"] and review["checkpoint_sha256"] == PARENT_11000_SHA
    sources = sorted((ROOT / "src/intact_tracking").rglob("*.py")) + [
        Path(__file__).resolve(), ROOT / "scripts/evaluate_memory350_response_window.py",
        ROOT / "scripts/evaluate_memory350_weak_pairs.py"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json(root / "source_sha256.json", hashes)
    for source in sources:
        target = root / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plans, children, checked = {}, {}, set()
    active_eval = None
    done_evals = set()
    final_done = False
    try:
        while True:
            if (root / "STOP").exists():
                for child, _ in children.values():
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                if active_eval and active_eval[0].poll() is None:
                    os.killpg(active_eval[0].pid, signal.SIGTERM)
                write_json(root / "state.json", {"status": "stop_requested", "heartbeat": time.time()})
                return
            for horizon in (10, 5):
                if horizon in children:
                    continue
                if horizon == 5:
                    completion = read_json(parent / "stage1_8192/completion.json")
                    previous_state = read_json(parent / "state.json") or {}
                    if not completion or completion["completed_updates"] != 12000:
                        continue
                    if previous_state.get("status") not in ("complete", "complete_with_evaluation_failures"):
                        continue
                gpus = list(range(4)) if horizon == 10 else list(range(4, 8))
                if any(card["free_mib"] < 60000 for card in gpu_status(gpus)):
                    continue
                folder = root / f"response{horizon}"
                plans[horizon] = prepare_arm(folder, parent, horizon)
                if args.prepare_only:
                    print(json.dumps(plans[horizon]), flush=True)
                    return
                child, record = child_start(plans[horizon]["formal_command"], environment(folder, gpus), folder / "training.log")
                write_json(folder / "training_process.json", record)
                children[horizon] = (child, record)
                print(json.dumps({"event": "arm_started", "horizon": horizon, **record}), flush=True)
            for horizon, (child, record) in children.items():
                if child.poll() is not None and horizon not in checked:
                    record.update(exit_code=child.returncode, finished_at=time.time())
                    write_json(root / f"response{horizon}/training_result.json", record)
                    completion = read_json(root / f"response{horizon}/stage1_8192/completion.json")
                    if child.returncode or not completion or completion["completed_updates"] != 15000 or not completion["hit_cap"]:
                        raise RuntimeError(f"response{horizon} did not finish u15000 cleanly")
                    checked.add(horizon)
            if active_eval and active_eval[0].poll() is not None:
                child, record, folder = active_eval
                summary = read_json(folder / "summary.json")
                record.update(exit_code=child.returncode, finished_at=time.time(),
                              complete=child.returncode == 0 and bool(summary and summary.get("complete")))
                write_json(folder.with_suffix(".result.json"), record)
                if not record["complete"]:
                    raise RuntimeError(f"Evaluation failed: {folder}")
                done_evals.add(str(folder))
                if folder.name == "comparison_015000":
                    final_done = True
                active_eval = None
            if active_eval is None:
                for horizon in (10, 5):
                    if horizon not in plans:
                        continue
                    folder = root / f"response{horizon}"
                    # Include control u12000 once, although its continuation starts there.
                    for update in range(12000, 15001, 1000):
                        output = folder / f"evaluation_{update:06d}"
                        if str(output) in done_evals:
                            continue
                        checkpoint = (parent if horizon == 5 and update == 12000 else folder) / f"stage1_8192/update_{update:06d}.pt"
                        if not checkpoint.exists():
                            continue
                        gpu = 3 if horizon == 10 else 7
                        if gpu_status([gpu])[0]["free_mib"] < 12000:
                            continue
                        active_eval = start_evaluation(output, checkpoint, parent / "stage1_8192/update_011000.pt",
                                                       plans[horizon]["cached_trajectories"], gpu)
                        break
                    if active_eval:
                        break
            if checked == {5, 10} and active_eval is None and len(done_evals) >= 8 and not final_done:
                active_eval = start_evaluation(root / "comparison_015000", root / "response10/stage1_8192/update_015000.pt",
                    root / "response5/stage1_8192/update_015000.pt", plans[10]["cached_trajectories"], 7)
            state = {"status": "stage1_training" if checked != {5, 10} else "stage1_evaluating",
                     "launcher_pid": os.getpid(), "heartbeat": time.time(), "parent_root": str(parent),
                     "arms": {str(h): {"job": children[h][1], "progress": read_json(root / f"response{h}/stage1_8192/progress.json")}
                              for h in children}, "completed_training_arms": sorted(checked),
                     "active_evaluation": str(active_eval[2]) if active_eval else None,
                     "completed_evaluations": sorted(done_evals)}
            if final_done:
                state["status"] = "awaiting_verified_ppo_launcher"
                if (root / "PPO_READY.json").exists():
                    ready = read_json(root / "PPO_READY.json")
                    if not ready.get("passed"):
                        raise RuntimeError("PPO readiness verification did not pass")
                    state["status"] = "starting_paired_ppo"
                    write_json(root / "state.json", state)
                    command = [PYTHON, "-B", "-u", str(ROOT / "scripts/run_memory350_compressed_ppo.py"), "--run-root", str(root)]
                    result = subprocess.run(command, cwd=ROOT, env=process_environment(), check=False)
                    if result.returncode:
                        raise RuntimeError(f"Paired PPO supervisor exited with {result.returncode}")
                    return
            write_json(root / "state.json", state)
            time.sleep(10)
    except BaseException as error:
        write_json(root / "launcher_error.json", {"error": repr(error), "unix_time": time.time()})
        for child, _ in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        if active_eval and active_eval[0].poll() is None:
            os.killpg(active_eval[0].pid, signal.SIGTERM)
        raise


if __name__ == "__main__":
    main()
