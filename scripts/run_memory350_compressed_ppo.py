"""Select the u15000 encoder by fixed cross-motion DR readout, then run paired PPO without a cap."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from run_limb_context_experiment import ROOT, PYTHON, DATASET, process_environment
from resume_memory350_weak_pairs import child_start, sha256
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json

ASSIGNMENTS = {"baseline": [0, 1, 2, 3], "concat": [4, 5, 6, 7]}


def select_context(root):
    summary = read_json(root / "comparison_015000/summary.json")
    if not summary or not summary["complete"]:
        raise ValueError("Both completed u15000 endpoints must be evaluated first")
    if summary["update"] != 15000 or summary["reference_update"] != 15000:
        raise ValueError("Selection is restricted to u15000 endpoints")
    main = summary["readout"]["memory_training"]["disjoint"]
    common = summary["readout"]["common"]["disjoint"]
    if (main["worlds"], main["test_samples"], common["worlds"], common["test_samples"]) != (292, 2761, 409, 3152):
        raise ValueError("The fixed classification protocol changed")
    scores = {name: {"dr_payload_top1": main["models"][name]["top1_accuracy"],
                     "ordinary_dr_top1": common["models"][name]["top1_accuracy"],
                     "dr_payload_top5": main["models"][name]["top5_accuracy"]}
              for name in ("baseline", "memory350")}
    winner = max(scores, key=lambda name: tuple(scores[name].values()))
    chosen = summary["checkpoints"][winner]
    if sha256(Path(chosen["path"])) != chosen["sha256"]:
        raise ValueError("The selected checkpoint differs from the evaluated artifact")
    selection = {"checkpoint": chosen["path"], "sha256": chosen["sha256"], "update": 15000,
                 "selected_arm": "response10" if winner == "memory350" else "response5",
                 "scores": scores, "decision_rule": "maximum DR+payload strict cross-motion Top1; ties use ordinary DR Top1, then DR+payload Top5, then response5",
                 "fixed_diagnostic_known_environments_only": True,
                 "comparison_summary": str(root / "comparison_015000/summary.json"),
                 "comparison_sha256": sha256(root / "comparison_015000/summary.json"),
                 "prediction_diagnostic": summary["prediction"],
                 "selected_at": time.time()}
    write_json(root / "context_selection.json", selection)
    return selection


def command_for(root, fusion, checkpoint, *, resume=None, smoke=False, motion=None, iterations=2):
    output = root / ("smoke" if smoke else "ppo") / f"{fusion}_121"
    command = [PYTHON, "-B", "-u", "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=4", "--max-restarts=0", "-m", "intact_tracking.cli.memory350_compressed_policy_train",
               "--fusion", fusion, "--output-dir", str(output), "--training-ranks", "4",
               "--num-envs", "128" if smoke else "8192", "--seed", "121", "--episode-steps", "1000",
               "--motion-sampling", "uniform" if smoke else "adaptive", "--adaptive-after-update", "1000",
               "--training-terminations", "no_ee_body_pos", "--dr-profile", "tracker_dr_plus_limb_payload",
               "--save-interval", "1" if smoke else "100", "--wandb-group", root.name,
               "--wandb-name", root.name + f"-{fusion}-" + ("smoke" if smoke else "ppo")]
    if smoke:
        if not motion:
            raise ValueError("A smoke run needs one fixed motion")
        command += ["--motion-file", motion, "--iterations", str(iterations)]
    else:
        command += ["--motion-path", DATASET, "--until-user-stop",
                    "--endpoint-eval-protocol", str(root / "protocols/periodic.json")]
    if fusion == "concat":
        command += ["--context-checkpoint", checkpoint]
    if resume:
        command += ["--resume", str(resume)]
    return command


def launch(root, fusion, checkpoint, *, resume=None, smoke=False, motion=None, iterations=2):
    output = root / ("smoke" if smoke else "ppo") / f"{fusion}_121"
    output.parent.mkdir(exist_ok=True)
    command = command_for(root, fusion, checkpoint, resume=resume, smoke=smoke, motion=motion, iterations=iterations)
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, ASSIGNMENTS[fusion])),
               WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
               WANDB_RESUME="allow", WANDB_RUN_ID="m350compressed-" + hashlib.sha256(str(output).encode()).hexdigest()[:12])
    phase = "resume" if resume else "initial"
    log = output.parent / f"{fusion}_{phase}.log"
    child, record = child_start(command, env, log)
    record.update(fusion=fusion, phase=phase, maximum_updates=None if not smoke else iterations)
    write_json(output.parent / f"{fusion}_{phase}_process.json", record)
    return child, record


def update_comparison(root):
    rows = {}
    for fusion in ASSIGNMENTS:
        path = root / "ppo" / f"{fusion}_121/endpoint_eval_metrics.jsonl"
        saved = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        rows[fusion] = {row["completed_updates"]: row for row in saved}
    updates = sorted(rows["baseline"].keys() & rows["concat"].keys())
    result = []
    different_precision = []
    for update in updates:
        a, b = rows["baseline"][update], rows["concat"][update]
        if a["protocol_sha256"] != b["protocol_sha256"]:
            raise ValueError("PPO comparison protocols differ")
        if a.get("policy_precision", "tf32") != b.get("policy_precision", "tf32"):
            different_precision.append(update)
            continue
        result.append({"completed_updates": update, "baseline": a["metrics"], "latent": b["metrics"],
                       "policy_precision": a.get("policy_precision", "tf32"),
                       "same_update_comparison": True})
    write_json(root / "ppo_comparison.json", {"matched_updates": [row["completed_updates"] for row in result], "rows": result,
              "excluded_different_precision_updates": different_precision,
              "maximum_updates": None, "updated_at": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    lock = (root / ".ppo_launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = read_json(root / "PPO_READY.json")
    if not ready or not ready["passed"]:
        raise ValueError("The tested paired PPO implementation must be ready")
    for name, digest in ready["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"PPO source changed after validation: {name}")
    if (root / "ppo").exists():
        raise FileExistsError("Formal PPO already exists; do not silently restart from scratch")
    selection = select_context(root)
    if any(card["free_mib"] < 60000 for card in gpu_status(list(range(8)))):
        raise ValueError("Both stage1 groups must release their GPUs before formal PPO")
    checkpoint = selection["checkpoint"]
    plan = {"maximum_updates": None, "training_ranks_per_arm": 4, "num_envs_per_rank": 8192,
            "physical_gpus": ASSIGNMENTS, "selected_context": selection,
            "actor_compression": [1645, 512, 256, 128],
            "critic_compression": [6330, 1024, 512, 256, 128],
            "latent_dim": 64, "head_input_dim": 192, "head_hidden_dims": [256, 128],
            "latent_in_actor_and_critic": True, "baseline_latent_slot": "zero",
            "predictor_executed": False, "context_encoder_frozen": True,
            "actor_and_critic_initialization": "from scratch; identical seed and network shape",
            "formal_commands": {fusion: command_for(root, fusion, checkpoint) for fusion in ASSIGNMENTS},
            "comparison": "same completed PPO updates; fixed cold/warm 512-motion endpoint tests every 100 updates"}
    write_json(root / "ppo_launch_contract.json", plan)
    children = {}
    try:
        for fusion in ASSIGNMENTS:
            children[fusion] = launch(root, fusion, checkpoint)
        while True:
            stopping = (root / "STOP").exists() or (root / "STOP_PPO").exists()
            if stopping:
                for child, _ in children.values():
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                write_json(root / "state.json", {"status": "ppo_stop_requested", "heartbeat": time.time()})
                return
            for fusion, (child, record) in list(children.items()):
                if child.poll() is None:
                    continue
                record.update(exit_code=child.returncode, finished_at=time.time())
                write_json(root / "ppo" / f"{fusion}_{record['phase']}_result.json", record)
                completion = read_json(root / "ppo" / f"{fusion}_121/completion.json")
                if (child.returncode == 0 and completion and completion.get("planned_sampling_transition")
                        and completion["completed_updates"] == 1000):
                    resume = root / "ppo" / f"{fusion}_121/checkpoint_update_001000.pt"
                    children[fusion] = launch(root, fusion, checkpoint, resume=resume)
                else:
                    raise RuntimeError(f"{fusion} PPO stopped unexpectedly; inspect completion and log")
            update_comparison(root)
            write_json(root / "state.json", {"status": "paired_ppo_training", "maximum_updates": None,
                       "launcher_pid": os.getpid(), "heartbeat": time.time(),
                       "selected_context": selection,
                       "arms": {fusion: {"job": record, "progress": read_json(root / "ppo" / f"{fusion}_121/progress.json")}
                                for fusion, (_, record) in children.items()}})
            time.sleep(20)
    except BaseException as error:
        for child, _ in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        write_json(root / "ppo_launcher_error.json", {"error": repr(error), "unix_time": time.time()})
        raise


if __name__ == "__main__":
    main()
