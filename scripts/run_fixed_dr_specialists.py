"""Eight independent single-GPU specialists, each paired with the same saved A."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from run_limb_context_experiment import ROOT, PYTHON, DATASET, process_environment
from resume_memory350_weak_pairs import child_start
from run_memory350_scale_nominal_stage1 import gpu_status, read_json, write_json


def training_command(root, profile, *, motion=None, iterations=None, eval_protocol=None, resume=None):
    name = f"B{profile['id']:02d}_{profile['name']}"
    output = root / ("smoke" if motion else "ppo") / name
    command = [PYTHON, "-B", "-u", "-m", "intact_tracking.cli.fixed_dr_specialist_train",
               "--fusion", "baseline", "--output-dir", str(output), "--training-ranks", "1",
               "--num-envs", "128" if motion else "8192", "--seed", "121",
               "--episode-steps", "1000", "--motion-sampling", "adaptive", "--adaptive-after-update", "0",
               "--training-terminations", "original", "--dr-profile", "tracker_dr_plus_limb_payload",
               "--policy-precision", "fp32", "--dr-bank", str(root / "protocols/dr_bank.json"),
               "--specialist-id", str(profile["id"]), "--save-interval", "1" if motion else "100",
               "--router-bootstrap-steps", "500", "--wandb-group", root.name,
               "--wandb-name", root.name + "-" + name + ("-smoke" if motion else "")]
    command += ["--motion-file", motion] if motion else ["--motion-path", DATASET]
    command += ["--iterations", str(iterations)] if iterations is not None else ["--until-user-stop"]
    if eval_protocol or not motion:
        command += ["--specialist-eval-protocol", str(eval_protocol or root / "protocols/evaluation.json")]
    if resume:
        command += ["--resume", str(resume)]
    return command


def job_environment(gpu, root, profile_id):
    from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment
    env = evaluation_environment(gpu, process_environment())
    env.update(WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
               WANDB_RESUME="allow", WANDB_RUN_ID="dr-specialist-" + hashlib.sha256(
                   f"{root}/{profile_id}".encode()).hexdigest()[:12])
    return env


def last_record(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 131072))
        lines = stream.read().split(b"\n")[:-1]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def refresh(root, jobs, protocol, stopping=False):
    arms, histories, issues = {}, {}, []
    for profile_id, job in jobs.items():
        directory = Path(job["output"])
        metrics = last_record(directory / "metrics.jsonl")
        evaluation = last_record(directory / "specialist_eval_metrics.jsonl")
        config = read_json(directory / "run_config.json")
        arm = {key: value for key, value in job.items() if key != "child"}
        arm.update(progress=read_json(directory / "progress.json"),
                   latest_evaluation_update=evaluation["completed_updates"] if evaluation else None,
                   evaluation_status=read_json(directory / "evaluation_status.json"),
                   wandb=read_json(directory / "wandb_run.json"))
        if metrics:
            arm.update(global_transitions=metrics["global_transitions"],
                       seconds_per_update=metrics["collect_seconds"] + metrics["learn_seconds"],
                       metric_time=metrics["unix_time"])
            if not all(math.isfinite(float(value)) for value in metrics["loss"].values()):
                issues.append(f"B{profile_id:02d}: nonfinite training metrics")
        if config:
            audit = config["input_audit"]
            if (config["distributed"]["world_size"] != 1 or audit["latent_dimensions"] != 0
                    or audit["actor_experts"] != 1 or audit["critic_experts"] != 1
                    or audit["actor_critic_parameters_shared"]):
                issues.append(f"B{profile_id:02d}: independent MLP invariant failed")
            actual = config["physics"]["runtime_audits_by_rank"][0]["physics"]["fixed_dr"]
            if actual["id"] != profile_id or actual["replicas"] != 8192:
                issues.append(f"B{profile_id:02d}: fixed DR replication invariant failed")
        if job["phase"] == "failed":
            issues.append(f"B{profile_id:02d}: {job.get('error', 'process failed')}")
        arms[str(profile_id)] = arm
        path = directory / "specialist_eval_metrics.jsonl"
        # An evaluator can still be writing the final, large comparison row.
        history = [json.loads(line) for line in path.read_text().split("\n")[:-1] if line] if path.exists() else []
        histories[str(profile_id)] = {row["completed_updates"]: row for row in history}
    common = sorted(set.intersection(*(set(rows) for rows in histories.values()))) if len(histories) == 8 else []
    write_json(root / "comparison.json", {"updated_at": time.time(), "reference_checkpoint": protocol["reference_checkpoint"],
               "matched_specialist_updates": common,
               "rows": [{"specialist_update": update, "by_dr": {key: rows[update] for key, rows in histories.items()}}
                        for update in common]})
    status = "stop_requested" if stopping else (
        "specialists_need_attention" if issues else (
            "eight_specialists_training" if all(job["phase"] == "training" for job in jobs.values()) else "specialists_starting"))
    state = {"status": status, "launcher_pid": os.getpid(), "heartbeat": time.time(),
             "maximum_updates": None, "reference_checkpoint": protocol["reference_checkpoint"],
             "reference_update": protocol["reference_update"], "evaluation_interval": protocol["interval_updates"],
             "arms": arms, "issues": issues}
    write_json(root / "state.json", state)
    health = root / "health"
    health.mkdir(exist_ok=True)
    write_json(health / "latest.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Run must remain in the workspace")
    lock = (root / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready, plan = read_json(root / "READY.json"), read_json(root / "launch_contract.json")
    if not ready or not ready["passed"]:
        raise ValueError("Finish the concrete physics/PPO/evaluation preflight before formal training")
    from intact_tracking.fixed_dr_profiles import file_sha256
    from intact_tracking.fixed_dr_specialists import load_protocol, evaluation_command, validate_evaluation
    for path, expected in ready["pinned_files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Validated file changed: {path}")
    protocol = load_protocol(root / "protocols/evaluation.json")
    bank = read_json(root / "protocols/dr_bank.json")
    if (root / "ppo").exists():
        raise FileExistsError("Formal specialist training already exists; do not restart from scratch")
    for profile in bank["profiles"]:
        if plan["commands"][str(profile["id"])] != training_command(root, profile):
            raise ValueError("Formal training command differs from the saved contract")
    cards = gpu_status(list(range(8)))
    if any(card["processes"] or card["free_mib"] < 60000 for card in cards):
        raise ValueError("The previous jobs must finish saving and release all eight GPUs")
    write_json(root / "gpu_before_launch.json", cards)
    (root / "ppo").mkdir()
    jobs, stopping = {}, False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        for job in jobs.values():
            if job["child"].poll() is None:
                # Direct Python workers, no torchrun or cross-specialist collectives.
                job["child"].terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for profile in bank["profiles"]:
        index = profile["id"]
        result = Path(protocol["reference_evaluations"]) / f"dr_{index:02d}.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        env = job_environment(index, root, index)
        command = evaluation_command(protocol["reference_checkpoint"], result, protocol, index)
        child, record = child_start(command, env, result.with_suffix(".log"))
        jobs[index] = {"child": child, "phase": "evaluating_A", "gpu": index,
                       "profile": profile["name"], "process": record, "reference_result": str(result),
                       "output": str(root / "ppo" / f"B{index:02d}_{profile['name']}")}
    pointer = ROOT / ".runtime/limb_context/current_ppo.json"
    previous = read_json(pointer)
    write_json(root / "previous_current_ppo.json", previous)
    while True:
        if not stopping and (root / "STOP").exists():
            stop(None, None)
        for index, job in jobs.items():
            code = job["child"].poll()
            if code is None or job["phase"] in ("failed", "stopped"):
                continue
            if stopping:
                job.update(phase="stopped", returncode=code)
            elif job["phase"] == "evaluating_A" and code == 0:
                try:
                    validate_evaluation(read_json(Path(job["reference_result"])), protocol, index, protocol["reference_checkpoint"])
                    child, record = child_start(plan["commands"][str(index)], job_environment(index, root, index),
                                               root / "ppo" / f"B{index:02d}.log")
                    job.update(child=child, process=record, phase="training")
                    write_json(root / "ppo" / f"B{index:02d}_process.json", record)
                except Exception as error:
                    job.update(phase="failed", error=repr(error))
            else:
                job.update(phase="failed", returncode=code, error=f"{job['phase']} exited with code {code}")
        state = refresh(root, jobs, protocol, stopping)
        write_json(pointer, {"run_root": str(root), "supervisor_pid": os.getpid(), "status": state["status"],
                   "updated_at": state["heartbeat"], "architecture": "8 independent single-GPU fixed-DR residual MLP specialists; no latent",
                   "reference_A": {"checkpoint": protocol["reference_checkpoint"], "update": protocol["reference_update"]},
                   "gpu_assignments": {str(index): job["profile"] for index, job in jobs.items()},
                   "periodic_updates": {"checkpoint": 100, "paired_A_B_evaluation": protocol["interval_updates"]},
                   "health": str(root / "health/latest.json")})
        if all(job["phase"] in ("stopped", "failed") for job in jobs.values()):
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
