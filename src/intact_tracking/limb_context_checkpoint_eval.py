"""Evaluate exact PPO checkpoints at the two fixed payload endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluation_due(completed_updates, interval):
    return interval > 0 and completed_updates > 0 and completed_updates % interval == 0


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol["version"] != "limb_context_periodic_endpoints_v1":
        raise ValueError("Unknown periodic endpoint protocol")
    if protocol["interval_updates"] != 100 or protocol["endpoints"] != {
        "all_0": [0, 0, 0, 0], "all_4": [4, 4, 4, 4]
    }:
        raise ValueError("Periodic evaluation requires both endpoints every 100 completed updates")
    manifest = Path(protocol["motion_manifest"])
    files = manifest.read_text().splitlines()
    if (digest(manifest) != protocol["manifest_sha256"] or len(files) != protocol["motions"]
            or len(set(files)) != len(files) or not 0 < len(files) <= 4096 or protocol["steps"] <= 0):
        raise ValueError("Periodic evaluation manifest changed")
    return protocol, files


def evaluation_environment(gpu, inherited=None):
    # A single-GPU evaluator must not inherit the training rank's motion shard
    # or rendezvous. Its cache/temp directories stay within the project.
    env = dict(os.environ if inherited is None else inherited)
    for key in tuple(env):
        if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "_WANDB_SERVICE", "WANDB_SERVICE"} or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID="0", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MUJOCO_GL="egl")
    return env


def validate_result(row, protocol, files, checkpoint_sha, update, case):
    if (row["checkpoint_sha256"] != checkpoint_sha or row["completed_training_updates"] != update
            or row["seed"] != protocol["seed"] or row["max_steps"] != protocol["steps"]
            or row["motion_files"] != files or row["episodes"] != len(files)
            or row["latent_intervention"] != "correct"
            or row["arguments"]["fixed_masses"] != protocol["endpoints"][case]
            or any(masses != protocol["endpoints"][case] for masses in row["actual_limb_masses_kg"])):
        raise ValueError(f"Checkpoint/endpoint/fixed-start contract mismatch in {case}")
    if not (row["reference_timeline_audited"] and row["partial_reset_survivor_state_audited"]
            and row["partial_reset_survivor_history_audited"]):
        raise ValueError("Endpoint evaluation lacks the survivor-state audit")


def audit_saved_evaluations(directory, completed_updates, protocol_path, enabled_after_update):
    directory = Path(directory)
    protocol, files = load_protocol(protocol_path)
    protocol_sha = digest(protocol_path)
    history = directory / "endpoint_eval_metrics.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
    by_update = {row["completed_updates"]: row for row in rows}
    if len(by_update) != len(rows):
        raise ValueError("Duplicate endpoint checkpoint updates")
    interval = protocol["interval_updates"]
    first = (enabled_after_update // interval + 1) * interval
    required = list(range(first, completed_updates + 1, interval))
    missing = sorted(set(required) - set(by_update))
    if missing:
        raise ValueError(f"Missing periodic endpoint tests for completed updates: {missing}")
    for update, row in by_update.items():
        if row["protocol_sha256"] != protocol_sha or not row.get("training_state_preserved"):
            raise ValueError("Endpoint protocol or training-state preservation audit changed")
        checkpoint = directory / f"checkpoint_update_{update:06d}.pt"
        if digest(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("An evaluated checkpoint was overwritten")
        for case in protocol["endpoints"]:
            target = directory / "endpoint_eval" / f"update_{update:06d}" / f"{case}.json"
            validate_result(json.loads(target.read_text()), protocol, files, row["checkpoint_sha256"], update, case)
            if not target.with_suffix(".traces.npz").exists():
                raise ValueError("Missing endpoint per-step traces")
    return {"passed": True, "enabled_after_update": enabled_after_update, "interval_updates": interval,
            "required_updates": required, "evaluated_updates": sorted(by_update), "motions_per_endpoint": len(files)}


def evaluate_checkpoint(checkpoint, directory, protocol_path, completed_updates, gpus):
    checkpoint, directory = Path(checkpoint).resolve(), Path(directory).resolve()
    if not checkpoint.is_relative_to(ROOT) or not directory.is_relative_to(ROOT):
        raise ValueError("Checkpoint/evaluation files must remain inside the project")
    if len(gpus) not in (2, 4) or len(set(gpus)) != len(gpus) or not set(map(int, gpus)).issubset(range(8)):
        raise ValueError("Use the policy's two or four distinct training GPUs from 0–7")
    protocol, files = load_protocol(protocol_path)
    checkpoint_sha = digest(checkpoint)
    output = directory / "endpoint_eval" / f"update_{completed_updates:06d}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    children, handles, rows = {}, [], {}
    try:
        for (case, masses), gpu in zip(protocol["endpoints"].items(), gpus[:2], strict=True):
            target = output / f"{case}.json"
            if target.exists():
                rows[case] = json.loads(target.read_text())
                validate_result(rows[case], protocol, files, checkpoint_sha, completed_updates, case)
                if not target.with_suffix(".traces.npz").exists():
                    raise ValueError("Endpoint result is missing per-step traces")
                continue
            command = [PYTHON, "-u", "-m", "intact_tracking.cli.limb_context_eval",
                       "--checkpoint", str(checkpoint), "--motion-manifest", protocol["motion_manifest"],
                       "--motion-path", protocol["motion_path"], "--output", str(target),
                       "--seed", str(protocol["seed"]), "--steps", str(protocol["steps"]),
                       "--fixed-masses", *map(str, masses)]
            handle = (output / f"{case}.log").open("a")
            handles.append(handle)
            child = subprocess.Popen(command, cwd=ROOT, env=evaluation_environment(gpu),
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children[case] = child
        write_json(output / "processes.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                   "completed_updates": completed_updates, "started_at": time.time(),
                   "children": {case: process.pid for case, process in children.items()}})
        while any(child.poll() is None for child in children.values()):
            for case, child in children.items():
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"Endpoint {case} failed; see {output / (case + '.log')}")
            if time.monotonic() - started > 1800:
                raise TimeoutError(f"Endpoint evaluation exceeded 30 minutes: {output}")
            time.sleep(1)
        for case, child in children.items():
            if child.returncode != 0:
                raise RuntimeError(f"Endpoint {case} failed; see {output / (case + '.log')}")
            rows[case] = json.loads((output / f"{case}.json").read_text())
            validate_result(rows[case], protocol, files, checkpoint_sha, completed_updates, case)
        for key in ("motion_files", "motion_ids", "start_frames", "horizons", "metric_names", "reward_contract"):
            if rows["all_0"][key] != rows["all_4"][key]:
                raise ValueError(f"Endpoint cases differ in fixed evaluation field {key}")
        metrics = {"checkpoint_update": completed_updates, "motions": len(files)}
        for case, row in rows.items():
            metrics[case] = {**row["mean"], **{key: row[key] for key in (
                "failure_rate", "coverage_fraction", "mean_episode_return", "residual_saturation_fraction")}}
        result = {"completed_updates": completed_updates, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": checkpoint_sha, "protocol_sha256": digest(protocol_path),
                  "metrics": metrics, "unix_time": time.time(), "wall_seconds": time.monotonic() - started}
        write_json(output / "summary.json", result)
        print(json.dumps({"event": "endpoint_evaluation_complete", **result}, allow_nan=False), flush=True)
        return result
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


class PeriodicEndpointEvaluator:
    def __init__(self, directory, protocol_path, distributed):
        self.directory, self.protocol_path = Path(directory), str(protocol_path)
        self.distributed = distributed
        self.protocol, _ = load_protocol(protocol_path)
        self.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if distributed.world_size not in (2, 4) or len(self.gpus) != distributed.world_size:
            raise ValueError("Periodic endpoints require the policy's two or four training GPUs")
        import datetime
        import torch
        # Long evaluation waits must not run an idle NCCL kernel on rank 1's
        # GPU, where an independent endpoint simulator is doing useful work.
        self.evaluation_group = torch.distributed.new_group(
            backend="gloo", timeout=datetime.timedelta(minutes=35))

    def __call__(self, runner, force=False):
        import torch
        from intact_tracking.limb_context_distributed import main_process_call, tensor_digest

        update = runner.completed_learning_updates
        if not force and not evaluation_due(update, self.protocol["interval_updates"]):
            return
        preparer = getattr(runner, "checkpoint_state_preparer", None)
        if preparer is not None:
            preparer(runner)
        before_cpu = torch.get_rng_state()
        before_cuda = torch.cuda.get_rng_state(self.distributed.device)
        before_steps = runner.env.unwrapped.common_step_counter
        before_normalization = tensor_digest(runner.alg.critic.obs_normalizer.state_dict().items())

        def execute():
            checkpoint = self.directory / f"checkpoint_update_{update:06d}.pt"
            if not checkpoint.exists():
                runner.save(str(checkpoint))
            return evaluate_checkpoint(checkpoint, self.directory, self.protocol_path, update, self.gpus)

        start = time.monotonic()
        result = main_process_call(self.distributed, execute, process_group=self.evaluation_group)
        preserved = (torch.equal(before_cpu, torch.get_rng_state())
                     and torch.equal(before_cuda, torch.cuda.get_rng_state(self.distributed.device))
                     and before_steps == runner.env.unwrapped.common_step_counter
                     and before_normalization == tensor_digest(runner.alg.critic.obs_normalizer.state_dict().items()))
        if not self.distributed.all_true(preserved):
            raise RuntimeError("Endpoint evaluation changed training RNG, environment steps or normalization")
        result["training_state_preserved"] = True

        def record():
            path = self.directory / "endpoint_eval_metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            existing = [row for row in rows if row["completed_updates"] == update]
            if existing:
                if any(row["checkpoint_sha256"] != result["checkpoint_sha256"]
                       or row["protocol_sha256"] != result["protocol_sha256"] for row in existing):
                    raise ValueError("An endpoint history update belongs to a different checkpoint/protocol")
            else:
                with path.open("a") as handle:
                    handle.write(json.dumps(result, allow_nan=False) + "\n")

        main_process_call(self.distributed, record)
        runner.pending_endpoint_evaluation = {**result["metrics"], "training_state_preserved": True}
        runner.pending_evaluation_seconds = time.monotonic() - start
