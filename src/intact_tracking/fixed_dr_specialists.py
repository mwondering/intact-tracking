"""Single-GPU specialist checkpoints and paired evaluation against a frozen A."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from intact_tracking.fixed_dr_profiles import audit_fixed_dr, file_sha256, load_profile
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.memory350_moe_training import OnlineMoERunner
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.memory350_policy_results import compare

VERSION = "complete_fixed_dr_independent_residual_mlp_v1"
EVAL_VERSION = "complete_fixed_dr_paired_tracking_v1"
PERIODIC_VERSION = "complete_fixed_dr_specialist_monitor_v1"
ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    files = Path(protocol["motion_manifest"]).read_text().splitlines()
    if (protocol["version"] != PERIODIC_VERSION or protocol["interval_updates"] <= 0
            or len(files) != protocol["motions"] or len(set(files)) != len(files)
            or not 0 < len(files) <= 4096 or protocol["steps"] <= 0):
        raise ValueError("Invalid specialist evaluation protocol")
    for path_key, hash_key in (("motion_manifest", "manifest_sha256"), ("dr_bank", "dr_bank_sha256"),
                               ("reference_checkpoint", "reference_sha256")):
        if file_sha256(protocol[path_key]) != protocol[hash_key]:
            raise ValueError(f"Pinned evaluation artifact changed: {path_key}")
    return protocol


def validate_evaluation(row, protocol, profile_id, checkpoint):
    selected, _, bank_sha = load_profile(protocol["dr_bank"], profile_id)
    actual = row["physics"]["runtime_audit"]["fixed_dr"]
    if (row["protocol"] != EVAL_VERSION or row["seed"] != protocol["seed"]
            or row["max_steps"] != protocol["steps"] or row["episodes"] != protocol["motions"]
            or row["motion_files"] != Path(protocol["motion_manifest"]).read_text().splitlines()
            or row["checkpoint_sha256"] != file_sha256(checkpoint)
            or row["fusion"] != "baseline" or row["policy_precision"] != "fp32"
            or row["memory_start"] != "cold" or row["warmup"]["steps"] != 0
            or actual["id"] != profile_id or actual["bank_sha256"] != bank_sha
            or actual["physics_fingerprint"] != selected["physics_fingerprint"]):
        raise ValueError("Fixed DR evaluation does not match its saved protocol/checkpoint")
    for key in ("reference_timeline_audited", "partial_reset_survivor_state_audited",
                "partial_reset_survivor_history_audited"):
        if not row[key]:
            raise ValueError(f"Missing evaluation invariant: {key}")


def evaluation_command(checkpoint, output, protocol, profile_id):
    return [PYTHON, "-B", "-u", "-m", "intact_tracking.cli.fixed_dr_specialist_eval",
            "--checkpoint", str(checkpoint), "--output", str(output),
            "--motion-manifest", protocol["motion_manifest"], "--motion-path", protocol["motion_path"],
            "--seed", str(protocol["seed"]), "--steps", str(protocol["steps"]),
            "--policy-precision", "fp32", "--dr-bank", protocol["dr_bank"], "--dr-id", str(profile_id)]


def run_evaluation(checkpoint, output, protocol, profile_id):
    output = Path(output)
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(gpus) != 1 or not gpus[0]:
            raise ValueError("The specialist evaluator must use exactly its own GPU")
        with output.with_suffix(".log").open("a") as log:
            child = subprocess.Popen(evaluation_command(checkpoint, output, protocol, profile_id),
                                     cwd=ROOT, env=evaluation_environment(gpus[0]),
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(output.with_suffix(".process.json"), {"pid": child.pid, "started_at": time.time(),
                       "checkpoint": str(checkpoint), "gpu": gpus[0]})
            try:
                code = child.wait(timeout=3600)
                if code:
                    raise RuntimeError(f"Fixed DR evaluation failed ({code}): {output.with_suffix('.log')}")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
    row = json.loads(output.read_text())
    validate_evaluation(row, protocol, profile_id, checkpoint)
    if not output.with_suffix(".traces.npz").is_file():
        raise ValueError("Fixed DR evaluation is missing per-step traces")
    return row


def compare_evaluations(reference_path, candidate_path):
    a, b = (json.loads(Path(path).read_text()) for path in (reference_path, candidate_path))
    af, bf = (row["physics"]["runtime_audit"]["fixed_dr"] for row in (a, b))
    for name in ("id", "bank_sha256", "physics_fingerprint"):
        if af[name] != bf[name]:
            raise ValueError(f"The two evaluations use different complete DR: {name}")
    with np.load(Path(reference_path).with_suffix(".traces.npz")) as ta, np.load(Path(candidate_path).with_suffix(".traces.npz")) as tb:
        result = compare(a, b, ta["body_joint"], tb["body_joint"])
    result.update(reference_checkpoint=a["checkpoint"], reference_update=a["completed_training_updates"],
                  specialist_checkpoint=b["checkpoint"], specialist_update=b["completed_training_updates"],
                  dr_id=af["id"], physics_fingerprint=af["physics_fingerprint"],
                  interpretation="Specialization relative to the existing continuous-DR universal A; not an equal-training-distribution or equal-total-samples ablation")
    return result


class SpecialistCheckpointMonitor:
    def __init__(self, runner):
        self.directory = Path(runner.logger.log_dir)
        self.settings = runner.residual_metadata["arguments"]
        self.profile_id = self.settings["specialist_id"]
        self.protocol_path = self.settings.get("specialist_eval_protocol")
        self.protocol = load_protocol(self.protocol_path) if self.protocol_path else None

    def __call__(self, runner):
        update = runner.completed_learning_updates
        due = self.protocol is not None and update % self.protocol["interval_updates"] == 0
        if update % self.settings["save_interval"] and not due:
            return
        physics = audit_fixed_dr(runner.env.unwrapped, runner.residual_metadata["physics"])
        if physics["fixed_dr"]["physics_fingerprint"] != runner.initial_fixed_fingerprint:
            raise RuntimeError("Static DR changed during PPO or episode resets")
        preparer = getattr(runner, "checkpoint_state_preparer", None)
        if preparer is not None:
            preparer(runner)
        checkpoint = self.directory / f"checkpoint_update_{update:06d}.pt"
        if not checkpoint.exists():
            runner.save(str(checkpoint))
        if not due:
            return
        before_cpu = torch.get_rng_state()
        before_cuda = torch.cuda.get_rng_state(runner.device)
        before_steps = runner.env.unwrapped.common_step_counter
        before_norm = tensor_digest(runner.alg.critic.obs_normalizer.state_dict().items())
        started = time.monotonic()
        status = self.directory / "evaluation_status.json"
        write_json(status, {"status": "evaluating", "completed_updates": update, "started_at": time.time()})
        a_path = Path(self.protocol["reference_evaluations"]) / f"dr_{self.profile_id:02d}.json"
        a = run_evaluation(self.protocol["reference_checkpoint"], a_path, self.protocol, self.profile_id)
        b_path = self.directory / "evaluation" / f"update_{update:06d}.json"
        b = run_evaluation(checkpoint, b_path, self.protocol, self.profile_id)
        if b["completed_training_updates"] != update:
            raise ValueError("Checkpoint update differs from evaluation label")
        comparison = compare_evaluations(a_path, b_path)
        preserved = (torch.equal(before_cpu, torch.get_rng_state())
                     and torch.equal(before_cuda, torch.cuda.get_rng_state(runner.device))
                     and before_steps == runner.env.unwrapped.common_step_counter
                     and before_norm == tensor_digest(runner.alg.critic.obs_normalizer.state_dict().items()))
        if not preserved:
            raise RuntimeError("Evaluation changed the training state")
        result = {"completed_updates": update, "unix_time": time.time(),
                  "dr_id": self.profile_id, "checkpoint_sha256": file_sha256(checkpoint),
                  "protocol_sha256": file_sha256(self.protocol_path), "training_state_preserved": True,
                  "global_transitions": update * runner.env.num_envs * runner.cfg["num_steps_per_env"],
                  "comparison": comparison}
        write_json(b_path.with_suffix(".comparison.json"), result)
        history = self.directory / "specialist_eval_metrics.jsonl"
        prior = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        if update not in {row["completed_updates"] for row in prior}:
            with history.open("a") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
        metrics = {"reference": {**a["mean"], "failure_rate": a["failure_rate"]},
                   "specialist": {**b["mean"], "failure_rate": b["failure_rate"]},
                   "reductions_percent": {key: row["reduction_percent"] for key, row in comparison.items()
                                          if isinstance(row, dict) and "reduction_percent" in row}}
        runner.pending_endpoint_evaluation = metrics
        runner.pending_evaluation_seconds = time.monotonic() - started
        write_json(status, {"status": "complete", "completed_updates": update, "unix_time": time.time(),
                            "wall_seconds": runner.pending_evaluation_seconds, "training_state_preserved": True})


class SpecialistRunner(OnlineMoERunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_distributed or self.alg.actor.residual_mlp.router is not None or self.alg.critic.mlp.router is not None:
            raise ValueError("B must be a single-process MLP without any latent/router")
        audit = audit_fixed_dr(self.env.unwrapped, self.residual_metadata["physics"])
        self.initial_fixed_fingerprint = audit["fixed_dr"]["physics_fingerprint"]

    def learn(self, *args, **kwargs):
        self.checkpoint_evaluator = SpecialistCheckpointMonitor(self)
        super().learn(*args, **kwargs)
        audit = audit_fixed_dr(self.env.unwrapped, self.residual_metadata["physics"])
        if audit["fixed_dr"]["physics_fingerprint"] != self.initial_fixed_fingerprint:
            raise RuntimeError("The specialist changed its static environment")
        write_json(Path(self.logger.log_dir) / "final_fixed_dr_audit.json", audit)
