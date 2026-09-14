"""Periodic evaluation of the new residual protocol on a separate simulator."""

import json
import os
from pathlib import Path
import subprocess
import time

import torch

from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.residual_uniform_protocol import EVAL_VERSION, physics_contract


def evaluate(runner, checkpoint):
    args = runner.residual_metadata["arguments"]
    if not args.get("eval_motion_manifest") or runner.completed_learning_updates % args["eval_interval"]:
        return
    if runner.is_distributed:
        raise ValueError("This monitor is for independent one-GPU specialists")
    root = Path(__file__).resolve().parents[2]
    directory = Path(runner.logger.log_dir)
    output = directory / "evaluation" / f"update_{runner.completed_learning_updates:06d}.json"
    output.parent.mkdir(exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise ValueError("Specialist evaluation must use its own single GPU")
    before = (torch.get_rng_state(), torch.cuda.get_rng_state(runner.device),
              runner.env.unwrapped.common_step_counter,
              tensor_digest(runner.alg.critic.obs_normalizer.state_dict().items()))
    started = time.monotonic()
    write_json(directory / "evaluation_status.json", {"status": "evaluating", "completed_updates": runner.completed_learning_updates})
    command = [str(root / ".venv/bin/python"), "-B", "-u", "-m", "intact_tracking.cli.residual_uniform_eval",
               "--checkpoint", str(checkpoint), "--output", str(output),
               "--motion-manifest", args["eval_motion_manifest"], "--steps", str(args["eval_steps"]),
               "--seed", str(args["eval_seed"]), "--policy-precision", "fp32"]
    if not output.exists():
        with output.with_suffix(".log").open("a") as log:
            child = subprocess.Popen(command, cwd=root, env=evaluation_environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            write_json(output.with_suffix(".process.json"), {"pid": child.pid, "gpu": gpu, "checkpoint": str(checkpoint)})
            try:
                code = child.wait(timeout=3600)
                if code:
                    raise RuntimeError(f"Residual evaluation failed: {output.with_suffix('.log')}")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
    result = json.loads(output.read_text())
    expected_files = Path(args["eval_motion_manifest"]).read_text().splitlines()
    if (result["protocol"] != EVAL_VERSION or result["completed_training_updates"] != runner.completed_learning_updates
            or result["checkpoint_sha256"] != file_sha256(checkpoint)
            or result["motion_files"] != expected_files or result["seed"] != args["eval_seed"]
            or result["max_steps"] != args["eval_steps"] or result["residual_output_bounded"]
            or result["physics"]["residual_physics_contract"] != physics_contract()):
        raise RuntimeError("Periodic evaluation does not match its checkpoint and protocol")
    expected_fixed = runner.residual_metadata["physics"].get("fixed_dr")
    if expected_fixed:
        actual = result["physics"]["runtime_audit"]["fixed_dr"]
        for key in ("id", "bank_sha256", "physics_fingerprint"):
            if actual[key] != expected_fixed[key]:
                raise RuntimeError(f"Evaluation changed fixed DR {key}")
    preserved = (torch.equal(before[0], torch.get_rng_state())
                 and torch.equal(before[1], torch.cuda.get_rng_state(runner.device))
                 and before[2] == runner.env.unwrapped.common_step_counter
                 and before[3] == tensor_digest(runner.alg.critic.obs_normalizer.state_dict().items()))
    if not preserved:
        raise RuntimeError("Evaluation changed training state")
    row = {"completed_updates": runner.completed_learning_updates, "unix_time": time.time(),
           "checkpoint_sha256": result["checkpoint_sha256"], "training_state_preserved": True,
           "evaluation": str(output), "mean": result["mean"], "failure_rate": result["failure_rate"],
           "coverage_fraction": result["coverage_fraction"], "mean_episode_return": result["mean_episode_return"]}
    with (directory / "specialist_eval_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    runner.pending_endpoint_evaluation = {"specialist": {**result["mean"], "failure_rate": result["failure_rate"],
                                                         "coverage_fraction": result["coverage_fraction"]}}
    runner.pending_evaluation_seconds = time.monotonic() - started
    write_json(directory / "evaluation_status.json", {"status": "complete", **row,
                                                       "wall_seconds": runner.pending_evaluation_seconds})
