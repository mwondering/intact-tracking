"""Run the existing online K-means16 MoE for 1000 matched 8x1024 PPO updates."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, process_environment
from run_residual_uniform_shared_comparison import training_command as baseline_command
from run_residual_uniform_specialists import read_optional_json, last_record
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.residual_uniform_protocol import physics_contract


def training_command(root, smoke=False):
    plan = json.loads((root / "plan.json").read_text())
    command = baseline_command(root, smoke=smoke)
    command[command.index("--fusion") + 1] = "concat"  # Existing hard-routing implementation.
    command[command.index("--output-dir") + 1] = str(root / ("smoke" if smoke else "moe"))
    command[command.index("--wandb-name") + 1] = root.name + ("-smoke8" if smoke else "-moe16")
    command += ["--context-checkpoint", plan["selected_context"]["checkpoint"], "--num-experts", "16",
                "--router-center-rate", "0.01", "--router-max-switch-fraction", "0.02"]
    return command


def audit_training(root, smoke=False, final=False):
    import torch
    from omegaconf import OmegaConf

    directory = root / ("smoke" if smoke else "moe")
    current = json.loads((directory / "run_config.json").read_text())
    baseline = json.loads((root / "references/baseline/run_config.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    allowed_args = {"fusion", "context_checkpoint", "output_dir", "wandb_group", "wandb_name"}
    if smoke:
        allowed_args |= {"motion_file", "motion_path", "iterations", "save_interval"}
    for key in set(current["arguments"]) | set(baseline["arguments"]):
        if key not in allowed_args and current["arguments"].get(key) != baseline["arguments"].get(key):
            raise ValueError(f"MoE/baseline arguments differ unexpectedly: {key}")
    for key in ("version", "tracker_sha256", "reward_contract", "reward_changes", "residual_output",
                "training_terminations", "training_starts", "initialization_seeds", "initialization_protocol",
                "policy_precision", "dr_profile", "dr_sampling", "distributed", "motion_sampling"):
        if current[key] != baseline[key]:
            raise ValueError(f"MoE/baseline protocol differs unexpectedly: {key}")
    assert current["fusion"] == "concat" and current["context_sha256"] == plan["selected_context"]["sha256"]
    assert current["encoder_frozen"] and current["context_normalization_frozen"]
    assert not current["predictor_executed_in_ppo"] and not current["extra_current_state_physics_privilege"]
    assert current["residual_output"]["bounded"] is False
    expected_updates = 2 if smoke else 1000
    assert current["maximum_updates"] == expected_updates
    audit = current["input_audit"]
    changing = {"actor_experts", "critic_experts", "latent_dimensions", "latent_use",
                "actor_trainable_parameters", "critic_trainable_parameters", "routing_bootstrap",
                "critic_initial_normalizer_sha256"}
    for key, value in baseline["input_audit"].items():
        if key not in changing and audit[key] != value:
            raise ValueError(f"MoE/baseline shared model input differs: {key}")
    assert audit["actor_experts"] == audit["critic_experts"] == 16
    assert audit["latent_dimensions"] == 64 and audit["latent_use"] == "online K-means hard routing only"
    assert audit["actor_trainable_parameters"] == 2240365 and audit["critic_trainable_parameters"] == 8347536
    assert not audit["actor_critic_parameters_shared"] and not audit["router_has_trainable_parameters"]
    assert audit["critic_tracker_action_matches_actor"]
    assert audit["routing_bootstrap"]["steps"] == 500
    assert audit["routing_bootstrap"]["bootstrap_samples"] == 8 * 1024 * 4
    assert len(audit["routing_bootstrap"]["bootstrap_counts"]) == 16
    assert all(count > 0 for count in audit["routing_bootstrap"]["bootstrap_counts"])
    if not smoke:
        for key in ("manifest_sha256", "loaded_motion_count", "loaded_frames", "root", "loaded_motion_counts_per_rank"):
            assert current["dataset"][key] == baseline["dataset"][key], key
    ranks = current["physics"]["runtime_audits_by_rank"]
    assert len(ranks) == 8 and len({r["physics"]["actual_mass_sha256"] for r in ranks}) == 8
    for rank, reference in zip(ranks, baseline["physics"]["runtime_audits_by_rank"], strict=True):
        physical = rank["physics"]
        assert physical["residual_physics_contract"] == physics_contract()
        assert physical["wrist_force_ranges_nm"] == [[-10., 10.]] * 4
        assert physical["pd_gains_and_tracker_action_scale_verified"]
        assert physical["actual_mass_sha256"] == reference["physics"]["actual_mass_sha256"]
    checkpoint = directory / ("checkpoint_final.pt" if final else "checkpoint_initial.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    reference = torch.load(root / "references/baseline/checkpoint_initial.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg, base_cfg = [OmegaConf.to_container(s["cfg"].agent, resolve=True) for s in (state, reference)]
    for key in ("algorithm", "obs_groups", "num_steps_per_env", "clip_actions", "seed", "check_for_nan", "torch_compile_mode"):
        assert cfg[key] == base_cfg[key], key
    for side in ("actor", "critic"):
        assert cfg[side]["fusion_mode"] == "concat"
        assert {k: v for k, v in cfg[side].items() if k != "fusion_mode"} == {
            k: v for k, v in base_cfg[side].items() if k != "fusion_mode"}, side
    actor, critic = state["actor_state_dict"], state["critic_state_dict"]
    for key, value in actor.items():
        if key.startswith("residual_mlp.router."):
            torch.testing.assert_close(value, critic[key.replace("residual_mlp.router.", "mlp.router.")], atol=0, rtol=0)
    assert bool(actor["residual_mlp.router.initialized"])
    assert int(actor["residual_mlp.router.update_count"]) == state["completed_updates"]
    result = {"passed": True, "smoke": smoke, "final": final, "checked_at": time.time(),
              "checkpoint": str(checkpoint), "completed_updates": state["completed_updates"],
              "allowed_argument_differences": sorted(allowed_args), "shared_model_and_ppo_settings_match_baseline": True,
              "eight_gpu_8192_global_envs": current["distributed"], "physics_contract": physics_contract(),
              "frozen_context_sha256": current["context_sha256"], "input_audit": audit,
              "dataset": current["dataset"], "actor_critic_router_buffers_agree": True}
    if final:
        completion = json.loads((directory / "completion.json").read_text())
        assert completion["complete"] and not completion["stopped"]
        assert completion["completed_updates"] == state["completed_updates"] == expected_updates
        agreement = completion["distributed_parameter_agreement"]
        assert agreement["passed"] and agreement["world_size"] == 8
        assert len({r["actor_router_sha256"] for r in agreement["ranks"]}) == 1
        assert all(r["actor_router_sha256"] == r["critic_router_sha256"] and r["router_update_count"] == expected_updates for r in agreement["ranks"])
        result["all_eight_model_normalizer_and_router_states_agree"] = True
    write_json(root / ("smoke_validation.json" if smoke else "training_final_audit.json" if final else "training_startup_audit.json"), result)
    return result


def evaluation_command(root, entry, arm, mode):
    plan = json.loads((root / "plan.json").read_text())
    checkpoint = (root / "moe/checkpoint_update_001000.pt" if arm == "moe" else
                  Path(plan["baseline"]["checkpoint"]) if arm == "baseline" else Path(entry["checkpoint"]))
    target = root / "evaluation" / mode / f"{arm}_dr_{entry['id']:02d}.json"
    return [str(ROOT / ".venv/bin/python"), "-B", "-u", "-m", "intact_tracking.cli.residual_uniform_eval",
            "--checkpoint", str(checkpoint), "--output", str(target),
            "--motion-manifest", str(root / "protocols/evaluation_motions.txt"),
            "--steps", "1000", "--seed", "20001", "--policy-precision", "fp32",
            "--dr-bank", str(root / "protocols/dr_bank.json"), "--dr-id", str(entry["id"]),
            "--global-metrics", "--memory-start", mode, "--warmup-steps", "500"]


def evaluate(root):
    entries = json.loads((root / "plan.json").read_text())["specialists"]
    for arm, mode in (("moe", "cold"), ("moe", "warm"), ("baseline", "warm"), ("specialist", "warm")):
        (root / "evaluation" / mode).mkdir(parents=True, exist_ok=True)
        children = []
        try:
            for entry in entries:
                command = evaluation_command(root, entry, arm, mode)
                target = Path(command[command.index("--output") + 1])
                if target.exists():
                    continue
                with target.with_suffix(".log").open("ab") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=evaluation_environment(entry["id"], process_environment()),
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
                write_json(target.with_suffix(".process.json"), {"pid": child.pid, "gpu": entry["id"], "command": command})
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError(f"{arm}/{mode} evaluation failed")
                write_json(root / "state.json", {"status": "evaluating", "arm": arm, "memory_start": mode, "heartbeat": time.time(),
                                                  "evaluation_pids": [child.pid for child in children]})
                time.sleep(10)
            if any(child.returncode for child in children):
                raise RuntimeError(f"{arm}/{mode} evaluation failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    lock = (root / ".moe_comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / "plan.json").read_text())
    if not args.evaluate_only:
        ready = json.loads((root / "READY.json").read_text())
        assert ready["passed"] and json.loads((root / "smoke_validation.json").read_text())["passed"]
        for name, expected in ready["source_sha256"].items():
            assert file_sha256(ROOT / name) == expected, name
        assert file_sha256(plan["selected_context"]["checkpoint"]) == plan["selected_context"]["sha256"]
        if (root / "moe").exists():
            raise FileExistsError("Formal MoE already exists; do not silently restart from scratch")
        command = training_command(root)
        write_json(root / "launch_contract.json", {"training": command, "plan": plan})
        env = process_environment()
        env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
                   WANDB_RUN_ID="uniform-moe16-" + hashlib.sha256(str(root).encode()).hexdigest()[:12], WANDB_RESUME="allow")
        with (root / "moe.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write_json(root / "training_process.json", {"pid": child.pid, "command": command, "started_at": time.time()})
        stopping, audited = False, False

        def stop(_signum=None, _frame=None):
            nonlocal stopping
            stopping = True
            # Signal only live workers belonging to this torchrun process.
            for path in Path("/proc").iterdir():
                if not path.name.isdigit():
                    continue
                try:
                    fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
                    command_line = (path / "cmdline").read_bytes()
                    if int(fields[1]) == child.pid and b"intact_tracking.cli.residual_uniform_train" in command_line:
                        os.kill(int(path.name), signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while child.poll() is None:
            if (root / "STOP").exists() and not stopping:
                stop()
            metrics = last_record(root / "moe/metrics.jsonl")
            issues = []
            if metrics:
                losses = metrics["loss"]
                if not all(math.isfinite(float(value)) for value in losses.values()):
                    issues.append("nonfinite loss")
                if losses.get("router_center_updates") != metrics["completed_updates"]:
                    issues.append("router update count differs from PPO")
                if losses.get("router_center_update_switch_fraction", 1) > .020001:
                    issues.append("router reassignment bound exceeded")
            if issues:
                write_json(root / "health_failure.json", {"issues": issues, "metrics": metrics})
                stop()
            if metrics and metrics["completed_updates"] >= 2 and not audited:
                try:
                    audit_training(root)
                    audited = True
                except Exception:
                    stop()
                    raise
            state = {"status": "stopping" if stopping else "training", "supervisor_pid": os.getpid(), "training_pid": child.pid,
                     "heartbeat": time.time(), "progress": read_optional_json(root / "moe/progress.json"), "target_updates": 1000,
                     "seconds_per_update": metrics["collect_seconds"] + metrics["learn_seconds"] if metrics else None,
                     "router": {k: v for k, v in metrics["loss"].items() if k.startswith("router_")} if metrics else None}
            write_json(root / "state.json", state)
            write_json(ROOT / ".runtime/limb_context/current_ppo.json", {**state, "run_root": str(root)})
            time.sleep(10)
        if stopping:
            write_json(root / "state.json", {"status": "stopped", "exit_code": child.returncode, "heartbeat": time.time()})
            return
        if child.returncode:
            raise RuntimeError(f"MoE training exited {child.returncode}; see moe.log")
    audit_training(root, final=True)
    evaluate(root)
    from report_residual_uniform_moe import report
    report(root)
    done = {"status": "complete", "completed_updates": 1000, "heartbeat": time.time(), "report": str(root / "README.md")}
    write_json(root / "state.json", done)
    write_json(ROOT / ".runtime/limb_context/current_ppo.json", {**done, "run_root": str(root)})


if __name__ == "__main__":
    main()
