"""Wait for the shared MoE comparison, then train and evaluate independent experts."""

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
from run_residual_uniform_moe_comparison import training_command as shared_command, evaluation_command
from run_residual_uniform_specialists import read_optional_json, last_record
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.memory350_independent_moe_policy import VERSION
from intact_tracking.residual_uniform_protocol import physics_contract

TRAIN_MODULE = "intact_tracking.cli.residual_independent_moe_train"
EVAL_MODULE = "intact_tracking.cli.residual_independent_moe_eval"


def training_command(root, smoke=False):
    command = shared_command(root, smoke=smoke)
    command[command.index("intact_tracking.cli.residual_uniform_train")] = TRAIN_MODULE
    command[command.index("--wandb-name") + 1] = root.name + ("-smoke8" if smoke else "-independent-moe16")
    return command


def audit_training(root, smoke=False, final=False):
    import torch
    from omegaconf import OmegaConf

    directory = root / ("smoke" if smoke else "moe")
    plan = json.loads((root / "plan.json").read_text())
    previous = Path(plan["shared_moe_run"])
    current = json.loads((directory / "run_config.json").read_text())
    shared = json.loads((previous / "moe/run_config.json").read_text())
    allowed = {"output_dir", "wandb_group", "wandb_name", "context_checkpoint"}
    if smoke:
        allowed |= {"iterations", "save_interval", "motion_file", "motion_path"}
    for key in set(current["arguments"]) | set(shared["arguments"]):
        if key not in allowed and current["arguments"].get(key) != shared["arguments"].get(key):
            raise ValueError(f"Independent/shared MoE argument differs: {key}")
    assert current["version"] == VERSION
    for key in ("fusion", "tracker_sha256", "context_sha256", "reward_contract", "reward_changes",
                "residual_output", "training_terminations", "training_starts", "initialization_seeds",
                "initialization_protocol", "policy_precision", "dr_profile", "dr_sampling", "distributed", "motion_sampling"):
        if current[key] != shared[key]:
            raise ValueError(f"Independent/shared MoE protocol differs: {key}")
    assert current["encoder_frozen"] and current["context_normalization_frozen"]
    assert not current["predictor_executed_in_ppo"] and not current["extra_current_state_physics_privilege"]
    assert current["context_sha256"] == plan["selected_context"]["sha256"]
    assert current["residual_output"]["bounded"] is False
    expected = 2 if smoke else 1000
    assert current["maximum_updates"] == expected
    audit = current["input_audit"]
    for key in ("actor_original_features", "critic_original_features", "actor_compression", "critic_compression",
                "actor_head", "critic_head", "actor_experts", "critic_experts", "latent_dimensions", "latent_use",
                "tracker_action_input_dim", "critic_tracker_action_input_dim", "critic_tracker_action_matches_actor",
                "critic_bootstrap_tracker_action", "action_std_initialization", "actor_initialization_seed", "critic_initialization_seed"):
        assert audit[key] == shared["input_audit"][key], key
    for key in ("actor_critic_parameters_shared", "expert_trainable_parameters_shared",
                "each_side_shares_one_obs_encoder_across_its_experts", "exploration_std_shared_across_experts",
                "router_has_trainable_parameters"):
        assert audit[key] is False
    assert audit["all_trainable_parameters_owned_by_one_expert"] and audit["dispatch_before_observation_encoding"]
    assert audit["actor_trainable_parameters"] == 17345440
    assert audit["critic_trainable_parameters"] == 115927056
    for side in ("actor", "critic"):
        assert audit[f"{side}_compressor_sha256_by_expert"] == [shared["input_audit"][f"{side}_compressor_sha256"]] * 16
    bootstrap = audit["routing_bootstrap"]
    assert bootstrap["steps"] == 500 and bootstrap["bootstrap_samples"] == 32768
    assert len(bootstrap["bootstrap_counts"]) == 16 and all(n > 0 for n in bootstrap["bootstrap_counts"])
    if not smoke:
        for key in ("manifest_sha256", "loaded_motion_count", "loaded_frames", "root", "loaded_motion_counts_per_rank"):
            assert current["dataset"][key] == shared["dataset"][key], key
    ranks = current["physics"]["runtime_audits_by_rank"]
    assert len(ranks) == 8
    for rank, ref in zip(ranks, shared["physics"]["runtime_audits_by_rank"], strict=True):
        physics = rank["physics"]
        assert physics["actual_mass_sha256"] == ref["physics"]["actual_mass_sha256"]
        assert physics["residual_physics_contract"] == physics_contract()
        assert physics["wrist_force_ranges_nm"] == [[-10., 10.]] * 4
        assert physics["pd_gains_and_tracker_action_scale_verified"]
    checkpoint = directory / ("checkpoint_final.pt" if final else "checkpoint_initial.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    reference = torch.load(previous / "moe/checkpoint_initial.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg, ref_cfg = [OmegaConf.to_container(s["cfg"].agent, resolve=True) for s in (state, reference)]
    for key in ("algorithm", "obs_groups", "num_steps_per_env", "clip_actions", "seed", "check_for_nan", "torch_compile_mode"):
        assert cfg[key] == ref_cfg[key], key
    for side in ("actor", "critic"):
        assert {k: v for k, v in cfg[side].items() if k != "class_name"} == {
            k: v for k, v in ref_cfg[side].items() if k != "class_name"}, side
    actor, critic = state["actor_state_dict"], state["critic_state_dict"]
    for name, value in actor.items():
        if name.startswith("tracker."):
            torch.testing.assert_close(value, reference["actor_state_dict"][name], atol=0, rtol=0)
        if name.startswith("residual_mlp.router."):
            torch.testing.assert_close(value, critic[name.replace("residual_mlp.router.", "mlp.router.")], atol=0, rtol=0)
    assert int(actor["residual_mlp.router.update_count"]) == state["completed_updates"]
    result = {"passed": True, "smoke": smoke, "final": final, "checked_at": time.time(),
              "checkpoint": str(checkpoint), "completed_updates": state["completed_updates"],
              "input_audit": audit, "dataset": current["dataset"], "ppo_and_physics_match_shared_moe": True,
              "all_initial_expert_encoders_match_shared_initial_encoder": True,
              "frozen_tracker_unchanged": True, "actor_critic_router_buffers_agree": True,
              "allowed_argument_differences": sorted(allowed)}
    if final:
        completion = json.loads((directory / "completion.json").read_text())
        assert completion["complete"] and not completion["stopped"]
        assert completion["completed_updates"] == state["completed_updates"] == expected
        agreement = completion["distributed_parameter_agreement"]
        assert agreement["passed"] and agreement["world_size"] == 8
        assert len({r["actor_router_sha256"] for r in agreement["ranks"]}) == 1
        assert all(r["actor_router_sha256"] == r["critic_router_sha256"] and r["router_update_count"] == expected for r in agreement["ranks"])
        result["all_eight_model_normalizer_and_router_states_agree"] = True
    name = "smoke_validation.json" if smoke else "training_final_audit.json" if final else "training_startup_audit.json"
    write_json(root / name, result)
    return result


def stop_workers(child):
    # No name-wide GPU killing: signal only workers of our own torchrun.
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[1]) == child.pid and TRAIN_MODULE.encode() in (path / "cmdline").read_bytes():
                os.kill(int(path.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass


def train(root, smoke=False):
    directory = root / ("smoke" if smoke else "moe")
    if directory.exists():
        raise FileExistsError(f"Refuse to silently restart existing training: {directory}")
    command = training_command(root, smoke)
    env = process_environment()
    run_id = "independent-moe16-" + hashlib.sha256(str(root).encode()).hexdigest()[:12] + ("-smoke" if smoke else "")
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
               WANDB_RUN_ID=run_id, WANDB_RESUME="allow")
    with (root / ("smoke.log" if smoke else "moe.log")).open("ab") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    write_json(root / ("smoke_process.json" if smoke else "training_process.json"),
               {"pid": child.pid, "command": command, "started_at": time.time(), "wandb_run_id": run_id})
    stopping, audited = False, False

    def stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        stop_workers(child)

    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while child.poll() is None:
            if (root / "STOP").exists() and not stopping:
                stop()
            metrics = last_record(directory / "metrics.jsonl")
            if metrics:
                loss = metrics["loss"]
                issues = []
                if not all(math.isfinite(float(v)) for v in loss.values()):
                    issues.append("nonfinite loss")
                if loss.get("router_center_updates") != metrics["completed_updates"]:
                    issues.append("router update count differs from PPO")
                if loss.get("router_center_update_switch_fraction", 1) > .020001:
                    issues.append("router reassignment bound exceeded")
                if issues:
                    write_json(root / "health_failure.json", {"issues": issues, "metrics": metrics})
                    stop()
                if metrics["completed_updates"] >= 2 and not audited:
                    audit_training(root, smoke=smoke)
                    audited = True
            state = {"status": "stopping" if stopping else "smoke_training" if smoke else "training",
                     "supervisor_pid": os.getpid(), "training_pid": child.pid, "heartbeat": time.time(),
                     "progress": read_optional_json(directory / "progress.json"), "target_updates": 2 if smoke else 1000,
                     "seconds_per_update": metrics["collect_seconds"] + metrics["learn_seconds"] if metrics else None,
                     "router": {k: v for k, v in metrics["loss"].items() if k.startswith("router_")} if metrics else None}
            write_json(root / "state.json", state)
            write_json(ROOT / ".runtime/limb_context/current_ppo.json", {**state, "run_root": str(root)})
            time.sleep(10)
        if stopping:
            raise RuntimeError("Independent MoE stopped; inspect saved completed-update checkpoint")
        if child.returncode:
            raise RuntimeError(f"Independent MoE training exited {child.returncode}")
        audit_training(root, smoke=smoke, final=True)
    finally:
        if child.poll() is None:
            stop_workers(child)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=30)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def evaluate(root):
    entries = json.loads((root / "plan.json").read_text())["specialists"]
    for mode in ("cold", "warm"):
        (root / "evaluation" / mode).mkdir(parents=True, exist_ok=True)
        children = []
        try:
            for entry in entries:
                command = evaluation_command(root, entry, "moe", mode)
                command[command.index("intact_tracking.cli.residual_uniform_eval")] = EVAL_MODULE
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
                    raise RuntimeError(f"Independent MoE {mode} evaluation failed")
                write_json(root / "state.json", {"status": "evaluating", "memory_start": mode, "heartbeat": time.time(),
                                                  "evaluation_pids": [child.pid for child in children]})
                time.sleep(10)
            if any(child.returncode for child in children):
                raise RuntimeError(f"Independent MoE {mode} evaluation failed")
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    lock = (root / ".independent_moe_comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / "plan.json").read_text())
    previous = Path(plan["shared_moe_run"])
    if not args.evaluate_only:
        prepared = json.loads((root / "PREPARED.json").read_text())
        assert prepared["passed"]
        while True:
            state = read_optional_json(previous / "state.json") or {}
            if state.get("status") == "complete":
                break
            if state.get("status") == "stopped" or (previous / "health_failure.json").exists():
                raise RuntimeError("The shared MoE did not complete; retain the sequential comparison")
            if (root / "STOP").exists():
                write_json(root / "state.json", {"status": "stopped_while_queued", "heartbeat": time.time()})
                return
            if state.get("heartbeat", 0) < time.time() - 600:
                raise RuntimeError("Shared MoE supervisor heartbeat is stale; inspect before starting")
            write_json(root / "state.json", {"status": "waiting_for_shared_moe_comparison", "heartbeat": time.time(),
                                              "supervisor_pid": os.getpid(), "shared_moe_state": state})
            time.sleep(10)
        assert json.loads((previous / "training_final_audit.json").read_text())["passed"]
        assert json.loads((previous / "comparison.json").read_text())["passed"]
        for name, expected in prepared["source_sha256"].items():
            assert file_sha256(ROOT / name) == expected, name
        assert file_sha256(plan["selected_context"]["checkpoint"]) == plan["selected_context"]["sha256"]
        plan["shared_moe_checkpoint_sha256"] = file_sha256(previous / "moe/checkpoint_update_001000.pt")
        write_json(root / "plan.json", plan)
        if not (root / "smoke_validation.json").exists():
            train(root, smoke=True)
        assert json.loads((root / "smoke_validation.json").read_text())["passed"]
        write_json(root / "READY.json", {**prepared, "smoke_validation": str(root / "smoke_validation.json"), "ready_at": time.time()})
        train(root)
    else:
        audit_training(root, final=True)
    evaluate(root)
    from report_residual_independent_moe import report
    report(root)
    done = {"status": "complete", "completed_updates": 1000, "heartbeat": time.time(), "report": str(root / "README.md")}
    write_json(root / "state.json", done)
    write_json(ROOT / ".runtime/limb_context/current_ppo.json", {**done, "run_root": str(root)})


if __name__ == "__main__":
    main()
