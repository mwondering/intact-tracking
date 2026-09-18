"""Train one 8x1024 uniform-DR baseline and compare it with eight u1000 specialists."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from run_limb_context_experiment import ROOT, PYTHON, DATASET, process_environment
from run_residual_uniform_specialists import read_optional_json, last_record
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.residual_uniform_protocol import EVAL_VERSION, physics_contract


def training_command(root, *, smoke=False):
    output = root / ("smoke" if smoke else "baseline")
    command = [PYTHON, "-B", "-u", "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0", "-m",
               "intact_tracking.cli.residual_uniform_train", "--fusion", "baseline",
               "--output-dir", str(output), "--training-ranks", "8", "--num-envs", "1024",
               "--iterations", "2" if smoke else "1000", "--seed", "121",
               "--rollout-steps", "24", "--epochs", "5", "--mini-batches", "4",
               "--actor-lr", "0.0001", "--critic-lr", "0.0005", "--entropy-coef", "0.0002",
               "--save-interval", "1" if smoke else "100", "--episode-steps", "1000",
               "--training-terminations", "original", "--policy-precision", "fp32",
               "--motion-sampling", "uniform", "--dr-sampling", "independent_uniform",
               "--router-bootstrap-steps", "500", "--wandb-group", root.name,
               "--wandb-name", root.name + ("-smoke8" if smoke else "-shared_baseline")]
    if smoke:
        motion = (root / "protocols/evaluation_motions.txt").read_text().splitlines()[0]
        command += ["--motion-file", motion]
    else:
        command += ["--motion-path", DATASET]
    return command


def audit_training(root, *, smoke=False, final=False):
    """Reject all training differences except the requested DR and rank layout."""
    import torch
    from omegaconf import OmegaConf

    output = root / ("smoke" if smoke else "baseline")
    config = json.loads((output / "run_config.json").read_text())
    selection = json.loads((root / "protocols/selection.json").read_text())
    allowed = {"output_dir", "dr_bank", "specialist_id", "eval_motion_manifest", "training_ranks",
               "num_envs", "iterations", "until_user_stop", "wandb_group", "wandb_name"}
    if smoke:
        allowed |= {"motion_path", "motion_file", "save_interval"}
    input_exceptions = {"critic_initial_normalizer_sha256", "routing_bootstrap"}
    for entry in selection["specialists"]:
        old = json.loads((root / "reference_specialists" / f"B{entry['id']:02d}" / "run_config.json").read_text())
        for key in set(old["arguments"]) | set(config["arguments"]):
            if key not in allowed and old["arguments"].get(key) != config["arguments"].get(key):
                raise ValueError(f"Shared/expert argument differs: {key}")
        for key in ("version", "fusion", "tracker_sha256", "residual_output", "reward_contract",
                    "reward_changes", "training_terminations", "training_starts", "initialization_seeds",
                    "initialization_protocol", "policy_precision", "dr_profile", "dr_sampling"):
            if old[key] != config[key]:
                raise ValueError(f"Shared/expert protocol differs: {key}")
        for key, value in old["input_audit"].items():
            if key not in input_exceptions and config["input_audit"][key] != value:
                raise ValueError(f"Shared/expert model/input differs: {key}")
        if not smoke:
            for key in ("manifest_sha256", "loaded_motion_count", "loaded_frames", "root"):
                if config["dataset"][key] != old["dataset"][key]:
                    raise ValueError(f"Shared/expert dataset differs: {key}")
    dist = config["distributed"]
    assert (dist["world_size"], dist["num_envs_per_rank"], dist["global_num_envs"]) == (8, 1024, 8192)
    assert config["maximum_updates"] == (2 if smoke else 1000)
    assert config["motion_sampling"]["active_mode"] == "uniform"
    assert config["residual_output"]["bounded"] is False and config["context_checkpoint"] is None
    assert "fixed_dr" not in config["physics"]
    ranks = config["physics"]["runtime_audits_by_rank"]
    assert len(ranks) == 8 and len({r["physics"]["actual_mass_sha256"] for r in ranks}) == 8
    for rank in ranks:
        physics = rank["physics"]
        assert physics["residual_physics_contract"] == physics_contract()
        assert physics["wrist_force_ranges_nm"] == [[-10., 10.]] * 4
        assert physics["pd_gains_and_tracker_action_scale_verified"]
    checkpoint = output / ("checkpoint_final.pt" if final else "checkpoint_initial.pt")
    current = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expert = torch.load(selection["specialists"][0]["frozen_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    left, right = [OmegaConf.to_container(c["cfg"].agent, resolve=True) for c in (current, expert)]
    for key in ("actor", "critic", "algorithm", "obs_groups", "num_steps_per_env", "clip_actions", "seed", "check_for_nan", "torch_compile_mode"):
        if left[key] != right[key]:
            raise ValueError(f"Shared/expert PPO configuration differs: {key}")
    result = {"passed": True, "smoke": smoke, "final": final,
              "checkpoint": str(checkpoint), "completed_updates": current["completed_updates"],
              "global_envs": 8192, "transitions_per_update": 8192 * 24,
              "model_and_ppo_settings_match_all_specialists": True,
              "independent_random_dr_ranks": 8, "physics_contract": physics_contract(),
              "motion_counts_by_rank": config["dataset"]["loaded_motion_counts_per_rank"],
              "allowed_argument_differences": sorted(allowed), "checked_at": time.time()}
    if final:
        completion = json.loads((output / "completion.json").read_text())
        assert completion["complete"] and not completion["stopped"]
        assert completion["completed_updates"] == current["completed_updates"] == (2 if smoke else 1000)
        assert completion["distributed_parameter_agreement"]["passed"]
        assert completion["distributed_parameter_agreement"]["world_size"] == 8
        result["all_eight_model_and_normalizer_hashes_agree"] = True
    write_json(root / ("smoke_validation.json" if smoke else "training_final_audit.json" if final else "training_startup_audit.json"), result)
    return result


def evaluate_command(root, entry):
    return [PYTHON, "-B", "-u", "-m", "intact_tracking.cli.residual_uniform_eval",
            "--checkpoint", str(root / "baseline/checkpoint_update_001000.pt"),
            "--output", str(root / "evaluation" / f"baseline_dr_{entry['id']:02d}.json"),
            "--motion-manifest", str(root / "protocols/evaluation_motions.txt"),
            "--steps", "1000", "--seed", "20001", "--policy-precision", "fp32",
            "--dr-bank", str(root / "protocols/dr_bank.json"), "--dr-id", str(entry["id"])]


def report(root):
    import numpy as np
    from intact_tracking.fixed_dr_specialists import compare_evaluations

    selection = json.loads((root / "protocols/selection.json").read_text())
    rows = []
    checkpoint = root / "baseline/checkpoint_update_001000.pt"
    checkpoint_sha = file_sha256(checkpoint)
    for entry in selection["specialists"]:
        a_path = root / "evaluation" / f"baseline_dr_{entry['id']:02d}.json"
        b_path = Path(entry["frozen_evaluation"])
        a, b = [json.loads(p.read_text()) for p in (a_path, b_path)]
        assert a["checkpoint_sha256"] == checkpoint_sha and b["checkpoint_sha256"] == entry["checkpoint_sha256"]
        for row in (a, b):
            assert row["protocol"] == EVAL_VERSION and row["completed_training_updates"] == 1000
            assert row["policy_precision"] == "fp32" and row["residual_output_bounded"] is False
            assert row["physics"]["residual_physics_contract"] == physics_contract()
            for key in ("reference_timeline_audited", "partial_reset_survivor_state_audited", "partial_reset_survivor_history_audited"):
                assert row[key], key
        comparison = compare_evaluations(a_path, b_path)
        comparison["interpretation"] = "Equal updates and equal total samples per policy; continuous-DR shared policy versus a fixed-complete-DR specialist; one training seed and different distributed layout."
        rows.append({"id": entry["id"], "profile": entry["profile"], "baseline": str(a_path),
                     "specialist": str(b_path), "comparison": comparison})
    result = {"passed": True, "completed_updates": 1000, "transitions_per_policy": 1000 * 8192 * 24,
              "baseline_checkpoint": str(checkpoint), "baseline_checkpoint_sha256": checkpoint_sha,
              "training_audit": json.loads((root / "training_final_audit.json").read_text()), "rows": rows,
              "equal_environment_mean_reduction_percent": {metric: float(np.mean([r["comparison"][metric]["reduction_percent"] for r in rows]))
                  for metric in ("common_error_body_pos", "common_error_joint_pos")}}
    write_json(root / "comparison.json", result)
    lines = ["# Uniform 随机 DR 共享 policy 与八个环境专家：1000 轮", "",
             "双方每轮总共 8192 环境 × 24 步，1000 轮各 196608000 个 PPO transition。共享 policy 为 8×1024，单个专家为 1×8192；其余网络、奖励、termination、动作协议和 PPO 超参数经逐项核对。", "",
             "评测复用同一 512-motion 列表、起点、噪声种子和完整静态 DR。body/joint 使用共同存活窗口；root 等指标按各自 episode 截断，应结合失败率解释。区间来自 motion 配对 bootstrap，不能代表训练 seed 的不确定性。", "",
             "| DR | body pos 共享 / 专家 (cm) | 专家改善 | joint pos 共享 / 专家 (rad) | root pos 共享 / 专家 (cm) | 失败率 共享 / 专家 |", "|---|---:|---:|---:|---:|---:|"]
    flat = []
    for row in rows:
        c = row["comparison"];body = c["common_error_body_pos"];joint = c["common_error_joint_pos"];anchor = c["truncated_error_anchor_pos"];failure = c["failure_rate"]
        lines.append(f"| B{row['id']:02d} {row['profile']} | {100*body['reference']:.3f} / {100*body['candidate']:.3f} | {body['reduction_percent']:+.2f}% | {joint['reference']:.4f} / {joint['candidate']:.4f} | {100*anchor['reference']:.3f} / {100*anchor['candidate']:.3f} | {100*failure['reference']:.2f}% / {100*failure['candidate']:.2f}% |")
        flat.append({"id": row["id"], "profile": row["profile"], "baseline_body_pos_m": body["reference"],
                     "specialist_body_pos_m": body["candidate"], "body_reduction_percent": body["reduction_percent"],
                     "body_reduction_ci95_lower": body["reduction_percent_ci95"][0], "body_reduction_ci95_upper": body["reduction_percent_ci95"][1],
                     "baseline_joint_pos_rad": joint["reference"], "specialist_joint_pos_rad": joint["candidate"],
                     "baseline_root_pos_m": anchor["reference"], "specialist_root_pos_m": anchor["candidate"],
                     "baseline_failure_rate": failure["reference"], "specialist_failure_rate": failure["candidate"]})
    (root / "README.md").write_text("\n".join(lines) + "\n")
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(); root = args.run_root.resolve()
    lock = (root / ".shared_comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((root / "READY.json").read_text())
    assert ready["passed"] and json.loads((root / "smoke_validation.json").read_text())["passed"]
    for path, expected in ready["source_sha256"].items():
        assert file_sha256(path) == expected, path
    if (root / "baseline").exists():
        raise FileExistsError("Formal baseline already exists")
    selection = json.loads((root / "protocols/selection.json").read_text())
    command = training_command(root)
    write_json(root / "launch_contract.json", {"training": command, "evaluations": [evaluate_command(root, e) for e in selection["specialists"]]})
    env = process_environment()
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", WANDB_API_KEY=(ROOT / ".runtime/limb_context/wandb_api_key").read_text().strip(),
               WANDB_RUN_ID="uniform-shared-" + hashlib.sha256(str(root).encode()).hexdigest()[:12], WANDB_RESUME="allow")
    with (root / "baseline.log").open("ab") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    write_json(root / "baseline_process.json", {"pid": child.pid, "command": command, "started_at": time.time()})
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
                cmd = (path / "cmdline").read_bytes()
                if int(fields[1]) == child.pid and b"intact_tracking.cli.residual_uniform_train" in cmd:
                    os.kill(int(path.name), signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    checked = False
    while child.poll() is None:
        if (root / "STOP").exists() and not stopping:
            stop(None, None)
        progress = read_optional_json(root / "baseline/progress.json")
        metrics = last_record(root / "baseline/metrics.jsonl")
        if metrics and not all(math.isfinite(float(value)) for value in metrics["loss"].values()):
            stop(None, None)
            raise RuntimeError("Shared baseline produced nonfinite metrics")
        if progress and progress["completed_updates"] >= 2 and not checked:
            audit_training(root); checked = True
        state = {"status": "shared_baseline_stopping" if stopping else "shared_baseline_training", "supervisor_pid": os.getpid(),
                 "heartbeat": time.time(), "training_pid": child.pid, "progress": progress,
                 "target_updates": 1000, "global_envs": 8192, "num_envs_per_rank": 1024,
                 "seconds_per_update": metrics["collect_seconds"] + metrics["learn_seconds"] if metrics else None}
        write_json(root / "state.json", state)
        write_json(ROOT / ".runtime/limb_context/current_ppo.json", {**state, "run_root": str(root), "version": physics_contract()["version"]})
        time.sleep(10)
    if stopping:
        write_json(root / "state.json", {"status": "stopped", "exit_code": child.returncode, "heartbeat": time.time()})
        return
    if child.returncode:
        raise RuntimeError(f"Shared baseline failed: exit {child.returncode}; see baseline.log")
    audit_training(root, final=True)
    (root / "evaluation").mkdir()
    children = []
    for entry in selection["specialists"]:
        gpu = entry["id"]; command = evaluate_command(root, entry)
        with (root / "evaluation" / f"baseline_dr_{gpu:02d}.log").open("ab") as log:
            job = subprocess.Popen(command, cwd=ROOT, env=evaluation_environment(gpu), stdout=log, stderr=subprocess.STDOUT)
        children.append(job)
    while any(job.poll() is None for job in children):
        codes = [job.poll() for job in children]
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError(f"Shared fixed-DR evaluation failed: {codes}")
        write_json(root / "state.json", {"status": "evaluating_eight_fixed_drs", "heartbeat": time.time(), "evaluation_pids": [job.pid for job in children], "exit_codes": codes})
        time.sleep(10)
    if any(job.returncode for job in children):
        raise RuntimeError("Fixed-DR evaluation failed")
    report(root)
    write_json(root / "state.json", {"status": "complete", "completed_updates": 1000, "heartbeat": time.time(), "report": str(root / "README.md")})
    write_json(ROOT / ".runtime/limb_context/current_ppo.json", {"status": "complete", "run_root": str(root), "updated_at": time.time(), "report": str(root / "README.md")})


if __name__ == "__main__":
    main()
