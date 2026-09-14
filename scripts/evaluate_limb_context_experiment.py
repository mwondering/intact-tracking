"""Restartable final evaluation, using independent paired starts and load samples."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from run_limb_context_experiment import ROOT, PYTHON, DATASET, TRACKER, available_gpus, process_environment, write_state, ppo_directory, PPO_INITIALIZATION, experiment_layout, formal_policies
from intact_tracking.limb_context_results import compare_files

POLICIES = ("frozen", "baseline_121", "baseline_122", "baseline_123", "film_121", "film_122", "film_123", "concat_121", "constant_121")
FIXED_CASES = {**{f"all_{mass}": [mass] * 4 for mass in range(5)},
               "hands_4_shins_2": [4, 4, 2, 2], "hands_2_shins_4": [2, 2, 4, 4],
               "left_4_right_2": [4, 2, 4, 2], "left_2_right_4": [2, 4, 2, 4],
               "hands_4_shins_0": [4, 4, 0, 0], "hands_0_shins_4": [0, 0, 4, 4]}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def subset(files, size, seed=813791):
    # Proportional stratification by top-level dataset directory; fixed before scores exist.
    groups = {}
    for path in files:
        groups.setdefault(Path(path).relative_to(DATASET).parts[0], []).append(path)
    quota = {key: size * len(rows) / len(files) for key, rows in groups.items()}
    counts = {key: int(value) for key, value in quota.items()}
    for key in sorted(groups, key=lambda k: (-(quota[k] - counts[k]), k))[:size - sum(counts.values())]:
        counts[key] += 1
    chosen = []
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda p: hashlib.sha256(f"{seed}/{p}".encode()).digest())
        chosen.extend(ordered[:counts[key]])
    return sorted(chosen)


def prepare_protocol(run_root):
    output = run_root / "eval"
    output.mkdir(exist_ok=True)
    manifest_root = output / "manifests"
    manifest_root.mkdir(exist_ok=True)
    files = sorted(str(path.resolve()) for path in Path(DATASET).rglob("*.npz"))
    if len(files) != 129827:
        raise ValueError(f"Dataset count changed: {len(files)}")
    cases = []

    def add_case(name, selected, masses, repeats=1, paired=False):
        batch_size = 4096 // repeats
        manifests = []
        for index, start in enumerate(range(0, len(selected), batch_size)):
            path = manifest_root / f"{name}_{index:03d}.txt"
            body = "\n".join(selected[start:start + batch_size]) + "\n"
            if path.exists() and path.read_text() != body:
                raise ValueError("Evaluation manifest changed")
            path.write_text(body)
            manifests.append(str(path))
        cases.append({"name": name, "motions": len(selected), "fixed_masses": masses,
                      "repeats": repeats, "paired_starts": paired, "manifests": manifests})

    add_case("iid_full", files, None)
    endpoints = subset(files, 4096)
    diagnostic = subset(files, 1024)
    for name, masses in FIXED_CASES.items():
        add_case(name, endpoints if name.startswith("all_") else diagnostic, masses)
    add_case("latent_intervention", diagnostic, None, repeats=2, paired=True)
    protocol = {"training_catalog_count": len(files), "cases": cases,
                "primary": "all-catalog IID independent per-limb U(0,4), one new start/load per motion",
                "endpoints": "same 4096 proportional-stratified motions/starts for all-0/1/2/3/4kg; same 1024 for asymmetric diagnostics",
                "latent_intervention": "three film training seeds; correct/zero/cross-load swap; swaps require matching motion/phase and full histories",
                "steps": 500, "seed": 20001,
                "interpretation": "three paired training seeds for baseline/film; other controls single-seed diagnostics; motion bootstrap is conditional uncertainty"}
    if not experiment_layout(run_root)["controls"]:
        protocol["interpretation"] = "three paired training seeds for baseline/film; motion bootstrap is conditional uncertainty; evaluation retains original terminations"
    if experiment_layout(run_root)["dr_profile"] != "load_only":
        protocol["dr_profile"] = experiment_layout(run_root)["dr_profile"]
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Locked evaluation protocol changed")
    write_state(path, protocol)
    return protocol


def evaluation_jobs(run_root, protocol, selected_policies=POLICIES):
    output = run_root / "eval"
    checkpoints = {policy: Path(TRACKER) if policy == "frozen" else ppo_directory(run_root) / policy / "checkpoint_final.pt" for policy in selected_policies}
    hashes = {policy: digest(path) for policy, path in checkpoints.items()}
    queue = []
    for case_index, case in enumerate(protocol["cases"]):
        intervention = case["name"] == "latent_intervention"
        policies = tuple(p for p in selected_policies if not intervention or p.startswith("film_"))
        for index, manifest in enumerate(case["manifests"]):
            seed = protocol["seed"] + index * 7919
            for policy in policies:
                for mode in (("correct", "zero", "paired-swap") if intervention else ("correct",)):
                    name = f"{case['name']}_{index:03d}_{policy}_{mode}"
                    target = output / "episodes" / f"{name}.json"
                    command = [PYTHON, "-u", "-m", "intact_tracking.cli.limb_context_eval",
                        "--motion-manifest", manifest, "--output", str(target), "--seed", str(seed),
                        "--steps", str(protocol["steps"]), "--repeats", str(case["repeats"]), "--latent-mode", mode]
                    if protocol.get("dr_profile", "load_only") != "load_only":
                        command += ["--dr-profile", protocol["dr_profile"]]
                    if policy != "frozen":
                        command += ["--checkpoint", str(checkpoints[policy])]
                    if case["fixed_masses"] is not None:
                        command += ["--fixed-masses", *map(str, case["fixed_masses"])]
                    if case["paired_starts"]:
                        command += ["--paired-starts"]
                    job = {"name": name, "case": case["name"], "policy": policy, "mode": mode,
                           "batch": index, "seed": seed, "output": str(target), "command": command,
                           "checkpoint_sha256": hashes[policy], "status": "pending"}
                    if target.exists():
                        row = json.loads(target.read_text())
                        if row["checkpoint_sha256"] != hashes[policy] or row["seed"] != seed or row["latent_intervention"] != mode:
                            raise ValueError("Existing evaluation belongs to a different checkpoint/protocol")
                        if row.get("physics", {}).get("dr_profile", "load_only") != protocol.get("dr_profile", "load_only"):
                            raise ValueError("Existing evaluation used a different DR profile")
                        if not target.with_suffix(".traces.npz").exists():
                            raise ValueError("Existing evaluation is missing step traces")
                        job["status"] = "complete"
                    queue.append(job)
    write_state(output / "checkpoint_sha256.json", hashes)
    return queue


def summarize_results(run_root, protocol, queue):
    comparisons = {}
    for case in protocol["cases"]:
        case_name = case["name"]
        available = [job for job in queue if job["case"] == case_name]

        def paths(policy, mode="correct"):
            return [job["output"] for job in sorted(available, key=lambda x: x["batch"])
                    if job["policy"] == policy and job["mode"] == mode]

        pairs = [(f"baseline_{seed}", f"film_{seed}") for seed in (121, 122, 123)]
        pairs += [("baseline_121", "concat_121"), ("constant_121", "film_121"), ("concat_121", "film_121")]
        pairs += [("frozen", policy) for policy in POLICIES if policy != "frozen"]
        if case_name == "latent_intervention":
            for seed in (121, 122, 123):
                policy = f"film_{seed}"
                for mode in ("zero", "paired-swap"):
                    key = f"{case_name}/{policy}/{mode}_vs_correct"
                    comparisons[key] = compare_files(paths(policy), paths(policy, mode))
        else:
            for reference, candidate in pairs:
                if not paths(reference) or not paths(candidate):
                    continue
                key = f"{case_name}/{candidate}_vs_{reference}"
                comparisons[key] = compare_files(paths(reference), paths(candidate))
    seeds = [comparisons[f"iid_full/film_{seed}_vs_baseline_{seed}"] for seed in (121, 122, 123)]
    ratios = np.asarray([row["candidate_over_reference_body_joint"] for row in seeds])
    failures = np.asarray([row["failure_rate_delta"] for row in seeds])
    result = {"protocol": protocol, "comparisons": comparisons,
              "primary_across_training_seeds": {
                  "seeds": [121, 122, 123], "body_joint_ratios": ratios.tolist(),
                  "mean_body_joint_ratios": ratios.mean(0).tolist(),
                  "sample_std_body_joint_ratios": ratios.std(0, ddof=1).tolist(),
                  "failure_rate_deltas": failures.tolist(),
                  "all_three_seeds_improve_both_tracking_metrics": bool((ratios < 1).all()),
                  "mean_failure_rate_does_not_increase": bool(failures.mean() <= 0)},
              "complete": True, "unix_time": time.time()}
    contract_path = run_root / "experiment_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        source_root = Path(contract["source_root"])
        previous_path = source_root / "archive/summary.json"
        previous = json.loads(previous_path.read_text())
        original_protocol = json.loads((source_root / "periodic_endpoints.json").read_text())
        if protocol["seed"] != original_protocol["seed"] or protocol["steps"] != original_protocol["steps"]:
            raise ValueError("Cross-batch gap comparison changed evaluation seed or horizon")
        gap = {"source_summary": str(previous_path), "source_sha256": digest(previous_path),
               "comparison_seed": 121, "old_worlds_per_policy": 16384, "new_worlds_per_policy": 32768,
               "interpretation": "Descriptive gap comparison at 5000 updates; doubled samples per update prevent attributing all gap changes to termination removal", "cases": {}}
        for mass in (0, 2, 4):
            case_name = f"all_{mass}"
            case = next(case for case in protocol["cases"] if case["name"] == case_name)
            if digest(case["manifests"][0]) != original_protocol["manifest_sha256"]:
                raise ValueError("Cross-batch gap comparison changed evaluation motions")
            key = f"{case_name}/film_121_vs_baseline_121"
            before, after = previous["comparisons"][key], comparisons[key]
            old_ratios, new_ratios = before["candidate_over_reference_body_joint"], after["candidate_over_reference_body_joint"]
            gap["cases"][case_name] = {"old_body_joint_ratios": old_ratios, "new_body_joint_ratios": new_ratios,
                "increase_in_relative_body_joint_advantage_pp": [100 * (old - new) for old, new in zip(old_ratios, new_ratios)],
                "old_failure_rate_delta": before["failure_rate_delta"], "new_failure_rate_delta": after["failure_rate_delta"]}
        result["gap_vs_original_termination_batch"] = gap
        write_state(run_root / "eval/gap_vs_previous_batch.json", gap)
    write_state(run_root / "eval/comparison.json", result)
    return result


def run(run_root, prepare_only=False, selected_policies=None, gpu_ids=None):
    layout = experiment_layout(run_root)
    expected_policies = ("frozen", *formal_policies(run_root))
    if selected_policies is None:
        selected_policies = expected_policies
    if not selected_policies or len(set(selected_policies)) != len(selected_policies) or not set(selected_policies).issubset(POLICIES):
        raise ValueError("Select unique policies from the fixed experiment matrix")
    partial = set(selected_policies) != set(expected_policies)
    if prepare_only:
        protocol = prepare_protocol(run_root)
        print(json.dumps(protocol, indent=2))
        return
    for policy in selected_policies:
        if policy == "frozen":
            continue
        completion = json.loads((ppo_directory(run_root) / policy / "completion.json").read_text())
        if not completion["complete"] or completion["completed_updates"] < 5000:
            raise ValueError(f"{policy} has not completed 5000 PPO updates")
        scale = completion.get("distributed", {})
        if scale.get("world_size") != layout["gpus_per_policy"] or scale.get("num_envs_per_rank") != 8192:
            raise ValueError(f"{policy} was not trained at the configured GPU/environment scale")
        if not completion.get("distributed_parameter_agreement", {}).get("passed"):
            raise ValueError(f"{policy} lacks the distributed model agreement audit")
        if completion.get("initialization_protocol") != PPO_INITIALIZATION:
            raise ValueError(f"{policy} did not train both residual actor and critic from scratch")
    output = run_root / "eval"
    output.mkdir(exist_ok=True)
    lock = (output / "evaluation.lock").open("a")
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            print("Waiting for the existing evaluation worker to finish", flush=True)
            time.sleep(20)
    protocol = prepare_protocol(run_root)
    queue = evaluation_jobs(run_root, protocol, selected_policies)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env, processes = process_environment(), {}
    while not all(job["status"] == "complete" for job in queue):
        for job in queue:
            if job["status"] == "running":
                code = processes[job["name"]].poll()
                if code is not None:
                    job["status"] = "complete" if code == 0 and Path(job["output"]).exists() else "failed"
                    job["exit_code"] = code
                    print(json.dumps({"name": job["name"], "status": job["status"]}), flush=True)
        reserved = {job["gpu"] for job in queue if job["status"] == "running"}
        available = available_gpus(reserved, layout["allowed_gpus"])
        if gpu_ids is not None:
            available = [gpu for gpu in available if gpu in gpu_ids]
        for job in queue:
            if not available:
                break
            if job["status"] != "pending":
                continue
            gpu = available.pop(0)
            with (logs / f"{job['name']}.log").open("a") as handle:
                process = subprocess.Popen(job["command"], cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes[job["name"]] = process
            job.update(status="running", pid=process.pid, gpu=gpu)
        write_state(output / ("state.partial.json" if partial else "state.json"),
                    {"updated_at": time.time(), "jobs": queue})
        if any(job["status"] == "failed" for job in queue) and not any(job["status"] == "running" for job in queue):
            raise RuntimeError("An evaluation failed; inspect its log")
        time.sleep(20)
    if partial:
        write_state(output / "partial_completion.json", {"policies": list(selected_policies),
                    "completed_jobs": len(queue), "complete": True, "unix_time": time.time()})
        return
    summarize_results(run_root, protocol, queue)
    subprocess.run([PYTHON, "-u", str(ROOT / "scripts/summarize_limb_context_experiment.py"), str(run_root)],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    parser.add_argument("--gpus", nargs="+", type=int)
    args = parser.parse_args()
    run(Path(args.run_root).resolve(), args.prepare_only, args.policies, args.gpus)
