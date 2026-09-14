"""Matched closed-loop latent interventions on held-out continuous DR samples."""

import json
import subprocess
from pathlib import Path

from intact_tracking.memory350_policy_checkpoint_eval import (
    ROOT, PYTHON, digest, evaluation_environment, write_json,
)


def validate_usage_result(row, spec, files, checkpoint_sha, update, mode):
    expected = {"checkpoint_sha256": checkpoint_sha, "completed_training_updates": update,
                "seed": spec["seed"], "max_steps": spec["steps"], "motion_files": files,
                "episodes": 2 * len(files), "repeats_per_motion": 2,
                "memory_start": "warm", "latent_intervention": mode}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Latent-usage result does not match its checkpoint/protocol")
    if (row["arguments"]["fixed_masses"] is not None or not row["arguments"]["paired_starts"]
            or row["warmup"]["steps"] != spec["warmup_steps"]
            or row["physics"]["dr_profile"] != "tracker_dr_plus_limb_payload"):
        raise ValueError("Latent usage requires matched starts and held-out continuous DR")
    for key in ("reference_timeline_audited", "partial_reset_survivor_state_audited",
                "partial_reset_survivor_history_audited"):
        if not row[key]:
            raise ValueError("Latent usage lacks a rollout-state audit")


def evaluate_latent_usage(checkpoint, directory, spec, update, fusion, gpus):
    from intact_tracking.limb_context_results import compare_files
    from intact_tracking.memory350_policy_results import assert_paired

    manifest = Path(spec["motion_manifest"])
    files = manifest.read_text().splitlines()
    if digest(manifest) != spec["manifest_sha256"] or len(files) != spec["motions"]:
        raise ValueError("Latent-usage motion manifest changed")
    if spec["version"] != "memory350_matched_latent_usage_v1" or not 0 < len(files) <= 2048:
        raise ValueError("Unsupported latent-usage protocol")
    checkpoint = Path(checkpoint).resolve()
    output = Path(directory).resolve() / "latent_usage_eval" / f"update_{update:06d}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = digest(checkpoint)
    modes = ("correct", "paired-swap") if fusion == "concat" else ("correct",)
    children, handles, rows = [], [], {}
    try:
        for i, mode in enumerate(modes):
            path = output / f"{mode}.json"
            if path.exists():
                rows[mode] = json.loads(path.read_text())
                validate_usage_result(rows[mode], spec, files, checkpoint_sha, update, mode)
                if not path.with_suffix(".traces.npz").exists():
                    raise ValueError("Latent-usage result lacks step traces")
                continue
            command = [PYTHON, "-B", "-u", "-m",
                spec.get("evaluation_module", "intact_tracking.cli.memory350_compressed_policy_eval"),
                "--checkpoint", str(checkpoint), "--motion-manifest", str(manifest),
                "--motion-path", spec["motion_path"], "--output", str(path),
                "--seed", str(spec["seed"]), "--steps", str(spec["steps"]),
                "--memory-start", "warm", "--warmup-steps", str(spec["warmup_steps"]),
                "--paired-starts", "--repeats", "2", "--latent-mode", mode]
            handle = path.with_suffix(".log").open("a")
            handles.append(handle)
            child = subprocess.Popen(command, cwd=ROOT, env=evaluation_environment(gpus[i % len(gpus)]),
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((mode, child))
        write_json(output / "processes.json", {"checkpoint": str(checkpoint), "update": update,
            "children": {mode: child.pid for mode, child in children}, "allocated_gpus": list(gpus)})
        for mode, child in children:
            if child.wait(timeout=1800) != 0:
                raise RuntimeError(f"Latent-usage evaluation failed: {output / (mode + '.log')}")
            rows[mode] = json.loads((output / f"{mode}.json").read_text())
            validate_usage_result(rows[mode], spec, files, checkpoint_sha, update, mode)
        metrics = {}
        for mode, row in rows.items():
            metrics[mode] = {**row["mean"], **{key: row[key] for key in (
                "failure_rate", "coverage_fraction", "mean_episode_return")}}
        if "paired-swap" in rows:
            assert_paired(rows["correct"], rows["paired-swap"])
            comparison = compare_files([output / "correct.json"], [output / "paired-swap.json"])
            metrics["swap_over_correct_common_body_ratio"] = comparison["common_prefix_body_joint_ratio"][0]
            metrics["swap_over_correct_common_joint_ratio"] = comparison["common_prefix_body_joint_ratio"][1]
            metrics["swap_minus_correct_failure_pp"] = 100 * comparison["failure_rate_delta"]
            swapped = rows["paired-swap"]
            metrics["swapped_fraction_of_valid_steps"] = sum(swapped["swapped_steps"]) / max(1, sum(swapped["episode_lengths"]))
        else:
            comparison = None
        result = {"completed_updates": update, "checkpoint_sha256": checkpoint_sha, "fusion": fusion,
            "policy_precision": rows["correct"].get("policy_precision", "tf32"),
            "protocol": spec, "metrics": metrics, "paired_motion_bootstrap": comparison,
            "scope": "Warm history; 2 matched-phase worlds per motion; independently sampled continuous DR. Swap only while both histories are full and both worlds are active. Ratios above one mean swapped latent worsens tracking."}
        write_json(output / "summary.json", result)
        return result
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()
