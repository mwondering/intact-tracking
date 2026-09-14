"""Bounded, restartable, paired full-catalog evaluation. Does not start training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from intact_tracking.cli.adaptation_eval import TRACKER
from intact_tracking.preview_protocol import FULL_DATASET, LIMB_PROFILE, dataset_identity
from intact_tracking.rollout.mjlab_adapter import _sha256


def validate_training_pair(checkpoints):
    """Require equal real experience and the same DR/reward/training protocol."""
    records = {}
    for arm in ("baseline", "preview"):
        checkpoint = torch.load(checkpoints[arm], map_location="cpu", weights_only=False)
        meta = checkpoint["residual_policy"]
        if meta["variant"] != arm or meta["physics_mode"] != "dr" or meta["reward_changes"]:
            raise ValueError(f"Not a fixed-reward DR {arm} checkpoint")
        records[arm] = {
            "completed_updates": checkpoint["completed_updates"],
            **{key: meta[key] for key in ("version", "tracker_sha256", "motion_sha256", "dr_profile")},
            "reward_sha256": meta["reward_contract"]["sha256"],
            "physics_details": meta["physics"]["details"],
            "arguments": {key: meta["arguments"][key] for key in (
                "num_envs", "seed", "rollout_steps", "actor_lr", "critic_lr", "critic_warmup_updates",
                "epochs", "mini_batches", "entropy_coef", "initial_action_std", "residual_scale", "hidden_dims")},
        }
        del checkpoint
    if records["baseline"] != records["preview"]:
        different = [key for key in records["baseline"]
                     if records["baseline"][key] != records["preview"][key]]
        raise ValueError(f"Training comparison is not matched: {different}")
    return records["preview"]


def paired_motion_rows(reference, candidate):
    for key in ("protocol", "seed", "motion_ids", "start_frames", "metric_names", "max_steps",
                "motion_files", "physics_world_fingerprints"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Unmatched evaluation field: {key}")
    for key in ("physics", "dr_profile"):
        if reference["physics"][key] != candidate["physics"][key]:
            raise ValueError(f"Unmatched evaluation physics: {key}")
    ids = np.asarray(reference["motion_ids"])
    names = reference["metric_names"]
    columns = [names.index(name) for name in ("error_body_pos", "error_joint_pos")]
    r = np.asarray(reference["per_episode_metrics"])[:, columns]
    c = np.asarray(candidate["per_episode_metrics"])[:, columns]
    rf, cf = np.asarray(reference["failed"], bool), np.asarray(candidate["failed"], bool)
    if not np.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError("Nonfinite evaluation metrics")
    # 4 tracking columns + baseline/candidate failure fractions, one row per motion.
    values = np.column_stack((r, c, rf, cf))
    sums = np.zeros((reference["motions"], 6))
    np.add.at(sums, ids, values)
    counts = np.bincount(ids, minlength=reference["motions"])
    if not (counts > 0).all():
        raise ValueError("Missing motion evaluation episodes")
    return sums / counts[:, None], int((cf & ~rf).sum()), int((rf & ~cf).sum())


def summarize(rows, new_failures, rescued, samples=2000):
    mean = rows.mean(0)
    rng = np.random.default_rng(7712)
    draws = []
    for _ in range(samples):
        sample = rows[rng.integers(len(rows), size=len(rows))].mean(0)
        draws.append([*(sample[2:4] / sample[:2]), sample[5] - sample[4]])
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "motions": len(rows), "baseline_body_joint": mean[:2].tolist(),
        "preview_body_joint": mean[2:4].tolist(),
        "preview_over_baseline_body_joint": (mean[2:4] / mean[:2]).tolist(),
        "ratio_ci95_low": lo[:2].tolist(), "ratio_ci95_high": hi[:2].tolist(),
        "baseline_failure_rate": float(mean[4]), "preview_failure_rate": float(mean[5]),
        "failure_rate_delta_ci95": [float(lo[2]), float(hi[2])],
        "new_failure_episodes": new_failures, "rescued_failure_episodes": rescued,
        "clear_tracking_gain_this_seed": bool((hi[:2] < 1).all()),
        "strict_no_regression_this_seed": bool((hi[:2] < 1).all() and mean[5] <= mean[4]
                                                and new_failures == 0),
        "uncertainty": "paired bootstrap over motions, not over independent training seeds",
        "scope": "in-distribution full-catalog evaluation, not an unseen-motion generalization test",
    }


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--preview-checkpoint", required=True)
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--motion-path", default=FULL_DATASET)
    parser.add_argument("--dr-profile", default=LIMB_PROFILE,
                        choices=("right-hand-1-3kg", LIMB_PROFILE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=4096, help="Upper bound, including repeats")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if min(args.repeats, args.steps, args.bootstrap_samples) <= 0 or args.num_envs < args.repeats:
        parser.error("Invalid batch size, repeats, steps, or bootstrap samples")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # No overlapping writers or checkpoint mixing on resumption.
    import fcntl

    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files, dataset = dataset_identity(motion_path=args.motion_path)
        checkpoints = {arm: str(Path(getattr(args, arm + "_checkpoint")).resolve())
                       for arm in ("baseline", "preview")}
        checkpoints["frozen"] = str(Path(args.tracker_checkpoint).resolve())
        training_pair = validate_training_pair(checkpoints)
        if training_pair["dr_profile"] != args.dr_profile:
            raise ValueError("Evaluation DR profile differs from the matched training profile")
        contract = {"dataset": dataset, "arguments": vars(args), "checkpoint_sha256": {
            arm: _sha256(Path(path)) for arm, path in checkpoints.items()}}
        if contract["checkpoint_sha256"]["frozen"] != training_pair["tracker_sha256"]:
            raise ValueError("Frozen reference is not the original tracker used by the residual pair")
        contract_path = output / "evaluation_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Evaluation contract changed; choose a new output directory")
        atomic_json(contract_path, contract)
        batch_size = args.num_envs // args.repeats
        rows, new_failures, rescued = [], 0, 0
        frozen_rows, frozen_added, frozen_rescued = [], 0, 0
        for batch, start in enumerate(range(0, len(files), batch_size)):
            selected = files[start:start + batch_size]
            manifest = output / f"batch_{batch:05d}.txt"
            manifest.write_text("\n".join(map(str, selected)) + "\n")
            results = {}
            for arm, checkpoint in checkpoints.items():
                result = output / f"{arm}_{batch:05d}.json"
                if not result.exists():
                    cmd = [sys.executable, "-m", "intact_tracking.cli.adaptation_eval",
                           "--tracker-checkpoint", checkpoints["frozen"],
                           "--motion-path", str(Path(args.motion_path).resolve()),
                           "--motion-manifest", str(manifest), "--physics", "dr",
                           "--dr-profile", args.dr_profile, "--seed", str(args.seed + batch),
                           "--repeats", str(args.repeats), "--steps", str(args.steps),
                           "--max-envs", str(args.num_envs), "--output", str(result)]
                    if arm != "frozen":
                        cmd.extend(["--checkpoint", checkpoint])
                    with (output / f"{arm}_{batch:05d}.log").open("a") as log:
                        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
                results[arm] = json.loads(result.read_text())
                if results[arm]["checkpoint_sha256"] != contract["checkpoint_sha256"][arm]:
                    raise ValueError("Batch result checkpoint mismatch")
                if results[arm]["motion_files"] != list(map(str, selected)):
                    raise ValueError("Batch result catalog mismatch")
                if arm != "frozen" and not results[arm]["training_reward_audit"]["eligible_fixed_reward_candidate"]:
                    raise ValueError("Batch candidate failed the original-reward lineage audit")
            row, added, removed = paired_motion_rows(results["baseline"], results["preview"])
            rows.append(row)
            new_failures += added
            rescued += removed
            row, added, removed = paired_motion_rows(results["frozen"], results["preview"])
            frozen_rows.append(row)
            frozen_added += added
            frozen_rescued += removed
            print(json.dumps({"finished_motions": start + len(selected), "total": len(files),
                              "new_failure_episodes_so_far": new_failures}), flush=True)
        result = summarize(np.concatenate(rows), new_failures, rescued, args.bootstrap_samples)
        result["preview_vs_frozen_dr"] = summarize(
            np.concatenate(frozen_rows), frozen_added, frozen_rescued, args.bootstrap_samples)
        result["preview_vs_frozen_dr"]["reference_label"] = "frozen tracker, same DR"
        result["contract"] = contract
        result["training_pair"] = training_pair
        atomic_json(output / "comparison.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
