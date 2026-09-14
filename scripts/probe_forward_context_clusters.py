"""Collect frozen forward-context latents across motions and phases.

Use the checkpoint's own rollout distribution and normalization. Main samples
have nonoverlapping 100-frame histories; optional +5-frame samples measure only
local consistency. No network or simulator parameter is optimized.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from intact_tracking.residual_context import (
    DynamicsContextInference,
    load_frozen_context_checkpoint,
)
from intact_tracking.rollout.online import FixedDRRolloutConfig, FixedDRTrackerRollout
from intact_tracking.rollout.mjlab_adapter import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=81208)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--neighbor-offset", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkpoint = args.checkpoint.resolve()
    run_config = json.loads((checkpoint.parent / "run_config.json").read_text())
    original = run_config["arguments"]
    frozen = load_frozen_context_checkpoint(checkpoint, device="cuda:0")
    if args.stride < frozen.config.context_history_steps:
        raise ValueError("Main sample histories must not overlap")
    if not 0 < args.neighbor_offset < args.stride:
        raise ValueError("Neighbor offset must lie between zero and stride")
    config = FixedDRRolloutConfig(
        checkpoint_file=original["checkpoint_file"],
        motion_path=original.get("motion_path"),
        motion_file=original.get("motion_file"),
        task_id=original.get("task_id"),
        num_envs=args.num_envs,
        device="cuda:0",
        seed=args.seed,
        stochastic_policy=original.get("stochastic_policy", False),
        randomize_initial_episode_phase=original.get("randomize_initial_episode_phase", True),
        nominal_fraction=original.get("nominal_fraction", 0.5),
        payload_enabled=original.get("payload", False),
        payload_body_name=original.get("payload_body_name", "right_wrist_yaw_link"),
        payload_mass_range_kg=tuple(original.get("payload_mass_range_kg", (1.0, 3.0))),
        payload_position_body_m=tuple(original.get("payload_position_body_m", (.12, 0., 0.))),
        payload_size_m=tuple(original.get("payload_size_m", (.10, .08, .08))),
        limb_payload_only=original.get("limb_payload_only", False),
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(json.dumps({"event": "initializing", "config": asdict(config)}), flush=True)
    started = time.monotonic()
    with FixedDRTrackerRollout(config) as rollout:
        tracker_hash = _sha256(rollout.checkpoint_path)
        if frozen.tracker_sha256 and tracker_hash != frozen.tracker_sha256:
            raise ValueError("Tracker differs from encoder training tracker")
        context = DynamicsContextInference(
            frozen, num_envs=args.num_envs, device="cuda:0", use_bfloat16=True
        )
        lengths = rollout.motion_command.motion.file_lengths
        metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": frozen.sha256,
            "checkpoint_update": 8000 if checkpoint.name == "update_008000.pt" else None,
            "tracker_sha256": tracker_hash,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "rollout_config": asdict(config),
            "rollout_metadata": rollout.metadata,
            "motion_files": list(rollout.motion_files),
            "motion_lengths": lengths.cpu().tolist(),
            "context_history_steps": context.history_steps,
            "contract": "Frozen checkpoint/normalization; original nominal/startup-DR distribution; "
            "new seed; no payload for v12; full causal histories only; reset and motion boundaries "
            "clear history; main windows spaced at least context length; +5 windows are local controls.",
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows = []
        coverage = []
        transitions_started = time.monotonic()
        numerical_check = None
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                batch = rollout.step(predictor_only=True)
                context.append(batch["robot_state"], batch["joint_target"], batch["reset_boundary"])
                offset = step % args.stride
                if step >= context.history_steps and offset in (0, args.neighbor_offset):
                    full = context.history_valid.all(dim=0)
                    current_state = batch["next_robot_state"]
                    latent = context.encode(current_state)
                    if numerical_check is None and int(full.sum()) >= 1:
                        context.use_bfloat16 = False
                        fp32 = context.encode(current_state)
                        context.use_bfloat16 = True
                        numerical_check = {
                            "bf16_fp32_full_context_rms_difference": float(
                                (latent[full] - fp32[full]).square().mean().sqrt()
                            ),
                            "bf16_fp32_full_context_mean_cosine": float(
                                torch.nn.functional.cosine_similarity(latent[full], fp32[full]).mean()
                            ),
                        }
                    motion_ids = rollout.motion_command.motion_idx
                    motion_steps = rollout.motion_command.time_steps
                    row = {
                        "latent": latent[full].cpu().numpy(),
                        "state": current_state[full].cpu().numpy(),
                        "world": rollout.world_ids[full].cpu().numpy(),
                        "episode": rollout.episode_ids[full].cpu().numpy(),
                        "nominal": rollout.is_nominal[full].cpu().numpy(),
                        "motion": motion_ids[full].cpu().numpy(),
                        "motion_step": motion_steps[full].cpu().numpy(),
                        "phase": (motion_steps[full] / (lengths[motion_ids[full]] - 1).clamp_min(1)).cpu().numpy(),
                        "step": np.full(int(full.sum()), step, dtype=np.int64),
                        "neighbor": np.full(int(full.sum()), offset != 0, dtype=bool),
                    }
                    if not np.isfinite(row["latent"]).all():
                        raise ValueError("Nonfinite encoded latent")
                    rows.append(row)
                    coverage.append({
                        "step": step,
                        "nominal_full": int((full & rollout.is_nominal).sum()),
                        "dr_full": int((full & ~rollout.is_nominal).sum()),
                    })
                if step % 100 == 0:
                    print(json.dumps({
                        "event": "collection", "step": step, "total_steps": args.steps,
                        "seconds": time.monotonic() - transitions_started,
                        "samples": sum(len(row["world"]) for row in rows),
                        **context.metrics,
                    }), flush=True)
                if step % 500 == 0 or step == args.steps:
                    if rows:
                        np.savez_compressed(args.output / "latents.npz", **{
                            key: np.concatenate([row[key] for row in rows]) for key in rows[0]
                        })
        rollout._assert_fixed_dr()
        np.savez_compressed(
            args.output / "physics.npz", values=rollout.privileged_dynamics.cpu().numpy(),
            names=np.asarray(rollout.privileged_dynamics_names),
        )
        metadata.update({
            "coverage": coverage, "numerical_check": numerical_check,
            "elapsed_seconds": time.monotonic() - started,
            "collection_seconds": time.monotonic() - transitions_started,
            "rollout_final_metadata": rollout.metadata,
            "complete": True,
        })
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"event": "complete", "output": str(args.output),
                          "seconds": metadata["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
