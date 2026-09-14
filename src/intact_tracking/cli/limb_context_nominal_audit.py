"""Independent zero-payload A/B consistency audit, never used as training data."""

import argparse
import json
from pathlib import Path

import torch

from intact_tracking.limb_context_protocol import TRACKER, PROJECT_ROOT
from intact_tracking.rollout import FixedDRRolloutConfig, FixedDRTrackerRollout, NominalPairRollout, NominalPairRolloutConfig


def run(args):
    output = Path(args.output).resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must remain inside the project")
    a = FixedDRTrackerRollout(FixedDRRolloutConfig(
        checkpoint_file=TRACKER, motion_file=args.motion_file, num_envs=args.num_envs, seed=77199,
        device=args.device, limb_payload_only=True, limb_fixed_masses=(0, 0, 0, 0)))
    b = None
    try:
        b = NominalPairRollout(NominalPairRolloutConfig(
            checkpoint_file=TRACKER, motion_file=args.motion_file, num_envs=args.num_envs,
            device=args.device, seed=99171, horizon=5))
        errors, diagnostics = [], []
        for block in range(30):
            steps = [a.step(predictor_only=True) for _ in range(5)]
            with torch.inference_mode():
                states, row = b.rollout_joint_targets(steps[0]["robot_state"],
                    torch.stack([s["joint_target"] for s in steps], 1))
            valid = ~torch.stack([s["reset_boundary"] for s in steps], 1).any(1)
            if valid.any():
                errors.append((torch.stack([s["next_robot_state"] for s in steps], 1) - states)[valid].cpu())
            diagnostics.append(row)
        error = torch.cat(errors).abs()
        components = {"root_position_m": slice(0, 3), "quaternion": slice(3, 7),
                      "root_linear_velocity": slice(7, 10), "root_angular_velocity": slice(10, 13),
                      "joint_position_rad": slice(13, 42), "joint_velocity": slice(42, 71)}
        result = {"protocol": "zero-payload nominal A minus action-matched nominal B",
                  "training_data": False, "valid_five_step_windows": len(error),
                  "a_physics": a.payload_configuration, "b_repeat": b.metadata.get("repeat_diagnostics"),
                  "components": {name: {"rms": float(error[..., index].square().mean().sqrt()),
                                          "max": float(error[..., index].max())}
                                 for name, index in components.items()},
                  "restore_max_abs_error": max(row["restore_max_abs_error"] for row in diagnostics)}
        # Numerical solver warm-start differences can exist; the pose audit must
        # remain much smaller than the motion-tracking errors being measured.
        result["passed"] = (result["components"]["root_position_m"]["rms"] < 1e-3
                            and result["components"]["joint_position_rad"]["rms"] < 1e-3
                            and result["restore_max_abs_error"] <= 1e-5)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("Nominal A/B consistency audit needs investigation before formal training")
    finally:
        if b is not None:
            b.close()
        a.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
