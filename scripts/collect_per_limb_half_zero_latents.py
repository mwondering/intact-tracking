"""Collect 50 seconds of mjwarp interaction with independently mixed nominal DR."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from intact_tracking import limb_context_protocol
from intact_tracking.memory350_inference import Memory350Inference, load_memory350_checkpoint
from intact_tracking.memory350_nominal_rollout import (
    NominalMemory350RolloutConfig, NominalMemory350TrackerRollout,
)
from intact_tracking.memory350_rollout import Memory350TrackerRollout
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.rollout.online import FixedDRRolloutConfig, _capture_privileged_dynamics_targets


SOURCE = Path("runs/limb_context_20260916_dr16384")
PARAMETERS = Path("runs/limb_context_20260917_per_limb_half_zero_parameters/parameters.npz")
REFERENCE = Path("runs/limb_context_20260917_response10_environment_bins")
OUTPUT = Path("runs/limb_context_20260917_per_limb_half_zero_50s")
STEPS = 2500


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, choices=range(8))
    parser.add_argument("--sampling", choices=("per-limb-half-zero", "all-fixed-half-nominal"),
                        default="per-limb-half-zero")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    all_fixed = args.sampling == "all-fixed-half-nominal"
    parameters = (Path("runs/limb_context_20260917_all_fixed_dr_half_nominal/parameter_bank.npz")
                  if all_fixed else PARAMETERS)
    if args.output is None:
        args.output = (Path("runs/limb_context_20260917_all_fixed_dr_half_nominal_50s")
                       if all_fixed else OUTPUT)
    torch.set_num_threads(2)
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_float32_matmul_precision("high")
    out = args.output / f"shard_{args.shard:02d}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_path = SOURCE / f"shard_{args.shard:02d}"
    baseline = json.loads((source_path / "metadata.json").read_text())
    encoding = json.loads((REFERENCE / "encoding_manifest.json").read_text())
    assert baseline["complete"]
    config_type = NominalMemory350RolloutConfig if baseline["nominal_worlds"] else FixedDRRolloutConfig
    rollout_type = NominalMemory350TrackerRollout if baseline["nominal_worlds"] else Memory350TrackerRollout
    config_values = dict(baseline["rollout_config"])
    if all_fixed:
        config_values["dr_nominal_probability"] = .5
    config = config_type(**config_values)
    count = config.num_envs
    with np.load(source_path / "physics.npz") as old:
        worlds, original, nominal = old["world"], old["values"], old["nominal"]
        names = old["names"]
    dr = ~nominal
    desired = original.copy()
    zero_mask = np.zeros((count, 4), dtype=bool)
    with np.load(parameters) as saved:
        ids = np.searchsorted(saved["world"], worlds[dr])
        np.testing.assert_array_equal(saved["world"][ids], worlds[dr])
        np.testing.assert_array_equal(saved["original"][ids], original[dr])
        desired[dr] = saved["mixed" if all_fixed else "per_limb_half_zero"][ids]
        zero_mask[dr] = saved["nominal_mask"][ids, 34:38] if all_fixed else saved["zero_payload_mask"][ids]
        expected_bias = saved["encoder_bias"][ids] if all_fixed else None
    assert dr.sum() == 2048
    checkpoint = load_memory350_checkpoint(encoding["checkpoint_path"], device="cuda:0")
    assert checkpoint.sha256 == encoding["checkpoint_sha256"]
    sampler = limb_context_protocol.sample_limb_masses
    calls = []

    def per_limb_half_zero(num_envs, seed, fixed_masses=None, *, max_masses_kg=(4., 4., 4., 4.)):
        assert num_envs == count and fixed_masses is None
        masses = sampler(num_envs, seed, fixed_masses, max_masses_kg=max_masses_kg)
        calls.append(masses.cpu().numpy().copy())
        masses[torch.from_numpy(zero_mask)] = 0
        return masses

    print(json.dumps({"event": "initializing", "shard": args.shard,
                      "device": torch.cuda.get_device_name(), "config": asdict(config)}), flush=True)
    # The all-fixed variant uses the production startup mixture; the earlier
    # payload-only variant retains its archived per-limb mask for reproducibility.
    with (nullcontext() if all_fixed else
          patch.object(limb_context_protocol, "sample_limb_masses", per_limb_half_zero)):
        rollout = rollout_type(config)
    with rollout:
        assert len(calls) == (0 if all_fixed else 1) and abs(rollout.env.step_dt - 0.02) < 1e-9
        assert checkpoint.tracker_sha256 == _sha256(rollout.checkpoint_path)
        assert list(rollout.motion_files) == baseline["motion_files"]
        np.testing.assert_array_equal(names, rollout.privileged_dynamics_names)
        physics = rollout.privileged_dynamics.cpu().numpy().copy()
        np.testing.assert_array_equal(rollout.is_nominal.cpu().numpy(), nominal)
        np.testing.assert_array_equal(physics, desired)
        if not all_fixed:
            np.testing.assert_allclose(calls[0][dr], original[dr, 34:38], rtol=0, atol=1e-6)
        bias_start = rollout.env.scene["robot"].data.encoder_bias.clone()
        if all_fixed:
            np.testing.assert_array_equal(bias_start.cpu().numpy()[dr], expected_bias)
        with np.load(REFERENCE / "environment_assignments.npz") as old:
            center, edges = old["nominal_center"], old["edges_16"][::2]
        rollout_metadata = rollout.metadata
        sampling = ("Each fixed scalar DR parameter independently takes nominal with probability .5, "
                    "otherwise its original draw; shared foot friction retained. Nominal controls unchanged."
                    if all_fixed else
                    "Each DR limb independently has 50% zero payload, otherwise its original uniform draw. Nominal controls unchanged.")
        rollout_metadata["collector_sampling"] = sampling
        metadata = {
            "complete": False, "shard": args.shard, "seed": config.seed,
            "checkpoint": checkpoint.path, "checkpoint_sha256": checkpoint.sha256,
            "tracker_sha256": checkpoint.tracker_sha256,
            "source_metadata_sha256": _sha256(source_path / "metadata.json"),
            "parameters": str(parameters), "parameters_sha256": _sha256(parameters),
            "script_sha256": _sha256(Path(__file__)), "rollout_config": asdict(config),
            "rollout_metadata": rollout_metadata, "motion_files": list(rollout.motion_files),
            "dr_worlds": int(dr.sum()), "nominal_worlds": int(nominal.sum()),
            "steps": STEPS, "step_dt": rollout.env.step_dt, "seconds_per_world": 50,
            "sampling_mode": args.sampling,
            "protocol": "Frozen original tracker, original observation noise/pulses and motion sampler. "
                        + sampling + " All DR parameters fixed throughout. 2500 steps at dt=.02; "
                        "queries every100 steps, only mature short50+long30 histories enter centers. "
                        "Original 1000-step episode cap with motion resets. No network training.",
            "parameters_match_saved_bank_exactly": True,
            "background_DR_matches_original_exactly": not all_fixed,
            "precision": "Encoder FP32 without TF32; original rollout backend settings restored after inference.",
        }
        write_json(out / "metadata.json", metadata)
        np.savez_compressed(out / "physics.npz", world=worlds, physics=physics,
                            nominal=nominal, names=names, zero_payload_mask=zero_mask)
        context = Memory350Inference(checkpoint, count, batch_size=256, use_bfloat16=False)
        records = []
        for step in range(1, STEPS + 1):
            context.append(rollout.step(predictor_only=True))
            if step % 50 == 0:
                write_json(out / "progress.json", {"step": step, "total_steps": STEPS,
                           "elapsed_seconds": time.monotonic() - started, "complete": False})
            if step % 100:
                continue
            memory = context.memory
            chunks = (memory.total_chunks - memory.session_start).clamp_max(30)
            valid = ((memory.short_count == 50) & (chunks == 30)).cpu().numpy()
            previous_matmul = torch.backends.cuda.matmul.allow_tf32
            previous_cudnn = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                z = context.encode().cpu().numpy()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul
                torch.backends.cudnn.allow_tf32 = previous_cudnn
            assert np.isfinite(z).all()
            records.append({"z": z, "valid": valid,
                "motion": rollout.motion_command.motion_idx.cpu().numpy().copy(),
                "motion_step": rollout.motion_command.time_steps.cpu().numpy().copy(),
                "episode": rollout.episode_ids.cpu().numpy().copy(),
                "short_steps": memory.short_count.cpu().numpy().copy(),
                "long_chunks": chunks.cpu().numpy().copy()})
            if step in (500, STEPS):
                # Small raw history sample for independent numerical replay, all worlds' z are saved.
                ids = memory._worlds[:32]
                short, short_valid = memory.ordered_short(ids)
                long, long_valid = memory.read_chunks(ids)
                torch.save({"step": step, "world": torch.from_numpy(worlds[:32]),
                            "short": short.cpu(), "short_valid": short_valid.cpu(),
                            "long": long.cpu(), "long_valid": long_valid.cpu(),
                            "z": torch.from_numpy(z[:32])}, out / f"encoder_audit_{step:06d}.pt")
            if step % 500 == 0 or step == STEPS:
                data = {k: np.stack([r[k] for r in records], axis=1) for k in records[0]}
                np.savez_compressed(out / "queries.npz", world=worlds,
                                    step=np.arange(100, step + 1, 100), **data)
            print(json.dumps({"event": "query", "shard": args.shard, "step": step,
                              "valid": int(valid.sum()), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        rollout._assert_fixed_dr()
        np.testing.assert_array_equal(physics, _capture_privileged_dynamics_targets(rollout.env).values.cpu().numpy())
        torch.testing.assert_close(bias_start, rollout.env.scene["robot"].data.encoder_bias, rtol=0, atol=0)
        assert context.memory.parameter_invalidations == 0
        full_count = data["valid"].sum(axis=1)
        assert full_count.min() >= 10
        unit = data["z"].astype(np.float64)
        unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
        unit[~data["valid"]] = np.nan
        centroid = np.nanmean(unit, axis=1)
        mean_radius = np.nanmean(np.linalg.norm(unit - center, axis=-1), axis=1)
        centroid_radius = np.linalg.norm(centroid - center, axis=1)
        np.savez_compressed(out / "environment_assignments.npz", world=worlds,
                            physics=physics, nominal=nominal, full_query_count=full_count,
                            mean_radius=mean_radius, centroid=centroid, centroid_radius=centroid_radius,
                            edges=edges, nominal_center=center)
        metadata.update(complete=True, elapsed_seconds=time.monotonic() - started,
                        full_queries=int(full_count[dr].sum()), minimum_queries=int(full_count[dr].min()),
                        maximum_queries=int(full_count[dr].max()), physics_unchanged=True,
                        encoder_bias_unchanged=True, memory_metrics=context.memory.metrics(),
                        physics_sha256=_sha256(out / "physics.npz"), queries_sha256=_sha256(out / "queries.npz"),
                        assignments_sha256=_sha256(out / "environment_assignments.npz"))
        write_json(out / "metadata.json", metadata)
        write_json(out / "progress.json", {"step": STEPS, "total_steps": STEPS,
                   "elapsed_seconds": time.monotonic() - started, "complete": True})
        print(json.dumps({"event": "complete", "shard": args.shard,
                          "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
