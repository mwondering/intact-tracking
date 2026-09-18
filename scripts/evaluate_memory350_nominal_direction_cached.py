"""Compare fixed checkpoints on cached, disjoint cross-motion histories using CPU."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_forward_context_clusters import normalized
from analyze_memory350 import sha256
from evaluate_memory350_weak_pairs import encode, environment_readout, write_json
from intact_tracking.memory350_inference import load_memory350_checkpoint


def selected_query(query, mask):
    count = len(query["world"])
    index = torch.from_numpy(np.flatnonzero(mask))
    return {key: value[index] if isinstance(value, torch.Tensor) and value.ndim and len(value) == count else value
            for key, value in query.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("common", "memory_training"), default=["memory_training"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = {"baseline": args.reference.resolve(), "memory350": args.candidate.resolve()}
    if any(path.name == "last.pt" for path in paths.values()):
        raise ValueError("Use immutable update checkpoints for paired evaluation")
    states = {name: torch.load(path, map_location="cpu", weights_only=False, mmap=True) for name, path in paths.items()}
    for field in ("model_config", "normalization", "tracker", "nominal_a_fraction", "supervision_horizons"):
        if states["baseline"][field] != states["memory350"][field]:
            raise ValueError(f"Candidate changed comparison contract: {field}")
    if states["memory350"]["loss_config"].get("nominal_direction_objective_version") != 1:
        raise ValueError("Expected a nominal-direction candidate")
    models = {name: load_memory350_checkpoint(path, device="cpu") for name, path in paths.items()}
    model_fields = {"baseline": "latent_baseline", "memory350": "latent_candidate"}
    summary = {
        "complete": False, "device": "cpu", "precision": "float32", "threads": args.threads,
        "reference_update": states["baseline"]["update"], "candidate_update": states["memory350"]["update"],
        "checkpoints": {name: {"path": str(path), "sha256": models[name].sha256,
                               "update": states[name]["update"]} for name, path in paths.items()},
        "same_model_config": True, "same_normalization": True, "same_queries": True,
        "profiles": {}, "script_sha256": sha256(Path(__file__)),
        "interpretation": "Known-environment identification on different motion families and disjoint full histories; not unseen-environment generalization or a matched-step causal ablation.",
    }
    write_json(output / "summary.json", summary)
    for profile in args.profiles:
        source = args.cache.resolve() / profile
        metadata = json.loads((source / "metadata.json").read_text())
        if not metadata["complete"] or not metadata["arguments"]["save_history"]:
            raise ValueError("Cached rollout is incomplete or has no saved raw histories")
        with np.load(source / "latents.npz") as cached:
            all_data = {key: cached[key] for key in cached.files if not key.startswith("latent")}
            calibration_latent = cached["latent"].copy()
        full = (all_data["short_steps"] == 50) & (all_data["long_chunks"] == 30)
        data = {key: value[full] for key, value in all_data.items()}
        data["source_row"] = np.flatnonzero(full)
        calibration = load_memory350_checkpoint(metadata["checkpoint"], device="cpu")
        if calibration.sha256 != metadata["checkpoint_sha256"]:
            raise ValueError("Original cache encoder checksum changed")
        parts, row_parts, query_records, checks = {name: [] for name in models}, [], [], []
        queries = sorted((source / "queries").glob("query_*.pt"))
        for number, path in enumerate(queries):
            query = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if query["format_version"] != "memory350_raw_query_v1":
                raise ValueError("Unexpected query format")
            rows = np.flatnonzero(all_data["step"] == query["step"])
            assert np.array_equal(query["world"].numpy(), all_data["world"][rows])
            assert np.array_equal(query["short_valid"].sum(-1).numpy(), all_data["short_steps"][rows])
            assert np.array_equal(query["long_valid"].sum(-1).numpy(), all_data["long_chunks"][rows])
            mask = full[rows]
            if not mask.any():
                continue
            query = selected_query(query, mask)
            selected_rows = rows[mask]
            if not checks or number == len(queries) - 1:
                audit = selected_query(query, np.arange(len(query["world"])) < 128)
                restored = encode(calibration, audit, bf16=False)
                expected = calibration_latent[selected_rows[:len(restored)]]
                cosine = float((normalized(restored) * normalized(expected)).sum(-1).mean())
                assert cosine > .9999, "CPU raw-query reconstruction differs from the saved cache"
                checks.append({"step": query["step"], "samples": len(restored), "cpu_saved_latent_cosine": cosine})
            for name, model in models.items():
                value = encode(model, query, bf16=False)
                assert np.isfinite(value).all()
                parts[name].append(value)
            row_parts.append(selected_rows)
            query_records.append({"path": str(path), "sha256": sha256(path),
                                  "step": query["step"], "full_history_samples": len(selected_rows)})
            if number % 8 == 0 or number == len(queries) - 1:
                print(json.dumps({"event": "encoded", "profile": profile, "query": number + 1}), flush=True)
        assert np.array_equal(np.concatenate(row_parts), data["source_row"])
        data.update({model_fields[name]: np.concatenate(values) for name, values in parts.items()})
        dr = {key: value[~data["nominal"]] for key, value in data.items()}
        meta = {"models": model_fields, "motion_files": metadata["motion_files"]}
        readout = environment_readout(dr, meta, disjoint=True)
        folder = output / profile
        folder.mkdir()
        np.savez_compressed(folder / "full_history_latents.npz", **data)
        summary["profiles"][profile] = {
            "full_history_samples": int(full.sum()), "dr_samples": len(dr["world"]),
            "source_metadata_sha256": sha256(source / "metadata.json"),
            "source_latents_sha256": sha256(source / "latents.npz"),
            "query_order_and_masks_verified": True, "cache_reconstruction": checks,
            "queries": query_records, "readout": readout,
        }
        write_json(output / "summary.json", summary)
    summary.update(complete=True, elapsed_seconds=time.monotonic() - started)
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "summary": str(output / "summary.json"),
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
