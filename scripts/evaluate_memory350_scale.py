"""Compare original and larger Memory350 at equal updates on identical held-out windows."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_memory350 import cluster_ci, error_metrics, sha256
from intact_tracking.forward_predictor_objective import _normalized_state_error
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_scale_model import parameter_counts


ROOT = Path(__file__).resolve().parents[1]


def summarize(samples):
    a, b = samples["original_mse"], samples["encoder2x_mse"]
    denominator = samples["unchanged_mse"]
    short, long = samples["short_steps"], samples["long_chunks"]
    masks = {
        "all": np.ones(len(a), dtype=bool), "long_available": long > 0,
        "short_incomplete_with_long": (short < 50) & (long > 0),
        "short_under10_with_long": (short < 10) & (long > 0),
        "short_full_with_long": (short == 50) & (long > 0),
        "no_long": long == 0, "full_30_chunks": long == 30,
    }
    if 'is_nominal' in samples:
        nominal = samples['is_nominal'].astype(bool)
        masks.update(nominal=nominal, dr=~nominal,
                     dr_full_30_chunks=(~nominal) & (long == 30),
                     nominal_full_30_chunks=nominal & (long == 30))
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        ratio = float(b[mask, -1].sum() / a[mask, -1].sum())
        result[name] = {
            "samples": int(mask.sum()), "worlds": len(np.unique(samples["world_id"][mask])),
            "original_five_step_nmse": float(a[mask, -1].sum() / denominator[mask, -1].sum()),
            "encoder2x_five_step_nmse": float(b[mask, -1].sum() / denominator[mask, -1].sum()),
            "encoder2x_to_original_error_ratio": ratio,
            "error_reduction_percent": 100 * (1 - ratio),
            "paired_world_bootstrap_ratio_ci95": cluster_ci(samples["world_id"], a[:, -1], b[:, -1], mask),
            "fraction_windows_encoder2x_better": float((b[mask, -1] < a[mask, -1]).mean()),
            "original_nmse_by_horizon": (a[mask].sum(0) / denominator[mask].sum(0)).tolist(),
            "encoder2x_nmse_by_horizon": (b[mask].sum(0) / denominator[mask].sum(0)).tolist(),
        }
    return result


def run(args):
    torch.set_num_threads(4)
    started = time.time()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Evaluation outputs must remain inside the current project")
    output.mkdir(parents=True, exist_ok=True)
    paths = {"original": Path(args.reference_checkpoint), "encoder2x": Path(args.scaled_checkpoint)}
    states = {name: torch.load(path, map_location="cpu", weights_only=False, mmap=True)
              for name, path in paths.items()}
    if len({value["update"] for value in states.values()}) != 1:
        raise ValueError("Compare checkpoints at the same completed update")
    mixture = states['original'].get('nominal_a_fraction', 0.)
    if states['encoder2x'].get('nominal_a_fraction', 0.) != mixture:
        raise ValueError('Comparison mismatch: training nominal A fraction')
    for field in ("optimizer_steps", "normalization", "loss_config", "tracker"):
        if states["original"][field] != states["encoder2x"][field]:
            raise ValueError(f"Comparison mismatch: {field}")
    expected = {**states["original"]["model_config"], "chunk_depth": 2, "memory_depth": 4, "context_depth": 4}
    if states["encoder2x"]["model_config"] != expected:
        raise ValueError("The comparison must change only encoder attention depths")
    device = torch.device(args.device)
    models = {}
    for name, state in states.items():
        model = Memory350Predictor(Memory350Config(**state["model_config"]))
        model.load_state_dict(state["model"], strict=True)
        models[name] = model.to(device).eval().requires_grad_(False)
    parts, hashes = [], {}
    for rank in range(4):
        filename = f"validation_broad_rank_{rank}.pt"
        path = paths["original"].parent / filename
        hashes[filename] = sha256(path)
        if sha256(paths["encoder2x"].parent / filename) != hashes[filename]:
            raise ValueError("Paired evaluation windows are not identical")
        full = torch.load(path, map_location="cpu", weights_only=False)
        count = len(full["state"])
        for start in range(0, count, args.batch_size):
            # Normalization vectors are not indexed along the sample axis.
            statistics = {"state_mean", "state_std", "action_mean", "action_std", "delta_mean", "delta_std"}
            batch = {key: (value[start:start + args.batch_size] if key not in statistics and value.ndim > 0
                           and value.shape[0] == count else value).to(device)
                     for key, value in full.items()}
            part = {
                "rank": np.full(len(batch["state"]), rank),
                "world_id": batch["world_id"].cpu().numpy(),
                "is_nominal": batch["is_nominal"].cpu().numpy(),
                "short_steps": batch["history_valid"].sum(1).cpu().numpy(),
                "long_chunks": batch["memory_valid"].sum(1).cpu().numpy(),
            }
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), precision:
                target = batch["state"][:, 1:]
                unchanged = batch["state"][:, :1].expand_as(target)
                error = _normalized_state_error(unchanged, target, batch["state_mean"],
                                                batch["state_std"], batch["delta_std"])
                part["unchanged_mse"] = error.float().square().mean(-1).cpu().numpy()
                for name, model in models.items():
                    latent = model.context_encoder(batch["history_state"], batch["history_action"],
                        batch["history_next_state"], batch["history_valid"],
                        batch["memory_interactions"], batch["memory_valid"])
                    part[name + "_mse"] = error_metrics(model, batch, latent)
            parts.append(part)
    samples = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if not all(np.isfinite(value).all() for value in samples.values()):
        raise RuntimeError("Non-finite prediction evaluation")
    np.savez_compressed(output / "paired_samples.npz", **samples)
    result = {
        "update": states["original"]["update"], "optimizer_steps": states["original"]["optimizer_steps"],
        "nominal_a_fraction": mixture,
        "checkpoints": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
        "parameter_counts": {name: parameter_counts(model) for name, model in models.items()},
        "validation_sha256": hashes, "same_normalization": True, "same_validation": True,
        "groups": summarize(samples), "device": str(device),
        "precision": "bfloat16 autocast" if device.type == "cuda" else "float32",
        "rank_mean_nmse": {}, "elapsed_seconds": time.time() - started,
        "uncertainty_scope": "paired validation-world bootstrap for one training seed; not across training seeds",
    }
    for name in models:
        result["rank_mean_nmse"][name] = float(np.mean([
            samples[name + "_mse"][samples["rank"] == rank, -1].sum()
            / samples["unchanged_mse"][samples["rank"] == rank, -1].sum() for rank in range(4)
        ]))
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.wandb:
        import wandb
        run = wandb.init(project="intact-forward-predictor", entity="2486344338-zhejiang-university",
            group=args.group, name=f'memory350_encoder2x_vs_original_u{result["update"]:06d}', dir=str(output),
            config={"update": result["update"], "checkpoints": result["checkpoints"],
                    "parameter_counts": result["parameter_counts"], "same_validation": True})
        payload = {"update": result["update"]}
        for group, values in result["groups"].items():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    payload[f"{group}/{key}"] = value
            low, high = values["paired_world_bootstrap_ratio_ci95"]
            payload[f"{group}/error_ratio_ci95_low"] = low
            payload[f"{group}/error_ratio_ci95_high"] = high
        run.log(payload)
        (output / "wandb_run.json").write_text(json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n")
        run.finish()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--scaled-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--group", default="memory350-encoder2x-paired-evaluation")
    run(parser.parse_args())
