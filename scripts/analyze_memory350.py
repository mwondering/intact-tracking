"""Paired, frozen-checkpoint memory interventions on saved held-out contexts.

No optimizer, simulator, training restart, or training-file mutation is performed.
Removal is an inference-time intervention, not a separately trained no-memory model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.forward_predictor_objective import _normalized_state_error


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cross_world_memory(batch, seed):
    """Change content while preserving each recipient's exact chunk padding mask."""
    counts = batch["memory_valid"].sum(1).cpu().numpy()
    worlds = batch["world_id"].cpu().numpy()
    rng = np.random.default_rng(seed)
    other = torch.zeros_like(batch["memory_interactions"])
    matched = np.ones(len(counts), dtype=bool)
    exact_count = np.ones(len(counts), dtype=bool)
    donors = np.arange(len(counts))
    for i, count in enumerate(counts):
        if not count:
            continue
        options = np.flatnonzero((worlds != worlds[i]) & (counts == count))
        if not len(options):
            options = np.flatnonzero((worlds != worlds[i]) & (counts >= count))
            exact_count[i] = False
        if not len(options):
            matched[i] = False
            other[i] = batch["memory_interactions"][i]
            continue
        donor = int(rng.choice(options))
        donors[i] = donor
        source = batch["memory_interactions"][donor, batch["memory_valid"][donor]][-count:]
        other[i, batch["memory_valid"][i]] = source
    return other, matched, exact_count, donors


def error_metrics(model, batch, latent):
    prediction = model.rollout(
        batch["state"][:, 0], batch["action"], batch["foot"][:, 0],
        batch["contact_force"][:, 0], batch["contact_binary"][:, 0],
        **{k: batch[k] for k in (
            "history_state", "history_action", "history_foot", "history_contact_force",
            "history_contact_binary", "history_valid", "state_mean", "state_std",
            "delta_mean", "delta_std")}, dynamics_latent=latent,
    )[0]
    error = _normalized_state_error(
        prediction, batch["state"][:, 1:], batch["state_mean"],
        batch["state_std"], batch["delta_std"],
    ).float()
    return error.square().mean(-1).cpu().numpy()


def encode(model, batch, memory, mask):
    return model.context_encoder(
        batch["history_state"], batch["history_action"], batch["history_next_state"],
        batch["history_valid"], memory, mask,
    )


def cluster_ci(world, natural, alternative, mask, seed=350, repeats=2000):
    """World-cluster bootstrap of paired mean-error ratios on a fixed context set."""
    if not np.any(mask):
        return None
    labels, inverse = np.unique(world[mask], return_inverse=True)
    a = np.bincount(inverse, weights=natural[mask], minlength=len(labels))
    b = np.bincount(inverse, weights=alternative[mask], minlength=len(labels))
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(labels), size=(repeats, len(labels)))
    ratios = b[sample].sum(1) / np.maximum(a[sample].sum(1), 1e-12)
    return np.quantile(ratios, [.025, .975]).tolist()


def summarize(samples):
    worlds = samples["world_id"]
    short = samples["short_steps"]
    chunks = samples["long_chunks"]
    natural = samples["natural_mse"][:, -1]
    unchanged = samples["unchanged_mse"][:, -1]
    conditions = sorted(k[:-4] for k in samples if k.endswith("_mse") and k != "unchanged_mse")
    groups = {
        "all": np.ones(len(short), dtype=bool),
        "long_available": chunks > 0,
        "short_incomplete_with_long": (short < 50) & (chunks > 0),
        "short_full_with_long": (short == 50) & (chunks > 0),
        "short_0_9_with_long": (short < 10) & (chunks > 0),
        "short_10_24_with_long": (short >= 10) & (short < 25) & (chunks > 0),
        "short_25_49_with_long": (short >= 25) & (short < 50) & (chunks > 0),
        "no_long_control": chunks == 0,
        "full_30_chunks": chunks == 30,
    }
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        entry = {"samples": int(mask.sum()), "worlds": len(np.unique(worlds[mask])),
                 "short_steps_mean": float(short[mask].mean()),
                 "long_chunks_mean": float(chunks[mask].mean()), "conditions": {}}
        for condition in conditions:
            errors = samples[condition + "_mse"][:, -1]
            entry["conditions"][condition] = {
                "pooled_five_step_nmse": float(errors[mask].sum() / unchanged[mask].sum()),
                "error_ratio_vs_natural": float(errors[mask].sum() / natural[mask].sum()),
                "paired_world_bootstrap_ratio_ci95": cluster_ci(worlds, natural, errors, mask),
                "fraction_samples_natural_better": float((errors[mask] > natural[mask]).mean()),
                "latent_cosine_to_natural_mean": float(samples[condition + "_latent_cosine"][mask].mean()),
            }
        result[name] = entry
    rank_nmse = []
    for rank in np.unique(samples["rank"]):
        m = samples["rank"] == rank
        rank_nmse.append(float(natural[m].sum() / unchanged[m].sum()))
    return {"coverage": {
        "samples": len(short), "worlds": len(np.unique(worlds)),
        "motions": len(np.unique(samples["motion_id"])),
        "short_full_fraction": float((short == 50).mean()),
        "long_available_fraction": float((chunks > 0).mean()),
        "full_30_chunks_fraction": float((chunks == 30).mean()),
        "long_chunks_mean": float(chunks.mean()),
        "cross_world_donor_matched_fraction": float(samples["donor_matched"].mean()),
        "cross_world_donor_exact_chunk_count_fraction": float(samples["donor_exact_count"].mean()),
    }, "natural_five_step_nmse_by_rank": rank_nmse,
        "natural_five_step_nmse_rank_mean": float(np.mean(rank_nmse)), "groups": result}


def run(args):
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.name.startswith("update_"), "Pin an immutable numbered checkpoint"
    started = time.time()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model = Memory350Predictor(Memory350Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device).eval().requires_grad_(False)
    result = {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
              "update": checkpoint["update"], "optimizer_steps": checkpoint["optimizer_steps"],
              "method": "same weights, same held-out inputs/targets, only long memory or latent intervened",
              "precision": "bfloat16 autocast", "device": args.device,
              "cross_world_shuffle": "different world, same number of newest chunks; recipient mask unchanged",
              "ci": "paired bootstrap resampling world IDs, 2000 draws; fixed validation contexts",
              "limitations": ["Inference-time removal is not a trained no-memory control.",
                              "Saved validation is fixed and collected during startup, not a fresh full-dataset evaluation.",
                              "Cross-world swaps also change motion history; they do not isolate physics-only content.",
                              "Short<50 plus nonempty long memory identifies memory carried across a short-history reset."],
              "suites": {}, "input_sha256": {}}
    for suite in args.suites:
        pieces = []
        for rank in range(4):
            prefix = "validation_broad" if suite == "broad" else "validation"
            path = checkpoint_path.parent / f"{prefix}_rank_{rank}.pt"
            result["input_sha256"][path.name] = sha256(path)
            batch = {k: v.to(args.device) for k, v in torch.load(path, map_location="cpu", weights_only=False).items()}
            size = batch["state"].shape[0]
            data = {"world_id": batch["world_id"].cpu().numpy(),
                    "motion_id": batch["motion_id"].cpu().numpy(),
                    "rank": np.full(size, rank),
                    "short_steps": batch["history_valid"].sum(1).cpu().numpy(),
                    "long_chunks": batch["memory_valid"].sum(1).cpu().numpy()}
            shuffled, matched, exact, donors = cross_world_memory(batch, 350 + rank)
            data.update(donor_matched=matched, donor_exact_count=exact, donor_index=donors)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latent = encode(model, batch, batch["memory_interactions"], batch["memory_valid"])
                no_change = batch["state"][:, :1].expand_as(batch["state"][:, 1:])
                error = _normalized_state_error(no_change, batch["state"][:, 1:], batch["state_mean"],
                                                batch["state_std"], batch["delta_std"])
                data["unchanged_mse"] = error.float().square().mean(-1).cpu().numpy()
                conditions = ["natural", "remove_long", "shuffle_long", "latest_1", "latest_5",
                              "latest_10", "latest_20", "shuffle_latent", "zero_latent"]
                for condition in conditions:
                    if condition == "natural":
                        z = latent
                    elif condition == "remove_long":
                        z = encode(model, batch, batch["memory_interactions"], torch.zeros_like(batch["memory_valid"]))
                    elif condition == "shuffle_long":
                        z = encode(model, batch, shuffled, batch["memory_valid"])
                    elif condition.startswith("latest_"):
                        n = int(condition.split("_")[1])
                        mask = batch["memory_valid"].clone()
                        mask[:, :-n] = False
                        z = encode(model, batch, batch["memory_interactions"], mask)
                    elif condition == "shuffle_latent":
                        z = latent.index_select(0, torch.as_tensor(donors, device=args.device))
                    else:
                        z = torch.zeros_like(latent)
                    data[condition + "_mse"] = error_metrics(model, batch, z)
                    data[condition + "_latent_cosine"] = F.cosine_similarity(z.float(), latent.float()).cpu().numpy()
                # No available long memory must be an exact intervention-negative control.
                empty = data["long_chunks"] == 0
                if empty.any():
                    np.testing.assert_array_equal(data["natural_mse"][empty], data["remove_long_mse"][empty])
                    np.testing.assert_array_equal(data["natural_mse"][empty], data["shuffle_long_mse"][empty])
            pieces.append(data)
            print(json.dumps({"suite": suite, "rank": rank,
                              "natural_nmse": float(data["natural_mse"][:, -1].sum()/data["unchanged_mse"][:, -1].sum()),
                              "remove_ratio": float(data["remove_long_mse"][:, -1].sum()/data["natural_mse"][:, -1].sum()),
                              "shuffle_ratio": float(data["shuffle_long_mse"][:, -1].sum()/data["natural_mse"][:, -1].sum()),
                              "elapsed_seconds": time.time()-started}), flush=True)
            del batch, shuffled, latent
        all_samples = {k: np.concatenate([part[k] for part in pieces]) for k in pieces[0]}
        np.savez_compressed(out / f"{suite}_paired_samples.npz", **all_samples)
        result["suites"][suite] = summarize(all_samples)
        if suite == "broad":
            rows = [json.loads(line) for line in (checkpoint_path.parent / "metrics.jsonl").read_text().splitlines() if line.strip()]
            recorded = next(r["fixed_probe"]["dr_five_step_nmse"] for r in rows if r["update"] == checkpoint["update"])
            actual = result["suites"][suite]["natural_five_step_nmse_rank_mean"]
            result["natural_reproduction"] = {"training_log": recorded, "offline": actual,
                                              "relative_difference": actual / recorded - 1}
            assert abs(actual / recorded - 1) < .01, "Offline natural forward does not reproduce recorded checkpoint validation"
        result["elapsed_seconds"] = time.time() - started
        (out / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "output": str(out.resolve()),
                      "elapsed_seconds": time.time()-started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suites", nargs="+", default=["broad", "positive"])
    run(parser.parse_args())
