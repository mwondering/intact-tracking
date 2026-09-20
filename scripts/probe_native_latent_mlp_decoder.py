"""Small CPU MLP readout of cached unit latents, with world-disjoint validation."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model():
    return nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128),
                         nn.SiLU(), nn.Linear(128, 92))


def metrics(truth, prediction, schema):
    # Two different motion views per world; both remain in the same split.
    errors = prediction - truth[:, None, :]
    mse_world = np.square(errors).mean(1)
    r2 = 1 - mse_world.mean(0) / truth.var(0)
    mae = np.abs(errors).mean((0, 1))
    scale = np.subtract(schema["upper"], schema["lower"])
    scale[:3] *= 100  # COM, cm.
    scale[3] = 2.0  # Native torso mass perturbation spans -1 to +1 kg.
    scale[5:] *= 100  # Motor multiplicative scales, percentage points.
    return {
        "factor_balanced_mse": float(mse_world.mean(0) @ schema["coordinate_weights"]),
        "groups": {name: {
            "mean_coordinate_r2": float(r2[cols].mean()),
            "mse_fraction_full_dr_range": float(mse_world[:, cols].mean()),
            "mae_fraction_full_dr_range": float(mae[cols].mean()),
            "mae_display_units": float((mae * scale)[cols].mean()),
            "r2_min": float(r2[cols].min()), "r2_max": float(r2[cols].max()),
        } for name, cols in schema["groups"].items()},
        "parameters": [{"name": name, "r2": float(r2[i]),
                        "mae_display_units": float(mae[i] * scale[i])}
                       for i, name in enumerate(schema["names"])],
    }


def fit(seed, x, y, schema, args):
    torch.manual_seed(seed)
    network = model()
    with torch.no_grad():
        network[-1].bias.copy_(y["train"].mean(0))
    weights = torch.tensor(schema["coordinate_weights"], dtype=torch.float32)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    history, best_state, best_epoch, best_value = [], None, None, float("inf")
    started = time.monotonic()

    def evaluate(epoch):
        nonlocal best_state, best_epoch, best_value
        network.eval()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        with torch.no_grad():
            for split in ("train", "validation"):
                mse = (network(x[split]) - y[split]).square().mean(0)
                row[split + "_mse"] = float((mse * weights).sum())
                row[split + "_groups"] = {
                    name: float(mse[cols].mean()) for name, cols in schema["groups"].items()}
        if not all(np.isfinite(row[k]) for k in ("train_mse", "validation_mse")):
            raise RuntimeError("Nonfinite decoder loss")
        if row["validation_mse"] < best_value:
            best_value, best_epoch = row["validation_mse"], epoch
            best_state = copy.deepcopy(network.state_dict())
        history.append(row)
        if epoch % 200 == 0 or epoch == args.epochs:
            write_json(args.output / f"history_seed_{seed}.json", history)
            print(json.dumps({"seed": seed, "epoch": epoch,
                              "train_mse": row["train_mse"],
                              "validation_mse": row["validation_mse"],
                              "best_epoch": best_epoch, "best_validation_mse": best_value,
                              "seconds": time.monotonic() - started}), flush=True)
        network.train()

    evaluate(0)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = ((network(x["train"]) - y["train"]).square() * weights).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 20 == 0 or epoch == args.epochs:
            evaluate(epoch)
    assert best_epoch > 0
    last_state = copy.deepcopy(network.state_dict())
    network.load_state_dict(best_state)
    record = {"seed": seed, "best_epoch": best_epoch, "best_validation_mse": best_value,
              "last_epoch": args.epochs, "seconds": time.monotonic() - started,
              "selected_epoch_metrics": next(r for r in history if r["epoch"] == best_epoch),
              "last_epoch_metrics": history[-1]}
    torch.save({"best_model_state_dict": best_state, "last_model_state_dict": last_state,
                "last_optimizer_state_dict": optimizer.state_dict(),
                "last_scheduler_state_dict": scheduler.state_dict(), "record": record},
               args.output / f"fit_seed_{seed}.pt")
    return network.eval(), last_state, record


def plot_curves(output, seeds, selected_seed, schema):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"base_com/com_offset/torso_link/x": "COM x",
              "base_com/com_offset/torso_link/y": "COM y",
              "base_com/com_offset/torso_link/z": "COM z",
              "base_mass/relative_mass/torso_link": "Torso mass",
              "foot_friction/friction/shared/0": "Foot friction",
              "joint_armature_scale_rms": "Armature (29 joints)",
              "joint_kd_scale_rms": "Kd (29 joints)",
              "joint_kp_scale_rms": "Kp (29 joints)"}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for ax, group in zip(axes.flat, [None, *schema["groups"]], strict=True):
        for seed in seeds:
            rows = json.loads((output / f"history_seed_{seed}.json").read_text())
            for split, color in (("train", "#2466a6"), ("validation", "#cf6823")):
                values = [r[split + "_groups"][group] if group else r[split + "_mse"]
                          for r in rows]
                ax.plot([r["epoch"] for r in rows], values, color=color,
                        alpha=1 if seed == selected_seed else .25,
                        linewidth=1.5 if seed == selected_seed else .8,
                        label=split if seed == selected_seed else None)
        ax.set_title(labels[group] if group else "Factor-balanced objective")
        ax.set_xlabel("Full-batch epoch")
        ax.set_ylabel("MSE (DR-range normalized)")
        ax.set_yscale("log")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle("Frozen u7179 latent: MLP decoder train / validation\n"
                 "Solid = validation-selected seed; faint = other seeds")
    fig.savefig(output / "learning_curves.png", dpi=170)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def main(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.source / "results.json").read_text())
    prior_protocol = json.loads((args.source / "protocol.json").read_text())
    schema = json.loads(args.config.read_text())["batch_a_rollout"]["dr_metric_schema"]
    assert digest(reference["source"]) == reference["source_sha256"]
    assert hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest() == reference["schema_sha256"]
    assert digest(prior_protocol["checkpoint"]["path"]) == prior_protocol["checkpoint"]["sha256"]
    data = np.load(reference["source"])
    ridge = np.load(args.source / "decoder_and_test_predictions.npz")
    splits = json.loads((args.source / "split_worlds.json").read_text())
    rows = data["retrieval_rows"]
    worlds = data["world"][rows]
    raw_x = np.stack((data["z"][rows], data["other"][rows]), 1)
    target = data["parameters"][rows] / np.sqrt(schema["coordinate_weights"])
    assert len(np.unique(worlds)) == len(worlds)
    assert (data["motion"][rows] != data["other_motion"][rows]).all()
    assert (data["age"][rows] >= 355).all()
    assert np.isfinite(raw_x).all() and np.isfinite(target).all()
    lookup = {int(w): i for i, w in enumerate(worlds)}
    ix = {name: np.array([lookup[w] for w in ids]) for name, ids in splits.items()}
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])
    assert sum(len(v) for v in ix.values()) == len(worlds)
    train_x = raw_x[ix["train"]].reshape(-1, 64)
    mean, std = train_x.mean(0), train_x.std(0).clip(1e-8)
    np.testing.assert_allclose(mean, ridge["mean_x"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(std, ridge["scale_x"], atol=1e-12, rtol=0)
    x = {name: torch.tensor(((raw_x[indices] - mean) / std).reshape(-1, 64),
                            dtype=torch.float32) for name, indices in ix.items()}
    y = {name: torch.tensor(np.repeat(target[indices], 2, axis=0), dtype=torch.float32)
         for name, indices in ix.items()}
    protocol = {
        "started_at": time.time(), "checkpoint": prior_protocol["checkpoint"],
        "source": reference["source"], "source_sha256": reference["source_sha256"],
        "schema_sha256": reference["schema_sha256"],
        "split_sha256": digest(args.source / "split_worlds.json"),
        "script_sha256": digest(__file__), "encoder_frozen": True,
        "device": "cpu", "cpu_threads": 2, "architecture": [64, 128, 128, 92],
        "parameters": sum(p.numel() for p in model().parameters()), "activation": "SiLU",
        "input": "single 64D unit latent, standardization fitted on train worlds only",
        "target": "92 static DR coordinates normalized by their configured ranges",
        "loss": "mean of per-factor MSEs, each motor block contributes its coordinate mean",
        "optimizer": "AdamW", "learning_rate": args.lr, "weight_decay": 1e-4,
        "scheduler": "cosine decay to 1e-5", "epochs_per_seed": args.epochs,
        "batch": "full batch; both views of each training world, equal world weights",
        "seeds": args.seeds, "worlds": {k: len(v) for k, v in ix.items()},
        "views_per_world": 2, "validation_frequency_epochs": 20,
        "selection": "lowest validation factor-balanced MSE selects epoch and seed; test after all fits",
        "predictions": "unclipped; no refit on validation", "training_process_changed": False,
        "limitations": reference["limitations"][:-1],
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "split_worlds.json", splits)
    print(json.dumps(protocol), flush=True)
    fits = [fit(seed, x, y, schema, args) for seed in args.seeds]
    selected = int(np.argmin([f[2]["best_validation_mse"] for f in fits]))
    # All fitting and selection are complete before any test metric is computed.
    report = {"protocol": protocol, "selected_seed": args.seeds[selected],
              "selected_epoch": fits[selected][2]["best_epoch"], "seeds": []}
    predictions = []
    with torch.no_grad():
        for network, last_state, record in fits:
            record["best"] = {}
            for name in ("train", "validation", "test"):
                prediction = network(x[name]).numpy().astype(np.float64).reshape(-1, 2, 92)
                record["best"][name] = metrics(target[ix[name]], prediction, schema)
                if name == "test":
                    predictions.append(prediction)
            last = model()
            last.load_state_dict(last_state)
            record["last"] = {name: metrics(target[ix[name]], last(x[name]).numpy()
                                            .astype(np.float64).reshape(-1, 2, 92), schema)
                              for name in ("train", "validation")}
            report["seeds"].append(record)
    test_truth = target[ix["test"]]
    np.testing.assert_array_equal(worlds[ix["test"]], ridge["test_world"])
    np.testing.assert_allclose(test_truth, ridge["target_range_normalized"], atol=0, rtol=0)
    baseline_prediction = ridge["prediction_range_normalized"]
    report["ridge_test"] = metrics(test_truth, baseline_prediction, schema)
    for name in schema["groups"]:
        assert abs(report["ridge_test"]["groups"][name]["mean_coordinate_r2"]
                   - reference["groups"][name]["mean_coordinate_r2"]) < 1e-12
    primary = predictions[selected]
    report["selected_test"] = metrics(test_truth, primary, schema)
    report["mean_predictor_test"] = metrics(
        test_truth, np.broadcast_to(target[ix["train"]].mean(0), primary.shape), schema)
    counts = np.random.default_rng(20260923).multinomial(
        len(test_truth), np.full(len(test_truth), 1 / len(test_truth)), size=2000) / len(test_truth)
    variance = counts @ np.square(test_truth) - np.square(counts @ test_truth)
    mlp_mse = np.square(primary - test_truth[:, None]).mean(1)
    ridge_mse = np.square(baseline_prediction - test_truth[:, None]).mean(1)
    sampled_r2 = 1 - (counts @ mlp_mse) / variance.clip(1e-12)
    sampled_delta = (counts @ (ridge_mse - mlp_mse)) / variance.clip(1e-12)
    report["comparison"] = {}
    for name, cols in schema["groups"].items():
        scores = [f[2]["best"]["test"]["groups"][name]["mean_coordinate_r2"] for f in fits]
        report["comparison"][name] = {
            "ridge_r2": reference["groups"][name]["mean_coordinate_r2"],
            "selected_mlp_r2": report["selected_test"]["groups"][name]["mean_coordinate_r2"],
            "mlp_seed_r2": scores, "mlp_seed_mean_r2": float(np.mean(scores)),
            "mlp_seed_std_r2": float(np.std(scores, ddof=1)),
            "selected_mlp_r2_ci95": np.quantile(sampled_r2[:, cols].mean(1), [.025, .975]).tolist(),
            "paired_r2_improvement_ci95": np.quantile(sampled_delta[:, cols].mean(1), [.025, .975]).tolist(),
        }
    report["bootstrap"] = "2000 paired test-world resamples; conditional on fitted models and existing split"
    checkpoint = {
        "model_state_dict": fits[selected][0].state_dict(),
        "input_mean": torch.tensor(mean), "input_std": torch.tensor(std),
        "target_lower": torch.tensor(schema["lower"]), "target_upper": torch.tensor(schema["upper"]),
        "parameter_names": schema["names"], "protocol": protocol,
        "selected_seed": args.seeds[selected], "selected_epoch": fits[selected][2]["best_epoch"],
    }
    torch.save(checkpoint, args.output / "decoder.pt")
    restored = torch.load(args.output / "decoder.pt", map_location="cpu", weights_only=False)
    restored_model = model().eval()
    restored_model.load_state_dict(restored["model_state_dict"])
    with torch.no_grad():
        restored_x = ((torch.tensor(raw_x[ix["test"]]) - restored["input_mean"])
                      / restored["input_std"]).float().reshape(-1, 64)
        restored_prediction = restored_model(restored_x).numpy().reshape(-1, 2, 92)
    np.testing.assert_allclose(restored_prediction, primary, atol=2e-6, rtol=0)
    report["checkpoint_reload_verified"] = True
    np.savez_compressed(args.output / "test_predictions.npz", world=worlds[ix["test"]],
                        truth=test_truth, mlp_predictions=np.stack(predictions),
                        ridge_prediction=baseline_prediction, seeds=args.seeds,
                        selected_seed=args.seeds[selected])
    plot_curves(args.output, args.seeds, args.seeds[selected], schema)
    report.update(complete=True, elapsed_seconds=time.time() - protocol["started_at"])
    write_json(args.output / "results.json", report)
    print(json.dumps({"complete": True, "selected_seed": report["selected_seed"],
                      "selected_epoch": report["selected_epoch"],
                      "comparison": report["comparison"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260920, 20260921, 20260922])
    main(parser.parse_args())
