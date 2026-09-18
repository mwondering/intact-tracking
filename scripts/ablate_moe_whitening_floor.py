"""Controlled covariance-floor sweep with world-separated router evaluation.

The context and source policy stay frozen. This script never trains PPO.
Only the whitening eigenvalue floor changes; KMeans is refitted under the
same settings. Fixed numerical temperature is the primary comparison.
A fixed relative-temperature rule is retained as a distance-scale control.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from confirm_moe_router_information import fine_stability, read_trace
from intact_tracking.memory350_metric_router import FrozenMetricRouter
from intact_tracking.router_information_probe import (
    STRONG_COORDINATES, affine_predict, causal_ema, fit_euclidean_centers,
    fit_projection, gate_summary, gate_weights, project, readout_amplification,
    regression_metrics, select_readout, squared_distances,
)


FLOORS = [.01, .008, .006, .005, .004, .003, .002, .001, .0003, .0001, .00001, .000001]
KMEANS_SEEDS = [731, 1731, 2731]
TEMPERATURE_MODES = ["fixed_absolute", "fixed_relative"]
BASE_NAME = "whiten_f0.01_k16_t3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.tolist())+"\n")


def model_name(floor, seed):
    return f"floor{floor:g}_seed{seed}"


def read_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def point_targets(trace, meta):
    lower = np.asarray(meta["dr_schema"]["lower"])
    width = np.asarray(meta["dr_schema"]["upper"])-lower
    return (trace["raw_dr"][trace["point_worlds"]]-lower)/width


def smooth_features(trace, projection):
    x = trace["unit_latent_samples"]
    features = project(x.reshape(-1, 64), projection).reshape(*x.shape[:-1], 64)
    return causal_ema(features, trace["full"], dt=trace["dt"], time_constant=2.)


def assert_trace_identity(meta, protocol):
    assert meta["context_sha256"] == protocol["context_sha256"]
    assert meta["checkpoint_sha256"] == protocol["policy_checkpoint_sha256"]
    assert meta["actual_actor_dispatch_verified_every_step"]
    assert meta["motion_count"] == 129827
    assert meta["static_DR_unchanged_verified"] and meta["router_state_unchanged_verified"]


def fit(parent, output):
    if (output/"selection_seal.json").exists():
        raise RuntimeError("This experiment is already sealed; use a new output directory for refitting.")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("models", "readouts", "validation_predictions", "source_snapshot"):
        (output/name).mkdir(exist_ok=True)
    old = json.loads((parent/"protocol.json").read_text())
    base = next(row for row in json.loads((parent/"final_selection.json").read_text())["candidates"]
                if row["name"] == BASE_NAME)
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parent": str(parent.resolve()), "parent_selection_sha256": sha(parent/"final_selection.json"),
        "context_sha256": old["context_sha256"], "policy_checkpoint_sha256": old["policy_checkpoint_sha256"],
        "source": old["source"], "fit_seed": 30402,
        "fit_world_ids": old["fit_world_ids"], "validation_world_ids": old["validation_world_ids"],
        "baseline": base, "floors": FLOORS, "kmeans_seeds": KMEANS_SEEDS,
        "temperature_modes": TEMPERATURE_MODES,
        "fixed_absolute_temperature": base["temperature"], "fixed_relative_temperature_multiplier": 3.,
        "kmeans": {"centers": 16, "restarts": 3, "max_iterations": 60},
        "EMA_seconds": 2., "active_experts": 16, "samples_per_world": 32,
        "readout": "Linear ridge to all67 DR values; regularization selected on validation worlds only.",
        "selection": "For each temperature mode, choose best lower floor by mean strong8 validation R2 over3 KMeans seeds. All grid results reported on independent seed30403. Baseline, chosen lower floors and1e-4 also checked at50Hz on seed30404.",
        "test_data_status": "Existing original-policy traces, previously used in the broader exploration; no test world enters current fitting or floor selection. These are not newly collected traces.",
        "test_seeds": [30403, 30404], "router_uses_numerical_DR_labels": False,
        "DR_labels_used_for_readout_and_validation_selection": True,
        "PPO_trained": False, "production_router_changed": False,
        "source_sha256": {str(p): sha(p) for p in [Path(__file__), Path("src/intact_tracking/router_information_probe.py"),
                                                    Path("scripts/confirm_moe_router_information.py")]},
    }
    save_json(output/"protocol.json", protocol)
    trace, meta = read_trace(Path(old["source"])/"sample_seed30402")
    assert_trace_identity(meta, protocol)
    points = trace["unit_latent_samples"][trace["point_rows"], trace["point_worlds"]]
    fit_mask = np.isin(trace["point_worlds"], old["fit_world_ids"])
    assert fit_mask.sum() == 768*32 and (~fit_mask).sum() == 256*32
    x = points[fit_mask]
    y = point_targets(trace, meta)
    groups = meta["dr_schema"]["groups"]
    mean = x.astype(float).mean(0)
    eigenvalues, axes = np.linalg.eigh(np.cov(x.astype(float).T, bias=True))
    coordinates = ((x.astype(float)-mean)@axes).reshape(768, 32, 64)
    between = coordinates.mean(1).var(0)
    within = np.square(coordinates-coordinates.mean(1, keepdims=True)).mean((0, 1))
    np.testing.assert_allclose(between+within, eigenvalues, atol=1e-12)
    save_json(output/"spectrum.json", {
        "eigenvalue_fraction_of_max": eigenvalues/eigenvalues.max(),
        "between_world_fraction_per_axis": between/np.maximum(eigenvalues, 1e-30),
        "within_world_fraction_per_axis": within/np.maximum(eigenvalues, 1e-30),
        "interpretation": "Same-world variation includes motion/history/measurement variability; it is not a proof of pure noise or of control relevance.",
    })
    rows, spectral = [], []
    old_projection = read_npz(parent/"models/whiten_f0.01.npz")
    old_temporal = json.loads((parent/"temporal_sweep.json").read_text())
    old_validation = next(row for row in old_temporal if row["name"] == base["evaluation_name"])["validation"]["strong8_r2"]
    for floor in FLOORS:
        projection = fit_projection(x, trace["point_worlds"][fit_mask], "whiten", floor=floor)
        if floor == .01:
            np.testing.assert_allclose(projection["matrix"], old_projection["matrix"], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(projection["offset"], old_projection["offset"], atol=1e-8, rtol=1e-8)
        denominator = np.maximum(eigenvalues, eigenvalues.max()*floor)
        spectral.append({"floor": floor, "clipped_directions": int((eigenvalues<floor*eigenvalues.max()).sum()),
                         "transformed_total_variance": float((eigenvalues/denominator).sum()),
                         "transformed_within_world_variance_fraction": float((within/denominator).sum()/(eigenvalues/denominator).sum()),
                         "max_relative_axis_gain": float(np.sqrt(denominator.max()/denominator.min()))})
        features = smooth_features(trace, projection)
        smoothed_points = features[trace["point_rows"], trace["point_worlds"]]
        raw_fit_features = project(x, projection)
        for seed in KMEANS_SEEDS:
            name = model_name(floor, seed)
            if floor == .01 and seed == 731:
                centers = old_projection["centers"].copy()
                inertia = float(squared_distances(raw_fit_features, centers).min(-1).mean())
            else:
                centers, inertia = fit_euclidean_centers(raw_fit_features, k=16, seed=seed, restarts=3, iterations=60)
            distances = squared_distances(smoothed_points, centers)
            center_distances = squared_distances(centers, centers)
            np.fill_diagonal(center_distances, np.inf)
            scale = float(np.median(center_distances.min(-1)))
            np.savez(output/"models"/f"{name}.npz", matrix=projection["matrix"], offset=projection["offset"], centers=centers)
            for mode in TEMPERATURE_MODES:
                temperature = base["temperature"] if mode == "fixed_absolute" else 3*scale
                candidate = f"{name}_{mode}"
                weights = gate_weights(distances, top_k=16, temperature=temperature)
                readout, metrics, grid = select_readout(weights[fit_mask], y[fit_mask], weights[~fit_mask], y[~fit_mask], groups)
                np.savez(output/"readouts"/f"{candidate}.npz", **readout)
                prediction = affine_predict(weights[~fit_mask], readout)[:, STRONG_COORDINATES]
                np.savez_compressed(output/"validation_predictions"/f"{candidate}.npz", prediction=prediction.astype(np.float32),
                                    targets=y[~fit_mask][:, STRONG_COORDINATES].astype(np.float32), worlds=trace["point_worlds"][~fit_mask])
                row = {"name": candidate, "model": name, "floor": floor, "kmeans_seed": seed,
                       "temperature_mode": mode, "temperature": temperature, "temperature_scale": scale,
                       "kmeans_inertia": inertia, "validation": metrics, "readout_grid": grid,
                       "validation_gate": gate_summary(weights[~fit_mask]), "amplification": readout_amplification(readout)}
                if floor == .01 and seed == 731:
                    row["baseline_validation_R2_error"] = abs(metrics["strong8_r2"]-old_validation)
                    assert row["baseline_validation_R2_error"] < 1e-6
                rows.append(row)
                print(json.dumps({"floor": floor, "seed": seed, "temperature_mode": mode,
                                  "validation_R2": metrics["strong8_r2"], "loads": metrics["r2"][:4]}), flush=True)
        save_json(output/"fit_results.partial.json", rows)
        save_json(output/"spectral_floor_effects.json", spectral)
    aggregate = []
    for mode in TEMPERATURE_MODES:
        for floor in FLOORS:
            values = [row["validation"]["strong8_r2"] for row in rows if row["floor"] == floor and row["temperature_mode"] == mode]
            aggregate.append({"floor": floor, "temperature_mode": mode, "mean_validation_R2": float(np.mean(values)),
                              "min_validation_R2": min(values), "max_validation_R2": max(values)})
    chosen = {mode: max([row for row in aggregate if row["floor"] < .01 and row["temperature_mode"] == mode],
                        key=lambda row: row["mean_validation_R2"])["floor"] for mode in TEMPERATURE_MODES}
    selection = {"chosen_lower_floors": chosen, "aggregate_validation": aggregate,
                 "runtime50hz_floors": sorted(set([.01, .0001, *chosen.values()]), reverse=True),
                 "rule": protocol["selection"], "test_data_used_for_this_selection": False}
    save_json(output/"fit_results.json", rows)
    save_json(output/"selection.json", selection)
    save_json(output/"selection_seal.json", {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           "selection_sha256": sha(output/"selection.json"),
                                           "fit_results_sha256": sha(output/"fit_results.json"),
                                           "protocol_sha256": sha(output/"protocol.json")})
    print(json.dumps({"sealed_lower_floor_choices": chosen}), flush=True)


def evaluate(output):
    protocol = json.loads((output/"protocol.json").read_text())
    parent = Path(protocol["parent"])
    selection = json.loads((output/"selection.json").read_text())
    seal = json.loads((output/"selection_seal.json").read_text())
    for name in ("selection", "fit_results", "protocol"):
        assert sha(output/f"{name}.json") == seal[f"{name}_sha256"]
    fitted = json.loads((output/"fit_results.json").read_text())
    for tag, directory in [("heldout30403", parent/"heldout_sample_seed30403"),
                           ("runtime50hz30404", parent/"runtime50hz_sample_seed30404")]:
        target = output/tag
        target.mkdir(exist_ok=True)
        trace, meta = read_trace(directory)
        assert_trace_identity(meta, protocol)
        y = point_targets(trace, meta)
        assert meta["arguments"]["seed"] != protocol["fit_seed"]
        np.savez_compressed(target/"targets.npz", targets=y[:, STRONG_COORDINATES].astype(np.float32), worlds=trace["point_worlds"])
        floors = FLOORS if tag == "heldout30403" else selection["runtime50hz_floors"]
        old_tag = "heldout_confirmation" if tag == "heldout30403" else "runtime50hz_confirmation"
        old_row = next(row for row in json.loads((parent/old_tag/"summary.json").read_text())["rows"] if row["name"] == BASE_NAME)
        rows = []
        for floor in floors:
            projection = read_npz(output/"models"/f"{model_name(floor,731)}.npz")
            features = smooth_features(trace, projection)
            feature_points = features[trace["point_rows"], trace["point_worlds"]]
            for seed in KMEANS_SEEDS:
                model = read_npz(output/"models"/f"{model_name(floor,seed)}.npz")
                distance_points = squared_distances(feature_points, model["centers"])
                for mode in TEMPERATURE_MODES:
                    candidate = next(x for x in fitted if x["floor"] == floor and x["kmeans_seed"] == seed and x["temperature_mode"] == mode)
                    name = candidate["name"]
                    readout = read_npz(output/"readouts"/f"{name}.npz")
                    point_weights = gate_weights(distance_points, top_k=16, temperature=candidate["temperature"])
                    prediction = affine_predict(point_weights, readout)
                    metrics = regression_metrics(y, prediction, meta["dr_schema"]["groups"])
                    row = {"name": name, "floor": floor, "kmeans_seed": seed, "temperature_mode": mode,
                           "temperature": candidate["temperature"], "metrics": metrics,
                           "gate": gate_summary(point_weights), "amplification": candidate["amplification"]}
                    np.savez_compressed(target/f"{name}.npz", prediction=prediction[:, STRONG_COORDINATES].astype(np.float32))
                    if tag == "runtime50hz30404":
                        distance = squared_distances(features.reshape(-1, 64), model["centers"])
                        weights = gate_weights(distance, top_k=16, temperature=candidate["temperature"]).reshape(*features.shape[:-1], 16)
                        row["stability"] = fine_stability(weights, trace, readout)
                        if seed == 731 and mode == "fixed_absolute":
                            router = FrozenMetricRouter(model["matrix"], model["offset"], model["centers"][None],
                                                        [candidate["temperature"]], group_top_k=16, time_constant=2., clip_unit_range=False)
                            maximum, total_error = 0., 0.
                            with torch.inference_mode():
                                for step in range(len(features)):
                                    actual = router.advance(torch.from_numpy(trace["unit_latent_samples"][step]),
                                                            torch.from_numpy(trace["full"][step]), dt=trace["dt"]).numpy()
                                    error = np.abs(actual-weights[step])
                                    maximum = max(maximum, float(error.max()))
                                    total_error += float(error.sum())
                            row["prototype_replay"] = {"device": "cpu", "maximum_absolute_error": maximum,
                                                       "mean_absolute_error": total_error/weights.size,
                                                       "passed": maximum < 5e-4, "has_trainable_parameters": bool(list(router.parameters()))}
                            assert row["prototype_replay"]["passed"] and not list(router.parameters())
                    if floor == .01 and seed == 731:
                        row["baseline_reference_R2_error"] = abs(metrics["strong8_r2"]-old_row["metrics"]["strong8_r2"])
                        assert row["baseline_reference_R2_error"] < 1e-6
                    rows.append(row)
                    print(json.dumps({"trace": tag, "floor": floor, "seed": seed, "temperature_mode": mode,
                                      "R2": metrics["strong8_r2"], "loads": metrics["r2"][:4],
                                      "TV": row.get("stability", {}).get("mean_gate_TV")}), flush=True)
            save_json(target/"results.partial.json", rows)
        save_json(target/"results.json", {"trace": str(directory.resolve()), "seed": meta["arguments"]["seed"],
                                           "latent_recording_interval": trace["dt"], "source_result_sha256": sha(directory/"result.json"),
                                           "parameters_fitted_on_this_trace": False, "selection_sha256": seal["selection_sha256"], "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("runs/limb_context_20260914_router_information_exploration"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if args.stage == "fit":
        fit(args.parent, args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
