"""Create independently checked tables, world-bootstrap intervals and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from confirm_moe_router_information import make_prototype, read_trace
from explore_moe_router_temporal import load_projection, route_features
from intact_tracking.router_information_probe import causal_ema, project, readout_amplification


METHODS = {
    "saved_k1_t1": "Original hard 16",
    "saved_k4_t0.3": "Original metric, top 4",
    "whiten_f0.01_k16_t3": "Whitened, all 16",
    "decoded_dr8_k8_t1": "DR metric, joint top 8",
    "decoded_dr8_k16_t3": "DR metric, joint all 16",
    "product_dr8_gk2_t0.3": "Grouped KMeans, 2/group",
    "product_dr8_gk3_t1": "Grouped KMeans, 3/group",
    "product_dr8_gk4_t1": "Grouped KMeans, 4/group",
    "factorized_triangular": "Fixed triangles, 3/group",
}
MAIN = ["saved_k1_t1", "saved_k4_t0.3", "whiten_f0.01_k16_t3",
        "decoded_dr8_k8_t1", "product_dr8_gk4_t1", "factorized_triangular"]
COORDINATES = [0, 1, 2, 3, 5, 6, 7, 8]
LABELS = ["Left hand", "Right hand", "Left shin", "Right shin", "COM x", "COM y", "COM z", "Friction"]


def check_predictions(root, tag, draws):
    report = json.loads((root/tag/"summary.json").read_text())
    rows = {row["name"]: row for row in report["rows"]}
    intervals, checked, bootstrap = {}, {}, {}
    with np.load(root/tag/"predictions.npz") as saved:
        worlds, targets = saved["worlds"], saved["targets"].astype(np.float64)
        assert np.array_equal(worlds, np.repeat(np.arange(1024), 32))
        grouped_y = targets.reshape(1024, 32, 8)
        world_y = grouped_y.mean(1)
        # Each world is resampled as a whole: its 32 time points are not independent trials.
        mean_y, mean_y2 = draws@world_y/1024, draws@np.square(world_y)/1024
        variance = mean_y2-np.square(mean_y)
        for name, row in rows.items():
            predicted = saved[name].astype(np.float64)
            error = np.square(predicted-targets)
            r2 = 1-error.mean(0)/targets.var(0)
            expected = np.asarray(row["metrics"]["r2"])[COORDINATES]
            discrepancy = float(np.abs(expected-r2).max())
            assert discrepancy < 1e-6
            checked[name] = discrepancy
            world_mse = error.reshape(1024, 32, 8).mean(1)
            resampled = 1-(draws@world_mse/1024)/variance
            bootstrap[name] = resampled.mean(1)
            intervals[name] = {
                "mean8_R2_95pct": np.quantile(resampled.mean(1), [.025, .975]).tolist(),
                "per_coordinate_R2_95pct": np.quantile(resampled, [.025, .975], axis=0).T.tolist(),
            }
        delta = bootstrap["product_dr8_gk4_t1"]-bootstrap["saved_k1_t1"]
        prediction = saved["product_dr8_gk4_t1"].reshape(1024, 32, 8)
        shuffled = prediction[np.random.default_rng(4931).permutation(1024)].reshape(-1, 8)
        shuffled_r2 = 1-np.square(shuffled-targets).mean(0)/targets.var(0)
    return {
        "seed": report["seed"], "resampling_unit": "whole DR world; all 32 observations remain together",
        "bootstrap_replicates": len(draws),
        "scope": "Conditional on this frozen policy and trace distribution; not policy-training-seed uncertainty.",
        "independent_saved_prediction_max_errors": checked,
        "intervals": intervals,
        "group4_minus_original_mean8_R2_95pct": np.quantile(delta, [.025, .975]).tolist(),
        "shuffled_world_prediction_mean8_R2": float(shuffled_r2.mean()),
    }


def save_figure(figure, root, name):
    figure.savefig(root/f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(root/f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def check_group_geometry(trace, model, candidate):
    """Invert softmax distance ratios, independently of any fitted DR readout."""
    router, _ = make_prototype(model, candidate)
    points = trace["unit_latent_samples"][trace["point_rows"], trace["point_worlds"]]
    observed = project(points, model)
    corners = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.float32)
    random = np.random.default_rng(2207).uniform(size=(4096, 8)).astype(np.float32)
    x = np.concatenate((observed, corners, random))
    w = router.weights_from_features(torch.from_numpy(x)).numpy().reshape(-1, 4, 4)
    restored = np.empty_like(x, dtype=np.float64)
    geometry = []
    for g in range(4):
        centers = model["centers"][g].astype(np.float64)
        basis = 2*(centers[1:]-centers[:1])
        singular = np.linalg.svd(basis, compute_uv=False)
        assert singular[-1] > 1e-6
        # T log(w_j/w_0) = 2(c_j-c_0)·x - (||c_j||²-||c_0||²).
        rhs = candidate["temperatures"][g]*np.log(w[:, g, 1:].astype(float)/w[:, g, :1])
        rhs += np.square(centers[1:]).sum(-1)-np.square(centers[0]).sum()
        restored[:, 2*g:2*g+2] = rhs@np.linalg.pinv(basis).T
        geometry.append({"rank": 2, "singular_values": singular.tolist(),
                         "condition_number": float(singular[0]/singular[-1])})
    maximum = float(np.abs(restored-x).max())
    assert maximum < 2e-6
    return {"claim": "Full group softmax retains all 8 projected coordinates when each center-difference matrix has rank 2 and weights remain positive. This is not a claim to preserve all64 latent coordinates.",
            "points": len(x), "point_sources": "32768 heldout latent projections, all256 unit-cube corners,4096 seeded random points",
            "maximum_coordinate_reconstruction_error": maximum,
            "minimum_expert_weight": float(w.min()), "group_geometry": geometry,
            "uses_fitted_DR_readout_for_inverse": False, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    heldout = json.loads((root/"heldout_confirmation/summary.json").read_text())
    runtime = json.loads((root/"runtime50hz_confirmation/summary.json").read_text())
    a = {x["name"]: x for x in heldout["rows"]}
    b = {x["name"]: x for x in runtime["rows"]}
    draws = np.random.default_rng(7702).multinomial(1024, np.full(1024, 1/1024), size=1000).astype(float)
    statistical = {tag: check_predictions(root, tag, draws)
                   for tag in ("heldout_confirmation", "runtime50hz_confirmation")}
    (root/"statistical_checks.json").write_text(json.dumps(statistical, indent=2)+"\n")

    table = []
    for name, row in a.items():
        readout_path = root/"temporal_readouts"/f"{row['settings']['evaluation_name']}.npz"
        with np.load(readout_path) as f:
            readout = {key: f[key] for key in f.files}
        amplitude = readout_amplification(readout)["strong8_mean_prototype_span_in_DR_ranges"]
        gate = row["coarse_stability"]["gate"]
        item = {
            "name": name, "active_heads": gate["active_experts_per_point"],
            "time_constant_seconds": row["settings"]["time_constant_seconds"],
            "heldout_seed30403_mean8_R2": row["metrics"]["strong8_r2"],
            "runtime_seed30404_mean8_R2": b[name]["metrics"]["strong8_r2"],
            "gate_TV_per_20ms": b[name]["native_cadence_stability"]["mean_gate_TV"],
            "decoded_DR_RMS_change_per_20ms": b[name]["native_cadence_stability"]["mean_decoded_DR_change_RMS_in_full_ranges"],
            "prototype_span_in_DR_ranges": amplitude,
            "minimum_marginal_weight": gate["minimum_expert_weight"],
            "maximum_marginal_weight": gate["maximum_expert_weight"],
            **{label.replace(" ", "_")+"_R2": row["metrics"]["r2"][coordinate]
               for label, coordinate in zip(LABELS, COORDINATES)},
        }
        table.append(item)
    with (root/"comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (root/"comparison.json").write_text(json.dumps(table, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [3.3, 1.45]})
    values = np.asarray([a[name]["metrics"]["r2"] for name in MAIN])[:, COORDINATES]
    im = axes[0].imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(range(8), LABELS, rotation=35, ha="right")
    axes[0].set_yticks(range(len(MAIN)), [METHODS[x] for x in MAIN])
    for y, row in enumerate(values):
        for x, v in enumerate(row):
            axes[0].text(x, y, f"{v:.2f}", ha="center", va="center", color="white" if v>.65 else "black")
    axes[0].set_title("DR prediction from router outputs only | held-out seed 30403")
    fig.colorbar(im, ax=axes[0], shrink=.6, label="R² (negative values printed, color clipped at 0)")
    avg = values.mean(1)
    ci = np.asarray([statistical["heldout_confirmation"]["intervals"][x]["mean8_R2_95pct"] for x in MAIN])
    axes[1].barh(range(len(MAIN)), avg, color=["#929aa5"]*4+["#14846d", "#467aad"])
    axes[1].errorbar(avg, range(len(MAIN)), xerr=[avg-ci[:, 0], ci[:, 1]-avg], fmt="none", ecolor="black", capsize=3)
    axes[1].set_yticks([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Mean R² of 8 DR coordinates")
    axes[1].set_title("1,024 new DR worlds\n95% world-bootstrap intervals")
    for i, v in enumerate(avg):
        axes[1].text(v+.025, i, f"{v:.3f}", va="center")
    fig.text(.5, -.04, "Frozen payload-trained context; no parameter fitting on this seed. R² is not classification accuracy or policy tracking performance.", ha="center")
    fig.tight_layout()
    save_figure(fig, root, "router_information_heldout")

    temporal_names = ["saved_k1_t1", "decoded_dr8_k8_t1", "product_dr8_gk4_t1", "factorized_triangular"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, key, title in (
        (axes[0], "mean_gate_TV", "Mean change of routing weights per 20 ms"),
        (axes[1], "mean_decoded_DR_change_RMS_in_full_ranges", "Mean decoded DR change per 20 ms"),
    ):
        vals = [b[n]["native_cadence_stability"][key] for n in temporal_names]
        axis.bar(range(4), vals, color=["#929aa5", "#d99a37", "#14846d", "#467aad"])
        axis.set_xticks(range(4), [METHODS[n] for n in temporal_names], rotation=25, ha="right")
        axis.set_yscale("log")
        axis.set_title(title)
        for i, value in enumerate(vals):
            axis.text(i, value*1.12, f"{value:.5f}", ha="center")
        axis.set_ylim(min(vals)*.55, max(vals)*2)
    axes[0].set_ylabel("Total variation = 0.5 × sum |Δ weight|")
    axes[1].set_ylabel("RMS across 8 coordinates, in full DR ranges")
    fig.suptitle("50 Hz causal replay | new seed 30404 | 714,454 full-memory same-motion intervals")
    fig.tight_layout()
    save_figure(fig, root, "router_stability_50hz")

    # One fixed world, chosen by index, not by routing outcome.
    trace, _ = read_trace(root/"runtime50hz_sample_seed30404")
    geometry = check_group_geometry(trace, load_projection(root, "product_dr8"),
                                   b["product_dr8_gk4_t1"]["settings"])
    (root/"group_geometry_audit.json").write_text(json.dumps(geometry, indent=2)+"\n")
    world = 0
    latent = trace["unit_latent_samples"][:, world:world+1]
    curves = {}
    for name in ("saved_k1_t1", "product_dr8_gk4_t1"):
        model = load_projection(root, b[name]["settings"]["metric"])
        features = project(latent.reshape(-1, 64), model).reshape(len(latent), 1, -1)
        filtered = causal_ema(features, trace["full"][:, world:world+1], dt=.02,
                              time_constant=b[name]["settings"]["time_constant_seconds"])
        curves[name] = route_features(filtered, model, b[name]["settings"])[:, 0]
    times = trace["sampled_steps"]*.02
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    axes[0].step(times, curves["saved_k1_t1"].argmax(-1), where="post", color="#4d5664")
    axes[0].set_ylabel("Hard ID")
    axes[0].set_yticks([0, 5, 10, 15])
    axes[0].set_ylim(-.5, 15.5)
    names = ["Hand loads", "Shin loads", "COM x / y", "COM z / friction"]
    for g in range(4):
        for k in range(4):
            axes[g+1].plot(times, curves["product_dr8_gk4_t1"][:, g*4+k]*4, linewidth=1, label=f"Center {k}")
        axes[g+1].set_ylabel(names[g]+"\nwithin-group weight")
        axes[g+1].set_ylim(0, 1)
    motion = trace["motion_ids"][trace["sampled_steps"], world]
    changed = np.flatnonzero(motion[1:]!=motion[:-1])+1
    for axis in axes:
        for index in changed:
            axis.axvline(times[index], color="#777777", linestyle=":", alpha=.6)
        axis.fill_between(times, 0, 1, where=~trace["full"][:, world], color="#dddddd",
                          alpha=.45, transform=axis.get_xaxis_transform())
    axes[-1].legend(ncol=4, loc="upper right")
    axes[-1].set_xlabel("Policy interaction time (seconds); dotted = motion change; gray = incomplete context")
    loads = trace["raw_dr"][world, :4]
    load_text = ", ".join(f"{value:.2f}" for value in loads)
    fig.suptitle(f"World 0, seed 30404 | fixed limb payloads [{load_text}] kg\nCounterfactual routers on the same original-policy trajectory; global argmax is not the new controller selection")
    fig.tight_layout()
    save_figure(fig, root, "router_fixed_world_timeline")
    print(json.dumps({"figures": 3, "tables": len(table),
                      "heldout_group4_R2_CI": statistical["heldout_confirmation"]["intervals"]["product_dr8_gk4_t1"]["mean8_R2_95pct"],
                      "shuffled_world_R2": statistical["heldout_confirmation"]["shuffled_world_prediction_mean8_R2"]}))


if __name__ == "__main__":
    main()
