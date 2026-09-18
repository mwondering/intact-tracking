"""Measure physical DR overlap within frozen whitening-router experts.

Replay existing held-out trajectories; do not fit centers, readouts, or PPO.
Aggregate equally sampled observations per world before pairing distinct worlds.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from confirm_moe_router_information import read_trace
from intact_tracking.router_information_probe import (
    STRONG_COORDINATES, causal_ema, gate_summary, gate_weights, project,
    squared_distances,
)


TEMPERATURES = [18.401254177093506, 6.133751392364502,
                1.8401254177093505, .6133751392364503]
FLOORS = [.01, .008]
KM_SEEDS = [731, 1731, 2731]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.tolist())+"\n")


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    cumulative = np.cumsum(weights)
    return np.interp(np.asarray(qs)*cumulative[-1], cumulative, values)


def proximity(raw_dr, membership, schema, *, detailed=False):
    """Choose expert by total mass, then two distinct worlds by its membership.

    membership[w,e] is the world's mean gate over its 32 equal-weight snapshots.
    For nearest-center routing this is the fraction of snapshots assigned to e.
    A distinct-world pair's squared difference is exactly
    2 * weighted_variance / (1 - sum(normalized_world_weights**2)).
    """
    membership = np.asarray(membership, np.float64)
    np.testing.assert_allclose(membership.sum(1), 1., atol=2e-7, rtol=0.)
    raw_dr = np.asarray(raw_dr, np.float64)
    nworld, nexpert = membership.shape
    lower = np.asarray(schema["lower"], np.float64)
    width = np.asarray(schema["upper"], np.float64)-lower
    y = (raw_dr-lower)/width
    mass = membership.sum(0)
    assert np.all(mass > 0)
    pi = mass/mass.sum()
    conditional = membership/mass[None]
    means = conditional.T@y
    variances = np.maximum(conditional.T@(y*y)-means*means, 0.)
    within = pi@variances
    total = y.var(0)
    between = pi@np.square(means-y.mean(0))
    np.testing.assert_allclose(within+between, total, atol=2e-9, rtol=2e-7)
    distinct_probability = 1-np.square(conditional).sum(0)
    assert np.all(distinct_probability > 0)
    pair_msd = 2*variances/distinct_probability[:, None]
    pooled_msd = pi@pair_msd
    random_msd = 2*total/(1-1/nworld)
    ratio = np.sqrt(pooled_msd/random_msd)
    groups = {"primary8": STRONG_COORDINATES, **schema["groups"]}
    result = {
        "mean_weight_by_expert": pi,
        "global_effective_experts": float(np.exp(-(pi*np.log(pi)).sum())),
        "per_parameter_names": schema["names"],
        "random_pair_rms_physical": np.sqrt(random_msd)*width,
        "same_expert_pair_rms_physical": np.sqrt(pooled_msd)*width,
        "same_expert_to_random_pair_rms_ratio": ratio,
        "within_variance_fraction": within/total,
        "between_expert_variance_fraction": between/total,
        "group_pair_rms_ratio": {
            k: float(np.sqrt(pooled_msd[ids].mean()/random_msd[ids].mean()))
            for k, ids in groups.items()
        },
        "group_mean_between_expert_variance_fraction": {
            k: float((between/total)[ids].mean()) for k, ids in groups.items()
        },
        "per_expert": [],
    }
    for e in range(nexpert):
        row = {
            "expert": e, "global_mass": float(pi[e]),
            "effective_distinct_worlds": float(1/np.square(conditional[:, e]).sum()),
            "worlds_with_positive_membership": int((membership[:, e] > 0).sum()),
            "mean_physical": means[e]*width+lower,
            "std_physical": np.sqrt(variances[e])*width,
            "distinct_world_pair_rms_physical": np.sqrt(pair_msd[e])*width,
        }
        if detailed:
            row["p10_p90_physical"] = np.asarray([
                weighted_quantiles(raw_dr[:, j], conditional[:, e], [.1, .9])
                for j in range(raw_dr.shape[1])
            ])
        result["per_expert"].append(row)
    return result


def verify_exact_pair_formula(raw_dr, membership, result):
    # Independent direct world-by-world calculation on the actual data.
    difference2 = np.square(raw_dr[:, None, :4]-raw_dr[None, :, :4])
    for e in [0, 7, 15]:
        a = membership[:, e].astype(np.float64)
        a /= a.sum()
        pairs = a[:, None]*a[None, :]
        np.fill_diagonal(pairs, 0.)
        direct = np.sqrt(np.einsum("ij,ijd->d", pairs, difference2)/pairs.sum())
        expected = np.asarray(result["per_expert"][e]["distinct_world_pair_rms_physical"][:4])
        np.testing.assert_allclose(direct, expected, atol=2e-7, rtol=2e-7)


def group_null(raw_dr, membership, schema, *, seed=20260915, repeats=100):
    """Permute complete world labels, retaining all within-world assignments."""
    y = ((raw_dr-np.asarray(schema["lower"])) /
         (np.asarray(schema["upper"])-np.asarray(schema["lower"])))[:, STRONG_COORDINATES]
    mass = membership.sum(0)
    conditional = membership/mass[None]
    pi = mass/mass.sum()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        yp = y[rng.permutation(len(y))]
        means = conditional.T@yp
        between = pi@np.square(means-yp.mean(0))
        values.append(between/yp.var(0))
    return {"repeats": repeats, "world_labels_permuted_together": True,
            "mean_between_fraction_primary8": np.asarray(values).mean(0),
            "p95_between_fraction_primary8": np.quantile(values, .95, axis=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("runs/limb_context_20260914_router_information_exploration"))
    parser.add_argument("--floor-root", type=Path, default=Path("runs/limb_context_20260915_whitening_floor_ablation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"results.json").exists():
        raise RuntimeError("Use a fresh output directory; completed audits are not overwritten.")
    protocol = json.loads((args.floor_root/"protocol.json").read_text())
    own_protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "context_sha256": protocol["context_sha256"],
        "policy_checkpoint_sha256": protocol["policy_checkpoint_sha256"],
        "center_fit_seed": 30402, "evaluation_seeds": [30403, 30404],
        "floors": FLOORS, "kmeans_seeds": KM_SEEDS,
        "EMA_seconds": [0., 2.], "temperatures_absolute": TEMPERATURES,
        "center_fit_or_update_on_evaluation_data": False,
        "PPO_or_readout_training": False, "GPU_used": False,
        "world_sampling": "32 full-memory observations/world, policy step>=500, equal world weight; same samples as previous audits.",
        "hard_assignment": "Nearest center at each observation; softmax argmax is independent of positive temperature.",
        "soft_assignment": "Expected DR exposure under each expert's gate coefficient, averaged over the 32 observations/world. This is not a measurement of actual PPO gradient magnitudes.",
        "pair_metric": "Choose an expert with probability equal to its global gate mass, draw two worlds in proportion to their membership, condition on distinct world IDs, compute RMS DR differences. Baseline draws two distinct uniform worlds.",
        "physical_normalization": "Normalize each parameter by its full sampling range when averaging different parameters; also report each coordinate in physical units.",
        "limitations": "Reuses existing frozen-original-policy traces. No retrained soft-MoE controller, no new trajectories, no assumption that similar DR guarantees identical optimal control.",
        "source_sha256": {str(p): sha(p) for p in [Path(__file__), Path("scripts/confirm_moe_router_information.py"), Path("src/intact_tracking/router_information_probe.py")]},
    }
    save_json(args.output/"protocol.json", own_protocol)
    rows, traces, verification = [], [], []
    for directory_name in ["heldout_sample_seed30403", "runtime50hz_sample_seed30404"]:
        directory = args.parent/directory_name
        trace, meta = read_trace(directory)
        assert meta["context_sha256"] == protocol["context_sha256"]
        assert meta["checkpoint_sha256"] == protocol["policy_checkpoint_sha256"]
        assert meta["motion_count"] == 129827 and meta["actual_actor_dispatch_verified_every_step"]
        seed = meta["arguments"]["seed"]
        target = args.output/f"seed{seed}"
        target.mkdir(exist_ok=True)
        raw = trace["raw_dr"].astype(np.float64)
        nworld = len(raw)
        np.testing.assert_array_equal(trace["point_worlds"], np.repeat(np.arange(nworld), 32))
        saved = load_npz(args.floor_root/("heldout30403" if seed == 30403 else "runtime50hz30404")/"targets.npz")
        np.testing.assert_array_equal(saved["worlds"], trace["point_worlds"])
        y8 = ((raw-np.asarray(meta["dr_schema"]["lower"])) /
              (np.asarray(meta["dr_schema"]["upper"])-np.asarray(meta["dr_schema"]["lower"])))[:, STRONG_COORDINATES]
        np.testing.assert_allclose(saved["targets"], y8[trace["point_worlds"]], atol=6e-8, rtol=0.)
        np.savez_compressed(target/"worlds.npz", raw_dr=raw, point_rows=trace["point_rows"],
                            point_worlds=trace["point_worlds"])
        save_json(target/"dr_schema.json", meta["dr_schema"])
        traces.append({"seed": seed, "worlds": nworld, "directory": str(directory.resolve()),
                       "trace_sha256": sha(directory/"traces.npz"),
                       "metadata_sha256": sha(directory/"result.json"),
                       "latent_recording_dt": trace["dt"]})
        for floor in FLOORS:
            reference = load_npz(args.floor_root/"models"/f"floor{floor:g}_seed731.npz")
            latent = trace["unit_latent_samples"]
            features = project(latent.reshape(-1, 64), reference).reshape(*latent.shape)
            for tau in [0., 2.]:
                filtered = features if tau == 0 else causal_ema(features, trace["full"], dt=trace["dt"], time_constant=tau)
                points = filtered[trace["point_rows"], trace["point_worlds"]]
                for km_seed in KM_SEEDS:
                    model_path = args.floor_root/"models"/f"floor{floor:g}_seed{km_seed}.npz"
                    model = load_npz(model_path)
                    np.testing.assert_array_equal(model["matrix"], reference["matrix"])
                    np.testing.assert_array_equal(model["offset"], reference["offset"])
                    distances = squared_distances(points, model["centers"])
                    hard_ids = distances.argmin(-1)
                    hard = np.eye(16, dtype=np.float32)[hard_ids]
                    stem = f"floor{floor:g}_km{km_seed}_ema{tau:g}"
                    np.savez_compressed(target/f"{stem}_distances.npz", distances=distances)
                    for temperature in [None, *TEMPERATURES]:
                        mode = "hard" if temperature is None else f"soft_t{temperature:.8g}"
                        gates = hard if temperature is None else gate_weights(distances, top_k=16, temperature=temperature)
                        np.testing.assert_array_equal(gates.argmax(-1), hard_ids)
                        membership = gates.astype(np.float64).reshape(nworld, 32, 16).mean(1)
                        detailed = temperature is None and km_seed == 731
                        metrics = proximity(raw, membership, meta["dr_schema"], detailed=detailed)
                        if detailed:
                            metrics["world_mean_max_assignment_fraction"] = float(membership.max(1).mean())
                            metrics["worlds_routed_to_only_one_expert_over_32_samples_fraction"] = float(((membership > 0).sum(1) == 1).mean())
                            metrics["world_label_permutation_null"] = group_null(raw, membership, meta["dr_schema"])
                            verify_exact_pair_formula(raw, membership, metrics)
                            verification.append({"seed": seed, "model": stem, "exact_distinct_world_pair_formula_verified": True})
                        row = {"name": f"seed{seed}_{stem}_{mode}", "trace_seed": seed, "floor": floor,
                               "kmeans_seed": km_seed, "EMA_seconds": tau, "temperature": temperature,
                               "mode": "hard" if temperature is None else "soft", "model_sha256": sha(model_path),
                               "gate": gate_summary(gates), "proximity": metrics}
                        np.savez_compressed(target/f"{stem}_{mode}_membership.npz", membership=membership)
                        rows.append(row)
                    print(json.dumps({"trace": seed, "floor": floor, "EMA": tau, "KM": km_seed,
                                      "hard_pair_rms_ratios": rows[-5]["proximity"]["group_pair_rms_ratio"],
                                      "hard_coordinate_ratios": np.asarray(rows[-5]["proximity"]["same_expert_to_random_pair_rms_ratio"])[STRONG_COORDINATES].tolist()}), flush=True)
                if tau:
                    del filtered
            del features
        del trace
        save_json(args.output/"results.partial.json", rows)
    save_json(args.output/"results.json", {"protocol_sha256": sha(args.output/"protocol.json"),
                                           "traces": traces, "rows": rows})
    save_json(args.output/"verification.json", {
        "passed": True, "rows": len(rows), "expected_rows": 2*2*2*3*5,
        "all_temperature_argmax_assignments_identical": True,
        "variance_decomposition_checked_every_row": True,
        "previous_world_targets_reproduced": True,
        "distinct_world_pair_formula_checks": verification,
    })
    assert len(rows) == 120


if __name__ == "__main__":
    main()
