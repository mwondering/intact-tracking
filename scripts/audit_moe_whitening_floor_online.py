"""Compare selected whitening floors with causal online center updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate_moe_whitening_floor import (
    affine_predict, assert_trace_identity, model_name, point_targets, read_npz,
    regression_metrics, save_json, sha,
)
from confirm_moe_router_information import fine_stability, read_trace
from intact_tracking.memory350_metric_router import FrozenMetricRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    protocol = json.loads((root/"protocol.json").read_text())
    selection = json.loads((root/"selection.json").read_text())
    seal = json.loads((root/"selection_seal.json").read_text())
    assert sha(root/"selection.json") == seal["selection_sha256"]
    floors = [.01, selection["chosen_lower_floors"]["fixed_absolute"]]
    audit_protocol = {
        "floors": floors, "kmeans_seed": 731, "temperature_mode": "fixed_absolute",
        "rollout_steps": 24, "center_rate": .01, "max_mean_weight_tv": .02,
        "selection_sha256": seal["selection_sha256"],
        "scope": "Same original-policy50Hz trace; no PPO update; original per-floor readouts frozen.",
    }
    save_json(root/"online_audit_protocol.json", audit_protocol)
    trace, meta = read_trace(Path(protocol["parent"])/"runtime50hz_sample_seed30404")
    assert_trace_identity(meta, protocol)
    assert trace["dt"] == .02
    fit_rows = json.loads((root/"fit_results.json").read_text())
    latent = torch.from_numpy(trace["unit_latent_samples"])
    valid = torch.from_numpy(trace["full"])
    y = point_targets(trace, meta)
    rows = []
    for floor in floors:
        candidate = next(row for row in fit_rows if row["floor"] == floor and row["kmeans_seed"] == 731
                         and row["temperature_mode"] == "fixed_absolute")
        model = read_npz(root/"models"/f"{model_name(floor,731)}.npz")
        readout = read_npz(root/"readouts"/f"{candidate['name']}.npz")
        router = FrozenMetricRouter(model["matrix"], model["offset"], model["centers"][None],
                                    [candidate["temperature"]], group_top_k=16, time_constant=2., clip_unit_range=False)
        initial = router.centers.clone()
        weights = np.empty((*trace["full"].shape, 16), dtype=np.float32)
        updates, buffer = [], []
        with torch.inference_mode():
            for step in range(len(latent)):
                gate = router.advance(latent[step], valid[step], dt=.02)
                weights[step] = gate.numpy()
                buffer.append(router.filtered_features[valid[step]].clone())
                if (step+1) % 24 == 0:
                    stored = gate.clone()
                    update = router.update_centers_from_features(torch.cat(buffer), center_rate=.01, max_mean_weight_tv=.02)
                    torch.testing.assert_close(stored, gate, atol=0, rtol=0)
                    assert update["mean_weight_tv"] <= .02 + 1e-7
                    update["after_step"] = step
                    updates.append(update)
                    buffer.clear()
        prediction = affine_predict(weights[trace["point_rows"], trace["point_worlds"]], readout)
        assert len(updates) == 62 and not list(router.parameters())
        row = {"floor": floor, "candidate": candidate["name"], "updates": updates,
               "center_shift_RMS": float((router.centers-initial).square().mean().sqrt()),
               "metrics": regression_metrics(y, prediction, meta["dr_schema"]["groups"]),
               "stability": fine_stability(weights, trace, readout),
               "recorded_gates_unchanged_verified": True, "has_trainable_parameters": False}
        np.savez_compressed(root/f"online_floor{floor:g}.npz", final_centers=router.centers.numpy(),
                            prediction=prediction[:, [0,1,2,3,5,6,7,8]].astype(np.float32))
        rows.append(row)
        print(json.dumps({"floor": floor, "R2": row["metrics"]["strong8_r2"], "loads": row["metrics"]["load_r2"],
                          "updates": len(updates), "maximum_update_TV": max(x["mean_weight_tv"] for x in updates)}), flush=True)
    save_json(root/"online_audit.json", {"protocol": audit_protocol, "trace_seed": meta["arguments"]["seed"],
                                        "rows": rows, "source_sha256": sha(Path(__file__))})


if __name__ == "__main__":
    main()
