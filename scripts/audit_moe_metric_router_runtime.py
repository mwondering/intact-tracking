"""Replay the selected continuous router with explicit post-rollout center updates.

This is a causal, fixed-policy-trajectory audit, not a new PPO experiment.
No readout, temperature, projection, or update hyperparameter is fitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_moe_metric_heads import mixed, restore_router
from confirm_moe_router_information import fine_stability, read_trace
from intact_tracking.memory350_moe_policy import HardRoutedMLP
from intact_tracking.router_information_probe import affine_predict, regression_metrics


def dense_equivalence(checkpoint, latent, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    results = {}
    for label, width, output, dims, key, prefix in (
        ("actor", 1645, 29, (512, 256, 128), "actor_state_dict", "residual_mlp."),
        ("critic", 6330, 1, (1024, 512, 256, 128), "critic_state_dict", "mlp."),
    ):
        module = HardRoutedMLP(width, output, compression_dims=dims,
                               fusion="concat", tracker_action_dim=29).to(device)
        module.load_state_dict({k.removeprefix(prefix): v for k, v in state[key].items()
                                if k.startswith(prefix)}, strict=True)
        observation = torch.cat((torch.randn(len(latent), width + 29, device=device), latent), -1)
        onehot = torch.nn.functional.one_hot(module.router(latent), 16).float()
        with torch.no_grad():
            reference = module(observation)
            sparse = mixed(module, observation, onehot, dense=False)
            dense = mixed(module, observation, onehot, dense=True)
            torch.testing.assert_close(sparse, reference, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(dense, reference, atol=2e-6, rtol=2e-6)
        results[label] = {
            "strict_checkpoint_load": True,
            "sparse_max_absolute_error": float((sparse-reference).abs().max()),
            "dense_max_absolute_error": float((dense-reference).abs().max()),
            "passed": True,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(681)
    root = args.root
    name = "product_dr8_gk4_t1"
    selection_sha = hashlib.sha256((root/"final_selection.json").read_bytes()).hexdigest()
    seal = json.loads((root/"selection_seal.json").read_text())
    assert selection_sha == seal["final_selection_sha256"]
    candidate = next(x for x in json.loads((root/"final_selection.json").read_text())["candidates"]
                     if x["name"] == name)
    # Inherit the original training's cadence/rate, rather than select on test DR.
    policy_root = Path("runs/limb_context_20260914_uniform_moe16_8x1024_u1000/moe")
    original = json.loads((policy_root/"run_config.json").read_text())["arguments"]
    rollout_steps, rate = original["rollout_steps"], original["router_center_rate"]
    protocol = {
        "candidate": name, "selection_sha256": selection_sha,
        "rollout_steps": rollout_steps, "center_rate": rate,
        "max_mean_weight_tv": .02,
        "update_timing": "After each complete 24-step batch; all gates already recorded remain unchanged.",
        "fit_on_test_labels": False,
        "scope": "Fixed original-policy trajectories; centers adapt to latent features only. No PPO/expert learning.",
    }
    (root/"online_update_audit_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    trace, meta = read_trace(args.trace)
    assert trace["dt"] == .02
    with np.load(root/"temporal_readouts"/f"{candidate['evaluation_name']}.npz") as f:
        readout = {k: f[k] for k in f.files}
    router = restore_router(root/"exports"/f"{name}.pt", args.device)
    initial = router.centers.detach().clone()
    latent = torch.from_numpy(trace["unit_latent_samples"]).to(args.device)
    valid = torch.from_numpy(trace["full"]).to(args.device)
    weights = np.empty((*trace["full"].shape, 16), dtype=np.float32)
    updates, buffer = [], []
    with torch.inference_mode():
        for step in range(len(latent)):
            gate = router.advance(latent[step], valid[step], dt=trace["dt"])
            weights[step] = gate.cpu().numpy()
            buffer.append(router.filtered_features[valid[step]].clone())
            if (step+1) % rollout_steps == 0:
                cached = gate.clone()
                update = router.update_centers_from_features(
                    torch.cat(buffer), center_rate=rate, max_mean_weight_tv=.02)
                torch.testing.assert_close(gate, cached, atol=0, rtol=0)
                assert update["mean_weight_tv"] <= .02 + 1e-7
                update["after_step"] = step
                updates.append(update)
                buffer.clear()
        final = router.centers.detach().clone()
    schema = meta["dr_schema"]
    lower, upper = np.asarray(schema["lower"]), np.asarray(schema["upper"])
    y = (trace["raw_dr"][trace["point_worlds"]]-lower)/(upper-lower)
    prediction = affine_predict(weights[trace["point_rows"], trace["point_worlds"]], readout)
    report = {
        "protocol": protocol, "trace_seed": meta["arguments"]["seed"],
        "updates": updates,
        "center_shift_RMS": float((final-initial).square().mean().sqrt()),
        "center_shift_max_abs": float((final-initial).abs().max()),
        "mean_update_TV": float(np.mean([x["mean_weight_tv"] for x in updates])),
        "max_update_TV": max(x["mean_weight_tv"] for x in updates),
        "metrics_using_frozen_original_readout": regression_metrics(y, prediction, schema["groups"]),
        "native_cadence_stability": fine_stability(weights, trace, readout),
        "recorded_gates_immutable_verified": True,
        "has_trainable_router_parameters": bool(list(router.parameters())),
        "dense_onehot_matches_original": dense_equivalence(
            policy_root/"checkpoint_update_001000.pt", latent[500], args.device),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    np.savez_compressed(root/"online_update_audit.npz", initial_centers=initial.cpu().numpy(),
                        final_centers=final.cpu().numpy(), predictions=prediction,
                        world0_gate_trace=weights[:, 0])
    (root/"online_update_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"updates": len(updates), "center_shift": report["center_shift_RMS"],
                      "R2": report["metrics_using_frozen_original_readout"]["strong8_r2"],
                      "dense_equivalence": report["dense_onehot_matches_original"]}))


if __name__ == "__main__":
    main()
