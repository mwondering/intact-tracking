"""Independently replay saved raw histories and verify the 50s sampling audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from intact_tracking.memory350_inference import load_memory350_checkpoint


ROOT = Path("runs/limb_context_20260917_per_limb_half_zero_50s")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((root / "shard_00/metadata.json").read_text())
    all_fixed = manifest.get("sampling_mode") == "all-fixed-half-nominal"
    cp = load_memory350_checkpoint(manifest["checkpoint"], device="cuda")
    assert cp.sha256 == manifest["checkpoint_sha256"]
    records = []
    for shard in range(8):
        folder = root / f"shard_{shard:02d}"
        meta = json.loads((folder / "metadata.json").read_text())
        assert meta["complete"] and meta["steps"] == 2500
        for step in (500, 2500):
            saved = torch.load(folder / f"encoder_audit_{step:06d}.pt", weights_only=False)

            def normalized(raw):
                raw = raw.cuda()
                return torch.cat(((raw[..., :71]-cp.state_mean)/cp.state_std,
                                  (raw[..., 71:100]-cp.action_mean)/cp.action_std,
                                  (raw[..., 100:]-cp.state_mean)/cp.state_std), dim=-1)

            short, long = normalized(saved["short"]), normalized(saved["long"])
            actual = cp.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                                saved["short_valid"].cuda(), long, saved["long_valid"].cuda())
            expected = saved["z"].cuda()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            error = torch.linalg.vector_norm(torch.nn.functional.normalize(actual, dim=-1)
                                             - torch.nn.functional.normalize(expected, dim=-1), dim=-1)
            records.append(dict(shard=shard, step=step, worlds=len(actual),
                                max_raw_error=float((actual-expected).abs().max()),
                                max_unit_l2_error=float(error.max())))
    with np.load(root / "analysis/environment_comparison.npz") as saved:
        bank = {k: saved[k] for k in saved.files}
    with np.load(manifest["parameters"]) as saved:
        np.testing.assert_array_equal(bank["new_world"], saved["world"])
        np.testing.assert_array_equal(bank["new_physics"], saved["mixed" if all_fixed else "per_limb_half_zero"])
        np.testing.assert_array_equal(bank["old_physics"], saved["original"])
    assert len(np.unique(bank["new_world"])) == 16384
    if not all_fixed:
        np.testing.assert_array_equal(bank["new_physics"][:, :34], bank["old_physics"][:, :34])
    for name in (("old", "payload_only", "new") if all_fixed else ("old", "new")):
        score = np.linalg.norm(bank[f"{name}_centroid"]-bank["nominal_center"], axis=1)
        np.testing.assert_array_equal(score, bank[f"{name}_centroid_radius"])
        width = bank["edges"][1] - bank["edges"][0]
        assert ((score >= bank["edges"][0]) & (score <= bank["edges"][-1])).all()
        labels = np.minimum(np.floor(score/width).astype(int), 7)
        np.testing.assert_array_equal(labels, bank[f"{name}_centroid_radius_class"])
    result = dict(encoder_replay=records, replayed_queries=sum(r["worlds"] for r in records),
                  all_16384_worlds_accounted_for=True, independent_center_distance_and_bins_match=True,
                  all_background_DR_unchanged=not all_fixed, actual_physics_matches_new_sampling_bank=True,
                  all_shards_complete_2500_steps_50s=True)
    (root / "analysis/verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
