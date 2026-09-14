"""Interpolate aligned pretrained policy cores into one inference network.

The receiver's privileged encoder, residual correction and feature input mode
are preserved. No ensembles or per-environment model selection are introduced.
This is a development candidate, not evidence that tracking equivalence holds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def merge(receiver, donor, fraction, module="tracker.mlp"):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Donor fraction must be between zero and one")
    if module not in ("tracker.mlp", "tracker.reference_encoder"):
        raise ValueError("Only aligned policy-core or reference-encoder transfer is supported")
    result = copy.deepcopy(receiver)
    for key, value in receiver.items():
        if key.startswith("tracker.") and "normalizer." in key:
            if key not in donor or not torch.equal(value, donor[key]):
                raise ValueError(f"Frozen preprocessing normalization differs: {key}")
    keys = [key for key in receiver if key.startswith(module + ".")]
    if not keys:
        raise ValueError(f"No aligned module weights: {module}")
    for key in keys:
        value = donor[key]
        if value.shape != receiver[key].shape or value.dtype != receiver[key].dtype:
            raise ValueError(f"Policy-core schema mismatch: {key}")
        result[key] = (
            value.clone() if fraction == 1.0 else torch.lerp(receiver[key], value, fraction)
        )
        if not torch.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite merged policy weight: {key}")
    return result, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-fraction", type=float, required=True)
    parser.add_argument(
        "--module", choices=("tracker.mlp", "tracker.reference_encoder"), default="tracker.mlp"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    receiver = torch.load(args.receiver, map_location="cpu", weights_only=False)
    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    if receiver["cfg"].agent.actor.class_name != donor["cfg"].agent.actor.class_name:
        raise ValueError("Receiver and donor must have the same actor class")
    receiver["actor_state_dict"], keys = merge(
        receiver["actor_state_dict"], donor["actor_state_dict"], args.donor_fraction, args.module
    )
    # PPO checkpoints expose aliases used by different loaders. Keep every
    # action state synchronized; never leave an old actor behind in rsl_rl.
    if "policy" in receiver:
        receiver["policy"] = receiver["actor_state_dict"]
    if isinstance(receiver.get("rsl_rl"), dict):
        receiver["rsl_rl"] = dict(receiver["rsl_rl"])
        receiver["rsl_rl"]["actor_state_dict"] = receiver["actor_state_dict"]
        receiver["rsl_rl"].pop("optimizer_state_dict", None)
    metadata = OmegaConf.to_container(receiver["cfg"].residual_policy, resolve=True)
    metadata.update(
        {
            "version": "single_module_parameter_interpolation_v2",
            "merge": {
                "receiver": str(args.receiver.resolve()),
                "receiver_sha256": sha256(args.receiver),
                "donor": str(args.donor.resolve()),
                "donor_sha256": sha256(args.donor),
                "donor_fraction": args.donor_fraction,
                "module": args.module,
                "merged_keys": keys,
                "input_mode": "receiver unchanged",
                "inference_contract": "one policy core, one receiver encoder, one residual; no ensemble",
                "training_contract": "weights-only warm start; optimizer intentionally removed",
                "source_sha256": sha256(__file__),
            },
        }
    )
    receiver["cfg"].residual_policy = OmegaConf.create(metadata)
    receiver["residual_policy"] = metadata
    receiver["iter"] = 0
    receiver.pop("optimizer_state_dict", None)
    args.output_dir.mkdir(parents=True)
    torch.save(receiver, args.output_dir / "checkpoint_0.pt")
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["merge"]), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
