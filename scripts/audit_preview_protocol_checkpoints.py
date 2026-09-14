"""CPU-only engineering checks for a matched full-data preview/baseline pair."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    checkpoints = {arm: torch.load(getattr(args, arm), map_location="cpu", weights_only=False)
                   for arm in ("preview", "baseline")}
    metadata = {arm: checkpoint["residual_policy"] for arm, checkpoint in checkpoints.items()}
    preview, baseline = metadata["preview"], metadata["baseline"]
    for key in ("tracker_sha256", "motion_sha256", "motion_count", "dr_profile", "dataset", "reward_contract"):
        if preview[key] != baseline[key]:
            raise ValueError(f"Pair differs in {key}")
    if preview["physics"]["runtime_payload_audit"] != baseline["physics"]["runtime_payload_audit"]:
        raise ValueError("Pair's actual four-limb payload samples differ")
    source = torch.load(preview["tracker_checkpoint"], map_location="cpu", weights_only=False)
    tracker_state = source["actor_state_dict"]
    results = {}
    for arm, checkpoint in checkpoints.items():
        actor = checkpoint["actor_state_dict"]
        for key, original in tracker_state.items():
            if not torch.equal(actor["tracker." + key], original):
                raise ValueError(f"Frozen tracker was changed: {arm} {key}")
        for network in ("actor_state_dict", "critic_state_dict"):
            if any(not torch.isfinite(value).all() for value in checkpoint[network].values()
                   if isinstance(value, torch.Tensor)):
                raise ValueError(f"Nonfinite weights: {arm} {network}")
        meta = metadata[arm]
        if meta["reward_changes"] or meta["distance_scalars"] or meta["extra_current_state_physics_privilege"]:
            raise ValueError("Unexpected rewards or extra privileges")
        initial_path = Path(getattr(args, arm)).parent / "checkpoint_initial.pt"
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        changed = {network: sum(not torch.equal(value, initial[network][key])
                               for key, value in checkpoint[network].items())
                   for network in ("actor_state_dict", "critic_state_dict")}
        results[arm] = {"input_audit": meta["input_audit"], "changed_state_tensors": changed,
                        "completed_updates": checkpoint["completed_updates"],
                        "frozen_tracker_tensors_unchanged": len(tracker_state),
                        "all_weights_finite": True}
    print(json.dumps({"dataset": preview["dataset"], "dr_profile": preview["dr_profile"],
                      "actual_payload_samples_identical": True,
                      "reward_sha256": preview["reward_contract"]["sha256"],
                      "arms": results, "engineering_only_not_efficacy": True}, indent=2))


if __name__ == "__main__":
    main()
