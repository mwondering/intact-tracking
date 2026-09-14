"""Add causal static-code averaging to an audited student; no weights are fitted."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from intact_tracking.adaptation_context_memory import physics_mean_only_configuration
from intact_tracking.adaptation_reward_contract import (
    assert_fixed_reward_checkpoint,
    capture_original_rewards,
)
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.rollout.mjlab_adapter import _sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("Output must be a new subdirectory of this repository")
    if output.exists():
        raise FileExistsError(output)
    source_sha = _sha256(args.checkpoint)
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prepared = prepare_rollout(checkpoint_file=TRACKER, num_envs=1, motion_path=DATASET, motion_file=None)
    assert_fixed_reward_checkpoint(original, capture_original_rewards(prepared.env))
    actor = physics_mean_only_configuration(
        OmegaConf.to_container(original["cfg"].agent.actor, resolve=True), args.horizon
    )
    payload = dict(original)
    payload["cfg"] = copy.deepcopy(original["cfg"])
    payload["cfg"].agent.actor = OmegaConf.create(actor)
    # The source optimizer trained the unfrozen encoder, and cannot be resumed
    # under this inference-only frozen-encoder contract.
    payload.pop("optimizer_state_dict", None)
    metadata = copy.deepcopy(original["residual_policy"])
    audit = {
        "source_checkpoint": str(args.checkpoint.resolve()), "source_sha256": source_sha,
        "method": "causal mean of seven static environment coordinates; never action averaging",
        "horizon": args.horizon, "learned_tensors_changed": 0,
        "training_updates_after_architecture_change": 0,
        "reset": "zero mean/count only for worlds starting a new episode",
        "no_future_frames_or_simulator_labels": True,
        "architecture_source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/intact_tracking/adaptation_context_memory.py",
                ROOT / "src/intact_tracking/adaptation_policy.py",
                ROOT / "src/intact_tracking/context_export.py",
            )
        },
    }
    metadata.update(context_latent_mean=True, context_mean_only=True,
                    context_mean_horizon=args.horizon, architecture_postprocessing=audit)
    payload["residual_policy"] = metadata
    payload["cfg"].residual_policy = OmegaConf.create(metadata)
    if _sha256(args.checkpoint) != source_sha:
        raise RuntimeError("Source checkpoint changed during conversion")
    output.mkdir(parents=True)
    torch.save(payload, output / "checkpoint_final.pt")
    restored = torch.load(output / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    for key, value in original["actor_state_dict"].items():
        torch.testing.assert_close(restored["actor_state_dict"][key], value, atol=0, rtol=0)
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
