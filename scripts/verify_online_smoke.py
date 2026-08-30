"""Verify artifacts produced by the pure-online smoke launcher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _assert_finite_metrics(name: str, metrics: dict[str, Any]) -> None:
    if not metrics:
        raise RuntimeError(f"{name} metrics are empty")
    for metric, value in metrics.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"{name}.{metric} is not finite: {value!r}")


def verify(
    run_dir: Path,
    expected_num_envs: int,
    expected_world_size: int = 1,
) -> dict[str, Any]:
    run_config = _read_json(run_dir / "run_config.json")
    history = _read_json(run_dir / "history.json")
    normalization = _read_json(run_dir / "normalization.json")
    checkpoint_path = run_dir / "last.pt"

    if run_config.get("method") != "INTACT-online":
        raise RuntimeError("Training run is not marked as pure-online INTACT")
    model = run_config.get("model", {})
    if model.get("context_tokens") != 16:
        raise RuntimeError("Model did not preserve the fixed 16-token context contract")
    arguments = run_config.get("arguments", {})
    expected_arguments = {
        "num_envs": expected_num_envs,
        "updates": 1,
        "gradient_steps_per_update": 1,
        "batch_size": 1,
        "block_size": 5,
        "horizon": 5,
    }
    for name, expected in expected_arguments.items():
        if arguments.get(name) != expected:
            raise RuntimeError(f"Unexpected smoke argument {name}: {arguments.get(name)!r}")
    distributed = run_config.get("distributed", {})
    if distributed.get("world_size") != expected_world_size:
        raise RuntimeError(f"Unexpected distributed world size: {distributed.get('world_size')!r}")
    if distributed.get("global_num_envs") != expected_num_envs * expected_world_size:
        raise RuntimeError("Smoke run recorded an incorrect global environment count")
    if distributed.get("global_batch_size") != expected_world_size:
        raise RuntimeError("Smoke run recorded an incorrect global DDP batch size")

    rollout = run_config.get("rollout", {})
    if rollout.get("tracker_frozen") is not True:
        raise RuntimeError("Smoke rollout did not certify a frozen tracker")
    contract = rollout.get("domain_randomization_contract", "")
    if "fixed per vector slot" not in contract or "never resampled" not in contract:
        raise RuntimeError(f"Unexpected DR contract: {contract!r}")
    if not rollout.get("startup_events"):
        raise RuntimeError("No startup DR events were retained")
    if not rollout.get("fixed_dr_model_fields"):
        raise RuntimeError("No randomized MuJoCo model fields were monitored")

    if len(history) != 1:
        raise RuntimeError(f"Expected one online update, got {len(history)}")
    record = history[0]
    if record.get("optimizer_steps") != 1 or record.get("update") != 1:
        raise RuntimeError(f"Smoke run did not execute exactly one optimizer step: {record}")
    minimum_steps = (16 + 5) * 5
    if record.get("env_steps", 0) < minimum_steps:
        raise RuntimeError(f"Online update started before full context: {record['env_steps']}")
    minimum_transitions = expected_num_envs * expected_world_size * minimum_steps
    if record.get("transitions", 0) < minimum_transitions:
        raise RuntimeError("Online transition count is smaller than the causal warmup")
    if record.get("replay_size", 0) < 1 or record.get("samples_generated", 0) < 1:
        raise RuntimeError("Online replay did not contain a trainable causal sample")
    _assert_finite_metrics("train", record.get("train", {}))
    _assert_finite_metrics("optimizer", {"gradient_norm": record.get("gradient_norm")})

    for name in (
        "observation_mean",
        "observation_std",
        "proprio_mean",
        "proprio_std",
        "action_mean",
        "action_std",
    ):
        values = normalization.get(name, [])
        if not values or not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Invalid online normalization field: {name}")

    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint is missing or empty: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("distributed", {}).get("world_size") != expected_world_size:
        raise RuntimeError("Checkpoint has an incorrect distributed world size")
    if checkpoint.get("tracker", {}).get("frozen") is not True:
        raise RuntimeError("Checkpoint does not mark the tracker as frozen")
    online_state = checkpoint.get("online_state", {})
    if online_state.get("replay_size", 0) < 1:
        raise RuntimeError("Checkpoint has no online replay progress")
    if online_state.get("dr_invariance_checks") != online_state.get("synchronous_resets"):
        raise RuntimeError("Not every episode reset received a fixed-DR invariance check")

    return {
        "status": "passed",
        "env_steps": record["env_steps"],
        "transitions": record["transitions"],
        "replay_size": record["replay_size"],
        "motions_seen": record["motions_seen"],
        "train_loss": record["train"]["loss"],
        "checkpoint": str(checkpoint_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-num-envs", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, default=1)
    args = parser.parse_args()
    result = verify(
        args.run_dir.resolve(),
        args.expected_num_envs,
        args.expected_world_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
