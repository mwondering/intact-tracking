"""Verify artifacts produced by the end-to-end smoke launcher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


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


def verify(manifest_path: Path, run_dir: Path, expected_transitions: int) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    run_config = _read_json(run_dir / "run_config.json")
    history = _read_json(run_dir / "history.json")
    checkpoint = run_dir / "last.pt"

    if manifest.get("total_transitions") != expected_transitions:
        raise RuntimeError(
            "Unexpected rollout size: "
            f"{manifest.get('total_transitions')} != {expected_transitions}"
        )
    worlds = manifest.get("world_ids", [])
    if len(set(worlds)) < 3:
        raise RuntimeError(f"Smoke rollout needs at least three worlds, got {worlds}")
    if run_config.get("method") != "INTACT":
        raise RuntimeError("Training run is not marked as INTACT")
    if run_config.get("model", {}).get("context_tokens") != 16:
        raise RuntimeError("Model did not preserve the fixed 16-token context contract")
    arguments = run_config.get("arguments", {})
    expected_arguments = {
        "batch_size": 1,
        "max_train_batches": 1,
        "max_validation_batches": 1,
        "block_size": 5,
        "horizon": 5,
        "allow_padded_context": False,
    }
    for name, expected in expected_arguments.items():
        if arguments.get(name) != expected:
            raise RuntimeError(f"Unexpected smoke argument {name}: {arguments.get(name)!r}")
    if len(history) != 1:
        raise RuntimeError(f"Expected one smoke epoch, got {len(history)}")
    epoch = history[0]
    if epoch.get("train_batches") != 1 or epoch.get("validation_batches") != 1:
        raise RuntimeError(
            "Smoke run did not execute exactly one train and one validation batch: "
            f"{epoch.get('train_batches')}, {epoch.get('validation_batches')}"
        )
    _assert_finite_metrics("train", epoch.get("train", {}))
    _assert_finite_metrics("validation", epoch.get("validation", {}))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint is missing or empty: {checkpoint}")

    return {
        "status": "passed",
        "transitions": manifest["total_transitions"],
        "world_ids": worlds,
        "train_batches": epoch["train_batches"],
        "validation_batches": epoch["validation_batches"],
        "train_loss": epoch["train"]["loss"],
        "validation_loss": epoch["validation"]["loss"],
        "checkpoint": str(checkpoint.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-transitions", type=int, required=True)
    args = parser.parse_args()
    result = verify(args.manifest.resolve(), args.run_dir.resolve(), args.expected_transitions)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
