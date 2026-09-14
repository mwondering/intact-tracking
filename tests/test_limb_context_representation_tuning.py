from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from intact_tracking.cli.forward_predictor_train import (
    _loss_weight_payload,
    _validate_arguments,
    _wandb_payload,
    build_parser,
)
from intact_tracking.forward_predictor_objective import (
    ForwardPredictorLossConfig,
    _counterfactual_representation_loss,
)

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/experiments/limb_context_memory350_representation.json"


def _angle_gradient(scale: float, multiplier: float = 1.0, response_difference: float = 0.5):
    # Two unit embeddings at distance 0.75. Increasing the target through this
    # distance must change gradient descent from attraction to separation.
    angle = torch.tensor(2.0 * math.asin(0.375), requires_grad=True)
    latent = torch.stack((torch.tensor([1.0, 0.0]), torch.stack((angle.cos(), angle.sin()))))
    response = torch.zeros(2, 5, 70)
    response[1] = response_difference
    valid = torch.ones(2, dtype=torch.bool)
    loss, metrics, _ = _counterfactual_representation_loss(
        latent, latent.detach(), response, valid, valid, valid, torch.arange(2),
        response_distance_scale=scale,
        representation_relation_weight=multiplier,
    )
    return torch.autograd.grad(loss, angle)[0], metrics


def test_larger_distance_target_changes_gradient_from_attraction_to_separation():
    old_gradient, old = _angle_gradient(1.0)
    new_gradient, new = _angle_gradient(0.75)
    stronger_gradient, _ = _angle_gradient(0.75, 2.0)
    assert old_gradient > 0
    assert new_gradient < 0
    torch.testing.assert_close(stronger_gradient, 2.0 * new_gradient)
    assert old["latent_target_distance_mean"].item() == pytest.approx(2.0 / 3.0)
    assert new["latent_target_distance_mean"].item() == pytest.approx(0.8)
    for percentile in (10, 50, 90):
        assert new[f"latent_target_distance_batch_p{percentile}"].item() == pytest.approx(0.8)


def test_identical_responses_still_have_zero_target_distance():
    for scale in (1.0, 0.75):
        gradient, metrics = _angle_gradient(scale, response_difference=0.0)
        assert gradient > 0
        assert metrics["latent_target_distance_mean"].item() == 0.0


def test_relation_multiplier_does_not_strengthen_same_world_attraction():
    results = []
    for multiplier in (1.0, 2.0):
        latent = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        positive = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        valid = torch.ones(2, dtype=torch.bool)
        loss, metrics, _ = _counterfactual_representation_loss(
            latent, positive, torch.zeros(2, 5, 70), valid, valid, valid,
            torch.zeros(2, dtype=torch.long),
            response_distance_scale=0.75,
            representation_relation_weight=multiplier,
        )
        results.append((loss.detach(), torch.autograd.grad(loss, latent)[0]))
        assert metrics["latent_relation_pairs"].item() == 0
        assert all(torch.isfinite(value) for value in metrics.values())
    torch.testing.assert_close(results[0], results[1])


def test_experiment_preset_reaches_stage1_cli_and_logging(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "scheduler_representation_tuning", ROOT / "scripts/run_limb_context_experiment.py"
    )
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    legacy = {job["name"]: job["command"] for job in scheduler.jobs(tmp_path)}
    preset = json.loads(PRESET.read_text())
    # Isolate the loss override here; the full preset's DR contract is tested separately.
    (tmp_path / "experiment_layout.json").write_text(json.dumps({"stage1_loss": preset["stage1_loss"]}))
    current = {job["name"]: job["command"] for job in scheduler.jobs(tmp_path)}
    for name in ("smoke_stage1", "stage1"):
        command = current[name]
        args = build_parser().parse_args(
            command[command.index("intact_tracking.cli.forward_predictor_train") + 1:]
        )
        _validate_arguments(args)
        assert args.representation_weight == 0.01
        assert args.representation_relation_weight == 2.0
        assert args.response_distance_scale == 0.75
    for name in legacy.keys() - {"smoke_stage1", "stage1"}:
        assert current[name] == legacy[name]
    loss_config = ForwardPredictorLossConfig(**preset["stage1_loss"])
    recorded = _loss_weight_payload(loss_config)
    assert recorded["representation_relation_weight"] == 2.0
    assert recorded["response_distance_scale"] == 0.75
    logged = _wandb_payload({
        "update": 1, "optimizer_steps": 4, "learning_rate_model": 0.0003,
        "replay_size": 8, "samples_generated": 8, "transitions": 40,
        "optimization_train": {}, "fixed_probe": {"latent_target_distance_mean": 0.8},
    })
    assert logged["fixed_probe/latent_target_distance_mean"] == 0.8
