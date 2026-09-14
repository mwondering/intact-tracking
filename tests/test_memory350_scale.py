"""Check encoder scaling, matched initialization and controlled training settings."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch
import numpy as np

from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_scale_model import Memory350ScaleConfig, Memory350ScalePredictor, parameter_counts
from intact_tracking.memory350_scale_comparison import reference_contract
from intact_tracking.memory350_objective import Memory350Objective
from intact_tracking.forward_predictor_objective import ForwardPredictorLossConfig
from test_memory350 import _replay_batch


def test_full_size_counts_common_initial_parameters_and_rng():
    torch.manual_seed(717)
    reference = Memory350Predictor(Memory350Config())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(717)
    scaled = Memory350ScalePredictor()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert parameter_counts(reference) == {
        "total": 20095126, "context_encoder": 1035328, "predictor": 19059798,
    }
    assert parameter_counts(scaled) == {
        "total": 21086486, "context_encoder": 2026688, "predictor": 19059798,
    }
    for name, tensor in reference.state_dict().items():
        assert torch.equal(tensor, scaled.state_dict()[name]), name
    added = set(scaled.state_dict()) - set(reference.state_dict())
    assert added
    assert all(name.startswith(("context_encoder.chunk_encoder.layers.1.",
                                "context_encoder.memory_encoder.layers.2.",
                                "context_encoder.memory_encoder.layers.3.",
                                "context_encoder.transformer.layers.2.",
                                "context_encoder.transformer.layers.3.")) for name in added)


def test_expanded_model_trains_all_layers_and_preserves_existing_checkpoint_schema():
    config = Memory350ScaleConfig(transformer_dim=32, transformer_depth=1,
                                  transformer_heads=4, context_dim=16, context_heads=4)
    model = Memory350ScalePredictor(config)
    objective = Memory350Objective(model, ForwardPredictorLossConfig(
        representation_relation_weight=2, response_distance_scale=.75))
    batch = _replay_batch()
    result = objective(batch)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for module in (model.context_encoder.chunk_encoder,
                   model.context_encoder.memory_encoder, model.context_encoder.transformer):
        for layer in module.layers:
            assert sum(p.grad.abs().sum() for p in layer.parameters()) > 0
    restored = Memory350Predictor(Memory350Config(**asdict(config)))
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval(); restored.eval()
    args = (batch["history_state"], batch["history_action"], batch["history_next_state"],
            batch["history_valid"], batch["memory_interactions"], batch["memory_valid"])
    torch.testing.assert_close(restored.context_encoder(*args), model.context_encoder(*args), rtol=0, atol=0)


def test_training_defaults_only_change_context_depths_and_require_fixed_reference():
    from intact_tracking.cli.forward_memory_train import build_parser as original_parser
    from intact_tracking.cli.forward_memory_scale_train import build_parser, _validate_arguments
    flags = ["--checkpoint-file", "tracker.pt", "--motion-path", "motions", "--output-dir", "new"]
    original = vars(original_parser().parse_args(flags))
    new = build_parser().parse_args(flags)
    with pytest.raises(ValueError, match="reference directory"):
        _validate_arguments(new)
    new.comparison_reference_dir = "reference"
    _validate_arguments(new)
    observed = vars(new).copy()
    for name in ("comparison_reference_dir", "chunk_depth", "memory_depth"):
        observed.pop(name)
    observed["context_depth"] = 2
    assert observed == original


def test_reference_guard_allows_only_planned_capacity_change(tmp_path):
    source = {
        "arguments": {"context_depth": 2, "num_envs": 8192, "batch_size": 1024,
                      "micro_batch_size": 256, "seed": 717,
                      "payload_mass_range_kg": [1.0, 3.0]},
        "model": asdict(Memory350Config()), "loss": {"weight": .01},
        "dataset": {"runtime_audits_by_rank": []},
    }
    (tmp_path / "run_config.json").write_text(json.dumps(source))
    (tmp_path / "normalization.json").write_text("{}")
    actual = copy.deepcopy(source)
    actual["arguments"]["context_depth"] = 4
    actual["arguments"]["payload_mass_range_kg"] = (1.0, 3.0)
    actual["model"] = asdict(Memory350ScaleConfig())
    assert reference_contract(tmp_path, actual)["physics_and_training_controls_passed"]
    for area, key, value in (("arguments", "batch_size", 512),
                             ("arguments", "num_envs", 4096),
                             ("arguments", "payload_mass_range_kg", (1.0, 4.0)),
                             ("loss", "weight", .02),
                             ("model", "transformer_depth", 7)):
        changed = copy.deepcopy(actual)
        changed[area][key] = value
        with pytest.raises(ValueError, match="changed settings"):
            reference_contract(tmp_path, changed)


def test_paired_report_direction_and_world_bootstrap():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_memory350_scale import summarize
    samples = {
        "original_mse": np.ones((4, 5)), "encoder2x_mse": np.full((4, 5), .8),
        "unchanged_mse": np.full((4, 5), 2.), "world_id": np.array([10, 10, 11, 12]),
        "short_steps": np.array([5, 50, 20, 50]), "long_chunks": np.array([30, 30, 0, 5]),
    }
    result = summarize(samples)
    assert result["all"]["original_five_step_nmse"] == .5
    assert result["all"]["encoder2x_five_step_nmse"] == .4
    assert result["all"]["error_reduction_percent"] == pytest.approx(20.)
    assert result["all"]["paired_world_bootstrap_ratio_ci95"] == pytest.approx([.8, .8])
    assert result["no_long"]["samples"] == 1
    assert result["short_incomplete_with_long"]["samples"] == 1


def test_launcher_waits_for_external_processes_and_preserves_training_budget(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_memory350_scale_stage1 import available, command_for, MILESTONES
    cards = [{"index": i, "free_mib": 75000, "processes": []} for i in range(4)]
    assert available(cards)
    cards[2]["processes"] = [{"pid": 100, "executable": "/another/project/python"}]
    assert not available(cards)
    command = command_for(tmp_path / "stage1_8192", False)
    assert "--nproc-per-node=4" in command
    assert command[command.index("--num-envs") + 1] == "8192"
    assert "--until-user-stop" in command
    assert command[command.index("--updates") + 1] == "8000"
    assert command[command.index("--context-depth") + 1] == "4"
    assert 22700 in MILESTONES
