import json
from pathlib import Path

import pytest
import torch

from intact_tracking.fixed_dr_profiles import representative_bank, load_profile, FixedStaticRandomization
from intact_tracking.fixed_dr_specialists import compare_evaluations
from intact_tracking.cli.fixed_dr_specialist_train import build_parser


def test_specialist_parser_requires_single_rank_and_no_latent(tmp_path):
    args = ["--fusion", "baseline", "--output-dir", str(tmp_path / "run"),
            "--dr-bank", str(tmp_path / "bank.json"), "--specialist-id", "3"]
    parsed = build_parser().parse_args(args)
    assert parsed.training_ranks == 1
    assert parsed.adaptive_after_update == 0
    assert parsed.training_terminations == "original"
    for override in (["--training-ranks", "4"], ["--fusion", "concat"], ["--specialist-id", "8"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(args + override)


def test_bank_identity_changes_if_complete_profile_changes(tmp_path):
    path = tmp_path / "bank.json"
    bank = representative_bank()
    path.write_text(json.dumps(bank))
    selected, _, original = load_profile(path, 6)
    assert selected["masses_kg"] == [4, 0, 4, 0]
    bank["profiles"][6]["static_seed"] += 1
    path.write_text(json.dumps(bank))
    assert load_profile(path, 6)[2] != original


def test_copy_prototype_preserves_per_joint_structure_and_no_alias():
    values = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    FixedStaticRandomization.repeat_first(values, 3)
    assert torch.equal(values, torch.tensor([[1., 2., 3.]]).expand(3, 3))
    values[2, 0] = 9
    assert values[0, 0] == 1


def test_comparison_rejects_hidden_physics_difference(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, fingerprint in zip(paths, ("encoder_bias_A", "encoder_bias_B")):
        path.write_text(json.dumps({"physics": {"runtime_audit": {"fixed_dr": {
            "id": 0, "bank_sha256": "same_bank", "physics_fingerprint": fingerprint}}}}))
    with pytest.raises(ValueError, match="physics_fingerprint"):
        compare_evaluations(*paths)


def test_eight_commands_are_independent_and_each_uses_full_catalog(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_fixed_dr_specialists import training_command, DATASET
    outputs = []
    for profile in representative_bank()["profiles"]:
        command = training_command(tmp_path, profile)
        assert "torch.distributed.run" not in command
        assert command[command.index("--training-ranks") + 1] == "1"
        assert command[command.index("--motion-path") + 1] == DATASET
        assert command[command.index("--num-envs") + 1] == "8192"
        assert command[command.index("--specialist-id") + 1] == str(profile["id"])
        outputs.append(command[command.index("--output-dir") + 1])
    assert len(set(outputs)) == 8


def test_monitor_tolerates_an_evaluation_row_being_written(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_fixed_dr_specialists import refresh
    directory = tmp_path / "B00"
    directory.mkdir()
    (directory / "specialist_eval_metrics.jsonl").write_text(
        '{"completed_updates":1000}\n{"completed_updates":2000')
    jobs = {0: {"output": str(directory), "phase": "training", "child": object()}}
    protocol = {"reference_checkpoint": "A.pt", "reference_update": 4000, "interval_updates": 1000}
    state = refresh(tmp_path, jobs, protocol)
    assert state["issues"] == []
    assert state["arms"]["0"]["latest_evaluation_update"] == 1000
