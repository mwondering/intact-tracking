import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_memory350_compressed_ppo import command_for, select_context


def comparison(root, a=.81, b=.79):
    checkpoints = {}
    for key in ("baseline", "memory350"):
        path = root / f"{key}.pt"
        path.write_bytes(key.encode())
        checkpoints[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    def readout(worlds, samples, first, second):
        return {"disjoint": {"worlds": worlds, "test_samples": samples, "models": {
            "baseline": {"top1_accuracy": first, "top5_accuracy": .9},
            "memory350": {"top1_accuracy": second, "top5_accuracy": .92}}}}
    result = {"complete": True, "update": 15000, "reference_update": 15000,
              "checkpoints": checkpoints, "prediction": {}, "readout": {
                  "memory_training": readout(292, 2761, a, b), "common": readout(409, 3152, .26, .70)}}
    target = root / "comparison_015000/summary.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result))
    return target, result


def test_selection_uses_declared_primary_metric_even_when_secondary_favors_other_arm(tmp_path):
    comparison(tmp_path)
    selected = select_context(tmp_path)
    assert selected["selected_arm"] == "response5"


def test_response10_can_win_and_exact_primary_tie_uses_secondary(tmp_path):
    comparison(tmp_path, a=.80, b=.82)
    assert select_context(tmp_path)["selected_arm"] == "response10"
    comparison(tmp_path, a=.81, b=.81)
    assert select_context(tmp_path)["selected_arm"] == "response10"


@pytest.mark.parametrize("change", ["incomplete", "wrong_update", "changed_protocol", "changed_checkpoint"])
def test_selection_rejects_incomplete_unmatched_or_changed_artifacts(tmp_path, change):
    target, value = comparison(tmp_path)
    if change == "incomplete": value["complete"] = False
    elif change == "wrong_update": value["reference_update"] = 14000
    elif change == "changed_protocol": value["readout"]["memory_training"]["disjoint"]["worlds"] = 16
    else: Path(value["checkpoints"]["baseline"]["path"]).write_bytes(b"changed")
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        select_context(tmp_path)


def test_formal_commands_are_four_gpu_unbounded_with_zero_and_real_latent(tmp_path):
    commands = {fusion: command_for(tmp_path, fusion, "/selected.pt") for fusion in ("baseline", "concat")}
    for cmd in commands.values():
        assert "--until-user-stop" in cmd and "--iterations" not in cmd
        assert "--nproc-per-node=4" in cmd
        assert cmd[cmd.index("--num-envs") + 1] == "8192"
        assert "--endpoint-eval-protocol" in cmd
    assert "--context-checkpoint" not in commands["baseline"]
    assert commands["concat"][commands["concat"].index("--context-checkpoint") + 1] == "/selected.pt"
