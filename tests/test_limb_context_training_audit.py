import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("limb_training_audit", SCRIPTS / "audit_limb_context_training.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def initial_statistics():
    return {"sum": torch.tensor([1234., 2., 0., 16384.]),
            "ssq": torch.tensor([48000., 600., 0., 16384.]),
            "count": torch.tensor([16384.])}


def test_cross_run_normalization_accepts_float32_rounding_but_rejects_different_observations():
    reference = initial_statistics()
    candidate = {key: value.clone() for key, value in reference.items()}
    candidate["sum"][0] = torch.nextafter(candidate["sum"][0], torch.tensor(float("inf")))
    result = audit.compare_initial_normalization(reference, candidate)
    assert result["passed"] and not result["exact_equal"]
    assert result["differences"]["sum"]["max_moment_difference"] < 1e-8
    candidate["sum"][0] += .01 * candidate["count"].item()
    with pytest.raises(ValueError, match="beyond float32 tolerance"):
        audit.compare_initial_normalization(reference, candidate)


def test_cross_run_normalization_rejects_count_changes_and_invalid_statistics():
    reference = initial_statistics()
    candidate = {key: value.clone() for key, value in reference.items()}
    candidate["count"] += 1
    with pytest.raises(ValueError, match="sample counts differ"):
        audit.compare_initial_normalization(reference, candidate)
    candidate["count"] = reference["count"].clone()
    candidate["ssq"][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        audit.compare_initial_normalization(reference, candidate)


@pytest.mark.parametrize("count", [16384, 32768])
def test_near_zero_rounding_tolerance_does_not_depend_on_world_count(count):
    reference = {"sum": torch.zeros(1), "ssq": torch.tensor([float(count)]),
                 "count": torch.tensor([float(count)])}
    candidate = {key: value.clone() for key, value in reference.items()}
    candidate["sum"] += count * 1e-9
    assert audit.compare_initial_normalization(reference, candidate)["passed"]
    candidate["sum"] += count * 1e-4
    with pytest.raises(ValueError, match="beyond float32 tolerance"):
        audit.compare_initial_normalization(reference, candidate)
