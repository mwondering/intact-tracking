from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/aggregate_physics_input_controls.py"
spec = importlib.util.spec_from_file_location("physics_input_aggregation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_evaluation_worlds_are_not_independent_training_replications():
    reference = np.ones((2, 3, 4, 1))
    candidate = reference * np.asarray([0.8, 1.0, 1.2])[None, :, None, None]
    failures = np.zeros((2, 3, 4))
    result = module.crossed_summary(reference, candidate, failures, failures)
    assert result["ratios"] == pytest.approx([1.0])
    assert result["ratio_intervals"][0] == pytest.approx([0.8, 1.2])


def test_motion_repeats_do_not_erase_training_variation():
    reference = np.ones((2, 3, 4, 1))
    candidate = reference * np.asarray([0.9, 1.1])[:, None, None, None]
    failures = np.zeros((2, 3, 4))
    result = module.crossed_summary(reference, candidate, failures, failures)
    assert result["ratio_intervals"][0] == pytest.approx([0.9, 1.1])
    with pytest.raises(ValueError, match=">=2"):
        module.crossed_summary(reference[:1], candidate[:1], failures[:1], failures[:1])
