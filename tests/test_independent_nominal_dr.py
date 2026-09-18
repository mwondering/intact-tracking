import copy

import pytest
import torch

from intact_tracking.independent_nominal_dr import EVENT, nominal_mask, validate_probability
from intact_tracking.limb_context_dr import TRACKER_DR, configure_limb_dr
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT
from intact_tracking.rollout.online import FixedDRRolloutConfig
from test_limb_context_dr import _original_cfg


def test_independent_masks_keep_global_rng_and_coordinate_mixture():
    before = torch.get_rng_state().clone()
    mask = nominal_mask((16384, 4), seed=73, probability=.5, device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(mask, nominal_mask(mask.shape, seed=73, probability=.5, device="cpu"))
    assert (abs(mask.float().mean(0)-.5) < .02).all()
    assert abs(mask.all(1).float().mean() - 1/16) < .01
    assert (mask.any(1) & ~mask.all(1)).any()
    original = 4 * torch.rand(mask.shape, generator=torch.Generator().manual_seed(99))
    changed = torch.where(mask, 0., original)
    assert torch.equal(changed[~mask], original[~mask])
    assert (changed > 3.9).any()  # The continuous branch still covers the far end.


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), float("inf")])
def test_bad_mixture_probability_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        validate_probability(probability)


@pytest.mark.parametrize("probability", [0., 1.])
def test_mixture_endpoints(probability):
    mask = nominal_mask((100, 29), seed=5, probability=probability, device="cpu")
    assert torch.equal(mask, torch.full_like(mask, bool(probability)))


def test_nominal_mixture_appended_after_original_physics_and_payload():
    cfg = _original_cfg()
    original = copy.deepcopy(cfg)
    metadata = configure_limb_dr(cfg, 73, profile=TRACKER_DR, nominal_probability=.5)
    assert list(cfg.events)[-2:] == [PAYLOAD_EVENT, EVENT]
    for name, event in original.events.items():
        assert cfg.events[name] == event
    assert cfg.observations == original.observations
    assert metadata["independent_nominal_mixture"]["probability"] == .5
    assert cfg.events[EVENT].params["probability"] == .5
    default = _original_cfg()
    configure_limb_dr(default, 73, profile=TRACKER_DR)
    assert EVENT not in default.events


def test_rollout_configuration_and_explicit_fixed_loads():
    args = dict(checkpoint_file="tracker.pt", motion_file="test.motion.npz",
                tracker_dr_plus_limb_payload=True, dr_nominal_probability=.5)
    assert FixedDRRolloutConfig(**args).dr_nominal_probability == .5
    with pytest.raises(ValueError, match="random tracker DR"):
        FixedDRRolloutConfig(**args, limb_fixed_masses=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="random tracker DR"):
        FixedDRRolloutConfig(**{**args, "tracker_dr_plus_limb_payload": False})


def test_encoder_training_cli_exposes_mixture_and_resume_rejects_changed_sampling():
    from intact_tracking.cli.forward_memory_scale_nominal_train import (
        _validate_resume_run_config, build_parser,
    )

    args = build_parser().parse_args([
        "--checkpoint-file", "tracker.pt", "--motion-file", "test.motion.npz",
        "--output-dir", "out", "--tracker-dr-plus-limb-payload", "--dr-nominal-probability", ".5",
    ])
    assert args.dr_nominal_probability == .5
    baseline = dict(arguments={"num_envs": 16, "validation_worlds": 0, "seed": 1,
        "gradient_steps_per_update": 1, "motion_path": None, "motion_file": "test.motion.npz",
        "limb_payload_only": False, "nominal_fraction": .5,
        "tracker_dr_plus_limb_payload": True, "batch_size": 8, "updates": 10})
    _validate_resume_run_config(baseline, baseline)
    changed = copy.deepcopy(baseline)
    changed["arguments"]["dr_nominal_probability"] = .5
    with pytest.raises(ValueError, match="nominal-mixture"):
        _validate_resume_run_config(baseline, changed)
