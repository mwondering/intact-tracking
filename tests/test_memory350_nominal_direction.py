from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import math

import pytest
import torch

from intact_tracking.cli.forward_memory_nominal_direction_train import (
    build_parser, prepare_objective, validate_resume_losses,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_nominal_direction import (
    NominalDirectionLossConfig, NominalDirectionObjective, direction_response_loss,
)
from intact_tracking.memory350_response_window import ResponseWindowObjective
from intact_tracking.memory350_weak_pairs import WeakPairLossConfig
from test_memory350 import _normalization
from test_memory350_weak_pairs import make_long_replay


def model():
    return Memory350Predictor(Memory350Config(
        transformer_dim=32, transformer_depth=1, transformer_heads=4,
        context_dim=16, context_heads=4, memory_depth=1))


def batch():
    result = make_long_replay().sample_batch(8, _normalization())
    result["label_response"] = torch.full((8, 10, 70), .3)
    result["label_response_valid"] = torch.ones(8, dtype=torch.bool)
    result["is_nominal"][:4] = True
    return result


def test_relation_uses_own_response_and_fixed_anchor_not_other_worlds():
    z = torch.tensor([[.5, math.sqrt(.75)], [.5, -math.sqrt(.75)]], requires_grad=True)
    anchor = torch.tensor([1., 0.], requires_grad=True)
    response = torch.full((2, 10, 70), .3, requires_grad=True)
    valid = torch.ones(2, dtype=torch.bool)
    loss, distance, target, _ = direction_response_loss(z, anchor, response, valid)
    torch.testing.assert_close(distance, torch.ones(2))
    torch.testing.assert_close(target, torch.ones(2))
    assert loss < 1e-12
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert response.grad is None and anchor.grad is None
    torch.testing.assert_close(direction_response_loss(7*z, anchor, response, valid)[0], loss)
    assert (z[0] - z[1]).norm() > 1.7  # Their mutual distance is deliberately irrelevant.


def test_relation_zero_response_pulls_toward_nominal_and_invalid_rows_have_no_gradient():
    angle = torch.tensor([.4, .7], requires_grad=True)
    z = torch.stack((angle.cos(), angle.sin()), -1)
    loss, *_ = direction_response_loss(z, torch.tensor([1., 0.]),
        torch.zeros(2, 10, 70), torch.tensor([True, False]))
    loss.backward()
    assert angle.grad[0] > 0 and angle.grad[1] == 0
    exact = torch.tensor([[1., 0.]], requires_grad=True)
    zero, *_ = direction_response_loss(exact, torch.tensor([1., 0.]),
        torch.zeros(1, 10, 70), torch.tensor([True]))
    zero.backward()
    assert zero == 0 and torch.isfinite(exact.grad).all()


def test_only_nominal_valid_rows_are_anchored_and_empty_masks_are_finite():
    m = model()
    anchor = torch.eye(64)[0]
    objective = NominalDirectionObjective(m, anchor=anchor)
    z = torch.zeros(3, 64)
    z[:, 1] = 1
    z.requires_grad_()
    b = {"history_valid": torch.tensor([[True], [True], [False]]),
         "memory_valid": torch.zeros(3, 1, dtype=torch.bool),
         "is_nominal": torch.tensor([True, False, True])}
    loss, metrics = objective._extra_representation_loss(b, (z, z))
    assert metrics["nominal_anchor_samples"] == 1
    assert metrics["nominal_anchor_loss"] == 1
    loss.backward()
    assert z.grad[0].abs().sum() > 0 and not z.grad[1:].any()
    b["history_valid"].zero_()
    loss, metrics = objective._extra_representation_loss(b, (z, z))
    assert loss == 0 and all(torch.isfinite(v) for v in metrics.values())


def test_real_objective_weights_prediction_preservation_and_no_negative_computation(monkeypatch):
    torch.set_num_threads(2)
    b, m = batch(), model()
    anchor = torch.eye(64)[0]
    objective = NominalDirectionObjective(m, anchor=anchor)
    old = ResponseWindowObjective(m, WeakPairLossConfig())
    old_result = old(b, compute_metrics=False)
    monkeypatch.setattr(torch, "cdist", lambda *a, **k: pytest.fail("Negative pairs must not be computed"))
    before = anchor.clone()
    result = objective(b, compute_metrics=False)
    torch.testing.assert_close(result["prediction_loss"], old_result["prediction_loss"], atol=0, rtol=0)
    expected = (result["prediction_loss"] + .01*result["representation_positive_loss"]
                + .04*result["dr_nominal_relation_loss"] + .008*result["weak_positive_loss"]
                + .01*result["nominal_anchor_loss"])
    torch.testing.assert_close(result["loss"].detach(), expected)
    assert result["weak_positive_pairs"] > 0 and result["nominal_anchor_samples"] == 4
    assert result["latent_relation_pairs"] == 4
    assert not any("negative" in key for key in result)
    result["loss"].backward()
    for level in ("chunk_encoder", "memory_encoder", "transformer"):
        grads = [p.grad for n,p in m.context_encoder.named_parameters() if n.startswith(level)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert objective.nominal_anchor.grad is None
    torch.testing.assert_close(objective.nominal_anchor, before, atol=0, rtol=0)
    changed = deepcopy(b)
    changed["label_response"][:, 5:] = 3.
    after = objective(changed, compute_metrics=False)
    torch.testing.assert_close(result["prediction_loss"], after["prediction_loss"], atol=0, rtol=0)
    assert result["dr_nominal_relation_loss"] != after["dr_nominal_relation_loss"]
    changed["label_response_valid"].zero_()
    masked = objective(changed, compute_metrics=False)
    assert masked["latent_relation_pairs"] == 0 and masked["dr_nominal_relation_loss"] == 0
    assert masked["nominal_anchor_samples"] == 4 and masked["weak_positive_pairs"] > 0


def test_anchor_calibration_uses_full_training_nominal_and_restore_does_not_sample(tmp_path):
    torch.set_num_threads(2)
    b, m = batch(), model()
    objective = NominalDirectionObjective(m)
    class Replay:
        def sample_batch(self, size, normalization):
            assert size == 8
            return b
    distributed = DistributedContext(0, 0, 1, torch.device("cpu"), None)
    args = SimpleNamespace(anchor_calibration_batches=2, anchor_calibration_batch_size=8)
    state = {}
    result = prepare_objective(objective, Replay(), None, distributed, tmp_path, None,
                               args=args, anchor_state=state)
    eligible = b["is_nominal"] & b["history_valid"].all(1) & b["memory_valid"].all(1)
    assert state["nominal_samples_total"] == 2 * int(eligible.sum())
    assert state["validation_used"] is False
    assert objective.anchor_ready and objective.nominal_anchor.norm() == pytest.approx(1.)
    restored = NominalDirectionObjective(model())
    new_state = {}
    prepare_objective(restored, None, None, distributed, tmp_path,
                      {"nominal_direction_anchor": result["nominal_direction_anchor"]},
                      args=args, anchor_state=new_state)
    torch.testing.assert_close(restored.nominal_anchor, objective.nominal_anchor, atol=0, rtol=0)
    assert new_state["sha256"] == state["sha256"]


def test_resume_allows_only_agreed_loss_transition_and_parser_defaults():
    current = asdict(NominalDirectionLossConfig())
    old = asdict(WeakPairLossConfig(representation_relation_weight=2.,
        response_distance_scale=.3, weak_positive_weight=.008,
        weak_negative_weight=.008, weak_negative_margin=1.1))
    validate_resume_losses(old, current, new_stage=True)
    with pytest.raises(ValueError):
        validate_resume_losses(old, current, new_stage=False)
    with pytest.raises(ValueError):
        validate_resume_losses(dict(old, foot_weight=2.), current, new_stage=True)
    validate_resume_losses(current, current, new_stage=False)
    with pytest.raises(ValueError):
        validate_resume_losses(current, dict(current, nominal_anchor_weight=.02), new_stage=True)
    with pytest.raises(ValueError):
        NominalDirectionLossConfig(weak_negative_weight=.008)
    args = build_parser().parse_args(["--checkpoint-file","tracker.pt",
                                      "--motion-path","motions","--output-dir","out"])
    assert args.representation_weight * args.representation_relation_weight == .04
    assert args.nominal_anchor_weight == .01 and args.weak_positive_weight == .008
    assert args.batch_size == 512 and args.micro_batch_size == 256
    assert args.stop_after_updates is None and args.until_user_stop


def test_nominal_tenth_partition_preserves_legacy_half_and_validates_both_groups():
    from intact_tracking.memory350_nominal_rollout import nominal_world_ids, NominalMemory350RolloutConfig
    from intact_tracking.cli.forward_memory_scale_nominal_train import _validate_arguments
    ids = nominal_world_ids(8192, .1)
    assert len(ids) == 819 and len(ids.unique()) == 819
    assert int((ids < 8064).sum()) == 806 and int((ids >= 8064).sum()) == 13
    torch.testing.assert_close(nominal_world_ids(8192, .5), torch.arange(0, 8192, 2))
    config = NominalMemory350RolloutConfig(checkpoint_file='tracker.pt', motion_file='motion.npz',
        num_envs=8192, nominal_fraction=.1, tracker_dr_plus_limb_payload=True,
        dr_nominal_probability=.5, limb_max_masses_kg=(2.5, 2.5, 4., 4.))
    assert config.nominal_fraction == .1 and config.dr_nominal_probability == .5
    command = ['--checkpoint-file','tracker.pt','--motion-path','motions','--output-dir','out',
               '--nominal-fraction','.1','--dr-nominal-probability','.5',
               '--limb-max-masses-kg','2.5','2.5','4','4']
    args = build_parser().parse_args(command)
    _validate_arguments(args, require_comparison_reference=False, allow_nominal_fraction=True)
    with pytest.raises(ValueError):
        _validate_arguments(args, require_comparison_reference=False)
    for fraction in (0., 1., float('nan'), 1e-6):
        with pytest.raises(ValueError):
            nominal_world_ids(8192, fraction)
