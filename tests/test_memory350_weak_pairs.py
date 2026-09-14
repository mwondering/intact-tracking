import math

import pytest
import torch

from intact_tracking.cli import forward_memory_scale_nominal_train as trainer
from intact_tracking.cli.forward_memory_weak_pairs_train import build_parser
from intact_tracking.memory350_model import Memory350Config, Memory350Predictor
from intact_tracking.memory350_weak_pairs import (
    WEAK_BATCH_FIELDS, WeakPairLossConfig, WeakPairObjective, WeakPairReplayBuffer, weak_pair_losses,
)
from test_memory350 import _step, _normalization


def test_direct_negatives_exclude_nominal_and_same_world_and_stop_at_margin():
    angles = torch.tensor([0., .1, .2, math.pi / 2], requires_grad=True)
    latent = torch.stack((angles.cos(), angles.sin()), -1)
    full_dr = torch.tensor([True, True, True, False])
    positive, negative, metrics = weak_pair_losses(
        latent, latent, torch.zeros(4, dtype=torch.bool), full_dr,
        torch.tensor([1, 1, 3, 4]), margin=1.)
    assert metrics['weak_negative_pairs'] == 2
    assert positive == 0 and negative > 0
    negative.backward()
    assert torch.isfinite(angles.grad).all() and angles.grad[:3].abs().sum() > 0
    assert angles.grad[-1] == 0
    far = torch.eye(2, requires_grad=True)
    _, loss, metrics = weak_pair_losses(far, far, torch.zeros(2, dtype=torch.bool),
                                       torch.ones(2, dtype=torch.bool), torch.tensor([1, 3]), margin=1.)
    assert metrics['weak_negative_pairs'] == 1 and loss == 0
    loss.backward()
    assert torch.equal(far.grad, torch.zeros_like(far))


def test_positive_gradient_pulls_different_views_together_and_empty_pairs_are_finite():
    angle = torch.tensor(.7, requires_grad=True)
    anchor = torch.stack((angle.cos(), angle.sin()))[None]
    positive, negative, metrics = weak_pair_losses(
        anchor, torch.tensor([[1., 0.]]), torch.tensor([True]),
        torch.tensor([True]), torch.tensor([1]), margin=1.)
    positive.backward()
    assert angle.grad > 0  # Gradient descent reduces the angle.
    assert negative == 0 and metrics['weak_negative_pairs'] == 0
    zero = torch.zeros(2, 64, requires_grad=True)
    a, b, metrics = weak_pair_losses(zero, zero, torch.zeros(2, dtype=torch.bool),
                                    torch.zeros(2, dtype=torch.bool), torch.tensor([0, 2]), margin=1.)
    (a + b).backward()
    assert all(torch.isfinite(v) for v in metrics.values())
    assert torch.isfinite(zero.grad).all()


def make_long_replay():
    replay = WeakPairReplayBuffer(num_worlds=1, capacity=16, sampling_mode='uniform',
                                  weak_archive_slots=4, weak_archive_interval=200)
    for step in range(1985):
        local = step % 100
        replay.add_step(_step(step, episode=step // 100, episode_step=local, reset=local == 99))
    return replay


def test_archived_cross_motion_views_survive_ring_overwrite_and_share_no_interactions():
    replay = make_long_replay()
    batch = replay.sample_batch(16, _normalization())
    valid = batch['weak_pair_valid']
    assert valid.any()
    assert (batch['weak_motion_id'][valid] != batch['motion_id'][valid]).all()
    assert (batch['weak_world_id'][valid] == batch['world_id'][valid]).all()
    assert (batch['weak_session'][valid] == batch['physics_session'][valid]).all()
    assert (batch['weak_age_steps'][valid] > replay.ring_steps).any()
    for i in valid.nonzero().flatten().tolist():
        anchor = torch.cat((batch['history_state'][i, :, 0],
                            batch['memory_interactions'][i, :, :, 0].flatten()))
        positive = torch.cat((batch['weak_history_state'][i, :, 0],
                              batch['weak_memory_interactions'][i, :, :, 0].flatten()))
        assert not set(anchor.tolist()) & set(positive.tolist())
        assert positive.max() < anchor.min()
    before = replay.weak_archive['raw'].clone()
    replay.memory.chunks.fill_(-999)
    torch.testing.assert_close(replay.weak_archive['raw'], before, equal_nan=True)


def test_physics_change_invalidates_old_positive_views():
    replay = make_long_replay()
    for step in range(1985, 2055):
        local = step % 100
        batch = _step(step, episode=step // 100, episode_step=local, reset=local == 99)
        if step == 1985:
            batch['parameters_changed'] = torch.tensor([True])
        replay.add_step(batch)
    batch = replay.sample_batch(8, _normalization())
    assert (batch['physics_session'] == 1).all()
    assert not batch['weak_pair_valid'].any()


def test_real_archived_views_backpropagate_through_all_encoder_levels():
    torch.set_num_threads(2)
    replay = make_long_replay()
    batch = replay.sample_batch(8, _normalization())
    assert batch['weak_pair_valid'].any()
    model = Memory350Predictor(Memory350Config(
        transformer_dim=32, transformer_depth=1, transformer_heads=4,
        context_dim=16, context_heads=4, memory_depth=1))
    objective = WeakPairObjective(model, WeakPairLossConfig())
    result = objective(batch, compute_metrics=False)
    expected = (result['prediction_loss'] + .01 * result['representation_loss']
                + .002 * result['weak_positive_loss'] + .002 * result['weak_negative_loss'])
    torch.testing.assert_close(result['loss'].detach(), expected)
    assert result['weak_pair_weighted_loss'].requires_grad is False
    result['loss'].backward()
    for level in ('chunk_encoder', 'memory_encoder', 'transformer'):
        gradients = [p.grad for name, p in model.context_encoder.named_parameters() if name.startswith(level)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_weak_fields_are_microbatched_and_cosine_horizon_stays_8000():
    args = build_parser().parse_args(['--checkpoint-file', 'tracker.pt', '--motion-path', 'motions', '--output-dir', 'out'])
    assert args.stop_after_updates == 5000 and args.updates == 8000
    assert args.response_distance_scale == .5 and args.nominal_fraction == .5
    fields = {name: torch.arange(8) for name in WEAK_BATCH_FIELDS}
    old = trainer._BATCH_FIELDS
    try:
        trainer._BATCH_FIELDS = old | WEAK_BATCH_FIELDS
        sliced = trainer._slice_predictor_batch(fields, 2, 4)
        assert all(torch.equal(value, torch.tensor([2, 3])) for value in sliced.values())
    finally:
        trainer._BATCH_FIELDS = old


@pytest.mark.parametrize('kwargs', [{'weak_positive_weight': -1}, {'weak_negative_weight': float('nan')}, {'weak_negative_margin': 2}])
def test_invalid_weak_loss_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        WeakPairLossConfig(**kwargs)
