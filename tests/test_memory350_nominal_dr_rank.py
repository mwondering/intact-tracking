from copy import deepcopy
from dataclasses import asdict
import math

import pytest
import torch

from intact_tracking.cli.forward_memory_nominal_dr_rank_train import (
    build_parser, validate_resume_losses,
)
from intact_tracking.memory350_nominal_direction import (
    NominalDirectionLossConfig, NominalDirectionObjective, direction_response_loss,
)
from intact_tracking.memory350_nominal_dr_rank import (
    NominalDRRankLossConfig, NominalDRRankObjective, NominalDRRankReplay,
    dr_center_rank_loss, rank_view_valid,
)
from intact_tracking.memory350_response_window import ResponseWindowCollector
from test_memory350 import _normalization, _step
from test_memory350_nominal_direction import batch as nominal_batch, model
from test_memory350_response_window import FakeB


def rank(z, theta, other=None, worlds=None, valid=None, nominal=None):
    n = len(z)
    return dr_center_rank_loss(
        z, z if other is None else other, theta,
        torch.arange(n) if worlds is None else worlds, torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.bool) if valid is None else valid,
        torch.zeros(n, dtype=torch.bool) if nominal is None else nominal)


def circle(angles):
    return torch.stack((angles.cos(), angles.sin()), -1)


def test_rank_rewards_parameter_order_and_backpropagates_to_both_views_only():
    theta = torch.tensor([[0.], [.1], [.7]], requires_grad=True)
    correct = circle(torch.tensor([0., .1, .7]))
    angles = torch.tensor([0., .7, .1], requires_grad=True)
    z = circle(angles)
    other = z.detach().clone().requires_grad_()
    good, gm = rank(correct, theta)
    bad, bm = rank(z, theta, other)
    assert good < bad and gm['dr_center_rank_accuracy'] == 1
    assert bm['dr_center_rank_accuracy'] < 1
    bad.backward()
    assert torch.isfinite(angles.grad).all() and angles.grad.abs().sum() > 0
    assert torch.isfinite(other.grad).all() and other.grad.abs().sum() > 0
    assert theta.grad is None
    improved, _ = rank(circle(angles.detach() - .05*angles.grad), theta,
                       other.detach() - .05*other.grad)
    assert improved < bad


def test_distinct_centers_can_stay_at_the_same_small_nominal_distance():
    # Ranking acts inside the existing unit sphere; no minimum distance of 1.1.
    radius = .2
    axial = 1 - radius**2/2
    transverse = math.sqrt(1 - axial**2)
    phase = torch.tensor([0., 1.2, .2], requires_grad=True)
    theta = torch.tensor([[0.], [.1], [.7]])

    def z():
        return torch.stack((torch.full_like(phase, axial), transverse*phase.cos(),
                            transverse*phase.sin()), -1)

    initial, _ = rank(z(), theta)
    optimizer = torch.optim.SGD([phase], lr=1.)
    for _ in range(15):
        optimizer.zero_grad()
        loss, _ = rank(z(), theta)
        loss.backward()
        optimizer.step()
    final, _ = rank(z(), theta)
    assert final < initial
    torch.testing.assert_close((z() - torch.tensor([1., 0., 0.])).norm(dim=-1),
                               torch.full((3,), radius), atol=1e-6, rtol=0)
    response_rms = .3 * radius / (2 - radius)
    ab, *_ = direction_response_loss(z(), torch.tensor([1., 0., 0.]),
                                     torch.full((3, 10, 70), response_rms),
                                     torch.ones(3, dtype=torch.bool))
    assert ab < 1e-12


def test_pooling_duplicate_worlds_masks_nominal_and_preserves_scale_and_permutation():
    z = circle(torch.tensor([0., .1, .7]))
    theta = torch.tensor([[0.], [.1], [.7]])
    loss, metrics = rank(z, theta)
    permutation = torch.tensor([2, 0, 1])
    reordered, _ = rank(z[permutation]*7, theta[permutation])
    torch.testing.assert_close(reordered, loss)
    extended = torch.cat((z, z[:1], torch.tensor([[0., 1.], [-1., 0.]]))).requires_grad_()
    extended_theta = torch.cat((theta, theta[:1], torch.tensor([[2.], [9.]])))
    nominal = torch.tensor([False, False, False, False, True, False])
    valid = torch.tensor([True, True, True, True, True, False])
    pooled, pm = rank(extended, extended_theta, worlds=torch.tensor([0, 1, 2, 0, 4, 5]),
                      valid=valid, nominal=nominal)
    torch.testing.assert_close(pooled, loss)
    assert pm['dr_center_rank_worlds'] == metrics['dr_center_rank_worlds'] == 3
    pooled.backward()
    assert not extended.grad[-2:].any()


@pytest.mark.parametrize('mode', ['one_world', 'equal_parameters', 'no_valid', 'all_nominal'])
def test_absent_ordering_is_reported_as_zero_comparisons_with_finite_gradients(mode):
    z = torch.zeros(3, 64, requires_grad=True)
    theta = torch.arange(3).float()[:, None]
    kwargs = {}
    if mode == 'one_world':
        kwargs['worlds'] = torch.zeros(3, dtype=torch.long)
    elif mode == 'equal_parameters':
        theta.zero_()
    elif mode == 'no_valid':
        kwargs['valid'] = torch.zeros(3, dtype=torch.bool)
    else:
        kwargs['nominal'] = torch.ones(3, dtype=torch.bool)
    loss, metrics = rank(z, theta, **kwargs)
    assert loss == 0 and metrics['dr_center_rank_comparisons'] == 0
    loss.backward()
    assert torch.isfinite(z.grad).all() and not z.grad.any()
    assert all(torch.isfinite(v) and not v.requires_grad for v in metrics.values())


def test_added_objective_preserves_all_six_losses_and_labels_never_reach_predictor():
    torch.set_num_threads(2)
    b, m = nominal_batch(), model()
    from intact_tracking.cli.forward_memory_scale_nominal_train import _BATCH_FIELDS
    from intact_tracking.memory350_weak_pairs import WEAK_BATCH_FIELDS
    from intact_tracking.memory350_response_window import RESPONSE_FIELDS
    usable = (b['weak_pair_valid'] & b['history_valid'].all(1) & b['memory_valid'].all(1)).nonzero().flatten()
    assert len(usable)
    indices = usable[torch.arange(8) % len(usable)]
    b = {k: v[indices].clone() if k in (_BATCH_FIELDS | WEAK_BATCH_FIELDS | RESPONSE_FIELDS)
         else v for k, v in b.items()}
    b['is_nominal'] = torch.arange(8) < 4
    # Distinct input histories exercise actual encoder gradients in this plumbing test.
    b['history_state'][:, :, 0] += torch.arange(8).float()[:, None] * .1
    b['world_id'] = b['weak_world_id'] = torch.arange(8)
    b['dr_metric'] = torch.arange(8).float()[:, None] * .1
    anchor = torch.eye(64)[0]
    baseline = NominalDirectionObjective(m, anchor=anchor)
    candidate = NominalDRRankObjective(m, anchor=anchor)
    zero_weight = NominalDRRankObjective(m, NominalDRRankLossConfig(dr_center_rank_weight=0), anchor=anchor)
    old = baseline(b, compute_metrics=False)
    new = candidate(b, compute_metrics=False)
    off = zero_weight(b, compute_metrics=False)
    assert new['dr_center_rank_comparisons'] > 0
    for key in old:
        torch.testing.assert_close(off[key], old[key], atol=0, rtol=0)
        if key != 'loss':
            torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)
    torch.testing.assert_close(new['loss'], old['loss'] + new['dr_center_rank_weighted_loss'])
    changed = deepcopy(b)
    changed['dr_metric'] = b['dr_metric'].flip(0).square()
    result = candidate(changed, compute_metrics=False)
    torch.testing.assert_close(result['prediction_loss'], new['prediction_loss'], atol=0, rtol=0)
    torch.testing.assert_close(result['dr_nominal_relation_loss'], new['dr_nominal_relation_loss'], atol=0, rtol=0)
    assert candidate.nominal_anchor.grad is None
    # Isolate the added term, proving it updates encoder and not predictor weights.
    views = candidate._encode_views(b)
    only, _ = dr_center_rank_loss(views[0], views[2], b['dr_metric'], b['world_id'],
                                  b['physics_session'], rank_view_valid(b), b['is_nominal'])
    only.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.context_encoder.parameters())
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith('context_encoder.'))


@pytest.mark.parametrize('field', ['weak_motion_id', 'weak_session', 'weak_world_id', 'weak_history_valid'])
def test_invalid_center_views_are_rejected(field):
    b = nominal_batch()
    assert rank_view_valid(b)[4:].any()
    if field == 'weak_motion_id':
        b[field] = b['motion_id'].clone()
    elif field == 'weak_history_valid':
        b[field].zero_()
    else:
        b[field] = b[field] + 1
    assert not rank_view_valid(b).any()


def test_resume_requires_explicit_new_stage_and_cannot_change_original_losses():
    old, new = asdict(NominalDirectionLossConfig()), asdict(NominalDRRankLossConfig())
    validate_resume_losses(old, new, new_stage=True)
    validate_resume_losses(new, new, new_stage=False)
    with pytest.raises(ValueError):
        validate_resume_losses(old, new, new_stage=False)
    with pytest.raises(ValueError):
        validate_resume_losses(old, dict(new, nominal_anchor_weight=.02), new_stage=True)
    with pytest.raises(ValueError):
        validate_resume_losses(new, dict(new, dr_rank_min_gap=.02), new_stage=True)
    for kwargs in [{'dr_rank_temperature': 0}, {'dr_rank_min_gap': float('nan')},
                   {'dr_center_rank_weight': -1}]:
        with pytest.raises(ValueError):
            NominalDRRankLossConfig(**kwargs)
    args = build_parser().parse_args(['--checkpoint-file', 'tracker.pt', '--motion-path', 'motions',
                                     '--output-dir', 'new_run'])
    assert args.nominal_fraction == .1 and args.dr_nominal_probability == .5
    assert args.dr_center_rank_weight == .002 and args.until_user_stop


def test_new_stage_retunes_only_explicitly_allowed_weights():
    old = asdict(NominalDirectionLossConfig())
    new = asdict(NominalDRRankLossConfig(
        representation_relation_weight=8., dr_center_rank_weight=.02))
    with pytest.raises(ValueError):
        validate_resume_losses(old, new, new_stage=True)
    validate_resume_losses(old, new, new_stage=True, allow_weight_retuning=True)
    validate_resume_losses(old, dict(new, nominal_anchor_weight=.02),
                           new_stage=True, allow_weight_retuning=True)
    with pytest.raises(ValueError):
        validate_resume_losses(old, new, new_stage=False, allow_weight_retuning=True)
    for key, value in [('representation_weight', .02), ('weak_positive_weight', .02),
                       ('response_distance_scale', .6), ('joint_position_weight', 2.)]:
        with pytest.raises(ValueError):
            validate_resume_losses(old, dict(new, **{key: value}),
                                   new_stage=True, allow_weight_retuning=True)
    previous_rank = asdict(NominalDRRankLossConfig())
    validate_resume_losses(previous_rank, new, new_stage=True, allow_weight_retuning=True)
    with pytest.raises(ValueError):
        validate_resume_losses(previous_rank, dict(new, dr_rank_temperature=.2),
                               new_stage=True, allow_weight_retuning=True)


def test_doubling_ab_changes_only_its_weighted_contribution():
    torch.set_num_threads(2)
    b, m = nominal_batch(), model()
    b['dr_metric'] = torch.arange(len(b['world_id'])).float()[:, None] * .1
    anchor = torch.eye(64)[0]
    before = NominalDRRankObjective(m, NominalDRRankLossConfig(dr_center_rank_weight=.02),
                                    anchor=anchor)(b, compute_metrics=False)
    after = NominalDRRankObjective(m, NominalDRRankLossConfig(
        representation_relation_weight=8., dr_center_rank_weight=.02), anchor=anchor)(
            b, compute_metrics=False)
    for key in ['prediction_loss', 'representation_positive_loss', 'nominal_anchor_loss',
                'nominal_anchor_weighted_loss', 'weak_positive_loss', 'dr_center_rank_loss',
                'dr_center_rank_weighted_loss', 'dr_nominal_relation_loss']:
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
    torch.testing.assert_close(after['dr_nominal_weighted_loss'],
                               2 * before['dr_nominal_weighted_loss'], atol=0, rtol=0)
    torch.testing.assert_close(after['loss'], before['loss'] + before['dr_nominal_weighted_loss'])


def test_response_scale_retuning_requires_its_own_new_stage_flag():
    from intact_tracking.cli.forward_memory_nominal_direction_train import validate_response_contract

    before = asdict(NominalDRRankLossConfig(dr_center_rank_weight=.02))
    after = dict(before, response_distance_scale=.6)
    for options in ({}, {'allow_weight_retuning': True}):
        with pytest.raises(ValueError):
            validate_resume_losses(before, after, new_stage=True, **options)
    with pytest.raises(ValueError):
        validate_resume_losses(before, after, new_stage=False, allow_response_scale_retuning=True)
    validate_resume_losses(before, after, new_stage=True, allow_response_scale_retuning=True)
    with pytest.raises(ValueError):
        validate_resume_losses(before, dict(after, dr_center_rank_weight=.04),
                               new_stage=True, allow_response_scale_retuning=True)
    old_contract = {'response_scale': .3, 'anchor': 'frozen', 'response_label_horizon': 10}
    new_contract = dict(old_contract, response_scale=.6)
    with pytest.raises(ValueError):
        validate_response_contract(old_contract, new_contract, new_stage=True)
    with pytest.raises(ValueError):
        validate_response_contract(old_contract, new_contract, allow_response_scale_retuning=True)
    validate_response_contract(old_contract, new_contract, new_stage=True,
                               allow_response_scale_retuning=True)
    with pytest.raises(ValueError):
        validate_response_contract(old_contract, dict(new_contract, anchor='recalibrated'),
                                   new_stage=True, allow_response_scale_retuning=True)


def test_response_replay_labels_align_with_worlds_and_physics_change_invalidates_archives():
    torch.set_num_threads(2)
    class MultiA:
        motion_files = ()
        collector_step = 0
        changed = False

        def step(self, predictor_only):
            step = self.collector_step
            self.collector_step += 1
            local = step % 100
            rows = [_step(step + 10000*i, episode=step//100, episode_step=local,
                          reset=local == 99, world=i) for i in range(4)]
            b = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            b['is_nominal'] = torch.tensor([True, False, False, False])
            b['dr_metric'] = torch.tensor([[0., 0.], [.1, .2], [.3, .4], [.8, .6]])
            if self.changed:
                b['dr_metric'][1:] += .01
            return b
    a, collector = MultiA(), ResponseWindowCollector()
    replay = NominalDRRankReplay(num_worlds=4, capacity=512, sampling_mode='uniform', dr_metric_dim=2)
    for _ in range(398):
        collector(a, FakeB(), replay)
    assert replay.can_sample_positive_pairs(32)
    b = replay.sample_batch(128, _normalization(), positive_ready_only=True)
    valid = rank_view_valid(b)
    assert torch.unique(b['world_id'][valid]).numel() == 3
    assert b['is_nominal'].any() and b['label_response'].shape == (128, 10, 70)
    expected = torch.tensor([[0., 0.], [.1, .2], [.3, .4], [.8, .6]])
    torch.testing.assert_close(b['dr_metric'], expected[b['world_id']])
    for i in valid.nonzero().flatten().tolist():
        before = torch.cat((b['weak_history_state'][i, :, 0], b['weak_memory_interactions'][i, :, :, 0].flatten()))
        after = torch.cat((b['history_state'][i, :, 0], b['memory_interactions'][i, :, :, 0].flatten()))
        assert not set(before.tolist()) & set(after.tolist())
    a.changed = True
    for _ in range(15):
        collector(a, FakeB(), replay)
    assert not replay.can_sample_positive_pairs(32)
    b = replay.sample_batch(16, _normalization())
    assert not rank_view_valid(b).any()
