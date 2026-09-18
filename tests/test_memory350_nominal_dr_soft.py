from dataclasses import asdict
import math

import pytest
import torch

from intact_tracking.memory350_nominal_dr_soft import (
    NominalDRSoftLossConfig, NominalDRSoftObjective, dr_soft_targets, dr_center_soft_loss,
)
from intact_tracking.memory350_nominal_dr_rank import NominalDRRankLossConfig
from intact_tracking.memory350_nominal_direction import NominalDirectionObjective
from intact_tracking.cli.forward_memory_nominal_dr_soft_train import build_parser, validate_resume_losses


def loss(z, theta, other=None, worlds=None, valid=None, nominal=None):
    n = len(z)
    return dr_center_soft_loss(z, z if other is None else other, theta,
        torch.arange(n) if worlds is None else worlds, torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.bool) if valid is None else valid,
        torch.zeros(n, dtype=torch.bool) if nominal is None else nominal)


def test_bandwidth_half_weight_and_detached_normalized_labels():
    theta = torch.tensor([[0.], [.15], [.30]], requires_grad=True)
    q = dr_soft_targets(theta)
    torch.testing.assert_close(q[0] / q[0, 0], torch.tensor([1., .5, .0625]))
    torch.testing.assert_close(q.sum(-1), torch.ones(3))
    assert not q.requires_grad
    # Equal physical labels are soft positives even with different world IDs.
    q = dr_soft_targets(torch.zeros(3, 2))
    torch.testing.assert_close(q, torch.full((3, 3), 1/3))


def test_cross_view_kl_reaches_known_optimum_and_backpropagates():
    # For two worlds, choose cosine difference so logits exactly match Gaussian ratios.
    d, h, temperature = .15, .15, .1
    cosine = 1 - temperature * math.log(2) * (d/h)**2
    z = torch.tensor([[1., 0.], [cosine, math.sqrt(1-cosine**2)]])
    theta = torch.tensor([[0.], [d]], requires_grad=True)
    value, metrics = loss(z, theta)
    assert abs(value.item()) < 1e-6
    torch.testing.assert_close(metrics['dr_soft_target_self_mass'], torch.tensor(2/3))
    current = torch.tensor([[1., .2], [.2, 1.]], requires_grad=True)
    archive = z.clone().requires_grad_()
    bad, _ = loss(current, theta, archive)
    bad.backward()
    assert current.grad.abs().sum() > 0 and archive.grad.abs().sum() > 0
    assert theta.grad is None
    improved, _ = loss(current.detach()-.01*current.grad, theta, archive.detach()-.01*archive.grad)
    assert improved < bad


def test_world_pooling_symmetry_permutation_and_exclusion():
    torch.manual_seed(1)
    z, a, theta = torch.randn(3, 8), torch.randn(3, 8), torch.tensor([[0.], [.1], [.4]])
    expected, _ = loss(z, theta, a)
    torch.testing.assert_close(loss(a, theta, z)[0], expected)
    p = torch.tensor([2, 0, 1])
    torch.testing.assert_close(loss(z[p]*3, theta[p], a[p]*2)[0], expected)
    extended = torch.cat([z, z[:1], torch.randn(2, 8)]).requires_grad_()
    archives = torch.cat([a, a[:1], torch.randn(2, 8)]).requires_grad_()
    got, metrics = loss(extended, torch.cat([theta, theta[:1], torch.ones(2, 1)]), archives,
        worlds=torch.tensor([0,1,2,0,4,5]), valid=torch.tensor([1,1,1,1,1,0]).bool(),
        nominal=torch.tensor([0,0,0,0,1,0]).bool())
    torch.testing.assert_close(got, expected)
    assert metrics['dr_soft_worlds'] == 3
    got.backward()
    assert not extended.grad[4:].any() and not archives.grad[4:].any()
    for n in (0,1):
        x=torch.randn(n,8,requires_grad=True)
        value,_=loss(x,torch.zeros(n,1));value.backward()
        assert value == 0 and torch.isfinite(x.grad).all()


def test_variant_preserves_original_losses_and_only_trains_encoder_with_dr_labels():
    from test_memory350_nominal_direction import batch, model
    from intact_tracking.cli.forward_memory_scale_nominal_train import _BATCH_FIELDS
    from intact_tracking.memory350_weak_pairs import WEAK_BATCH_FIELDS
    from intact_tracking.memory350_response_window import RESPONSE_FIELDS
    torch.set_num_threads(2)
    b,m=batch(),model()
    usable=(b['weak_pair_valid'] & b['history_valid'].all(1) & b['memory_valid'].all(1)).nonzero().flatten()
    idx=usable[torch.arange(8)%len(usable)]
    b={k:v[idx].clone() if k in (_BATCH_FIELDS|WEAK_BATCH_FIELDS|RESPONSE_FIELDS) else v for k,v in b.items()}
    b['is_nominal']=torch.arange(8)<4
    b['history_state'][:,:,0]+=torch.arange(8).float()[:,None]*.1
    b['world_id']=b['weak_world_id']=torch.arange(8)
    b['dr_metric']=torch.arange(8).float()[:,None]*.1
    anchor=torch.eye(64)[0]
    old=NominalDirectionObjective(m,anchor=anchor)(b,compute_metrics=False)
    candidate=NominalDRSoftObjective(m,anchor=anchor)
    new=candidate(b,compute_metrics=False)
    off=NominalDRSoftObjective(m,NominalDRSoftLossConfig(dr_soft_weight=0),anchor=anchor)(b,compute_metrics=False)
    assert new['dr_soft_worlds'] == 4
    for k in old:
        torch.testing.assert_close(old[k],off[k],rtol=0,atol=0)
        if k!='loss': torch.testing.assert_close(old[k],new[k],rtol=0,atol=0)
    torch.testing.assert_close(new['loss'],old['loss']+new['dr_soft_weighted_loss'])
    views=candidate._encode_views(b)
    only,_=loss(views[0],b['dr_metric'],views[2],nominal=b['is_nominal'])
    only.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.context_encoder.parameters())
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('context_encoder.'))


def test_resume_replaces_ranking_only_and_defaults_match_current_stage():
    old=asdict(NominalDRRankLossConfig(response_distance_scale=.6,representation_relation_weight=8,dr_center_rank_weight=.02))
    new=asdict(NominalDRSoftLossConfig(response_distance_scale=.6,representation_relation_weight=8))
    validate_resume_losses(old,new,new_stage=True)
    validate_resume_losses(new,new,new_stage=False)
    with pytest.raises(ValueError): validate_resume_losses(old,new,new_stage=False)
    with pytest.raises(ValueError): validate_resume_losses(old,dict(new,response_distance_scale=.3),new_stage=True)
    with pytest.raises(ValueError): validate_resume_losses(new,dict(new,dr_soft_h=.2),new_stage=True)
    args=build_parser().parse_args(['--checkpoint-file','tracker.pt','--motion-path','motions','--output-dir','new_run'])
    assert (args.dr_soft_h,args.dr_soft_weight,args.response_distance_scale)==(.15,.02,.6)
    assert args.representation_relation_weight==8 and args.until_user_stop
    for kwargs in ({'dr_soft_h':0},{'dr_soft_h':float('nan')},{'dr_soft_temperature':-1},{'dr_soft_weight':float('inf')}):
        with pytest.raises(ValueError): NominalDRSoftLossConfig(**kwargs)


def test_anchor_retuning_requires_explicit_new_stage_and_preserves_other_losses():
    previous = asdict(NominalDRSoftLossConfig(response_distance_scale=.6, representation_relation_weight=8))
    actual = dict(previous, nominal_anchor_weight=.05)
    validate_resume_losses(previous, actual, new_stage=True, retune_nominal_anchor=True)
    validate_resume_losses(actual, actual, new_stage=False)
    with pytest.raises(ValueError):
        validate_resume_losses(previous, actual, new_stage=True)
    with pytest.raises(ValueError):
        validate_resume_losses(previous, actual, new_stage=False, retune_nominal_anchor=True)
    rank = asdict(NominalDRRankLossConfig(response_distance_scale=.6, representation_relation_weight=8))
    with pytest.raises(ValueError):
        validate_resume_losses(rank, actual, new_stage=True, retune_nominal_anchor=True)
    for key, value in (('dr_soft_h', .2), ('dr_soft_weight', .04),
                       ('response_distance_scale', .3), ('representation_relation_weight', 4.),
                       ('weak_positive_weight', .016), ('joint_position_weight', 2.)):
        with pytest.raises(ValueError):
            validate_resume_losses(previous, dict(actual, **{key: value}),
                                   new_stage=True, retune_nominal_anchor=True)
    args = build_parser().parse_args(['--checkpoint-file', 'tracker.pt', '--motion-path', 'motions',
        '--output-dir', 'new_run', '--resume', 'soft.pt', '--resume-new-stage',
        '--resume-retune-nominal-anchor', '--nominal-anchor-weight', '.05'])
    assert args.resume_retune_nominal_anchor and args.nominal_anchor_weight == .05
