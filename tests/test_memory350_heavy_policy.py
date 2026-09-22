from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_memory350_tracker_action_policy import models, observation, _algorithm
from test_residual_dr_aux import schema_and_params
from intact_tracking.memory350_heavy_policy import native_aux_schema, heavy_aux_schema
from intact_tracking.memory350_heavy_dr import EVENT, MASS_LIMITS
from intact_tracking.memory350_native_dr import native_metric_schema
from intact_tracking.preview_protocol import LIMBS
from intact_tracking.residual_dr_aux import (
    capture_dr_aux_targets, DRAuxiliaryObjective, PAYLOAD_GROUPS,
    DR_TARGET_GROUP, DR_HISTORY_WEIGHT_GROUP,
)


def heavy_schema():
    base, params = schema_and_params()
    names = base['names'] + [f'{EVENT}/added_mass_kg/{limb}' for limb in LIMBS]
    names += [f'{EVENT}/payload_com_offset/{limb}/{axis}' for limb in LIMBS for axis in 'xyz']
    params[EVENT] = {'max_masses_kg': MASS_LIMITS, 'com_half_width_m': .05}
    return native_metric_schema(names, params, {'torso_link': 10.}), params


def test_zero_mode_preserves_parameter_shapes_and_blocks_all_latent_paths(models):
    learned, learned_critic, obs = models(residual_output_mode='unbounded')
    zero, zero_critic, _ = models(residual_output_mode='unbounded', latent_input_mode='zero')
    for left, right in ((learned, zero), (learned_critic, zero_critic)):
        assert {k: tuple(v.shape) for k,v in left.state_dict().items()} == {
            k: tuple(v.shape) for k,v in right.state_dict().items()}
        for k,v in left.state_dict().items():
            torch.testing.assert_close(v, right.state_dict()[k], rtol=0, atol=0)
    with torch.no_grad():
        learned.residual_mlp.base[-1].weight.normal_(std=.1)
        learned.residual_mlp.latent_input.weight.normal_(std=.1)
        learned_critic.mlp.latent_input.weight.normal_(std=.1)
    zero.load_state_dict(learned.state_dict())
    zero_critic.load_state_dict(learned_critic.state_dict())
    altered = obs.clone()
    altered['dynamics_latent'] = torch.randn_like(obs['dynamics_latent']) * 100
    assert not torch.allclose(learned(obs), learned(altered))
    assert not torch.allclose(learned_critic(obs), learned_critic(altered))
    torch.testing.assert_close(zero(obs), zero(altered), rtol=0, atol=0)
    torch.testing.assert_close(zero_critic(obs), zero_critic(altered), rtol=0, atol=0)
    (zero(obs).square().mean() + zero_critic(obs).square().mean()).backward()
    for mlp in (zero.residual_mlp, zero_critic.mlp):
        assert mlp.latent_input.weight.grad[:, :320].eq(0).all()
        assert mlp.latent_input.weight.grad[:, 320:].abs().sum() > 0
    from intact_tracking.cli.memory350_policy_train import audit_initial_models
    from intact_tracking.memory350_tracker_action_policy import audit_tracker_action_models
    initial, critic, obs = models(latent_input_mode='zero')
    report = audit_tracker_action_models(initial, critic, obs, 'concat', original_audit=audit_initial_models)
    assert report['latent_input_mode'] == 'zero' and report['latent_dimensions'] == 320


def test_heavy_auxiliary_retains_original_92_coordinates_by_name(monkeypatch):
    import intact_tracking.rollout.online as online
    schema, params = heavy_schema()
    selected = native_aux_schema(schema)
    original, _ = schema_and_params()
    for key in ('names','lower','upper'):
        assert selected[key] == original[key]
    values = torch.tensor([schema['lower'],schema['upper']])
    # Extra payload parameters and shuffled capture order must never shift labels.
    permutation = torch.randperm(108)
    names = [schema['names'][i] for i in permutation]
    # Actual schema construction requires native order; test label capture with
    # its real event order and separately permute the requested auxiliary schema.
    requested = deepcopy(selected)
    order = torch.randperm(92)
    for key in ('names','lower','upper'):
        requested[key] = [selected[key][i] for i in order]
    monkeypatch.setattr(online, '_capture_privileged_dynamics_targets',
        lambda env: SimpleNamespace(names=schema['names'],values=values))
    monkeypatch.setattr(online, '_entity_indices_and_names', lambda *a: (torch.tensor([0]), ['torso_link']))
    monkeypatch.setattr(online, '_expanded_and_default_field', lambda *a: (None, torch.tensor([10.])))
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params=params[name])))
    with pytest.raises(ValueError, match='names/order'):
        capture_dr_aux_targets(env, requested)
    result = capture_dr_aux_targets(env, requested, allow_extra_parameters=True)
    torch.testing.assert_close(result, torch.stack((torch.zeros(92),torch.ones(92))))


def test_heavy_entry_exposes_matched_four_rank_full_scale_contract():
    from intact_tracking.cli.memory350_proprio_heavy_policy_train import build_parser
    args = build_parser().parse_args(['--fusion','concat','--context-checkpoint','heavy.pt',
        '--output-dir','unused','--latent-input-mode','zero'])
    assert (args.training_ranks,args.num_envs) == (4,8192)
    assert args.until_user_stop and args.motion_sampling == 'adaptive' and args.adaptive_after_update == 0
    assert args.entropy_coef == .005 and args.initial_action_std == 1.
    assert args.dr_aux_coef is None and args.dr_aux_motor_weight == 0.


def test_heavy_supervision_balances_families_and_excludes_inactive_coordinates():
    schema, _ = heavy_schema()
    objective = DRAuxiliaryObjective(heavy_aux_schema(schema))
    assert objective.output_dim == 108 and len(objective.active) == 20
    assert set(objective.supervised_groups) == {'com_x','com_y','com_z','friction', *PAYLOAD_GROUPS}
    target = torch.full((2,108), float('nan'))
    target[:, objective.active] = 0
    prediction = torch.ones(2,108, requires_grad=True)
    loss, stats = objective(prediction, target, torch.ones(2,1))
    assert loss.item() == pytest.approx(1.)
    loss.backward()
    # .1 / 8 leaves each original group's effective coefficient at .0125,
    # while four limbs together give each new family the same .0125.
    assert prediction.grad[:,0].sum().item() == pytest.approx(2 / 8)
    assert prediction.grad[:,92:96].sum().item() == pytest.approx(2 / 8)
    for axis in range(3):
        assert prediction.grad[:,96+axis:108:3].sum().item() == pytest.approx(2 / 8)
    assert prediction.grad[:,3].eq(0).all() and prediction.grad[:,5:92].eq(0).all()
    groups = len(objective.groups)
    for limb in LIMBS:
        i = objective.groups.index(f'payload_mass_{limb}')
        assert stats[groups+i].item() == pytest.approx(2 * MASS_LIMITS[list(LIMBS).index(limb)])
        for axis in 'xyz':
            i = objective.groups.index(f'payload_com_{limb}_{axis}')
            assert stats[groups+i].item() == pytest.approx(.2)


def test_heavy_targets_follow_physical_names_and_ranges(monkeypatch):
    import intact_tracking.rollout.online as online
    schema, params = heavy_schema()
    lower, upper = torch.tensor(schema['lower']), torch.tensor(schema['upper'])
    expected = torch.linspace(.05,.95,108)[None]
    values = lower + expected * (upper-lower)
    monkeypatch.setattr(online, '_capture_privileged_dynamics_targets',
        lambda env: SimpleNamespace(names=schema['names'],values=values))
    monkeypatch.setattr(online, '_entity_indices_and_names', lambda *a: (torch.tensor([0]), ['torso_link']))
    monkeypatch.setattr(online, '_expanded_and_default_field', lambda *a: (None, torch.tensor([10.])))
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params=params[name])))
    order = torch.randperm(108)
    requested = {key:[schema[key][i] for i in order] for key in ('names','lower','upper')}
    captured = capture_dr_aux_targets(env, requested, allow_extra_parameters=True)
    torch.testing.assert_close(captured, expected[:,order])
    pred = torch.randn(1,108)
    left,_ = DRAuxiliaryObjective(schema)(pred,expected,torch.ones(1,1))
    right,_ = DRAuxiliaryObjective(requested)(pred[:,order],captured,torch.ones(1,1))
    torch.testing.assert_close(left,right)


@pytest.mark.parametrize('mode,coefficient', [('learned',.1),('zero',0.)])
def test_heavy_ppo_auxiliary_on_off_and_checkpoint_restore(models, mode, coefficient):
    schema, _ = heavy_schema()
    options = dict(dr_aux_schema=heavy_aux_schema(schema), residual_output_mode='unbounded', latent_input_mode=mode)
    actor, critic, obs = models(**options)
    before = deepcopy(actor.state_dict())
    def labels(obs):
        if coefficient:
            obs[DR_TARGET_GROUP] = torch.rand(8,108)
            obs[DR_HISTORY_WEIGHT_GROUP] = torch.ones(8,1)
        return obs
    algorithm = _algorithm(actor,critic,labels(obs), dr_aux_coef=coefficient)
    assert (DR_TARGET_GROUP in algorithm.storage.observations) == bool(coefficient)
    with torch.inference_mode():
        for step in range(2):
            algorithm.act(obs)
            obs = labels(observation(step+1))
            algorithm.process_env_step(obs,torch.randn(8),torch.zeros(8),
                {'motion_resample_boundary':torch.zeros(8,dtype=torch.bool)})
        algorithm.compute_returns(obs)
    result = algorithm.update()
    assert result['AuxDR/enabled'] == bool(coefficient) and result['AuxDR/coefficient'] == coefficient
    assert ('AuxDR/loss' in result) == bool(coefficient)
    assert not torch.equal(actor.residual_mlp.base[-1].weight,before['residual_mlp.base.4.weight'])
    active = actor.dr_aux_objective.active
    changed = (actor.dr_aux_head.weight != before['dr_aux_head.weight']).any(dim=1)
    assert changed.sum().item() == (20 if coefficient else 0)
    if coefficient:
        assert changed[active].all()
        assert len([k for k in result if k.startswith('AuxDR/payload_')]) == 32
        assert result['AuxDR/shared_aux_gradient_norm'] > 0
    else:
        assert all(p.grad is None for p in actor.dr_aux_head.parameters())
    assert actor.dr_aux_features is None
    new_actor,new_critic,fresh = models(**options)
    restored = _algorithm(new_actor,new_critic,labels(fresh),dr_aux_coef=coefficient)
    restored.load(algorithm.save(),None,strict=True)
    evaluation = observation(7)
    torch.testing.assert_close(actor(evaluation),new_actor(evaluation),atol=0,rtol=0)
    torch.testing.assert_close(actor.predict_dr(evaluation),new_actor.predict_dr(evaluation),atol=0,rtol=0)


@pytest.mark.parametrize('mode,coefficient', [('learned',.5),('zero',0.)])
def test_heavy_entry_mode_defaults_keep_same_head_but_baseline_has_no_labels(monkeypatch, mode, coefficient):
    from intact_tracking.cli import memory350_proprio_heavy_policy_train as entry
    schema,_ = heavy_schema()
    args = entry.build_parser().parse_args(['--fusion','concat','--context-checkpoint','heavy.pt',
        '--output-dir','unused','--latent-input-mode',mode])
    fake_base = SimpleNamespace(ManagerBasedRlEnv=lambda *a,**kw:None)
    monkeypatch.setattr(entry,'base',fake_base)
    monkeypatch.setattr(entry.torch,'load',lambda *a,**kw:{'dr_metric_schema':schema})
    monkeypatch.setattr(entry.heavy,'validate_context',lambda *a:None)
    entry.configure(args)
    cfg = fake_base.configure_context_models({'actor':{},'critic':{}},'concat',scratch_seed=121)
    assert cfg['algorithm']['dr_aux_coef'] == coefficient == args.dr_aux_coef
    assert len(cfg['actor']['dr_aux_schema']['names']) == 108
    assert cfg['actor']['dr_aux_payload_com_enabled'] is False
    assert (fake_base.LimbContextWrapper.keywords['dr_aux_schema'] is not None) == bool(coefficient)


def test_mass_only_loss_preserves_denominator_and_exact_fivefold_gradients():
    schema,_ = heavy_schema()
    original = DRAuxiliaryObjective(schema)
    current = DRAuxiliaryObjective(schema,payload_com_enabled=False)
    assert len(current.active) == 8 and current.output_dim == 108
    assert current.normalization_denominator == 8
    torch.testing.assert_close(current.weights[:11],original.weights[:11],atol=0,rtol=0)
    assert current.weights[11:].eq(0).all()
    prediction = torch.ones(4,108,requires_grad=True)
    target = torch.zeros_like(prediction)
    # Compare the same retained errors: old COM errors are zero. New COM
    # targets can even be NaN because these coordinates are excluded first.
    target[:,96:] = 1
    old,_ = original(prediction,target,torch.ones(4,1))
    old_gradient = torch.autograd.grad(.1*old,prediction)[0]
    target[:,96:] = float('nan')
    new,stats = current(prediction,target,torch.ones(4,1))
    new_gradient = torch.autograd.grad(.5*new,prediction)[0]
    torch.testing.assert_close(.5*new,5*.1*old)
    torch.testing.assert_close(new_gradient,5*old_gradient)
    assert new_gradient[:,96:].eq(0).all() and torch.isfinite(stats).all()
    assert new.item() == pytest.approx(5/8)


@pytest.mark.parametrize('mode,coefficient', [('learned',.1),('zero',0.)])
def test_mass_only_resume_restores_weights_optimizer_and_action_distribution(models, mode, coefficient):
    from intact_tracking.cli.memory350_policy_train import state_digest
    schema,_ = heavy_schema()
    options = dict(dr_aux_schema=heavy_aux_schema(schema),residual_output_mode='unbounded',latent_input_mode=mode)
    actor,critic,obs = models(**options)
    def labels(value):
        if coefficient:
            value[DR_TARGET_GROUP] = torch.rand(8,108)
            value[DR_HISTORY_WEIGHT_GROUP] = torch.ones(8,1)
        return value
    old = _algorithm(actor,critic,labels(obs),dr_aux_coef=coefficient)
    with torch.inference_mode():
        for step in range(2):
            old.act(obs)
            obs = labels(observation(step+1))
            old.process_env_step(obs,torch.randn(8),torch.zeros(8),
                {'motion_resample_boundary':torch.zeros(8,dtype=torch.bool)})
        old.compute_returns(obs)
    old.update()
    saved = deepcopy(old.save())
    restored_actor,restored_critic,fresh = models(**options,dr_aux_payload_com_enabled=False)
    current = _algorithm(restored_actor,restored_critic,labels(fresh),dr_aux_coef=5*coefficient)
    current.load(saved,None,strict=True)
    assert state_digest(current.save()) == state_digest(saved)
    obs = observation(13)
    torch.testing.assert_close(actor(obs),restored_actor(obs),atol=0,rtol=0)
    torch.testing.assert_close(actor.distribution.std_param,restored_actor.distribution.std_param,atol=0,rtol=0)
    assert len(restored_actor.dr_aux_objective.active) == 8


@pytest.mark.parametrize('mode', ['learned','zero'])
def test_auxiliary_resume_transition_rejects_unrelated_configuration_changes(mode):
    from intact_tracking.memory350_heavy_policy import validate_resume_models
    schema,_ = heavy_schema()
    old = {'actor':{'latent_input_mode':mode,'dr_aux_schema':heavy_aux_schema(schema),
                    'dr_aux_motor_weight':0.,'hidden_dims':[512,256,128]},
           'critic':{'hidden_dims':[512,256,128]},'obs_groups':{'actor':['policy']},
           'algorithm':{'dr_aux_coef':.1 if mode == 'learned' else 0.,'entropy_coef':.005}}
    new = deepcopy(old)
    new['actor']['dr_aux_payload_com_enabled'] = False
    new['algorithm']['dr_aux_coef'] *= 5
    args = SimpleNamespace(allow_dr_aux_change=True,resume='old.pt',reset_adaptive_sampling=True)
    result = validate_resume_models(old,new,args)
    assert result['active_coordinates_to'] == (8 if mode == 'learned' else 0)
    args.allow_dr_aux_change = False
    with pytest.raises(ValueError,match='Resume changed'):
        validate_resume_models(old,new,args)
    args.allow_dr_aux_change = True
    for key,field,value in [('actor','hidden_dims',[256]),('critic','hidden_dims',[256]),
                            ('algorithm','entropy_coef',.1),('actor','dr_aux_motor_weight',.1)]:
        changed = deepcopy(new);changed[key][field]=value
        with pytest.raises(ValueError,match='Resume changed'):
            validate_resume_models(old,changed,args)
    args.reset_adaptive_sampling = False
    with pytest.raises(ValueError,match='reset-adaptive-sampling'):
        validate_resume_models(old,new,args)
