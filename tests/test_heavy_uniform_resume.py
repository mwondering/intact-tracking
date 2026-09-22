from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from intact_tracking.limb_context_sampling import SamplingCheckpoint, state_digest, validate_sampling_resume
from intact_tracking.memory350_native_policy import configure_sampling
from test_tracker144000_sampling_parity import source_cfg


def sampling(mode):
    cfg = SimpleNamespace(commands={'motion':source_cfg()})
    return cfg, configure_sampling(cfg,mode,0,2001)


def test_uniform_sampling_disables_rewind_without_changing_the_adaptive_profile():
    _, adaptive = sampling('adaptive')
    cfg, uniform = sampling('uniform')
    assert cfg.commands['motion'].sampling_mode == 'uniform'
    assert not cfg.commands['motion'].rewind.enabled
    assert uniform['parameters'] == adaptive['parameters']
    assert uniform['adaptive_sampling'] == adaptive['adaptive_sampling']
    assert uniform['rewind'] == {**adaptive['rewind'],'enabled':False}
    previous = {'motion_sampling':adaptive}
    with pytest.raises(ValueError,match='exact curriculum'):
        validate_sampling_resume(previous,uniform,2001)
    assert validate_sampling_resume(previous,uniform,2001,switch_to_uniform=True) == 'adaptive'
    changed = deepcopy(uniform)
    changed['parameters']['adaptive_probability_max_over_mean'] += 1
    with pytest.raises(ValueError,match='unrelated sampler'):
        validate_sampling_resume(previous,changed,2001,switch_to_uniform=True)


def test_uniform_resume_never_reads_old_adaptive_files_or_modifies_model_state(tmp_path,monkeypatch):
    cfg, uniform = sampling('uniform')
    weights = {'weights':torch.ones(3),'optimizer':{'exp_avg':torch.tensor([.2])}}
    before = state_digest(weights)
    runner = SimpleNamespace(completed_learning_updates=2001, alg=weights,
        loaded_motion_sampling_state={'ranks':[{'path':'/missing/adaptive.pt'}]},
        env=SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(
            get_term=lambda _:cfg.commands['motion']))))
    # The runtime command exposes its configuration under cfg.
    command = SimpleNamespace(cfg=cfg.commands['motion'])
    runner.env.unwrapped.command_manager.get_term = lambda _:command
    def reject(*args,**kwargs):
        pytest.fail('Uniform resume must not load old sampler files')
    monkeypatch.setattr(torch,'load',reject)
    sampler = SamplingCheckpoint(tmp_path,SimpleNamespace(rank=0),uniform)
    report = sampler.restore(runner,'adaptive')
    assert report['adaptive_statistics_discarded'] and not report['restored']
    assert runner.loaded_motion_sampling_state is None and runner.motion_sampling_state is None
    sampler.prepare(runner)
    assert not list(tmp_path.iterdir()) and before == state_digest(weights)
    assert sampler.restore(runner,'uniform')['previous_mode'] == 'uniform'


@pytest.mark.parametrize('resume,mode,reset,align',[
    (None,'uniform',False,False), ('old.pt','adaptive',False,False),
    ('old.pt','uniform',True,False), ('old.pt','uniform',False,True)])
def test_explicit_uniform_resume_rejects_conflicting_flags(resume,mode,reset,align):
    from intact_tracking.cli.memory350_policy_train import _run
    args = SimpleNamespace(resume_uniform_sampling=True,resume=resume,motion_sampling=mode,
                           reset_adaptive_sampling=reset,align_sampling_to_tracker=align)
    with pytest.raises(ValueError,match='resume-uniform-sampling'):
        _run(args,None)


@pytest.mark.parametrize('mode',['learned','zero'])
def test_uniform_transition_keeps_the_authorized_mass_only_objective(mode):
    from intact_tracking.memory350_heavy_policy import validate_resume_models
    from test_memory350_heavy_policy import heavy_schema
    from intact_tracking.memory350_heavy_policy import heavy_aux_schema
    schema,_ = heavy_schema()
    old = {'actor':{'latent_input_mode':mode,'dr_aux_schema':heavy_aux_schema(schema),
                    'dr_aux_motor_weight':0.,'hidden_dims':[512,256,128]},
           'critic':{},'obs_groups':{},'algorithm':{'dr_aux_coef':.1 if mode=='learned' else 0.}}
    new = deepcopy(old); new['actor']['dr_aux_payload_com_enabled'] = False
    new['algorithm']['dr_aux_coef'] *= 5
    args = SimpleNamespace(allow_dr_aux_change=True,resume='old.pt',
                           reset_adaptive_sampling=False,resume_uniform_sampling=True)
    assert validate_resume_models(old,new,args)['coefficient_to'] == (.5 if mode=='learned' else 0.)
