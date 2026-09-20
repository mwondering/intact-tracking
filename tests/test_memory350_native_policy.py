from types import SimpleNamespace

import pytest
import torch

from intact_tracking.environment.mdp.actions import SpTrackingJointPositionAction
from intact_tracking.memory350_native_policy import (
    NativePolicyAction, NativePolicyWrapper, configure_sampling, validate_context, PROFILE, TRACKER_SHA256,
)


def test_ppo_sampled_action_does_not_overwrite_separately_recorded_mean(monkeypatch):
    obj = object.__new__(NativePolicyAction)
    obj.cfg = SimpleNamespace(prev_action_obs='mean', action_rate_source='mean', raw_action_clip=None)
    obj._raw_actions = torch.zeros(2, 29)
    obj._action_mean_history = torch.zeros(2, 8, 29)
    mean = torch.full((2, 29), .2)
    sample = torch.full((2, 29), .7)
    obj.record_policy_mean(mean)
    seen = []
    monkeypatch.setattr(SpTrackingJointPositionAction, 'process_actions', lambda self, x: seen.append(x.clone()))
    obj.process_actions(sample)
    torch.testing.assert_close(seen[0], sample, rtol=0, atol=0)
    torch.testing.assert_close(obj._action_mean_history[:, 0], mean, rtol=0, atol=0)
    assert not obj._action_mean_history[:, 1].any()


def test_native_wrapper_uses_all_physical_substeps_instead_of_last_target(monkeypatch):
    trace = torch.tensor([1., 2., 4., 9.]).reshape(1, 4, 1).expand(2, 4, 29).clone()
    action = SimpleNamespace(physical_target_trace=trace)
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: action))
    monkeypatch.setattr(NativePolicyWrapper, 'unwrapped', property(lambda _: env))
    obj = object.__new__(NativePolicyWrapper)
    target = obj._applied_joint_target()
    assert target.shape == (2, 29)
    torch.testing.assert_close(target, torch.full_like(target, 4.))
    assert not torch.equal(target, trace[:, -1])


def test_native_context_rejects_old_last_target_contract():
    state = {'native_dr_version': 1, 'tracker': {'checkpoint_sha256': TRACKER_SHA256},
             'episode_length_control_steps': 500,
             'predictor_action_contract': {'available': True, 'mode': 'captured_substep_physical_targets',
                                          'predictor_input': 'mean over one control step'}}
    validate_context(state, PROFILE)
    state['predictor_action_contract']['predictor_input'] = 'last target'
    with pytest.raises(ValueError, match='physical target'):
        validate_context(state, PROFILE)


def test_native_sampling_cannot_silently_start_uniform():
    for mode, after in [('uniform', 0), ('adaptive', 1000)]:
        with pytest.raises(ValueError, match='update zero'):
            configure_sampling(None, mode, after, 0)


def test_native_parser_supports_eight_ranks_without_changing_legacy_defaults():
    from intact_tracking.cli.memory350_native_policy_train import build_parser
    from intact_tracking.cli.memory350_policy_train import build_parser as legacy_parser
    args = build_parser().parse_args(['--fusion', 'concat', '--output-dir', 'unused', '--training-ranks', '8'])
    assert args.num_envs == 8192 and args.episode_steps == 500
    assert args.motion_sampling == 'adaptive' and args.adaptive_after_update == 0
    assert args.until_user_stop and args.training_terminations == 'original'
    legacy = legacy_parser().parse_args(['--fusion', 'concat', '--output-dir', 'unused'])
    assert legacy.episode_steps == 1000
