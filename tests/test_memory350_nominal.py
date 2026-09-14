from types import SimpleNamespace

import pytest
import torch

from intact_tracking.environment.mdp.randomizations import body_force_pulse
from intact_tracking.memory350_nominal_rollout import (
    NominalExcludedForcePulse, NominalMemory350RolloutConfig,
)
from intact_tracking.rollout.online import FixedDRRolloutConfig
from intact_tracking.cli.forward_memory_nominal_train import build_parser, _validate_arguments


def test_nominal_pulses_stay_disabled_through_forced_triggers_and_partial_resets():
    n = 8
    wrench = torch.zeros(n, 3)
    def write(*, forces, torques, env_ids, body_ids):
        wrench[env_ids] = forces[:, 0]
    asset = SimpleNamespace(find_bodies=lambda _: ([0], ['torso_link']),
                            write_external_wrench_to_sim=write)
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, scene={'robot': asset})
    cfg = SimpleNamespace(params={'body_name': 'torso_link', 'interval_range_s': (3, 6),
                                   'duration_range_s': (.3, .5), 'force_abs_range_n': (0, 10)})
    source = body_force_pulse(cfg, env)
    nominal = torch.arange(0, n, 2)
    dr = torch.arange(1, n, 2)
    masked = NominalExcludedForcePulse(source, nominal)
    dr_was_pushed = False
    for step in range(400):
        if step % 71 == 0:
            masked.reset(torch.tensor([0, 1, 4, 7]))
            # The original event ignores env_ids; force every slot's timer due.
            source.time_to_next_pulse_s.zero_()
        masked(env, None)
        assert not source.active[nominal].any()
        assert not wrench[nominal].any()
        assert not masked.observe()[nominal].any()
        assert torch.isinf(source.time_to_next_pulse_s[nominal]).all()
        dr_was_pushed |= bool(wrench[dr].abs().max() > 0)
    assert dr_was_pushed
    assert source.interval_range_s == (3., 6.)
    assert source.duration_range_s == (.3, .5)


def test_new_config_accepts_mixture_without_changing_original_all_dr_contract():
    common = dict(checkpoint_file='tracker.pt', motion_file='motion.npz', num_envs=128,
                  tracker_dr_plus_limb_payload=True)
    assert NominalMemory350RolloutConfig(**common).nominal_fraction == .5
    assert FixedDRRolloutConfig(**common).nominal_fraction == 0
    with pytest.raises(ValueError):
        NominalMemory350RolloutConfig(**common, nominal_fraction=0)
    with pytest.raises(ValueError):
        NominalMemory350RolloutConfig(**{**common, 'num_envs': 127})


def test_training_defaults_and_validation_partition_require_nominal_half():
    parser = build_parser()
    command = ['--checkpoint-file', 'tracker.pt', '--motion-file', 'motion.npz', '--output-dir', 'out']
    args = parser.parse_args(command)
    _validate_arguments(args)
    assert args.nominal_fraction == .5
    assert (args.context_depth, args.context_dim, args.dynamics_latent_dim) == (2, 128, 64)
    assert (args.representation_weight, args.representation_relation_weight, args.response_distance_scale) == (.01, 2., .75)
    with pytest.raises(ValueError, match='Validation worlds'):
        _validate_arguments(parser.parse_args(command + ['--validation-worlds', '127']))
