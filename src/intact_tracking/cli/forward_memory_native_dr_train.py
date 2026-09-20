"""Scratch Memory350 soft training on a checkpoint's original flat-world DR."""
from functools import partial
from pathlib import Path

from intact_tracking.cli import forward_memory_nominal_dr_soft_train as soft
from intact_tracking.memory350_native_dr import NativeDRTrackerRollout
from intact_tracking.rollout.online import FixedDRRolloutConfig


NATIVE_PHYSICAL_ACTION = 'predictor uses control-step mean physical PD target (29D); B replays all physical substeps exactly'


def validate_native_response_contract(previous, actual, *, base, **kwargs):
    """Validate the native action extension before the shared anchor contract."""
    def shared(contract):
        if contract is None:
            return None
        contract = dict(contract)
        physical = contract.pop('physical_action', NATIVE_PHYSICAL_ACTION)
        if physical != NATIVE_PHYSICAL_ACTION:
            raise ValueError('Continuation changed the native physical action contract')
        return contract
    base(shared(previous), shared(actual), **kwargs)


class FlatTerrainHeightOffset:
    """The source terrain-height event is exactly a no-op without a terrain plan."""
    model_fields = ()

    def __init__(self, cfg, env):
        if getattr(env.cfg.commands.get('motion'), 'terrain_motion_plan', None) is not None:
            raise ValueError('This entry point is explicitly flat-only')

    def __call__(self, env, env_ids, **kwargs):
        pass


def build_parser():
    parser = soft.build_parser()
    parser.description = __doc__
    parser.set_defaults(checkpoint_native_dr=True, tracker_dr_plus_limb_payload=False,
        limb_max_masses_kg=None, dr_nominal_probability=0., nominal_fraction=.1,
        nominal_anchor_weight=.08, representation_relation_weight=10.,
        batch_size=1024, micro_batch_size=256)
    return parser


def configure_metadata(actual):
    soft.configure_metadata(actual)
    args = actual['arguments']
    actual['method'] = 'Memory350 soft, checkpoint-native flat DR, no payloads, uniform motions'
    actual['architecture']['physics'] = (
        'ceil(10% of each rank) restored nominal; remaining worlds original checkpoint random DR; '
        'fixed physical parameters across resets; native stateful action chain captured at every substep')
    actual['nominal_direction_contract']['physical_action'] = NATIVE_PHYSICAL_ACTION
    actual['native_dr_contract'] = actual['batch_a_rollout']['payload']
    actual['native_dr_contract']['episode_length_control_steps'] = actual['batch_a_rollout']['episode_length_control_steps']
    actual['episode_length_control_steps'] = actual['batch_a_rollout']['episode_length_control_steps']
    actual['initialization'] = 'scratch' if not args['resume'] else 'native checkpoint continuation'
    actual['research_source_sha256'][str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]))] = soft.nominal.trainer._sha256(Path(__file__))


def configure_checkpoint(state, rollout, *, base):
    base(state, rollout)
    state['native_dr_version'] = 1
    state['predictor_action_contract'] = rollout.metadata['predictor_action_transform']
    state['episode_length_control_steps'] = rollout.env.max_episode_length


def configure_trainer(args):
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = FlatTerrainHeightOffset
    soft.configure_trainer(args)
    soft.nominal.validate_response_contract = partial(
        validate_native_response_contract, base=soft.nominal.validate_response_contract)
    trainer = soft.nominal.trainer
    trainer.FixedDRRolloutConfig = FixedDRRolloutConfig
    trainer.FixedDRTrackerRollout = NativeDRTrackerRollout
    trainer._configure_run_metadata = configure_metadata
    trainer._configure_checkpoint = partial(configure_checkpoint, base=trainer._configure_checkpoint)


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(soft.nominal.trainer.run(args), flush=True)


if __name__ == '__main__':
    main()
