"""Scratch proprio122 Memory350 on native DR with stratified limb mass and COM."""

from copy import deepcopy
from functools import partial
from pathlib import Path

from intact_tracking.cli import forward_memory_proprio_native_train as proprio
from intact_tracking.memory350_heavy_dr import HeavyDRRolloutConfig, MASS_LIMITS


def build_parser():
    parser = proprio.build_parser()
    parser.description = __doc__
    parser.add_argument('--heavy-max-masses-kg', type=float, nargs=4, default=MASS_LIMITS)
    parser.add_argument('--heavy-com-half-width-m', type=float, default=.05)
    parser.add_argument('--weak-archive-device', choices=('cpu', 'cuda'), default='cpu',
                        help='Raw float32 archive storage; sampling metadata stays on the training GPU')
    return parser


def configure_metadata(actual):
    proprio.configure_metadata(actual)
    actual['method'] = 'Memory350 noisy proprio122, native flat DR plus 256 stratified limb mass bins and independent xyz payload COM'
    actual['architecture']['physics'] = (
        'ceil(10% of each rank) exactly nominal; remaining worlds checkpoint-native DR plus '
        'four independently sampled payloads in balanced Cartesian mass bins; fixed physics across resets')
    actual['heavy_payload_contract'] = deepcopy(actual['native_dr_contract']['heavy_payload'])
    actual['replay']['weak_archive_storage'] = {
        'device': actual['arguments']['weak_archive_device'], 'dtype': 'float32',
        'sampling': 'unchanged; exact raw histories copied to GPU only for sampled pairs'}
    root = Path(__file__).resolve().parents[3]
    for path in (Path(__file__), root / 'src/intact_tracking/memory350_heavy_dr.py',
                 root / 'src/intact_tracking/memory350_native_dr.py',
                 root / 'src/intact_tracking/memory350_weak_pairs.py'):
        actual['research_source_sha256'][str(path.relative_to(root))] = proprio.native.soft.nominal.trainer._sha256(path)


def configure_checkpoint(state, rollout, *, base):
    base(state, rollout)
    state['heavy_payload_contract'] = deepcopy(rollout.payload_configuration['heavy_payload'])


def configure_trainer(args):
    proprio.configure_trainer(args)
    trainer = proprio.native.soft.nominal.trainer
    trainer.FixedDRRolloutConfig = partial(HeavyDRRolloutConfig,
        heavy_max_masses_kg=tuple(args.heavy_max_masses_kg),
        heavy_com_half_width_m=args.heavy_com_half_width_m)
    trainer.ForwardPredictorReplayBuffer = partial(trainer.ForwardPredictorReplayBuffer,
        weak_archive_device=args.weak_archive_device)
    trainer._configure_run_metadata = configure_metadata
    trainer._configure_checkpoint = partial(configure_checkpoint, base=trainer._configure_checkpoint)


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(proprio.native.soft.nominal.trainer.run(args), flush=True)


if __name__ == '__main__':
    main()
