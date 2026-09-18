"""Independent Memory350 variant: replace DR ranking with soft neighborhoods."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path

from intact_tracking.cli import forward_memory_nominal_direction_train as nominal
from intact_tracking.memory350_dr_center_rollout import DRCenterTrackerRollout
from intact_tracking.memory350_nominal_dr_rank import RANK_LOSS_FIELDS, NominalDRRankReplay
from intact_tracking.memory350_nominal_dr_soft import (
    SOFT_FIELDS, SOFT_METRICS, NominalDRSoftLossConfig, NominalDRSoftObjective,
)


def validate_resume_losses(previous, actual, *, new_stage, retune_nominal_anchor=False):
    if retune_nominal_anchor:
        if not new_stage or previous.get('dr_soft_version') != 1:
            raise ValueError('Nominal-anchor retuning requires a soft checkpoint and --resume-new-stage')
        if dict(previous, nominal_anchor_weight=actual['nominal_anchor_weight']) != actual:
            raise ValueError('Nominal-anchor retuning cannot change any other loss setting')
        return
    if previous.get('dr_soft_version') == 1:
        if previous != actual:
            raise ValueError('Soft-variant continuation must preserve all loss settings')
        return
    if not new_stage or previous.get('nominal_direction_objective_version') != 1:
        raise ValueError('Creating the soft variant requires nominal-direction source and --resume-new-stage')
    if 'dr_center_rank_version' in previous and previous['dr_center_rank_version'] != 1:
        raise ValueError('Unsupported source DR ranking version')
    before = {k: v for k, v in previous.items() if k not in RANK_LOSS_FIELDS}
    after = {k: v for k, v in actual.items() if k not in SOFT_FIELDS}
    if before != after:
        raise ValueError('Replacing DR ranking cannot change AB or other existing losses')


def soft_contract(loss, schema):
    return {
        'version': 1, 'schema': schema,
        'center': 'mean unit latents per world/session separately for current and archived views; normalize means for cosine',
        'eligibility': 'DR only; same world/session, different motion, full disjoint histories from existing weak archive',
        'candidates': 'all eligible world/session centers within one rank/microbatch; includes other-view same-world center',
        'target': 'row-normalize exp(-ln(2)*(parameter_distance/h)^2); diagonal raw weight 1; detached labels',
        'prediction': 'row-softmax of cross-view center cosine / temperature',
        'loss': 'mean of current-to-archive and archive-to-current KL(target || prediction); equal weight per world',
        'weight': loss['dr_soft_weight'], 'h': loss['dr_soft_h'],
        'temperature': loss['dr_soft_temperature'],
        'replaces_rank_loss': True, 'nominal_in_candidates': False,
        'inference': 'unchanged history-only encoder; DR parameters only supervise training',
    }


def configure_metadata(actual):
    nominal.configure_metadata(actual)
    args, loss = actual['arguments'], actual['loss']
    contract = soft_contract(loss, actual['batch_a_rollout']['dr_metric_schema'])
    if args.get('resume'):
        source = json.loads((Path(args['resume']).resolve().parent / 'run_config.json').read_text())
        validate_resume_losses(source['loss'], loss, new_stage=args['resume_new_stage'],
                               retune_nominal_anchor=args.get('resume_retune_nominal_anchor', False))
        prior = source.get('dr_soft_contract')
        if prior is not None and prior != contract:
            raise ValueError('Soft-variant continuation changed its supervision contract')
        schema = (prior or source.get('dr_center_rank_contract') or {}).get('schema')
        if schema is not None and schema != contract['schema']:
            raise ValueError('Soft variant must preserve DR parameter metric schema')
        actual['new_stage_loss_changes'] = {
            k: {'previous': source['loss'].get(k), 'current': loss.get(k)}
            for k in source['loss'].keys() | loss.keys() if source['loss'].get(k) != loss.get(k)}
    actual['method'] += ' plus DR soft neighborhoods v1 (ranking replaced)'
    actual['dr_soft_contract'] = contract
    actual['objective_weights']['dr_soft_weight'] = loss['dr_soft_weight']
    actual['validation']['dr_soft_probe'] = 'at least three eligible DR worlds; report target self mass, entropy and effective neighbors'
    for path in (Path(__file__), Path(__file__).parents[1] / 'memory350_nominal_dr_soft.py'):
        actual['research_source_sha256'][str(path.resolve().relative_to(Path(__file__).resolve().parents[3]))] = nominal.trainer._sha256(path)


def configure_checkpoint(state, rollout, *, base):
    base(state, rollout)
    state['representation_supervision'] = 'fixed_nominal_direction_response10_dr_soft_v1'
    state['dr_metric_schema'] = deepcopy(rollout.dr_metric_schema)
    state['dr_soft_supervision'] = {k: state['loss_config'][k] for k in SOFT_FIELDS}
    state['privileged_dynamics']['inference_contract'] = 'history-only encoder; DR parameters supervise soft neighborhoods during training only'


def build_parser():
    parser = nominal.build_parser()
    parser.description = __doc__
    parser.add_argument('--dr-soft-weight', type=float, default=.02)
    parser.add_argument('--dr-soft-h', type=float, default=.15)
    parser.add_argument('--dr-soft-temperature', type=float, default=.1)
    parser.add_argument('--resume-retune-nominal-anchor', action='store_true',
                        help='Allow only nominal anchor weight to change when resuming a soft checkpoint into a new stage')
    parser.set_defaults(nominal_fraction=.1, dr_nominal_probability=.5,
                        limb_max_masses_kg=(2.5, 2.5, 4., 4.), warmup_steps=1000,
                        representation_relation_weight=8., response_distance_scale=.6)
    return parser


def configure_trainer(args):
    if args.resume_retune_nominal_anchor and (not args.resume or not args.resume_new_stage):
        raise ValueError('Nominal-anchor retuning requires --resume and --resume-new-stage')
    config = NominalDRSoftLossConfig(dr_soft_weight=args.dr_soft_weight,
        dr_soft_h=args.dr_soft_h, dr_soft_temperature=args.dr_soft_temperature)
    if args.weak_archive_slots < 2 or args.weak_archive_interval < 1:
        raise ValueError('Cross-motion centers require at least two archive slots and a positive interval')
    nominal.configure_trainer(args)
    trainer = nominal.trainer
    trainer.FixedDRTrackerRollout = DRCenterTrackerRollout
    trainer.ForwardPredictorReplayBuffer = partial(NominalDRRankReplay,
        weak_archive_slots=args.weak_archive_slots, weak_archive_interval=args.weak_archive_interval,
        require_rank_probe=config.dr_soft_weight > 0)
    trainer.ForwardPredictorObjective = NominalDRSoftObjective
    trainer.ForwardPredictorLossConfig = partial(NominalDRSoftLossConfig,
        nominal_anchor_weight=args.nominal_anchor_weight, weak_positive_weight=args.weak_positive_weight,
        dr_soft_weight=args.dr_soft_weight, dr_soft_h=args.dr_soft_h,
        dr_soft_temperature=args.dr_soft_temperature)
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | {'dr_metric'}
    trainer._validate_resume_loss_config = partial(validate_resume_losses, new_stage=args.resume_new_stage,
        retune_nominal_anchor=args.resume_retune_nominal_anchor)
    trainer._configure_run_metadata = configure_metadata
    trainer._configure_checkpoint = partial(configure_checkpoint, base=trainer._configure_checkpoint)
    trainer._REPRESENTATION_PROBE_METRICS = (*trainer._REPRESENTATION_PROBE_METRICS, *SOFT_METRICS)


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(nominal.trainer.run(args), flush=True)


if __name__ == '__main__':
    main()
