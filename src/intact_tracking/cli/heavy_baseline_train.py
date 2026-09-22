"""RMA teacher / Any2Track under the matched 144000 HDR training protocol."""

from functools import partial
import os
from pathlib import Path

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking import memory350_native_policy as native, memory350_heavy_policy as heavy
from intact_tracking.heavy_rma_teacher import VERSION, configure_models, audit_initial_models
from intact_tracking.heavy_baseline_env import HeavyBaselineWrapper
from intact_tracking.memory350_native_dr import native_dataset_identity

METHODS = ('rma_teacher', 'any2track')
BOUNDARY_COMPLETED_UPDATES = 2001


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    for option in parser._actions:
        if option.dest == 'fusion':
            option.required = False
        if option.dest == 'dr_profile':
            option.choices = (heavy.PROFILE,)
        if option.dest == 'training_ranks':
            option.choices = (1, 2, 4)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--stage', choices=('stage1', 'resume_sampling_reset', 'continuous_uniform'), default='continuous_uniform')
    parser.add_argument('--bounded-smoke', action='store_true')
    parser.add_argument('--smoke-stage-boundary', type=int,
                        help='Only for bounded smoke: exercise the same stage transition at a small update count')
    parser.set_defaults(fusion='concat', dr_profile=heavy.PROFILE, training_ranks=4, num_envs=8192,
                        tracker_checkpoint=native.TRACKER,
                        episode_steps=500, training_terminations='original', motion_sampling='uniform',
                        adaptive_after_update=0, policy_precision='fp32', entropy_coef=.005,
                        initial_action_std=1., residual_scale=1., residual_output_mode='unbounded',
                        wandb_group='144000-exp-heavy', iterations=BOUNDARY_COMPLETED_UPDATES)
    return parser


def adapt_metadata(metadata, args, previous):
    if not args.bounded_smoke and (
            metadata['dataset']['motion_count'] != 220480 or
            metadata['dataset']['manifest_sha256'] != '59b8e336c152e4773133bcdcd86cf86fd6545c3c0e6cc00ce67912d107c740ac'):
        raise ValueError('The filtered full dataset differs from the matched heavy experiment')
    old = previous['residual_policy'] if previous else {}
    boundary = getattr(args, 'smoke_stage_boundary', None) or BOUNDARY_COMPLETED_UPDATES
    generation = int(old.get('sampling_reset_generation', 0))
    if args.reset_adaptive_sampling:
        if (args.stage != 'resume_sampling_reset' or generation != 0
                or previous['completed_updates'] != boundary
                or old.get('baseline_stage') != 'stage1'):
            raise ValueError('The single planned sampler reset must follow stage1 at completed update 2001')
        generation = 1
    if args.stage == 'continuous_uniform':
        if args.motion_sampling != 'uniform' or args.reset_adaptive_sampling or generation or args.resume_uniform_sampling:
            raise ValueError('Continuous uniform training has no adaptive sampler or planned reset')
        if previous and old.get('baseline_stage') != 'continuous_uniform':
            raise ValueError('Continuous uniform resume must use its own training checkpoint')
    if previous and old.get('method') != args.method:
        raise ValueError('Resume cannot change the baseline method')
    if args.stage == 'resume_sampling_reset' and generation != 1:
        raise ValueError('Stage B must restore the once-refreshed sampler or request its initial refresh')
    metadata.update(method=args.method, baseline_stage=args.stage, sampling_reset_generation=generation,
                    version=VERSION, context_protocol=None, frozen_inference='SPV5-2A tracker only',
                    execution_protocol='heavy_baselines_4gpu8192_v1',
                    encoder_frozen=False, context_normalization_frozen=None,
                    predictor_executed_in_ppo=False, extra_current_state_physics_privilege=args.method == 'rma_teacher',
                    baseline_contract={
                        'reference_commit': 'cb9b751993a2483e5d1805a2565ddbfe950c04c9' if args.method == 'any2track' else None,
                        'dr_auxiliary_loss': False, 'pretrained_memory350_weights': False,
                        'conditioning': 'theta108 normalized [-1,1]' if args.method == 'rma_teacher' else 'past79 state64+command29',
                        'encoder_objective': 'PPO' if args.method == 'rma_teacher' else '20-step autoregressive dynamics',
                        'action': 'frozen tracker + unbounded residual' if args.method == 'rma_teacher' else 'layer-adapted frozen tracker, no extra addition',
                        'training_schedule': ('continuous_until_user_stop' if args.stage == 'continuous_uniform'
                                              else 'checkpoint_2000_then_once_refreshed_resume'),
                        'stage1_boundary_completed_updates': None if args.stage == 'continuous_uniform' else boundary,
                        'stage2_first_completed_update': None if args.stage == 'continuous_uniform' else boundary+1,
                        'drift_metrics_scope': 'first 1024 worlds at first rollout step; embedding drift all transitions',
                        'world_model': None if args.method == 'rma_teacher' else {
                            'lr': 1e-4, 'optimizer': 'Adam', 'epochs': 5, 'mini_batches': 4,
                            'horizon': 20, 'history': 79, 'embedding': 128,
                            'hidden_dims': [512, 512, 256, 256, 256, 128],
                            'loss_weights': [500., 500., 1., .5, 500.],
                            'velocity_scale': .05, 'dt': .02, 'initialization_and_window_seed': 40130,
                            'gradient_clip_norm': 1., 'parameters': 676897,
                        },
                    })
    metadata['actor_initialization'] = ('fresh DR encoder and residual MLP; zero residual output; std=1.0'
                                        if args.method == 'rma_teacher' else
                                        'zero layer adapters; fresh history encoder and world model; std=1.0')
    metadata['encoder_frozen'] = args.method == 'any2track'
    metadata['encoder_freezing_scope'] = ('PPO only; trainable during the preceding world-model update'
                                         if args.method == 'any2track' else 'trainable by PPO')
    for path in (*Path(__file__).parents[1].glob('heavy_*.py'),
                 *Path(__file__).parents[1].glob('anyadapter_*.py'), Path(__file__)):
        metadata['research_source_sha256'][str(path.relative_to(base.PROJECT_ROOT))] = base._sha256(path)


def configure(args):
    if args.smoke_stage_boundary is not None and (not args.bounded_smoke or args.smoke_stage_boundary < 1):
        raise ValueError('A short stage boundary is allowed only in explicitly bounded smoke runs')
    if args.context_checkpoint:
        raise ValueError('These baselines train their own conditioning modules from scratch')
    if (args.fusion != 'concat' or args.dr_sampling != 'independent_uniform' or args.episode_steps != 500
            or args.training_terminations != 'original' or args.motion_sampling not in ('adaptive', 'uniform')
            or args.adaptive_after_update != 0 or args.endpoint_eval_protocol
            or args.align_sampling_to_tracker or args.policy_precision != 'fp32'):
        raise ValueError('The matched native tracker/HDR contract must be preserved')
    if ((args.stage == 'resume_sampling_reset' and args.motion_sampling != 'adaptive')
            or (args.stage == 'continuous_uniform' and args.motion_sampling != 'uniform')
            or (args.reset_adaptive_sampling and args.motion_sampling != 'adaptive')
            or args.resume_uniform_sampling):
        raise ValueError('The baseline stage and resume flags must match its motion sampling mode')
    if args.bounded_smoke:
        args.until_user_stop = False
        base.WandbLogger = partial(base.WandbLogger, mode='disabled')
    else:
        if args.training_ranks != 4 or args.num_envs != 8192 or args.motion_file or args.seed != 121:
            raise ValueError('Formal baselines require four ranks, 8192 environments per rank and the full catalog')
        if (args.rollout_steps, args.epochs, args.mini_batches, args.actor_lr, args.critic_lr,
            args.entropy_coef, args.initial_action_std, args.residual_scale, args.residual_output_mode) != (
                24, 5, 4, 1e-4, 5e-4, .005, 1., 1., 'unbounded'):
            raise ValueError('Formal PPO hyperparameters must match the heavy residual experiment')
        if args.stage == 'stage1':
            args.iterations, args.until_user_stop = BOUNDARY_COMPLETED_UPDATES, False
        else:
            args.until_user_stop = True
        if args.motion_path and Path(args.motion_path).resolve() != Path(native.FULL_DATASET):
            raise ValueError('Formal baseline requires the full filtered motion_data_correct catalog')
    if args.stage == 'resume_sampling_reset' and not args.resume:
        raise ValueError('Stage B requires a checkpoint')
    if args.until_user_stop and args.stage == 'stage1':
        raise ValueError('Stage A has a formal boundary at checkpoint_2000')
    args.wandb_name = args.wandb_name or ('144000-exp-heavy-' + args.method.replace('_', '-')
                                        + ('-uniform' if args.motion_sampling == 'uniform' else ''))
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    base.VERSION, base.CONTEXT_CHECKPOINT_REQUIRED = VERSION, False
    base.EMBED_TRACKER_WITHOUT_CONTEXT, base.METADATA_ADAPTER = True, adapt_metadata
    base.WANDB_TAGS = ('heavy', 'baseline', args.method)
    base.PPO_CLASS_NAME = ('intact_tracking.memory350_tracker_action_policy:TrackerActionPPO'
                           if args.method == 'rma_teacher' else 'intact_tracking.anyadapter_training:AnyAdapterPPO')
    base.configure_context_models = partial(configure_models, method=args.method)
    base.audit_initial_models = audit_initial_models
    base.TRACKER_SHA256, base.FULL_DATASET, base.EPISODE_STEPS = native.TRACKER_SHA256, native.FULL_DATASET, 500
    base.ALLOWED_DR_PROFILES, base.TRAINING_START_PROFILE = (heavy.PROFILE,), 'original'
    base.resolve_dr_profile = heavy.resolve_profile
    base.configure_limb_dr = partial(heavy.configure_physics, rank=int(os.environ.get('RANK', '0')))
    base.audit_limb_dr, base.configure_motion_sampling = heavy.audit_physics, native.configure_sampling
    base.LimbContextWrapper = partial(HeavyBaselineWrapper, method=args.method)
    base.ManagerBasedRlEnv = partial(heavy.environment_factory, base.ManagerBasedRlEnv)
    base.dataset_identity = partial(native_dataset_identity, native.TRACKER)
    base.TRACKER_WANDB_GROUPS = (*base.TRACKER_WANDB_GROUPS, 'Teacher', 'WorldModel', 'Adapter')


def main():
    args = build_parser().parse_args()
    configure(args)
    base.run(args)


if __name__ == '__main__':
    main()
