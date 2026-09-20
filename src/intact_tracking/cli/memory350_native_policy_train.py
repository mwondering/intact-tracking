"""144000 flat native DR: scratch residual PPO with directly concatenated Memory350."""
from functools import partial
import hashlib
from pathlib import Path

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking import memory350_native_policy as variant
from intact_tracking.memory350_native_dr import native_dataset_identity


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == 'training_ranks':
            action.choices = (1, 2, 4, 8)
        elif action.dest == 'dr_profile':
            action.choices = (variant.PROFILE,)
    parser.add_argument('--bounded-smoke', action='store_true')
    parser.set_defaults(training_ranks=8, fusion='concat', tracker_checkpoint=variant.TRACKER,
                        context_checkpoint=variant.CONTEXT, dr_profile=variant.PROFILE,
                        episode_steps=500, motion_sampling='adaptive', adaptive_after_update=0,
                        training_terminations='original', policy_precision='fp32', until_user_stop=True,
                        entropy_coef=0.005, initial_action_std=1.0,
                        residual_output_mode='unbounded', residual_scale=1.0,
                        wandb_group='144000_exp', wandb_name='144000_exp-residual-concat-u14500',
                        save_interval=100)
    return parser


def configure(args):
    if args.bounded_smoke:
        args.until_user_stop = False
    elif args.training_ranks != 8 or args.num_envs != 8192 or not args.until_user_stop:
        raise ValueError('Formal training uses eight ranks, 8192 environments each, without an update cap')
    if (args.fusion != 'concat' or args.dr_sampling != 'independent_uniform'
            or args.training_terminations != 'original' or args.episode_steps != 500
            or args.motion_sampling != 'adaptive' or args.adaptive_after_update != 0):
        raise ValueError('Native residual PPO contract changed')
    if args.endpoint_eval_protocol:
        raise ValueError('Legacy payload endpoint evaluation is incompatible with native DR')
    with Path(args.context_checkpoint).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != variant.CONTEXT_SHA256:
            raise ValueError('Use the selected, evaluated u14500 encoder')
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = variant.FlatTerrainHeightOffset
    base.VERSION = variant.VERSION
    base.TRACKER_SHA256 = variant.TRACKER_SHA256
    base.FULL_DATASET = variant.FULL_DATASET
    base.EPISODE_STEPS = 500
    base.ALLOWED_DR_PROFILES = (variant.PROFILE,)
    base.CONTEXT_SOURCE_DR_PROFILE = variant.PROFILE
    base.TRAINING_START_PROFILE = 'original'
    base.resolve_dr_profile = variant.resolve_profile
    base.validate_context_dr = variant.validate_context
    base.configure_limb_dr = variant.configure_physics
    base.audit_limb_dr = variant.audit_physics
    base.configure_motion_sampling = variant.configure_sampling
    base.LimbContextWrapper = variant.NativePolicyWrapper
    base.ManagerBasedRlEnv = partial(variant.environment_factory, base.ManagerBasedRlEnv)
    base.dataset_identity = partial(native_dataset_identity, variant.TRACKER)


def main():
    args = build_parser().parse_args()
    configure(args)
    base.run(args)


if __name__ == '__main__':
    main()
