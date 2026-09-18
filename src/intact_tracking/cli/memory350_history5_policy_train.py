"""Scratch uniform PPO: compressed original observations plus five frozen latents."""
from functools import partial
from intact_tracking.cli import memory350_policy_train as base
from intact_tracking import memory350_history5_policy as variant


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument("--bounded-smoke", action="store_true", help="Use --iterations only for preflight; formal runs stay unbounded")
    parser.set_defaults(training_ranks=4, motion_sampling='uniform', adaptive_after_update=0,
                        training_terminations='original', policy_precision='fp32', until_user_stop=True)
    return parser


def configure(args):
    if args.bounded_smoke:
        args.until_user_stop = False
    if args.fusion not in ('baseline','concat') or args.dr_sampling != 'independent_uniform':
        raise ValueError('History5 compares baseline/concat with independent nominal-mixture DR')
    if args.motion_sampling != 'uniform' or args.training_terminations != 'original':
        raise ValueError('This experiment requires uniform motions and complete original termination')
    base.VERSION = variant.VERSION
    base.configure_context_models = variant.configure_models
    base.configure_motion_sampling = variant.configure_motion_sampling
    base.audit_initial_models = variant.audit_initial_models
    base.LimbContextWrapper = variant.History5Wrapper
    base.configure_limb_dr = variant.configure_physics
    base.audit_limb_dr = variant.audit_physics
    base.ManagerBasedRlEnv = partial(variant.environment_factory, base.ManagerBasedRlEnv)


def main():
    args = build_parser().parse_args()
    configure(args)
    base.run(args)


if __name__ == '__main__':
    main()
