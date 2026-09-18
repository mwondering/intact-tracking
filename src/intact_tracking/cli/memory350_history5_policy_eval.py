"""Fixed-query evaluation of the five-frame compressed PPO comparison."""
from functools import partial
from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking import memory350_history5_policy as variant
from intact_tracking.memory350_compressed_policy import ACTOR_CLASS, CompressedContextActor


def main():
    args = base.build_parser().parse_args()
    base.VERSION = variant.VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = CompressedContextActor
    base.LimbContextWrapper = variant.History5Wrapper
    base.configure_limb_dr = variant.configure_physics
    base.audit_limb_dr = variant.audit_physics
    base.ManagerBasedRlEnv = partial(variant.environment_factory, base.ManagerBasedRlEnv,
                                    restore_nominal=args.fixed_masses is None)
    base.run(args)


if __name__ == '__main__':
    main()
