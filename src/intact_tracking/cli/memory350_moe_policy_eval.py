"""Fixed-center evaluation of a saved online-K-means MoE or its MLP baseline."""

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.memory350_moe_policy import VERSION, ACTOR_CLASS, HardMoEActor


def main():
    base.VERSION = VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = HardMoEActor
    base.run(base.build_parser().parse_args())


if __name__ == "__main__":
    main()
