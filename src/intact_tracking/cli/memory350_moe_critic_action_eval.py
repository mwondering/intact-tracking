"""Evaluate v2 actor checkpoints; critic action features affect PPO training only."""

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.memory350_moe_policy import ACTOR_CLASS, HardMoEActor
from intact_tracking.memory350_moe_critic_action import VERSION


def main():
    base.VERSION = VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = HardMoEActor
    base.run(base.build_parser().parse_args())


if __name__ == "__main__":
    main()
