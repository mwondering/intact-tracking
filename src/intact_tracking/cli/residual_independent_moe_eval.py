"""Evaluate independent experts using the unchanged uniform-residual physics."""

from intact_tracking.cli import residual_uniform_eval as uniform
from intact_tracking.memory350_independent_moe_policy import VERSION, ACTOR_CLASS, IndependentMoEActor


def main():
    uniform.VERSION = VERSION
    uniform.ACTOR_CLASS = ACTOR_CLASS
    uniform.UnboundedResidualActor = IndependentMoEActor
    # Keep the measurement/physics protocol ID equal across architectures so
    # the existing strict trial-pairing audit remains meaningful.
    uniform.main()


if __name__ == "__main__":
    main()
