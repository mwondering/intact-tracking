"""Fixed cold/warm evaluation of residual actors with raw tracker action inputs."""

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.memory350_compressed_policy import ACTION_VERSION, ACTOR_CLASS, CompressedContextActor


def main():
    base.VERSION = ACTION_VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = CompressedContextActor
    base.run(base.build_parser().parse_args())


if __name__ == "__main__":
    main()
