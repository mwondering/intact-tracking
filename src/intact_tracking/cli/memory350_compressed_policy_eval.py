"""Original fixed cold/warm tracking evaluation for the compressed residual actors."""

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.memory350_compressed_policy import VERSION, ACTOR_CLASS, CompressedContextActor


def main():
    base.VERSION = VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = CompressedContextActor
    base.run(base.build_parser().parse_args())


if __name__ == "__main__":
    main()
