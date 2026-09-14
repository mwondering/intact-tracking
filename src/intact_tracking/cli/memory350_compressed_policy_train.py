"""Four-GPU matched PPO with learned observation compression and concatenated context."""

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.memory350_compressed_policy import (
    VERSION, configure_compressed_models, audit_initial_models,
)


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(training_ranks=4)
    return parser


def main():
    args = build_parser().parse_args()
    if args.fusion not in ("baseline", "concat"):
        raise ValueError("Compressed PPO compares zero latent with concatenated learned latent")
    base.VERSION = VERSION
    base.configure_context_models = configure_compressed_models
    base.audit_initial_models = audit_initial_models
    base.run(args)


if __name__ == "__main__":
    main()
