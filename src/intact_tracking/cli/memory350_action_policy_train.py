"""Matched compressed residual PPO conditioned on the current raw tracker action."""

from functools import partial

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.cli.memory350_compressed_policy_train import build_parser
from intact_tracking.memory350_compressed_policy import (
    ACTION_VERSION, configure_compressed_models, audit_initial_models,
)


def main():
    args = build_parser().parse_args()
    if args.fusion not in ("baseline", "concat"):
        raise ValueError("Tracker-action PPO compares baseline and concatenated latent")
    base.VERSION = ACTION_VERSION
    base.configure_context_models = partial(configure_compressed_models, tracker_action_input=True)
    base.audit_initial_models = audit_initial_models
    base.run(args)


if __name__ == "__main__":
    main()
