"""Paired MLP/MoE PPO with independent encoders and tracker action in both heads."""

from functools import partial

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.cli.memory350_moe_policy_train import build_parser
from intact_tracking.memory350_moe_training import OnlineMoERunner
from intact_tracking.memory350_moe_critic_action import (
    VERSION, configure_critic_action_models, audit_critic_action_models,
    TrackerActionBaselineWrapper, TrackerActionContextWrapper,
)


def main():
    args = build_parser().parse_args()
    if args.fusion not in ("baseline", "concat") or args.dr_sampling != "independent_uniform":
        raise ValueError("Use the matched MLP/MoE independent-uniform DR experiment")
    base.VERSION = VERSION
    base.PPO_CLASS_NAME = "intact_tracking.memory350_moe_critic_action:TrackerActionMoEPPO"
    base.RslRlVecEnvWrapper = TrackerActionBaselineWrapper
    base.LimbContextWrapper = TrackerActionContextWrapper
    base.ResidualOnPolicyRunner = OnlineMoERunner
    base.configure_context_models = partial(configure_critic_action_models,
        num_experts=args.num_experts, center_rate=args.router_center_rate,
        max_switch_fraction=args.router_max_switch_fraction)
    base.audit_initial_models = audit_critic_action_models
    base.run(args)


if __name__ == "__main__":
    main()
