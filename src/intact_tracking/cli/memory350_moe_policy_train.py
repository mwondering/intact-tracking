"""Uniform-DR online K-means16 hard MoE versus a latent-free single-head MLP."""

from functools import partial

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.memory350_moe_policy import VERSION, configure_moe_models, audit_initial_models
from intact_tracking.memory350_moe_training import OnlineMoERunner


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(training_ranks=4, dr_sampling="independent_uniform", policy_precision="fp32",
                        adaptive_after_update=0, training_terminations="original")
    parser.add_argument("--num-experts", type=int, choices=(16,), default=16)
    parser.add_argument("--router-center-rate", type=float, default=0.01)
    parser.add_argument("--router-max-switch-fraction", type=float, default=0.02)
    parser.add_argument("--router-bootstrap-steps", type=int, default=500)
    return parser


def main():
    args = build_parser().parse_args()
    if args.fusion not in ("baseline", "concat") or args.dr_sampling != "independent_uniform":
        raise ValueError("Use baseline or latent routing with independently sampled uniform DR")
    base.VERSION = VERSION
    base.PPO_CLASS_NAME = "intact_tracking.memory350_moe_training:OnlineMoEPPO"
    base.ResidualOnPolicyRunner = OnlineMoERunner
    base.configure_context_models = partial(configure_moe_models, num_experts=args.num_experts,
        center_rate=args.router_center_rate, max_switch_fraction=args.router_max_switch_fraction)
    base.audit_initial_models = audit_initial_models
    base.run(args)


if __name__ == "__main__":
    main()
