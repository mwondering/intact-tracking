"""Train one fully independent residual MLP on one complete static DR profile."""

from functools import partial
import os
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.cli.memory350_moe_policy_train import build_parser as parent_parser
from intact_tracking.fixed_dr_profiles import SAMPLING, configure_fixed_dr, audit_fixed_dr, load_profile
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.fixed_dr_specialists import VERSION, SpecialistRunner
from intact_tracking.memory350_moe_critic_action import (
    configure_critic_action_models, audit_critic_action_models, TrackerActionBaselineWrapper,
)


def build_parser():
    parser = parent_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "training_ranks":
            action.choices = (1,)
        elif action.dest == "fusion":
            action.choices = ("baseline",)
        elif action.dest == "dr_sampling":
            action.choices = (SAMPLING,)
    parser.set_defaults(training_ranks=1, dr_sampling=SAMPLING)
    parser.add_argument("--dr-bank", required=True)
    parser.add_argument("--specialist-id", type=int, choices=range(8), required=True)
    parser.add_argument("--specialist-eval-protocol")
    return parser


def main():
    args = build_parser().parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.endpoint_eval_protocol:
        raise ValueError("Each specialist must run independently, using its own single-GPU evaluator")
    selected, _, bank_sha = load_profile(args.dr_bank, args.specialist_id)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)["residual_policy"]
        old = saved["physics"]["fixed_dr"]
        if old["bank_sha256"] != bank_sha or old["id"] != selected["id"]:
            raise ValueError("Cannot change a specialist's DR profile on resume")
    base.VERSION = VERSION
    base.PPO_CLASS_NAME = "intact_tracking.memory350_moe_critic_action:TrackerActionMoEPPO"
    base.RslRlVecEnvWrapper = TrackerActionBaselineWrapper
    base.ResidualOnPolicyRunner = SpecialistRunner
    base.configure_context_models = partial(configure_critic_action_models, num_experts=16)
    base.audit_initial_models = audit_critic_action_models
    base.configure_limb_dr = partial(configure_fixed_dr, bank_path=args.dr_bank, profile_id=args.specialist_id)
    base.audit_limb_dr = audit_fixed_dr
    original_configuration = base.memory_checkpoint_configuration

    def configuration(source, train, metadata):
        metadata.update(experiment="eight completely independent fixed-DR specialists vs saved universal A",
                        context_protocol=None, frozen_inference="frozen tracker only; no context encoder",
                        context_normalization_frozen=None,
                        independent_policy_state="actor, critic, learned observation encoders, optimizers and critic normalization are private to this process")
        metadata["motion_sampling"].update(
            scope="motion/bin sampling only; all worlds keep the same complete static DR profile",
            statistics="private full-catalog motion/bin statistics for this single specialist")
        for path in (Path(__file__), Path(__file__).with_name("fixed_dr_specialist_eval.py"),
                     Path(__file__).parents[1] / "fixed_dr_profiles.py",
                     Path(__file__).parents[1] / "fixed_dr_specialists.py"):
            metadata["research_source_sha256"][str(path.resolve().relative_to(base.PROJECT_ROOT))] = file_sha256(path)
        return original_configuration(source, train, metadata)

    base.memory_checkpoint_configuration = configuration
    base.run(args)


if __name__ == "__main__":
    main()
