"""Evaluate a fixed specialist or the existing universal MLP on the same full DR."""

from functools import partial

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.fixed_dr_profiles import configure_fixed_dr, audit_fixed_dr
from intact_tracking.fixed_dr_specialists import VERSION, EVAL_VERSION
from intact_tracking.memory350_moe_policy import ACTOR_CLASS, HardMoEActor
from intact_tracking.memory350_moe_critic_action import VERSION as UNIVERSAL_VERSION


def main():
    parser = base.build_parser()
    parser.add_argument("--dr-bank", required=True)
    parser.add_argument("--dr-id", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    if not args.checkpoint or args.fixed_masses is not None or args.memory_start != "cold":
        raise ValueError("Use a saved no-latent policy with the complete fixed profile and cold evaluation")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
    if saved["fusion"] != "baseline" or saved["version"] not in (VERSION, UNIVERSAL_VERSION):
        raise ValueError("Expected one of the matched universal/specialist MLP checkpoints")
    base.VERSION = saved["version"]
    base.EVAL_PROTOCOL = EVAL_VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = HardMoEActor
    base.configure_limb_dr = partial(configure_fixed_dr, bank_path=args.dr_bank, profile_id=args.dr_id)
    base.audit_limb_dr = audit_fixed_dr
    base.run(args)


if __name__ == "__main__":
    main()
