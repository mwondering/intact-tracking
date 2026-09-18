"""Paired evaluation of direct tracker FiLM or the original frozen tracker."""

from functools import partial
import json
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.limb_context_dr import TRACKER_DR
from intact_tracking.memory350_tracker_film import VERSION, ACTOR_CLASS, TrackerFiLMActor
from intact_tracking.residual_uniform_protocol import configure_physics, audit_physics, physics_contract


def main():
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(dr_profile=TRACKER_DR)
    parser.add_argument("--dr-bank")
    parser.add_argument("--dr-id", type=int, choices=range(8))
    args = parser.parse_args()
    if bool(args.dr_bank) != (args.dr_id is not None):
        raise ValueError("Provide both --dr-bank and --dr-id")
    metadata = None
    if args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
        if metadata["version"] != VERSION or metadata["physics"].get("residual_physics_contract") != physics_contract():
            raise ValueError("Checkpoint belongs to a different adaptation/physics experiment")
    base.VERSION = VERSION
    base.EVAL_PROTOCOL = VERSION + "_evaluation"
    base.ACTOR_CLASS_NAME, base.LimbContextResidualActor = ACTOR_CLASS, TrackerFiLMActor
    base.configure_limb_dr = partial(configure_physics, bank_path=args.dr_bank, profile_id=args.dr_id)
    base.audit_limb_dr = audit_physics
    base.run(args)
    output = Path(args.output)
    result = json.loads(output.read_text())
    result.update(actor_conditioning=metadata["actor_conditioning"] if metadata else "unmodified_tracker",
                  residual_network=False, residual_saturation_fraction=None,
                  critic_used_for_actions=False)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
