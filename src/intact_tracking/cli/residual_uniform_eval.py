"""Evaluate the saved uniform-motion, 10-Nm-wrist residual physics protocol."""

from functools import partial
import json
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.residual_uniform_protocol import (
    VERSION, EVAL_VERSION, ACTOR_CLASS, UnboundedResidualActor,
    configure_physics, audit_physics, physics_contract,
)


def main():
    parser = base.build_parser()
    parser.add_argument("--dr-bank")
    parser.add_argument("--dr-id", type=int, choices=range(8))
    args = parser.parse_args()
    if not args.checkpoint:
        raise ValueError("Use a saved checkpoint to select the physics protocol")
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
    if metadata["version"] != VERSION or metadata["physics"].get("residual_physics_contract") != physics_contract():
        raise ValueError("This evaluator requires the matching new physics checkpoint")
    if bool(args.dr_bank) != (args.dr_id is not None):
        raise ValueError("Provide both --dr-bank and --dr-id")
    saved_fixed = metadata["physics"].get("fixed_dr")
    if saved_fixed and not args.dr_bank:
        args.dr_bank, args.dr_id = saved_fixed["bank"], saved_fixed["id"]
    if saved_fixed and (args.dr_id != saved_fixed["id"] or file_sha256(args.dr_bank) != saved_fixed["bank_sha256"]):
        raise ValueError("Evaluate a specialist on its matching complete fixed DR")
    base.VERSION = VERSION
    base.EVAL_PROTOCOL = EVAL_VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = UnboundedResidualActor
    base.configure_limb_dr = partial(configure_physics, bank_path=args.dr_bank, profile_id=args.dr_id)
    base.audit_limb_dr = audit_physics
    base.run(args)
    # The reused historical evaluator records saturation relative to a scalar
    # cap. There is no such cap here; mark it inapplicable, not a measured zero.
    output = Path(args.output)
    result = json.loads(output.read_text())
    result.update(residual_output_bounded=False, residual_saturation_fraction=None)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
