"""Matched fixed-start evaluation for the load-prototype Top-5 experiment."""

from functools import partial
import json
from pathlib import Path
import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.payload_prototype_moe import VERSION, PayloadMoEActor, PayloadMoEWrapper
from intact_tracking.payload_prototype_physics import configure_physics, audit_physics


def main():
    args = base.build_parser().parse_args()
    if not args.checkpoint:
        raise ValueError("Select a saved payload Top-5 checkpoint")
    metadata = torch.load(args.checkpoint,map_location="cpu",weights_only=False,mmap=True)["residual_policy"]
    if metadata["version"] != VERSION: raise ValueError("Wrong policy architecture/version")
    base.VERSION = VERSION
    base.EVAL_PROTOCOL = VERSION+"_evaluation"
    base.ACTOR_CLASS_NAME = "intact_tracking.payload_prototype_moe:PayloadMoEActor"
    base.LimbContextResidualActor = PayloadMoEActor
    base.LimbContextWrapper = PayloadMoEWrapper
    # Evaluation may contain fewer than 256 worlds. Explicit fixed masses or
    # continuous draws have the same nominal-background physical contract.
    base.configure_limb_dr = partial(configure_physics,anchor_fraction=0.)
    base.audit_limb_dr = audit_physics
    base.run(args)
    output = Path(args.output)
    report = json.loads(output.read_text())
    report.update(residual_output_bounded=False,residual_saturation_fraction=None,
                  route_mode=metadata["arguments"]["route_mode"],prototype_sha256=metadata["prototype_sha256"])
    tmp = output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(output)


if __name__ == "__main__":
    main()
