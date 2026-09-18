"""Fixed-start evaluation and full latent-history interventions for temporal PPO."""

from functools import partial
import json
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.limb_context_dr import LOAD_ONLY, TRACKER_DR
from intact_tracking.memory350_token_env import evaluation_wrapper
from intact_tracking.memory350_token_policy import ACTOR_CLASS, VERSION, TemporalTokenActor
from intact_tracking.memory350_token_training import audit_physics


def main():
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument("--anchor-fraction", type=float, default=0.0,
                        help="Default evaluation uses continuous uniform loads, or --fixed-masses")
    args = parser.parse_args()
    metadata = None
    if args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
        if metadata["version"] != VERSION:
            raise ValueError("Expected a temporal Transformer/original-MLP checkpoint")
        profile = metadata["dr_profile"]
        if args.dr_profile is not None and args.dr_profile != profile:
            raise ValueError("Evaluation cannot silently change the training DR profile")
        args.dr_profile = profile
    else:
        args.dr_profile = args.dr_profile or LOAD_ONLY
    if args.dr_profile == LOAD_ONLY:
        from intact_tracking.payload_prototype_physics import configure_physics
        base.configure_limb_dr = partial(configure_physics, anchor_fraction=args.anchor_fraction)
    elif args.dr_profile == TRACKER_DR:
        from intact_tracking.residual_uniform_protocol import configure_physics
        base.configure_limb_dr = configure_physics
    else:
        raise ValueError(args.dr_profile)
    base.VERSION = VERSION
    base.EVAL_PROTOCOL = VERSION + "_evaluation"
    base.ACTOR_CLASS_NAME, base.LimbContextResidualActor = ACTOR_CLASS, TemporalTokenActor
    base.LimbContextWrapper = evaluation_wrapper
    base.audit_limb_dr = audit_physics
    base.run(args)
    output = Path(args.output)
    result = json.loads(output.read_text())
    result.update(architecture=metadata["architecture"] if metadata else "frozen_tracker",
                  residual_output_bounded=False, residual_saturation_fraction=None,
                  latent_intervention_scope="entire five-frame latent stream and current code",
                  critic_used_for_actions=False)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
