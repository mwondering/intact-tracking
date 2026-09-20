"""Residual PPO with the noisy-proprio122 encoder; an explicit new checkpoint is required."""

from functools import partial
import math

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.cli.memory350_native_policy_train import build_parser as native_parser
from intact_tracking import memory350_native_policy as native
from intact_tracking.memory350_native_dr import native_dataset_identity
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper, validate_proprio_context
from intact_tracking.memory350_proprio_inputs import INPUT_CONTRACT
from intact_tracking.memory350_tracker_action_policy import (
    VERSION, LATENT_HISTORY_FRAMES, configure_tracker_action_models, audit_tracker_action_models,
)

_original_input_audit = base.audit_initial_models


def configure_proprio_physics(*args, **kwargs):
    physics = native.configure_physics(*args, **kwargs)
    physics["encoder_action_input"] = INPUT_CONTRACT["action_source"]
    physics["encoder_state_input"] = INPUT_CONTRACT["state_sampling"]
    return physics


def build_parser():
    parser = native_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "context_checkpoint":
            action.required = True
            action.default = None
    parser.add_argument("--dr-aux-coef", type=float, default=0.05,
                        help="Shared actor DR loss coefficient; zero disables the head and labels")
    parser.add_argument("--dr-aux-motor-weight", type=float, default=0.0,
                        help="Weight per Kp/Kd/armature group; zero excludes all three (default); COM xyz and friction have weight 1")
    parser.set_defaults(wandb_name="144000_exp-residual-proprio122-history5-tracker-action-auxdr",
                        latent_history_frames=LATENT_HISTORY_FRAMES, tracker_action_observed=True)
    return parser


def configure(args):
    if args.bounded_smoke:
        args.until_user_stop = False
        base.WandbLogger = partial(base.WandbLogger, mode="disabled")
    elif args.training_ranks != 8 or args.num_envs != 8192 or not args.until_user_stop:
        raise ValueError("Formal PPO requires eight ranks, 8192 environments each and no update cap")
    if (args.fusion != "concat" or args.dr_sampling != "independent_uniform"
            or args.training_terminations != "original" or args.episode_steps != 500
            or args.motion_sampling != "adaptive" or args.adaptive_after_update != 0
            or args.endpoint_eval_protocol):
        raise ValueError("Native flat residual PPO contract changed")
    if any(not math.isfinite(value) or value < 0 for value in (args.dr_aux_coef, args.dr_aux_motor_weight)):
        raise ValueError("DR auxiliary weights must be finite and nonnegative")
    context_state = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)
    validate_proprio_context(context_state, args.dr_profile)
    schema = context_state.get("dr_metric_schema") if args.dr_aux_coef > 0 else None
    if args.dr_aux_coef > 0:
        from intact_tracking.residual_dr_aux import dr_aux_layout
        if schema is None:
            raise ValueError("DR auxiliary supervision requires dr_metric_schema in the context checkpoint")
        dr_aux_layout(schema, args.dr_aux_motor_weight)
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS["terrain_height_offset"] = native.FlatTerrainHeightOffset
    base.VERSION = VERSION
    base.PPO_CLASS_NAME = "intact_tracking.memory350_tracker_action_policy:TrackerActionPPO"
    base.configure_context_models = partial(configure_tracker_action_models,
                                            dr_aux_schema=schema, dr_aux_coef=args.dr_aux_coef,
                                            dr_aux_motor_weight=args.dr_aux_motor_weight)
    base.audit_initial_models = partial(audit_tracker_action_models, original_audit=_original_input_audit)
    base.TRACKER_SHA256 = native.TRACKER_SHA256
    base.FULL_DATASET = native.FULL_DATASET
    base.EPISODE_STEPS = 500
    base.ALLOWED_DR_PROFILES = (native.PROFILE,)
    base.CONTEXT_SOURCE_DR_PROFILE = native.PROFILE
    base.TRAINING_START_PROFILE = "original"
    base.resolve_dr_profile = native.resolve_profile
    base.validate_context_dr = validate_proprio_context
    base.configure_limb_dr = configure_proprio_physics
    base.audit_limb_dr = native.audit_physics
    base.configure_motion_sampling = native.configure_sampling
    base.LimbContextWrapper = partial(ProprioNativePolicyWrapper, latent_history_frames=LATENT_HISTORY_FRAMES,
                                      dr_aux_schema=schema)
    base.ManagerBasedRlEnv = partial(native.environment_factory, base.ManagerBasedRlEnv)
    base.dataset_identity = partial(native_dataset_identity, native.TRACKER)


def main():
    args = build_parser().parse_args()
    configure(args)
    base.run(args)


if __name__ == "__main__":
    main()
