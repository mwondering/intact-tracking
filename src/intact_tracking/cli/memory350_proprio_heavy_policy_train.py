"""Matched heavy residual PPO with learned or zero latent, identical networks."""

from functools import partial
import math
import os

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.cli.memory350_proprio_native_policy_train import build_parser as proprio_parser
from intact_tracking import memory350_native_policy as native
from intact_tracking import memory350_heavy_policy as heavy
from intact_tracking.memory350_native_dr import native_dataset_identity
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_tracker_action_policy import (
    VERSION, LATENT_HISTORY_FRAMES, configure_tracker_action_models, audit_tracker_action_models,
)

_original_input_audit = base.audit_initial_models


def configure_models(*args, **kwargs):
    result = configure_tracker_action_models(*args, **kwargs)
    # PPO minibatches and Warp graph temporaries share the same GPU. Return
    # unused PyTorch activation storage before the next physics rollout.
    result['release_cuda_cache_after_update'] = True
    return result


def build_parser():
    parser = proprio_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == 'dr_profile':
            action.choices = (heavy.PROFILE,)
    parser.add_argument('--latent-input-mode', choices=('learned', 'zero'), required=True)
    parser.add_argument('--dr-aux-payload-targets', choices=('all','mass'), default='mass')
    parser.add_argument('--allow-dr-aux-change', action='store_true',
                        help='Resume the existing 108D head with mass-only supervision and coefficient x5')
    parser.set_defaults(training_ranks=4, num_envs=8192, dr_profile=heavy.PROFILE, dr_aux_coef=None,
                        wandb_group='144000-exp-heavy')
    return parser


def configure(args):
    if args.dr_aux_coef is None:
        args.dr_aux_coef = (.5 if args.dr_aux_payload_targets == 'mass' else .1) if args.latent_input_mode == 'learned' else 0.
    if args.allow_dr_aux_change and (not args.resume or not (args.reset_adaptive_sampling or args.resume_uniform_sampling)):
        raise ValueError('Auxiliary transition requires resume and reset-adaptive-sampling or resume-uniform-sampling')
    if args.latent_input_mode == 'zero' and args.dr_aux_coef != 0:
        raise ValueError('The heavy zero-latent baseline disables auxiliary supervision')
    if args.bounded_smoke:
        args.until_user_stop = False
        base.WandbLogger = partial(base.WandbLogger, mode='disabled')
    elif args.training_ranks != 4 or args.num_envs != 8192 or not args.until_user_stop:
        raise ValueError('Heavy comparison uses four ranks per arm, 8192 worlds per rank, no update cap')
    if (args.fusion != 'concat' or args.dr_sampling != 'independent_uniform'
            or args.training_terminations != 'original' or args.episode_steps != 500
            or args.motion_sampling not in ('adaptive', 'uniform') or args.adaptive_after_update != 0
            or args.endpoint_eval_protocol):
        raise ValueError('Heavy residual PPO requires the original 144000 motion/reward contract')
    if any(not math.isfinite(v) or v < 0 for v in (args.dr_aux_coef, args.dr_aux_motor_weight)):
        raise ValueError('DR auxiliary weights must be finite and nonnegative')
    context = torch.load(args.context_checkpoint, map_location='cpu', weights_only=False)
    heavy.validate_context(context, args.dr_profile)
    schema = heavy.heavy_aux_schema(context['dr_metric_schema'])
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    base.VERSION = VERSION
    base.PPO_CLASS_NAME = 'intact_tracking.memory350_tracker_action_policy:TrackerActionPPO'
    base.configure_context_models = partial(configure_models,
        dr_aux_schema=schema, dr_aux_coef=args.dr_aux_coef, dr_aux_motor_weight=args.dr_aux_motor_weight,
        latent_input_mode=args.latent_input_mode, retain_dr_aux_head=True,
        dr_aux_payload_com_enabled=False if args.dr_aux_payload_targets == 'mass' else None)
    base.validate_resume_models = heavy.validate_resume_models
    base.audit_initial_models = partial(audit_tracker_action_models, original_audit=_original_input_audit,
                                       training_dr_aux_coef=args.dr_aux_coef)
    base.TRACKER_SHA256, base.FULL_DATASET, base.EPISODE_STEPS = native.TRACKER_SHA256, native.FULL_DATASET, 500
    base.ALLOWED_DR_PROFILES = (heavy.PROFILE,)
    base.CONTEXT_SOURCE_DR_PROFILE = heavy.PROFILE
    base.TRAINING_START_PROFILE = 'original'
    base.resolve_dr_profile = heavy.resolve_profile
    base.validate_context_dr = heavy.validate_context
    base.configure_limb_dr = partial(heavy.configure_physics, rank=int(os.environ.get('RANK', '0')))
    base.audit_limb_dr = heavy.audit_physics
    base.configure_motion_sampling = native.configure_sampling
    base.LimbContextWrapper = partial(ProprioNativePolicyWrapper, latent_history_frames=LATENT_HISTORY_FRAMES,
        dr_aux_schema=schema if args.dr_aux_coef > 0 else None,
        dr_aux_allow_extra_parameters=True, latent_input_mode=args.latent_input_mode)
    base.ManagerBasedRlEnv = partial(heavy.environment_factory, base.ManagerBasedRlEnv)
    base.dataset_identity = partial(native_dataset_identity, native.TRACKER)


def main():
    args = build_parser().parse_args()
    configure(args)
    base.run(args)


if __name__ == '__main__':
    main()
