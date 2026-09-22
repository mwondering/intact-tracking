"""Paired evaluation with the heavy training physics and saved latent mode."""

from functools import partial
import torch

from intact_tracking.cli import memory350_proprio_native_policy_eval as proprio
from intact_tracking import memory350_heavy_policy as heavy


configure_evaluation_physics = heavy.configure_evaluation_physics


def evaluation_world_metadata(env):
    result = proprio.evaluation_world_metadata(env)
    payload = env.heavy_policy_payload
    result.update(extra_payload=True, added_limb_payload_kg=payload.mass.cpu().tolist(),
                  payload_com_offsets_m=payload.offsets.cpu().tolist(), heavy_payload_audit=payload.audit())
    return result


def main():
    proprio.configure()
    base = proprio.base
    base.EVAL_PROTOCOL = 'memory350_proprio122_history5_heavy_dr_warm_v1'
    base.TRACKER_DR = heavy.PROFILE
    base.DR_PROFILES = (heavy.PROFILE,)
    base.configure_limb_dr = configure_evaluation_physics
    base.audit_limb_dr = heavy.audit_physics
    base.resolve_dr_profile = heavy.resolve_profile
    # configure() already installed the paired factory; unwrap it to avoid
    # restoring physics twice or wrapping the force reset callback twice.
    original_factory = base.ManagerBasedRlEnv.args[0]
    base.ManagerBasedRlEnv = partial(proprio.paired_environment, original_factory,
                                    physics_factory=heavy.environment_factory)
    base.evaluation_world_metadata = evaluation_world_metadata
    parser = base.build_parser()
    parser.add_argument('--physics', choices=('nominal', 'hdr', 'mixed'), default='mixed')
    parser.add_argument('--warmup-policy', choices=('frozen-tracker', 'evaluated-policy'),
                        default='frozen-tracker', help='Policy generating the warm context history')
    parser.add_argument('--frozen-tracker-only', action='store_true')
    parser.set_defaults(dr_profile=heavy.PROFILE, memory_start='warm', global_metrics=True,
                        policy_precision='fp32')
    args = parser.parse_args()
    base.configure_limb_dr = partial(configure_evaluation_physics, physics=args.physics)
    if args.checkpoint is None:
        parser.error('A heavy residual checkpoint is required')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    mode = state['residual_policy']['arguments']['latent_input_mode']
    base.LimbContextWrapper = partial(proprio.EvaluationWrapper,
        latent_history_frames=proprio.LATENT_HISTORY_FRAMES, latent_input_mode=mode)
    if args.frozen_tracker_only:
        base.LimbContextResidualActor = proprio.TrackerOnlyActor
    base.run(args)


if __name__ == '__main__':
    main()
