"""Paired full-nominal or full-HDR evaluation of the two new baselines."""

from functools import partial

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.cli import memory350_proprio_native_policy_eval as paired
from intact_tracking.cli.memory350_proprio_heavy_policy_eval import evaluation_world_metadata
from intact_tracking.heavy_baseline_env import HeavyBaselineWrapper
from intact_tracking.heavy_rma_teacher import VERSION, RMATeacherActor
from intact_tracking.heavy_anyadapter import AnyAdapterActor
from intact_tracking.heavy_rma_student import RMAStudentActor, VERSION as STUDENT_VERSION
from intact_tracking.rma_student_env import RMAStudentWrapper
from intact_tracking import memory350_native_policy as native, memory350_heavy_policy as heavy


configure_physics = heavy.configure_evaluation_physics


def main():
    base.VERSION, base.EVAL_PROTOCOL = VERSION, 'heavy_baselines_paired_v1'
    base.TRACKER, base.TRACKER_SHA256 = native.TRACKER, native.TRACKER_SHA256
    base.TRACKER_DR, base.DR_PROFILES, base.FULL_DATASET = heavy.PROFILE, (heavy.PROFILE,), native.FULL_DATASET
    base.EPISODE_STEPS, base.WARMUP_STEPS = 500, 1000
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument('--physics', choices=('nominal', 'hdr', 'mixed'), required=True)
    parser.add_argument('--frozen-tracker-only', action='store_true')
    parser.set_defaults(dr_profile=heavy.PROFILE, memory_start='cold', global_metrics=True, policy_precision='fp32')
    args = parser.parse_args()
    if not args.checkpoint or args.latent_mode != 'correct':
        parser.error('Supply a baseline checkpoint and use correct conditioning')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    method = state['residual_policy']['method']
    actor = {'rma_teacher': RMATeacherActor, 'any2track': AnyAdapterActor, 'rma_student': RMAStudentActor}[method]
    if method == 'rma_student':
        base.VERSION = STUDENT_VERSION
    base.ACTOR_CLASS_NAME = f'{actor.__module__}:{actor.__name__}'
    if args.frozen_tracker_only:
        class TrackerOnly(actor):
            def forward(self, obs, **kwargs):
                return self._base_features_and_action(obs)[1]
        base.LimbContextResidualActor = TrackerOnly
    else:
        base.LimbContextResidualActor = actor
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS['terrain_height_offset'] = native.FlatTerrainHeightOffset
    base.LimbContextWrapper = RMAStudentWrapper if method == 'rma_student' else partial(HeavyBaselineWrapper, method=method)
    base.configure_limb_dr = partial(configure_physics, physics=args.physics)
    base.audit_limb_dr, base.resolve_dr_profile = heavy.audit_physics, heavy.resolve_profile
    base.ManagerBasedRlEnv = partial(paired.paired_environment, base.ManagerBasedRlEnv,
                                     physics_factory=heavy.environment_factory)
    base.evaluation_world_metadata = evaluation_world_metadata
    base.run(args)


if __name__ == '__main__':
    main()
