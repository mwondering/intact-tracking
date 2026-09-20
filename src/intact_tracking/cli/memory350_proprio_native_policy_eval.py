"""Matched warm tracking evaluation for native-DR proprio122 history-five PPO."""

from functools import partial
import hashlib

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking.cli.memory350_proprio_native_policy_train import configure_proprio_physics
from intact_tracking import memory350_native_policy as native
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_proprio_inputs import current_proprio
from intact_tracking.memory350_tracker_action_policy import (
    VERSION, LATENT_HISTORY_FRAMES, TrackerActionResidualActor,
)
from intact_tracking.rollout.online import _capture_privileged_dynamics_targets


def paired_environment(factory, **kwargs):
    env = native.environment_factory(factory, **kwargs)
    env.native_paired_query = False
    pulse = env.event_manager.get_term_cfg("push_robot").func.source
    original_reset = pulse.reset

    def reset(env_ids=None):
        # Query failures are removed from scoring. Keep their independent pulse
        # clocks advancing, so their resets cannot change the number/order of
        # random force draws for surviving worlds in either policy arm.
        if not env.native_paired_query:
            original_reset(env_ids=env_ids)

    pulse.reset = reset
    return env


class EvaluationWrapper(ProprioNativePolicyWrapper):
    def __init__(self, *args, **kwargs):
        self.query_forces = []
        self._sensor_step = None
        self._sensor_frame = None
        super().__init__(*args, **kwargs)

    def _remember_sensors(self, obs):
        self._sensor_step = int(self.unwrapped.common_step_counter)
        self._sensor_frame = current_proprio(obs)

    def get_observations(self):
        obs = super().get_observations()
        self._remember_sensors(obs)
        return obs

    def reset(self):
        self._sensor_step = None
        result = super().reset()
        self._remember_sensors(result[0])
        return result

    def _context_state(self):
        # The manual reset of inactive evaluation worlds invalidates the
        # manager's cache for every world. Preserve survivors' already observed
        # noisy frame instead of drawing a second noise sample before env.step.
        if self._sensor_step != int(self.unwrapped.common_step_counter):
            self._remember_sensors(self.unwrapped.observation_manager.compute())
        return self._sensor_frame

    def step(self, actions):
        result = super().step(actions)
        if self.unwrapped.native_paired_query:
            pulse = self.unwrapped.event_manager.get_term_cfg("push_robot").func
            self.query_forces.append(pulse.current_force_w.detach().clone())
        return result

    @property
    def evaluation_diagnostics(self):
        forces = torch.stack(self.query_forces).cpu().contiguous().numpy()
        # One policy can lose every world before the other's query ends. Keep
        # cumulative digests so the comparator can verify the actual shared
        # time interval instead of comparing hashes of unequal-length arrays.
        digest = hashlib.sha256()
        prefixes = []
        for step in forces:
            digest.update(step.tobytes())
            prefixes.append(digest.hexdigest())
        return {"query_force_steps": len(forces),
                "query_force_sha256": digest.hexdigest(),
                "query_force_prefix_sha256": prefixes,
                "failed_world_pulse_clocks_preserved": True,
                "cached_proprio_preserved_after_partial_reset": True}


class TrackerOnlyActor(TrackerActionResidualActor):
    """Original tracker output, with identical construction/caching to the PPO arm."""

    def forward(self, obs, **kwargs):
        return self._base_features_and_action(obs)[1]


def configure_evaluation_physics(cfg, seed, *, profile, fixed_masses=None):
    if fixed_masses is not None:
        raise ValueError("Native DR evaluation has no additional limb payload")
    result = configure_proprio_physics(cfg, seed, profile=profile)
    # Query starts are injected through _uniform_sampling in the common
    # evaluator. Do not let the source tracker's adaptive sampler bypass them.
    cfg.commands["motion"].sampling_mode = "uniform"
    cfg.commands["motion"].rewind.enabled = False
    result["evaluation_sampling"] = "fixed paired query starts; uniform frozen-tracker warmup"
    result["query_force_pairing"] = "preserve failed-world pulse clocks; matched per-step seeds and force hash"
    return result


def evaluation_world_metadata(env):
    # Audit all supervised physical parameters plus noisy-sensor bias and the
    # controller state drawn at the query reset. No old payload-only sampler.
    action = env.action_manager.get_term("joint_pos")
    values = [_capture_privileged_dynamics_targets(env).values,
              env.scene["robot"].data.encoder_bias]
    for name in ("delay", "alpha", "joint_offset", "boot_delay"):
        values.append(getattr(action, name).reshape(env.num_envs, -1).float())
    physical = torch.cat(values, -1).detach().cpu().contiguous().numpy()
    nominal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    nominal[env.native_policy_nominal_ids] = True
    env.native_paired_query = True
    return {
        "physics_world_fingerprints": [hashlib.sha256(row.tobytes()).hexdigest() for row in physical],
        "physics_fingerprint_dimensions": physical.shape[-1],
        "is_nominal": nominal.cpu().tolist(),
        "extra_payload": False,
        "added_limb_payload_kg": [[0., 0., 0., 0.] for _ in range(env.num_envs)],
    }


def configure():
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS["terrain_height_offset"] = native.FlatTerrainHeightOffset
    base.VERSION = VERSION
    base.EVAL_PROTOCOL = "memory350_proprio122_history5_tracker_action_native_dr_warm_v2"
    base.TRACKER = native.TRACKER
    base.TRACKER_SHA256 = native.TRACKER_SHA256
    base.TRACKER_DR = native.PROFILE
    base.FULL_DATASET = native.FULL_DATASET
    base.DR_PROFILES = (native.PROFILE,)
    base.EPISODE_STEPS = 500
    base.WARMUP_STEPS = 1000
    base.ACTOR_CLASS_NAME = "intact_tracking.memory350_tracker_action_policy:TrackerActionResidualActor"
    base.LimbContextResidualActor = TrackerActionResidualActor
    base.LimbContextWrapper = partial(EvaluationWrapper, latent_history_frames=LATENT_HISTORY_FRAMES)
    base.configure_limb_dr = configure_evaluation_physics
    base.audit_limb_dr = native.audit_physics
    base.resolve_dr_profile = native.resolve_profile
    base.ManagerBasedRlEnv = partial(paired_environment, base.ManagerBasedRlEnv)
    base.evaluation_world_metadata = evaluation_world_metadata


def main():
    configure()
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument("--frozen-tracker-only", action="store_true")
    parser.set_defaults(dr_profile=native.PROFILE, memory_start="warm", global_metrics=True,
                        policy_precision="fp32")
    args = parser.parse_args()
    if args.frozen_tracker_only:
        if args.checkpoint is None:
            parser.error("Tracker-only paired evaluation needs the residual checkpoint for identical setup")
        base.LimbContextResidualActor = TrackerOnlyActor
    base.run(args)


if __name__ == "__main__":
    main()
