"""History5 policy evaluation in compiled nominal physics, with training sensor noise."""

from functools import partial

import torch

from intact_tracking.cli import memory350_policy_eval as base
from intact_tracking import memory350_history5_policy as variant
from intact_tracking.limb_context_protocol import PAYLOAD_EVENT
from intact_tracking.memory350_compressed_policy import ACTOR_CLASS, CompressedContextActor
from intact_tracking.memory350_nominal_rollout import NominalExcludedForcePulse
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper
from intact_tracking.rollout.online import _restore_nominal_physics


def configure_nominal(cfg, seed, *, profile, fixed_masses=None):
    if fixed_masses is not None:
        raise ValueError("Full nominal evaluation already sets every payload to zero")
    metadata = variant.configure_physics(cfg, seed, profile=profile)
    metadata.update(
        profile="compiled-nominal-physics-with-training-observation-noise",
        sampling="Every world restored to compiled nominal physics; all external force pulses disabled",
        nominal_population_fraction=1.0, nominal_population_rounding="all worlds",
        force_pulse_scope="none; every world is nominal",
        total_added_mass_range_kg=[0, 0],
        evaluation_override={
            "version": "history5_full_nominal_v1", "physical_nominal_fraction": 1.0,
            "observation_noise": "unchanged from training",
            "initial_sampler": "Training startup sampler, followed by full nominal restoration before any rollout",
        },
    )
    return metadata


def assert_nominal_clean(env, *, full=False):
    pulse = env.event_manager.get_term_cfg("push_robot").func
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    bias = env.scene["robot"].data.encoder_bias
    if (bool(pulse.active.any()) or bool(pulse.current_force_w.ne(0).any())
            or not bool(torch.isinf(pulse.time_to_next_pulse_s).all())):
        raise RuntimeError("Nominal evaluation received or re-enabled a force pulse")
    if bool(bias.ne(0).any()) or bool(payload.mass.ne(0).any()):
        raise RuntimeError("Nominal evaluation has nonzero encoder bias or payload")
    if not full:
        return
    errors = {}
    for raw_name in env.event_manager.domain_randomization_fields:
        name = str(raw_name)
        current = getattr(env.sim.model, name).clone()
        default = env.sim.get_default_field(name)
        if not torch.equal(current, default.expand_as(current)):
            raise RuntimeError(f"Nominal evaluation model field changed: {name}")
        errors[name] = 0.0
    if not errors or bool(payload.observe().ne(0).any()):
        raise RuntimeError("Nominal model verification is missing or actual payload is nonzero")
    action = env.action_manager.get_term("joint_pos")
    for name, expected in (("joint_offset", 0), ("alpha", 1)):
        value = getattr(action, name, None)
        if isinstance(value, torch.Tensor) and bool(value.ne(expected).any()):
            raise RuntimeError(f"Nominal action {name} changed")
    if int(getattr(action, "max_delay", 0)) != 0:
        raise RuntimeError("Nominal action delay is nonzero")
    audit = env.history5_nominal_audit
    audit["verified_model_field_max_abs_errors"] = errors
    audit["full_verifications"] += 1


def nominal_environment(factory, **kwargs):
    env = factory(**kwargs)
    try:
        ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        restoration = _restore_nominal_physics(env, ids)
        env.event_manager.get_term_cfg(PAYLOAD_EVENT).func.mass.zero_()
        term = env.event_manager.get_term_cfg("push_robot")
        if type(term.func).__name__ != "body_force_pulse":
            raise ValueError("Unexpected force pulse implementation")
        term.func = NominalExcludedForcePulse(term.func, ids)
        if not any(row is term for row in env.event_manager._mode_class_term_cfgs["step"]):
            raise RuntimeError("Nominal pulse exclusion must cover reset callbacks")
        env.history5_nominal_audit = {
            "count": env.num_envs, "num_envs": env.num_envs, "fraction": 1.0,
            "restoration": restoration, "pulse_exclusion": True,
            "full_verifications": 0, "warmup_steps_verified": 0, "query_steps_verified": 0,
        }
        assert_nominal_clean(env, full=True)
        return env
    except BaseException:
        env.close()
        raise


class NominalHistory5Wrapper(Memory350PolicyWrapper):
    def reset(self):
        result = super().reset()
        assert_nominal_clean(self.unwrapped, full=True)
        return result

    def step(self, actions):
        result = super().step(actions)
        assert_nominal_clean(self.unwrapped)
        key = "query_steps_verified" if self.encoding_enabled else "warmup_steps_verified"
        self.unwrapped.history5_nominal_audit[key] += 1
        return result

    @property
    def latent_metrics(self):
        # Called at both the query's initial and final states, after partial resets.
        assert_nominal_clean(self.unwrapped, full=True)
        return super().latent_metrics


def main():
    parser = base.build_parser()
    parser.description = __doc__
    args = parser.parse_args()
    if args.fixed_masses is not None:
        parser.error("Full nominal evaluation already fixes all payloads to zero")
    args.environment_case = "full_nominal_with_training_observation_noise"
    base.VERSION = variant.VERSION
    base.ACTOR_CLASS_NAME = ACTOR_CLASS
    base.LimbContextResidualActor = CompressedContextActor
    base.LimbContextWrapper = partial(NominalHistory5Wrapper, latent_history_frames=5)
    base.configure_limb_dr = configure_nominal
    base.audit_limb_dr = variant.audit_physics
    base.ManagerBasedRlEnv = partial(nominal_environment, base.ManagerBasedRlEnv)
    base.run(args)


if __name__ == "__main__":
    main()
