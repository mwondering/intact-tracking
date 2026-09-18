"""Independent nominal/continuous mixtures for the fixed tracker DR parameters."""

from __future__ import annotations

import math

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields


EVENT = "independent_nominal_dr"
VERSION = "independent_coordinate_nominal_mixture_v1"


def validate_probability(probability):
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("DR nominal probability must be finite and in [0, 1]")
    return probability


def nominal_mask(shape, *, seed, probability, device):
    """One independent decision per coordinate, without consuming global RNG."""
    probability = validate_probability(probability)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.rand(shape, generator=generator) < probability).to(device)


@requires_model_fields(
    "body_mass", "body_ipos", "geom_friction", "dof_armature",
    recompute=RecomputeLevel.set_const,
)
class IndependentNominalDR:
    """Last startup event: replace selected original draws with compiled defaults.

    The existing payload event rebuilds its composite inertia after masking its
    four mass inputs. EventManager recomputes derived MuJoCo constants after this
    event. Observation noise, pushes, motion sampling and reset poses are outside
    this fixed-parameter distribution.
    """

    def __init__(self, cfg, env):
        self.env = env
        self.seed = int(cfg.params["seed"])
        self.probability = validate_probability(cfg.params["probability"])
        self.masks = {}
        self.original = {}
        self.applied = False

    def _mix(self, name, actual, nominal, offset):
        if name not in self.masks:
            self.masks[name] = nominal_mask(
                actual.shape, seed=self.seed + offset, probability=self.probability,
                device=actual.device,
            )
            self.original[name] = actual.clone()
        result = torch.where(self.masks[name], nominal, actual)
        torch.testing.assert_close(result[~self.masks[name]], actual[~self.masks[name]], rtol=0, atol=0)
        return result

    def __call__(self, env, env_ids, **_):
        from intact_tracking.limb_context_protocol import PAYLOAD_EVENT
        from intact_tracking.rollout.online import _entity_indices_and_names

        if env_ids is not None:
            ids = torch.as_tensor(env_ids, device=env.device)
            if ids.numel() != env.num_envs or not torch.equal(ids, torch.arange(env.num_envs, device=env.device)):
                raise ValueError("IndependentNominalDR is a whole-population startup event")
        if self.applied:
            return
        events = env.event_manager
        payload = events.get_term_cfg(PAYLOAD_EVENT).func
        payload.mass.copy_(self._mix("limb_payload", payload.mass, 0., 101))
        payload(env, None)

        for event, field, name, offset in (
            ("base_com", "body_ipos", "com_xyz", 211),
            ("base_mass", "body_mass", "torso_mass", 307),
        ):
            cfg = events.get_term_cfg(event)
            body_ids, body_names = _entity_indices_and_names(env, cfg.params["asset_cfg"], "body")
            if tuple(body_names) != ("torso_link",):
                raise ValueError("Independent nominal mixture requires torso-only background mass/COM DR")
            model_field = getattr(env.sim.model, field)
            default = env.sim.get_default_field(field).to(model_field.device)
            selected = model_field[:, body_ids]
            model_field[:, body_ids] = self._mix(name, selected, default[body_ids], offset)

        friction_cfg = events.get_term_cfg("foot_friction")
        if not friction_cfg.params.get("shared_random"):
            raise ValueError("Expected one shared friction parameter for all foot geometries")
        geom_ids, _ = _entity_indices_and_names(env, friction_cfg.params["asset_cfg"], "geom")
        friction = env.sim.model.geom_friction
        default = env.sim.get_default_field("geom_friction").to(friction.device)[geom_ids, 0]
        torch.testing.assert_close(default, default[0].expand_as(default), rtol=0, atol=0)
        sampled = friction[:, geom_ids, 0]
        torch.testing.assert_close(sampled, sampled[:, :1].expand_as(sampled), rtol=0, atol=0)
        mixed = self._mix("foot_friction", sampled[:, :1], default[0], 401)
        friction[:, geom_ids, 0] = mixed.expand_as(sampled)

        motor = events.get_term_cfg("motor_params_implicit").func
        if motor.kp_ctrl_ids.numel() or motor.kd_ctrl_ids.numel() or motor.fric_dof_ids.numel():
            raise ValueError("This tracker mixture expects only armature motor randomization")
        armature = env.sim.model.dof_armature
        ids = motor.arm_dof_ids
        armature[:, ids] = self._mix("armature", armature[:, ids], motor.arm_def, 503)
        motor._obs_arm_scale.copy_(torch.where(self.masks["armature"], 1., motor._obs_arm_scale))

        bias = env.scene["robot"].data.encoder_bias
        bias.copy_(self._mix("encoder_bias", bias, 0., 601))
        self.applied = True

    def reset(self, env_ids=None):
        # Startup parameters, including the nominal masks, persist across episodes.
        pass

    def audit(self):
        if not self.applied:
            raise RuntimeError("Independent nominal DR event was not applied")
        return {
            "version": VERSION, "probability": self.probability, "seed": self.seed,
            "scope": "Independent per scalar fixed DR parameter; all foot geoms share one friction decision",
            "nominal_mask_fractions": {
                name: mask.float().reshape(self.env.num_envs, -1).mean(0).tolist()
                for name, mask in self.masks.items()
            },
            "reset_policy": "fixed across episodes",
            "unmodified": ["force pulses", "per-step observation noise", "motion and reset sampling"],
        }
