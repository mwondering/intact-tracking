"""A shared bank of 256 static DR profiles, one per four-limb load combination."""

from __future__ import annotations

import copy
import hashlib
import inspect

import torch
from mjlab.envs.mdp.dr import body, geom, joint
from mjlab.managers.event_manager import RecomputeLevel

from intact_tracking.environment.mdp.randomizations import motor_params_implicit
from intact_tracking.limb_context_protocol import LIMBS, PAYLOAD_EVENT, UniformLimbPayload

UNIFORM = "independent_uniform"
GRID256 = "grid256_shared"
SAMPLING_MODES = (UNIFORM, GRID256)
LEVELS = (0.0, 1.0, 2.0, 4.0)
GROUPS = 256
STATIC_EVENTS = {
    "base_com": (body.body_com_offset, ("body_ipos",)),
    "base_mass": (body.body_mass, ("body_mass",)),
    "encoder_bias": (joint.encoder_bias, ()),
    "foot_friction": (geom.geom_friction, ("geom_friction",)),
    "motor_params_implicit": (motor_params_implicit, (
        "actuator_gainprm", "actuator_biasprm", "dof_armature", "dof_frictionloss")),
}
PHYSICS_FIELDS = (
    "body_mass", "body_ipos", "body_inertia", "body_iquat", "geom_friction",
    "dof_armature", "dof_damping", "dof_frictionloss", "actuator_gainprm",
    "actuator_biasprm", "jnt_pos",
)


def group_ids(num_envs, *, device="cpu"):
    if num_envs < GROUPS or num_envs % GROUPS:
        raise ValueError("Grid256 requires a positive multiple of 256 worlds per rank")
    return torch.arange(num_envs, device=device) % GROUPS


def mass_table():
    values = torch.tensor(LEVELS, dtype=torch.float32)
    return torch.cartesian_prod(values, values, values, values)


def sample_grid_masses(num_envs):
    return mass_table()[group_ids(num_envs)]


def _repeat_prototypes(value, ids):
    if value.shape[0] != len(ids):
        raise ValueError("Expected a per-world tensor for grouped static DR")
    # Gather before overwriting; all replicas draw from the same 256 prototypes.
    value.copy_(value[:GROUPS].clone()[ids])


def world_parameter_view(value, num_envs, default_shape, *, expanded=False):
    """Read shared MuJoCo fields without allocating or mutating simulator storage."""
    shape = tuple(default_shape)
    if tuple(value.shape) == (num_envs, *shape):
        return value
    if not expanded and tuple(value.shape) == shape:
        return value.reshape(1, *shape).expand(num_envs, *shape)
    if not expanded and tuple(value.shape) == (1, *shape):
        return value.expand(num_envs, *shape)
    raise ValueError(f"Unrecognized physical field storage: {tuple(value.shape)}, default {shape}")


class GroupedStaticRandomization:
    # The event manager expands these fields and recomputes physical constants
    # once after all startup events (including composite limb payloads) have fired.
    model_fields = tuple(dict.fromkeys(
        field for function, _ in STATIC_EVENTS.values()
        for field in getattr(function, "model_fields", ())))
    recompute = RecomputeLevel.set_const

    def __init__(self, cfg, env):
        self.env = env
        self.ids = group_ids(env.num_envs, device=env.device)
        self.event = cfg.params["grouped_event"]
        self.seed = int(cfg.params["grouped_seed"])
        function, self.fields = STATIC_EVENTS[self.event]
        self.params = {key: value for key, value in cfg.params.items()
                       if key not in ("grouped_event", "grouped_seed")}
        missing = tuple(name for name in self.model_fields if name not in env.sim.expanded_fields)
        if missing:
            env.sim.expand_model_fields(missing)
        inner = copy.deepcopy(cfg)
        inner.func, inner.params = function, self.params
        self.function = function(inner, env) if inspect.isclass(function) else function
        if self.event == "motor_params_implicit":
            if self.function.kp_ctrl_ids.numel() or self.function.kd_ctrl_ids.numel():
                raise ValueError("Grid256 requires static motor DR; reset-randomized gains need a separate protocol")

    def __call__(self, env, env_ids, **_):
        if env_ids is not None and not isinstance(env_ids, slice):
            if not torch.equal(env_ids.to(self.ids.device), torch.arange(env.num_envs, device=self.ids.device)):
                raise ValueError("Grouped physics is initialized for all worlds at startup only")
        device = torch.device(env.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Rank-independent DR; leave motion/noise/action RNG streams untouched.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(self.seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(self.seed)
            self.function(env, torch.arange(GROUPS, device=env.device), **self.params)
        for field in self.fields:
            _repeat_prototypes(getattr(env.sim.model, field), self.ids)
        if self.event == "encoder_bias":
            _repeat_prototypes(env.scene["robot"].data.encoder_bias, self.ids)
        if self.event == "motor_params_implicit":
            for name in ("_obs_kp_scale", "_obs_kd_scale", "_obs_arm_scale", "_obs_frictionloss"):
                _repeat_prototypes(getattr(self.function, name), self.ids)

    def reset(self, env_ids=None):
        # The original armature-only motor term has a no-op reset. Preserve it;
        # neither physics nor the persistent Memory350 environment changes here.
        reset = getattr(self.function, "reset", None)
        if callable(reset):
            reset(env_ids=env_ids)

    def observe(self, **kwargs):
        observe = getattr(self.function, "observe", None)
        if not callable(observe):
            raise NotImplementedError("The original static event has no privileged observation")
        return observe(**kwargs)


class Grid256LimbPayload(UniformLimbPayload):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mass = sample_grid_masses(env.num_envs).to(env.device)


def configure_grid256_dr(env_cfg, common_seed):
    from intact_tracking.limb_context_dr import TRACKER_DR, _event_contract, configure_limb_dr

    metadata = configure_limb_dr(env_cfg, common_seed, profile=TRACKER_DR)
    for index, (name, (function, _)) in enumerate(STATIC_EVENTS.items()):
        term = env_cfg.events[name]
        if term.mode != "startup" or term.func is not function:
            raise ValueError(f"Unsupported original static DR event: {name}")
        term.func = GroupedStaticRandomization
        term.params.update(grouped_event=name, grouped_seed=int(common_seed) + 300001 + index * 1009)
    env_cfg.events[PAYLOAD_EVENT].func = Grid256LimbPayload
    metadata.update(
        profile="frozen-tracker-dr-plus-shared-grid256-limb-payload",
        sampling="All {0,1,2,4}^4 loads; full static DR repeated per profile across replicas and ranks",
        grouped_dr={
            "version": "shared_static_grid256_v1", "mode": GRID256,
            "profiles": GROUPS, "limb_order": list(LIMBS), "levels_kg": list(LEVELS),
            "common_seed": int(common_seed), "layout": "profile_id = world_id % 256",
            "static_parameters_shared_across_ranks": True,
            "motion_phase_resets_noise_pushes_and_memory": "independent per world",
            "masses_by_profile_kg": mass_table().tolist(),
        },
        effective_events={name: _event_contract(term) for name, term in env_cfg.events.items()},
    )
    return metadata


def audit_grid256_dr(env, configuration):
    from intact_tracking.limb_context_dr import _event_contract, _resolved_event_contract

    active = {name for names in env.event_manager.active_terms.values() for name in names}
    if active != set(configuration["effective_events"]):
        raise RuntimeError("Grid256 event set changed")
    for name, expected in configuration["effective_events"].items():
        actual = _event_contract(env.event_manager.get_term_cfg(name))
        if actual != _resolved_event_contract(expected, env.scene):
            raise RuntimeError(f"Grid256 event configuration changed: {name}")
    corruption = {name: bool(group.enable_corruption) for name, group in env.cfg.observations.items()}
    if corruption != configuration["observation_corruption"]:
        raise RuntimeError("Grid256 observation corruption changed")
    ids = group_ids(env.num_envs, device=env.device)
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    torch.testing.assert_close(payload.mass, sample_grid_masses(env.num_envs).to(env.device), atol=0, rtol=0)
    result = payload.audit()
    tensors = {name: world_parameter_view(getattr(env.sim.model, name), env.num_envs,
                   env.sim.get_default_field(name).shape, expanded=name in env.sim.expanded_fields)
               for name in PHYSICS_FIELDS}
    tensors["encoder_bias"] = env.scene["robot"].data.encoder_bias
    prototypes, errors, field_digests = [], {}, {}
    for name, value in tensors.items():
        if value.shape[0] != env.num_envs:
            raise RuntimeError(f"Grid256 audit requires per-world model view for {name}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite grouped physics: {name}")
        reference = value[:GROUPS]
        error = float((value - reference[ids]).abs().max()) if value.numel() else 0.0
        if error != 0.0:
            raise RuntimeError(f"Replicas have different {name}: {error}")
        errors[name] = error
        array = reference.detach().cpu().contiguous().numpy()
        field_digests[name] = hashlib.sha256(array.tobytes()).hexdigest()
        prototypes.append(reference.reshape(GROUPS, -1).float())
    prototype = torch.cat(prototypes, dim=1).cpu().contiguous().numpy()
    result.update(
        grouped_dr={
            "profiles": GROUPS, "replicas_per_profile_per_rank": env.num_envs // GROUPS,
            "counts_per_profile": torch.bincount(ids, minlength=GROUPS).cpu().tolist(),
            "static_replica_max_errors": errors,
            "prototype_fields_sha256": field_digests,
            "static_bank_sha256": hashlib.sha256(prototype.tobytes()).hexdigest(),
            "profile_fingerprints": [hashlib.sha256(row.tobytes()).hexdigest() for row in prototype],
        },
        sampled_mass_sha256=hashlib.sha256(payload.mass.cpu().numpy().tobytes()).hexdigest(),
        per_limb_min_kg=payload.mass.amin(0).tolist(), per_limb_max_kg=payload.mass.amax(0).tolist(),
        original_events=copy.deepcopy(configuration["original_events"]),
        observation_corruption=corruption,
    )
    return result


def audit_grouped_ranks(ranks):
    banks = [rank["physics"]["grouped_dr"] for rank in ranks]
    if len({bank["static_bank_sha256"] for bank in banks}) != 1:
        raise RuntimeError("Ranks sampled different Grid256 static DR banks")
    if any(bank["counts_per_profile"] != banks[0]["counts_per_profile"] for bank in banks):
        raise RuntimeError("Ranks have unequal Grid256 replication counts")
    return {"passed": True, "profiles": GROUPS,
            "replicas_per_profile_per_rank": banks[0]["replicas_per_profile_per_rank"],
            "replicas_per_profile_all_ranks": sum(bank["replicas_per_profile_per_rank"] for bank in banks),
            "static_bank_sha256": banks[0]["static_bank_sha256"]}
