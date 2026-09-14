"""One reproducible, complete static tracker DR profile per specialist process."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch
from mjlab.managers.event_manager import RecomputeLevel

from intact_tracking.limb_context_dr import (
    TRACKER_DR, _event_contract, _resolved_event_contract, configure_limb_dr,
)
from intact_tracking.limb_context_grouped_dr import STATIC_EVENTS, PHYSICS_FIELDS, world_parameter_view
from intact_tracking.limb_context_protocol import LIMBS, PAYLOAD_EVENT

BANK_VERSION = "eight_fixed_tracker_dr_profiles_v1"
SAMPLING = "fixed_single_profile"
MOTOR_OBSERVATIONS = ("_obs_kp_scale", "_obs_kd_scale", "_obs_arm_scale", "_obs_frictionloss")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def representative_bank(seed=20260914):
    cases = (
        ("all_0kg", [0, 0, 0, 0]), ("all_1kg", [1, 1, 1, 1]),
        ("all_2kg", [2, 2, 2, 2]), ("all_4kg", [4, 4, 4, 4]),
        ("arms_4kg", [4, 4, 0, 0]), ("shins_4kg", [0, 0, 4, 4]),
        ("left_4kg", [4, 0, 4, 0]), ("right_4kg", [0, 4, 0, 4]),
    )
    return {
        "version": BANK_VERSION, "limb_order": list(LIMBS), "selection_seed": seed,
        "selection": "Eight preselected payload patterns; remaining static DR independently sampled once in the original ranges",
        "zero_load_is_nominal": False,
        "stochastic_terms": "Original independent observation noise, motion resets and torso force pulses are retained",
        "profiles": [{"id": index, "name": name, "masses_kg": masses,
                      "static_seed": seed + 1009 * index}
                     for index, (name, masses) in enumerate(cases)],
    }


def load_profile(path, profile_id):
    path = Path(path).resolve()
    bank = json.loads(path.read_text())
    if (bank["version"] != BANK_VERSION or bank["limb_order"] != list(LIMBS)
            or [row["id"] for row in bank["profiles"]] != list(range(8))):
        raise ValueError("Invalid fixed DR profile bank")
    if profile_id not in range(8):
        raise ValueError("Specialist ID must be in [0,7]")
    profile = copy.deepcopy(bank["profiles"][profile_id])
    masses = torch.as_tensor(profile["masses_kg"], dtype=torch.float32)
    if masses.shape != (4,) or not torch.isfinite(masses).all() or not ((0 <= masses) & (masses <= 4)).all():
        raise ValueError("Invalid fixed limb payload")
    return profile, str(path), file_sha256(path)


class FixedStaticRandomization:
    model_fields = tuple(dict.fromkeys(
        field for function, _ in STATIC_EVENTS.values()
        for field in getattr(function, "model_fields", ())))
    recompute = RecomputeLevel.set_const

    def __init__(self, cfg, env):
        self.event = cfg.params["fixed_event"]
        self.seed = int(cfg.params["fixed_seed"])
        function, self.fields = STATIC_EVENTS[self.event]
        self.params = {key: value for key, value in cfg.params.items()
                       if key not in ("fixed_event", "fixed_seed")}
        missing = tuple(name for name in self.model_fields if name not in env.sim.expanded_fields)
        if missing:
            env.sim.expand_model_fields(missing)
        inner = copy.deepcopy(cfg)
        inner.func, inner.params = function, self.params
        self.function = function(inner, env) if inspect.isclass(function) else function
        if self.event == "motor_params_implicit":
            if self.function.kp_ctrl_ids.numel() or self.function.kd_ctrl_ids.numel():
                raise ValueError("Reset-randomized gains are outside the fixed-static protocol")

    @staticmethod
    def repeat_first(value, num_envs):
        if value.shape[0] != num_envs:
            raise ValueError("Expected expanded per-world DR storage")
        value.copy_(value[:1].clone().expand_as(value))

    def __call__(self, env, env_ids, **_):
        if env_ids is not None and not isinstance(env_ids, slice):
            if not torch.equal(env_ids, torch.arange(env.num_envs, device=env.device)):
                raise ValueError("Fixed DR must initialize all worlds together at startup")
        device = torch.device(env.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Always sample precisely ONE world. Sampling N and choosing its first
        # row can change the profile when evaluation uses a different batch size.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(self.seed)
            self.function(env, torch.zeros(1, dtype=torch.long, device=env.device), **self.params)
        for name in self.fields:
            self.repeat_first(getattr(env.sim.model, name), env.num_envs)
        if self.event == "encoder_bias":
            self.repeat_first(env.scene["robot"].data.encoder_bias, env.num_envs)
        if self.event == "motor_params_implicit":
            for name in MOTOR_OBSERVATIONS:
                self.repeat_first(getattr(self.function, name), env.num_envs)

    def reset(self, env_ids=None):
        # Original armature-only motor DR does not resample anything on reset.
        reset = getattr(self.function, "reset", None)
        if callable(reset):
            reset(env_ids=env_ids)

    def observe(self, **kwargs):
        return self.function.observe(**kwargs)


def configure_fixed_dr(env_cfg, seed, *, bank_path, profile_id, profile=TRACKER_DR, fixed_masses=None):
    if profile != TRACKER_DR or fixed_masses is not None:
        raise ValueError("Use the complete selected profile, without a payload-only override")
    selected, bank_path, bank_sha = load_profile(bank_path, profile_id)
    metadata = configure_limb_dr(env_cfg, seed, profile=TRACKER_DR, fixed_masses=selected["masses_kg"])
    for index, (name, (function, _)) in enumerate(STATIC_EVENTS.items()):
        term = env_cfg.events[name]
        if term.mode != "startup" or term.func is not function:
            raise ValueError(f"Unsupported original static event: {name}")
        term.func = FixedStaticRandomization
        term.params.update(fixed_event=name, fixed_seed=selected["static_seed"] + 300001 + index * 1009)
    metadata.update(
        profile="frozen-tracker-complete-fixed-static-dr-plus-limb-payload",
        sampling="All worlds repeat one fixed static DR; motion, phase, noise and push events remain independent",
        fixed_dr={"bank": bank_path, "bank_sha256": bank_sha, **selected},
        effective_events={name: _event_contract(term) for name, term in env_cfg.events.items()},
    )
    return metadata


def audit_fixed_dr(env, configuration):
    active = {name for names in env.event_manager.active_terms.values() for name in names}
    if active != set(configuration["effective_events"]):
        raise RuntimeError("The fixed DR event set changed")
    for name, expected in configuration["effective_events"].items():
        actual = _event_contract(env.event_manager.get_term_cfg(name))
        if actual != _resolved_event_contract(expected, env.scene):
            raise RuntimeError(f"Fixed DR event changed: {name}")
    corruption = {name: bool(group.enable_corruption) for name, group in env.cfg.observations.items()}
    if corruption != configuration["observation_corruption"]:
        raise RuntimeError("Observation noise changed")
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    result = payload.audit()
    expected = torch.tensor(configuration["fixed_dr"]["masses_kg"], device=env.device, dtype=payload.mass.dtype)
    torch.testing.assert_close(payload.mass, expected.expand_as(payload.mass), atol=0, rtol=0)
    tensors = {name: world_parameter_view(getattr(env.sim.model, name), env.num_envs,
               env.sim.get_default_field(name).shape, expanded=name in env.sim.expanded_fields)
               for name in PHYSICS_FIELDS}
    tensors["encoder_bias"] = env.scene["robot"].data.encoder_bias
    motor = env.event_manager.get_term_cfg("motor_params_implicit").func.function
    for name in MOTOR_OBSERVATIONS:
        tensors["motor" + name] = getattr(motor, name)
    digest, errors, prototypes = hashlib.sha256(), {}, {}
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite fixed physics: {name}")
        error = float((value - value[:1]).abs().max()) if value.numel() else 0.0
        if error != 0:
            raise RuntimeError(f"Worlds do not share one complete DR: {name}, error={error}")
        row = value[0].detach().cpu().contiguous()
        digest.update(f"{name}/{row.dtype}/{tuple(row.shape)}".encode())
        digest.update(row.numpy().tobytes())
        errors[name], prototypes[name] = error, row.tolist()
    fingerprint = digest.hexdigest()
    expected_fingerprint = configuration["fixed_dr"].get("physics_fingerprint")
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise RuntimeError("Physics differs from the preflight materialized DR profile")
    result.update(fixed_dr={**configuration["fixed_dr"], "physics_fingerprint": fingerprint,
                           "replicas": env.num_envs, "replica_max_errors": errors,
                           "prototype_fields": prototypes},
                  original_events=copy.deepcopy(configuration["original_events"]),
                  observation_corruption=corruption,
                  per_limb_min_kg=payload.observe().amin(0).tolist(),
                  per_limb_max_kg=payload.observe().amax(0).tolist())
    return result
