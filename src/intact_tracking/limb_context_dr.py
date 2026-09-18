"""Explicit DR profiles shared by context training, residual PPO and evaluation."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass


LOAD_ONLY = "load_only"
TRACKER_DR = "tracker_dr_plus_limb_payload"
DR_PROFILES = (LOAD_ONLY, TRACKER_DR)


def resolve_dr_profile(requested=None, metadata=None):
    """Old residual checkpoints used load-only; an override must agree on resume/eval."""
    saved = metadata.get("dr_profile", LOAD_ONLY) if metadata is not None else None
    profile = requested or saved or LOAD_ONLY
    if profile not in DR_PROFILES or (saved is not None and profile != saved):
        raise ValueError("DR profile differs from the checkpoint or is unsupported")
    return profile


def validate_context_dr(metadata, profile):
    center_supervised = metadata.get("representation_supervision") == "dr_parameter_center_distance_v1"
    if center_supervised:
        if (metadata.get("loss_config", {}).get("dr_center_objective_version") != 1
                or metadata.get("dr_metric_schema", {}).get("version") != 1
                or len(metadata.get("dr_metric_schema", {}).get("names", [])) != 38):
            raise ValueError("DR-center context has missing or incompatible supervision provenance")
    if not (metadata.get("nominal_counterfactual_representation_supervision") or center_supervised):
        raise ValueError("Context must use nominal-counterfactual or supported DR-center representation supervision")
    saved = metadata.get("dr_profile")
    if saved is None and metadata.get("load_only_experiment"):
        saved = LOAD_ONLY
    if saved != profile:
        raise ValueError("Context and residual policy must use the same DR profile")
    if metadata["loss_config"]["representation_weight"] <= 0:
        raise ValueError("Context representation supervision was disabled")


def _config_value(value):
    if type(value).__name__ == "SceneEntityCfg":
        # Entity resolution fills integer indices in place. Compare the original
        # selectors instead so initialization cannot hide an event/config change.
        return {name: _config_value(getattr(value, name)) for name in (
            "name", "body_names", "joint_names", "geom_names", "site_names", "actuator_names", "preserve_order",
        ) if hasattr(value, name)}
    if is_dataclass(value):
        return {field.name: _config_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported event configuration value: {type(value).__name__}")


def _event_contract(term):
    func = term.func
    target = func if hasattr(func, "__qualname__") else type(func)
    return {
        "function": f"{target.__module__}.{target.__qualname__}",
        **{name: _config_value(getattr(term, name)) for name in (
            "mode", "params", "interval_range_s", "is_global_time", "min_step_count_between_reset",
        )},
    }


def _resolved_event_contract(contract, scene):
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    result = copy.deepcopy(contract)
    asset = result["params"].get("asset_cfg")
    if asset is not None:
        # MJLab expands regex selectors into concrete body/geometry names.
        # Resolve the saved selector independently before comparing runtime DR.
        selector = SceneEntityCfg(**asset)
        selector.resolve(scene)
        result["params"]["asset_cfg"] = _config_value(selector)
    return result


def configure_limb_dr(env_cfg, seed, *, profile=LOAD_ONLY, fixed_masses=None, max_masses_kg=None,
                      nominal_probability=0.0):
    from mjlab.managers.event_manager import EventTermCfg
    from intact_tracking.limb_context_protocol import (
        LIMBS, PAYLOAD_EVENT, UniformLimbPayload, configure_load_only, validate_limb_max_masses,
    )

    from intact_tracking.independent_nominal_dr import validate_probability

    resolve_dr_profile(profile)
    nominal_probability = validate_probability(nominal_probability)
    if nominal_probability and (profile != TRACKER_DR or fixed_masses is not None):
        raise ValueError("Independent nominal mixtures require random tracker DR plus limb payloads")
    if profile == LOAD_ONLY:
        return {**configure_load_only(env_cfg, seed, fixed_masses=fixed_masses,
                                      max_masses_kg=max_masses_kg), "dr_profile": profile}
    required = {"push_robot", "base_com", "base_mass", "encoder_bias", "foot_friction", "motor_params_implicit"}
    if set(env_cfg.events) != required:
        raise ValueError("Expected the frozen tracker's complete original DR event set")
    # The payload implementation snapshots limb inertias during construction.
    # This pinned checkpoint randomizes torso inertias only, so appending limb
    # payloads cannot overwrite another startup event's mass/COM modification.
    for name in ("base_com", "base_mass"):
        if set(env_cfg.events[name].params["asset_cfg"].body_names) != {"torso_link"}:
            raise ValueError("Original inertia DR overlaps an unsupported payload attachment")
    original = {name: _event_contract(term) for name, term in env_cfg.events.items()}
    if original["push_robot"]["mode"] != "step":
        raise ValueError("Expected the checkpoint's torso force-pulse event")
    corruption = {name: bool(group.enable_corruption) for name, group in env_cfg.observations.items()}
    env_cfg.events[PAYLOAD_EVENT] = EventTermCfg(
        mode="startup", func=UniformLimbPayload,
        params={"seed": int(seed) + 91283, "fixed_masses": fixed_masses},
    )
    if max_masses_kg is not None:
        env_cfg.events[PAYLOAD_EVENT].params["max_masses_kg"] = validate_limb_max_masses(max_masses_kg)
    command = env_cfg.commands["motion"]
    command.motion_manifest_file = ""
    command.excluded_motion_files = ()
    command.motion_exclude_files = ()
    command.motion_exclude_file = ""
    command.adaptive_bin_snapshot_interval_iterations = 0
    command.sampling_mode = "uniform"
    command.rewind.enabled = False
    metadata = {
        "dr_profile": profile,
        "profile": "frozen-tracker-dr-plus-hands-shins-independent-uniform-0-4kg",
        "original_events": original,
        "observation_corruption": corruption,
        "original_reset_callbacks": "preserved, including force-pulse timer resets",
        "sampling": "independent U(0,4) per limb and world, fixed at startup",
        "fixed_masses": list(fixed_masses) if fixed_masses is not None else None,
        "limbs": {name: {"body": row[0], "position": list(row[1]), "size": list(row[2])}
                  for name, row in LIMBS.items()},
        "total_added_mass_range_kg": [0, 16],
        "removed_disturbances": [],
    }
    if max_masses_kg is not None:
        limits = list(validate_limb_max_masses(max_masses_kg))
        metadata.update(limb_max_masses_kg=limits, total_added_mass_range_kg=[0, sum(limits)],
                        profile="frozen-tracker-dr-plus-per-limb-uniform-limits",
                        sampling=f"independent U(0,max) kg per limb; maxima {limits}; fixed at startup")
    if nominal_probability:
        from intact_tracking.independent_nominal_dr import EVENT, VERSION, IndependentNominalDR

        # Append after payload: restore scalar background parameters, then rebuild
        # payload inertias from masked mass inputs before EventManager.set_const.
        env_cfg.events[EVENT] = EventTermCfg(
            mode="startup", func=IndependentNominalDR,
            params={"seed": int(seed) + 713917, "probability": nominal_probability},
        )
        metadata["independent_nominal_mixture"] = {
            "version": VERSION, "probability": nominal_probability,
            "event_contract": _event_contract(env_cfg.events[EVENT]),
            "original_payload_sampling": metadata["sampling"],
        }
        metadata["profile"] = "frozen-tracker-dr-independent-coordinate-nominal-mixture"
        metadata["sampling"] = (
            f"Each fixed scalar DR parameter independently takes its compiled nominal value with "
            f"probability {nominal_probability}; otherwise retains its original sample. "
            "COM xyz, limbs and each armature/encoder-bias joint decide separately; "
            "foot friction retains its shared-geometries contract. Fixed across resets."
        )
    return metadata


def audit_limb_dr(env, configuration):
    import hashlib
    import torch
    from intact_tracking.limb_context_protocol import PAYLOAD_EVENT, audit_load_only

    if configuration.get("grouped_dr"):
        from intact_tracking.limb_context_grouped_dr import audit_grid256_dr
        return audit_grid256_dr(env, configuration)

    if resolve_dr_profile(configuration["dr_profile"]) == LOAD_ONLY:
        return audit_load_only(env)
    active = {name for names in env.event_manager.active_terms.values() for name in names}
    extra = set()
    if configuration.get("independent_nominal_mixture"):
        from intact_tracking.independent_nominal_dr import EVENT

        extra.add(EVENT)
        expected = configuration["independent_nominal_mixture"]["event_contract"]
        if _event_contract(env.event_manager.get_term_cfg(EVENT)) != expected:
            raise RuntimeError("Independent nominal-mixture configuration changed")
    if active != set(configuration["original_events"]) | {PAYLOAD_EVENT} | extra:
        raise RuntimeError("An original tracker event or the added payload is missing")
    for name, expected in configuration["original_events"].items():
        if _event_contract(env.event_manager.get_term_cfg(name)) != _resolved_event_contract(expected, env.scene):
            raise RuntimeError(f"Original tracker DR configuration changed: {name}")
    corruption = {name: bool(group.enable_corruption) for name, group in env.cfg.observations.items()}
    if corruption != configuration["observation_corruption"]:
        raise RuntimeError("Original tracker observation corruption changed")
    payload = env.event_manager.get_term_cfg(PAYLOAD_EVENT).func
    result = payload.audit()
    actual = payload.observe()
    if "limb_max_masses_kg" in configuration:
        limits = actual.new_tensor(configuration["limb_max_masses_kg"])
        if not bool(torch.isfinite(actual).all() & (actual >= -1e-5).all()
                    & (actual <= limits + 1e-5).all()):
            raise RuntimeError("Actual payload mass is outside its configured per-limb range")
        result["limb_max_masses_kg"] = configuration["limb_max_masses_kg"]
    result.update(
        sampled_mass_sha256=hashlib.sha256(payload.mass.cpu().numpy().tobytes()).hexdigest(),
        per_limb_min_kg=actual.amin(0).tolist(), per_limb_max_kg=actual.amax(0).tolist(),
        original_events=copy.deepcopy(configuration["original_events"]),
        observation_corruption=corruption,
    )
    if env.num_envs > 4 and (actual.std(0) > 1e-5).all():
        result["correlation"] = torch.corrcoef(actual.T).cpu().tolist()
    if extra:
        result["independent_nominal_mixture"] = env.event_manager.get_term_cfg(EVENT).func.audit()
    return result
