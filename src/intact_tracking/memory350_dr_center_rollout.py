"""Training-only physical labels for the DR-center Memory350 variant."""

from collections.abc import Mapping
import math
import re

import torch

from intact_tracking.limb_context_protocol import LIMBS, PAYLOAD_EVENT, validate_limb_max_masses
from intact_tracking.memory350_nominal_rollout import NominalMemory350TrackerRollout
from intact_tracking.rollout.online import _entity_indices_and_names, _expanded_and_default_field


def dr_metric_schema(names, event_params, default_body_mass):
    """Fixed sampling-range normalization for the existing 38 causal coordinates.

    Ten equal-weight factors: four loads, three COM axes, torso mass, foot
    friction, and one block containing the 29 armature scales. The latter uses
    its own RMS, so its width alone cannot overwhelm the other factors.
    """
    lower, upper, groups = [], [], {}
    for index, name in enumerate(names):
        event, kind, *labels = name.split("/")
        params = event_params[event]
        group = name
        if event == PAYLOAD_EVENT and kind == "added_mass_kg" and labels[0] in LIMBS:
            limits = validate_limb_max_masses(params.get("max_masses_kg", (4.0, 4.0, 4.0, 4.0)))
            low, high = 0.0, limits[list(LIMBS).index(labels[0])]
        elif kind == "com_offset" and labels[0] == "torso_link":
            if params.get("operation") != "add":
                raise ValueError("COM metric requires additive offsets")
            axis = "xyz".index(labels[1])
            ranges = params["ranges"]
            low, high = ranges.get(axis, ranges.get(str(axis))) if isinstance(ranges, Mapping) else ranges
        elif kind == "relative_mass" and labels == ["torso_link"]:
            if params.get("operation") != "add":
                raise ValueError("Torso-mass metric requires additive mass DR")
            mass = default_body_mass[labels[0]]
            if mass <= 0:
                raise ValueError("Compiled torso mass must be positive")
            low, high = (float(x) / mass for x in params["ranges"])
        elif kind == "friction" and labels == ["shared", "0"]:
            if params.get("operation") != "abs" or not params.get("shared_random"):
                raise ValueError("Foot-friction metric requires shared absolute friction DR")
            low, high = params["ranges"]
        elif kind == "armature_scale":
            matches = [limits for pattern, limits in params["armature_range"].items()
                       if re.fullmatch(pattern, labels[0])]
            if len(matches) != 1 or params.get("mode", "uniform") != "uniform":
                raise ValueError(f"Unsupported armature sampling range for {name}")
            low, high = matches[0]
            group = "joint_armature_rms"
        else:
            raise ValueError(f"No audited DR metric normalization for {name}")
        if not all(math.isfinite(x) for x in (low, high)) or high <= low:
            raise ValueError(f"Invalid DR range for {name}: {(low, high)}")
        lower.append(float(low))
        upper.append(float(high))
        groups.setdefault(group, []).append(index)
    if (len(names) != 38 or len(groups) != 10
            or len(groups.get("joint_armature_rms", [])) != 29 or len(set(names)) != 38):
        raise ValueError("This variant requires all 38 existing causal DR coordinates in ten factors")
    weights = [0.0] * len(names)
    for columns in groups.values():
        for index in columns:
            weights[index] = 1 / (len(groups) * len(columns))
    return {
        "version": 1, "names": list(names), "lower": lower, "upper": upper,
        "coordinate_weights": weights, "groups": groups,
        "normalization": "x_k=(theta_k-lower_k)/(upper_k-lower_k); feature_k=x_k*sqrt(weight_k)",
        "distance": "sqrt(mean over ten factors of within-factor mean squared normalized differences)",
        "excluded": ["encoder bias: predictor input is the physical PD target",
                     "time-varying force pulses: disturbance, not a fixed DR parameter",
                     "payload-derived inertia and COM: already determined by the four masses"],
    }


def normalize_dr_metric(values, schema):
    if values.ndim != 2 or values.shape[-1] != len(schema["names"]):
        raise ValueError("DR label dimensions differ from the recorded schema")
    low = values.new_tensor(schema["lower"])
    high = values.new_tensor(schema["upper"])
    unit = (values.detach().float() - low) / (high - low)
    # Check once at rollout construction, not with repeated per-step GPU syncs.
    if not bool(torch.isfinite(unit).all() & (unit >= -1e-5).all() & (unit <= 1 + 1e-5).all()):
        raise ValueError("Measured DR labels are outside their configured sampling ranges")
    return unit * unit.new_tensor(schema["coordinate_weights"]).sqrt()


class DRCenterTrackerRollout(NominalMemory350TrackerRollout):
    def __init__(self, config):
        super().__init__(config)
        try:
            events = {name.split("/", 1)[0] for name in self.privileged_dynamics_names}
            params = {name: self.env.event_manager.get_term_cfg(name).params for name in events}
            defaults = {}
            for event, p in params.items():
                if any(n.startswith(event + "/relative_mass/") for n in self.privileged_dynamics_names):
                    ids, names = _entity_indices_and_names(self.env, p["asset_cfg"], "body")
                    _, masses = _expanded_and_default_field(self.env, "body_mass")
                    defaults.update(zip(names, masses[ids].cpu().tolist(), strict=True))
            self.dr_metric_schema = dr_metric_schema(self.privileged_dynamics_names, params, defaults)
            self.dr_metric = normalize_dr_metric(self.privileged_dynamics, self.dr_metric_schema)
        except BaseException:
            self.close()
            raise

    def step(self, **kwargs):
        batch = super().step(**kwargs)
        # This collector fixes physical parameters for its complete lifetime.
        # Resets only resample motion; the replay also handles explicit changes.
        batch["dr_metric"] = self.dr_metric
        return batch

    @property
    def metadata(self):
        return {**super().metadata, "dr_metric_schema": self.dr_metric_schema}
