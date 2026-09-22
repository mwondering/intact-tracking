"""Range-normalized physical supervision for the residual actor's shared features."""

from __future__ import annotations

import math

import torch
from torch import nn

from intact_tracking.preview_protocol import LIMBS

DR_TARGET_GROUP = "residual_dr_target"
DR_HISTORY_WEIGHT_GROUP = "residual_dr_history_weight"
DR_DIM = 92
GROUPS = ("com_x", "com_y", "com_z", "friction", "kp", "kd", "armature")
MAE_UNITS = ("m", "m", "m", "coefficient", "scale", "scale", "scale")
PAYLOAD_GROUPS = tuple(f"payload_mass_{limb}" for limb in LIMBS) + tuple(
    f"payload_com_{limb}_{axis}" for limb in LIMBS for axis in "xyz")
PAYLOAD_MAE_UNITS = ("kg",) * 4 + ("m",) * 12


def dr_aux_layout(schema, motor_weight=0.0):
    """Bind native/payload coordinates by name; torso mass remains excluded."""
    if not math.isfinite(motor_weight) or motor_weight < 0:
        raise ValueError("DR auxiliary motor weight must be finite and nonnegative")
    names = schema["names"]
    dimensions = len(names)
    if dimensions not in (DR_DIM, DR_DIM + 16) or len(set(names)) != dimensions:
        raise ValueError("DR auxiliary supervision requires 92 or 108 unique parameter names")
    groups = GROUPS + (PAYLOAD_GROUPS if dimensions == DR_DIM + 16 else ())
    lower, upper = schema["lower"], schema["upper"]
    if len(lower) != dimensions or len(upper) != dimensions or any(
        not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo
        for lo, hi in zip(lower, upper, strict=True)
    ):
        raise ValueError("DR auxiliary schema requires finite, increasing physical ranges")
    columns = {group: [] for group in (*groups, "mass")}
    for index, name in enumerate(names):
        _, kind, *labels = name.split("/")
        if kind == "com_offset" and len(labels) == 2 and labels[0] == "torso_link" and labels[1] in ("x", "y", "z"):
            group = "com_" + labels[1]
        elif kind == "relative_mass" and labels == ["torso_link"]:
            group = "mass"
        elif kind == "friction" and labels == ["shared", "0"]:
            group = "friction"
        elif kind in ("kp_scale", "kd_scale", "armature_scale") and len(labels) == 1:
            group = kind.removesuffix("_scale")
        elif kind == "added_mass_kg" and len(labels) == 1 and labels[0] in LIMBS:
            group = f"payload_mass_{labels[0]}"
        elif kind == "payload_com_offset" and len(labels) == 2 and labels[0] in LIMBS and labels[1] in "xyz":
            group = f"payload_com_{labels[0]}_{labels[1]}"
        else:
            raise ValueError(f"Unsupported DR auxiliary parameter: {name}")
        if group not in columns:
            raise ValueError(f"DR auxiliary schema has unexpected payload parameter: {name}")
        columns[group].append(index)
    if any(len(columns[group]) != (29 if group in GROUPS[4:] else 1) for group in columns):
        raise ValueError("Expected native COM/friction/mass/motor coordinates and, for 108 outputs, all 16 payload coordinates")
    weights = [1., 1., 1., 1., motor_weight, motor_weight, motor_weight]
    # Each payload family (mass, COM x/y/z) has the same weight as one native
    # group, divided equally over its four limbs. Log each limb separately.
    if dimensions == DR_DIM + 16:
        weights += [.25] * 16
    return columns, weights


class DRAuxiliaryObjective(nn.Module):
    """Motor dimensions are averaged within groups, then groups are weighted."""

    def __init__(self, schema, motor_weight=0.0, *, payload_com_enabled=True):
        super().__init__()
        columns, weights = dr_aux_layout(schema, motor_weight)
        self.output_dim = len(schema["names"])
        if not isinstance(payload_com_enabled, bool):
            raise ValueError("payload_com_enabled must be boolean")
        if not payload_com_enabled and self.output_dim != 108:
            raise ValueError("Disabling payload COM requires the 108-coordinate heavy schema")
        self.payload_com_enabled = payload_com_enabled
        # Preserve the original normalization when dropping supervision. The
        # remaining coordinates must not receive an additional renormalization.
        self.normalization_denominator = sum(weights)
        if not payload_com_enabled:
            weights[-12:] = [0.] * 12
        self.groups = tuple(group for group in columns if group != "mass")
        self.mae_units = MAE_UNITS + (PAYLOAD_MAE_UNITS if self.output_dim == DR_DIM + 16 else ())
        self.supervised_groups = tuple(group for group, weight in zip(self.groups, weights, strict=True) if weight > 0)
        # Exclude zero-weight groups before subtraction, so their outputs and
        # labels cannot contaminate losses/metrics through NaN * zero.
        active = [i for group in self.supervised_groups for i in columns[group]]
        self.register_buffer("active", torch.tensor(active), persistent=False)
        pooling = torch.zeros(len(active), len(self.groups))
        offset = 0
        for g, group in enumerate(self.groups):
            if group not in self.supervised_groups:
                continue
            width = len(columns[group])
            pooling[offset:offset + width, g] = 1. / width
            offset += width
        self.register_buffer("pooling", pooling, persistent=False)
        weights = torch.tensor(weights)
        self.register_buffer("weights", weights / self.normalization_denominator, persistent=False)
        ranges = torch.tensor(schema["upper"]) - torch.tensor(schema["lower"])
        self.register_buffer("ranges", ranges[active], persistent=False)

    def forward(self, prediction, target, history_weight):
        if prediction.ndim != 2 or prediction.shape[-1] != self.output_dim or target.shape != prediction.shape:
            raise ValueError(f"DR predictions and labels must have matching [batch, {self.output_dim}] shapes")
        if history_weight.shape != (prediction.shape[0], 1):
            raise ValueError("DR history weights must have shape [batch, 1]")
        weight = history_weight.detach().to(prediction)
        error = prediction[:, self.active] - target.detach().to(prediction)[:, self.active]
        # Empty histories contribute an exact differentiable zero on every rank.
        error = torch.where(weight > 0, error, torch.zeros_like(error))
        group_mse = error.square() @ self.pooling
        loss = ((group_mse @ self.weights) * weight[:, 0]).mean()
        with torch.no_grad():
            group_mae = (error.abs() * self.ranges) @ self.pooling
            stats = torch.cat(((group_mse * weight).sum(0), (group_mae * weight).sum(0),
                               weight.sum().view(1), (weight > 0).sum().to(prediction).view(1),
                               prediction.new_tensor([prediction.shape[0]])))
        return loss, stats


def dr_history_weight(memory):
    """Snapshot availability of the same short/long history used by the encoder."""
    chunks = (memory.total_chunks - memory.session_start).clamp(0, memory.long_chunks)
    steps = memory.short_count + chunks * memory.chunk_steps
    return (steps.float() / (memory.short_steps + memory.long_chunks * memory.chunk_steps)).clamp(0, 1)[:, None]


@torch.no_grad()
def capture_dr_aux_targets(env, schema, *, allow_extra_parameters=False):
    """Capture actual post-restoration physics and verify checkpoint range semantics."""
    from intact_tracking.memory350_native_dr import native_metric_schema
    from intact_tracking.rollout.online import (
        _capture_privileged_dynamics_targets, _entity_indices_and_names, _expanded_and_default_field,
    )

    dr_aux_layout(schema)
    physical = _capture_privileged_dynamics_targets(env)
    names = list(physical.names)
    if names != schema["names"] and not allow_extra_parameters:
        raise ValueError("Environment DR label names/order differ from the context checkpoint")
    if len(set(names)) != len(names) or not set(schema["names"]).issubset(names):
        raise ValueError("Environment is missing required DR auxiliary parameters")
    columns = [names.index(name) for name in schema["names"]]
    events = {name.split("/", 1)[0] for name in names}
    params = {name: env.event_manager.get_term_cfg(name).params for name in events}
    defaults = {}
    for event, config in params.items():
        if any(name.startswith(event + "/relative_mass/") for name in names):
            ids, body_names = _entity_indices_and_names(env, config["asset_cfg"], "body")
            _, masses = _expanded_and_default_field(env, "body_mass")
            defaults.update(zip(body_names, masses[ids].cpu().tolist(), strict=True))
    actual = native_metric_schema(names, params, defaults)
    for key in ("lower", "upper"):
        if not torch.allclose(torch.tensor(actual[key])[columns], torch.tensor(schema[key]), rtol=1e-6, atol=1e-8):
            raise ValueError(f"Environment DR {key} ranges differ from the context checkpoint")
    values = physical.values.detach().float()[:, columns]
    lower, upper = values.new_tensor(schema["lower"]), values.new_tensor(schema["upper"])
    targets = (values - lower) / (upper - lower)
    if not bool(torch.isfinite(targets).all()) or bool(((targets < -1e-4) | (targets > 1.0001)).any()):
        raise ValueError("Actual DR targets are nonfinite or outside configured physical ranges")
    # Do not use dr_metric: its sqrt(coordinate_weights) is for encoder distance.
    return targets.clone()
