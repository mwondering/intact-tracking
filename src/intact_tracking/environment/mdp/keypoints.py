"""Semantic keypoints backed by physical body frames and local transforms.

The tracking tasks use semantic points such as ``head`` and ``left_hand``.
Those points do not need dedicated MuJoCo bodies: a physical parent body plus
a rigid local transform is sufficient for pose and velocity observations.  An
optional additive position correction can be expressed in a second body frame
when two robot descriptions place a joint origin at different locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul


@dataclass(frozen=True)
class KeypointSpec:
    name: str
    asset_body_name: str
    reference_body_name: str
    asset_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reference_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    asset_local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    reference_local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    asset_correction_body_name: str | None = None
    reference_correction_body_name: str | None = None
    asset_correction_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reference_correction_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class KeypointKinematics:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


def _tuple(value: Sequence[float], size: int, *, field: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != size:
        raise ValueError(f"{field} must contain {size} values, got {result}")
    return result


def parse_keypoint_specs(
    raw_specs: Iterable[KeypointSpec | Mapping[str, Any]],
) -> tuple[KeypointSpec, ...]:
    specs: list[KeypointSpec] = []
    for raw in raw_specs:
        if isinstance(raw, KeypointSpec):
            specs.append(raw)
            continue
        data = dict(raw)
        name = str(data["name"])
        common_body = data.get("body_name")
        asset_body = str(data.get("asset_body_name", common_body))
        reference_body = str(data.get("reference_body_name", common_body))
        if asset_body == "None" or reference_body == "None":
            raise ValueError(f"Keypoint {name!r} must define asset/reference body names")
        common_pos = data.get("local_pos", (0.0, 0.0, 0.0))
        common_quat = data.get("local_quat", (1.0, 0.0, 0.0, 0.0))
        common_correction_body = data.get("correction_body_name")
        asset_correction_body = data.get("asset_correction_body_name", common_correction_body)
        reference_correction_body = data.get(
            "reference_correction_body_name", common_correction_body
        )
        common_correction_pos = data.get("correction_local_pos", (0.0, 0.0, 0.0))
        specs.append(
            KeypointSpec(
                name=name,
                asset_body_name=asset_body,
                reference_body_name=reference_body,
                asset_local_pos=_tuple(
                    data.get("asset_local_pos", common_pos), 3, field="asset_local_pos"
                ),
                reference_local_pos=_tuple(
                    data.get("reference_local_pos", common_pos),
                    3,
                    field="reference_local_pos",
                ),
                asset_local_quat=_tuple(
                    data.get("asset_local_quat", common_quat),
                    4,
                    field="asset_local_quat",
                ),
                reference_local_quat=_tuple(
                    data.get("reference_local_quat", common_quat),
                    4,
                    field="reference_local_quat",
                ),
                asset_correction_body_name=(
                    str(asset_correction_body) if asset_correction_body is not None else None
                ),
                reference_correction_body_name=(
                    str(reference_correction_body)
                    if reference_correction_body is not None
                    else None
                ),
                asset_correction_local_pos=_tuple(
                    data.get("asset_correction_local_pos", common_correction_pos),
                    3,
                    field="asset_correction_local_pos",
                ),
                reference_correction_local_pos=_tuple(
                    data.get("reference_correction_local_pos", common_correction_pos),
                    3,
                    field="reference_correction_local_pos",
                ),
            )
        )
    names = [spec.name for spec in specs]
    if not specs:
        raise ValueError("At least one semantic keypoint must be configured")
    if len(names) != len(set(names)):
        raise ValueError(f"Semantic keypoint names must be unique, got {names}")
    return tuple(specs)


def _rigid_points(
    parent_pos_w: torch.Tensor,
    parent_quat_w: torch.Tensor,
    parent_lin_vel_w: torch.Tensor,
    parent_ang_vel_w: torch.Tensor,
    local_pos: torch.Tensor,
    local_quat: torch.Tensor,
    correction_quat_w: torch.Tensor,
    correction_ang_vel_w: torch.Tensor,
    correction_local_pos: torch.Tensor,
) -> KeypointKinematics:
    while local_pos.ndim < parent_pos_w.ndim:
        local_pos = local_pos.unsqueeze(0)
        local_quat = local_quat.unsqueeze(0)
        correction_local_pos = correction_local_pos.unsqueeze(0)
    offset_w = quat_apply(parent_quat_w, local_pos.expand_as(parent_pos_w))
    correction_w = quat_apply(correction_quat_w, correction_local_pos.expand_as(parent_pos_w))
    pos_w = parent_pos_w + offset_w + correction_w
    quat_w = quat_mul(parent_quat_w, local_quat.expand_as(parent_quat_w))
    lin_vel_w = (
        parent_lin_vel_w
        + torch.linalg.cross(parent_ang_vel_w, offset_w, dim=-1)
        + torch.linalg.cross(correction_ang_vel_w, correction_w, dim=-1)
    )
    return KeypointKinematics(pos_w, quat_w, lin_vel_w, parent_ang_vel_w)


class SemanticKeypointResolver:
    """Resolve asset and reference indices independently for semantic points."""

    def __init__(self, asset, reference_body_names: Sequence[str], raw_specs):
        self.specs = parse_keypoint_specs(raw_specs)
        asset_names = tuple(str(name).split("/")[-1] for name in asset.body_names)
        reference_names = tuple(str(name).split("/")[-1] for name in reference_body_names)
        missing_asset = [
            spec.asset_body_name for spec in self.specs if spec.asset_body_name not in asset_names
        ]
        missing_reference = [
            spec.reference_body_name
            for spec in self.specs
            if spec.reference_body_name not in reference_names
        ]
        missing_asset.extend(
            spec.asset_correction_body_name
            for spec in self.specs
            if spec.asset_correction_body_name is not None
            and spec.asset_correction_body_name not in asset_names
        )
        missing_reference.extend(
            spec.reference_correction_body_name
            for spec in self.specs
            if spec.reference_correction_body_name is not None
            and spec.reference_correction_body_name not in reference_names
        )
        if missing_asset or missing_reference:
            raise ValueError(
                "Semantic keypoint parent bodies are missing: "
                f"asset={missing_asset}, reference={missing_reference}"
            )
        device = asset.data.body_link_pos_w.device
        self.asset_ids = torch.as_tensor(
            [asset_names.index(spec.asset_body_name) for spec in self.specs],
            device=device,
            dtype=torch.long,
        )
        self.reference_ids = torch.as_tensor(
            [reference_names.index(spec.reference_body_name) for spec in self.specs],
            device=device,
            dtype=torch.long,
        )
        self.asset_correction_ids = torch.as_tensor(
            [
                asset_names.index(spec.asset_correction_body_name or spec.asset_body_name)
                for spec in self.specs
            ],
            device=device,
            dtype=torch.long,
        )
        self.reference_correction_ids = torch.as_tensor(
            [
                reference_names.index(
                    spec.reference_correction_body_name or spec.reference_body_name
                )
                for spec in self.specs
            ],
            device=device,
            dtype=torch.long,
        )
        self.asset_local_pos = torch.tensor(
            [spec.asset_local_pos for spec in self.specs],
            device=device,
            dtype=torch.float32,
        )
        self.reference_local_pos = torch.tensor(
            [spec.reference_local_pos for spec in self.specs],
            device=device,
            dtype=torch.float32,
        )
        self.asset_local_quat = torch.tensor(
            [spec.asset_local_quat for spec in self.specs],
            device=device,
            dtype=torch.float32,
        )
        self.reference_local_quat = torch.tensor(
            [spec.reference_local_quat for spec in self.specs],
            device=device,
            dtype=torch.float32,
        )
        self.asset_correction_local_pos = torch.tensor(
            [spec.asset_correction_local_pos for spec in self.specs],
            device=device,
            dtype=torch.float32,
        )
        self.reference_correction_local_pos = torch.tensor(
            [spec.reference_correction_local_pos for spec in self.specs],
            device=device,
            dtype=torch.float32,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def current(self, asset) -> KeypointKinematics:
        ids = self.asset_ids
        correction_ids = self.asset_correction_ids
        return _rigid_points(
            asset.data.body_link_pos_w[:, ids],
            asset.data.body_link_quat_w[:, ids],
            asset.data.body_link_lin_vel_w[:, ids],
            asset.data.body_link_ang_vel_w[:, ids],
            self.asset_local_pos,
            self.asset_local_quat,
            asset.data.body_link_quat_w[:, correction_ids],
            asset.data.body_link_ang_vel_w[:, correction_ids],
            self.asset_correction_local_pos,
        )

    def reference(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
    ) -> KeypointKinematics:
        ids = self.reference_ids.to(body_pos_w.device)
        correction_ids = self.reference_correction_ids.to(body_pos_w.device)
        return _rigid_points(
            body_pos_w.index_select(-2, ids),
            body_quat_w.index_select(-2, ids),
            body_lin_vel_w.index_select(-2, ids),
            body_ang_vel_w.index_select(-2, ids),
            self.reference_local_pos.to(body_pos_w),
            self.reference_local_quat.to(body_quat_w),
            body_quat_w.index_select(-2, correction_ids),
            body_ang_vel_w.index_select(-2, correction_ids),
            self.reference_correction_local_pos.to(body_pos_w),
        )
