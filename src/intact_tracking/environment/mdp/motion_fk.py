from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


def _basename(name: str) -> str:
    return name.split("/")[-1]


def _as_torch(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if hasattr(value, "_tensor"):
        return value._tensor.to(device=device)
    try:
        import warp as wp  # type: ignore

        if isinstance(value, wp.array):  # type: ignore[arg-type]
            return wp.to_torch(value).to(device=device)
    except Exception:
        pass
    return torch.as_tensor(value, device=device)


def _as_scalar_1d(value: Any, *, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 2:
        out = out[0]
    if out.ndim != 1:
        raise ValueError(f"Expected scalar field [N] or [E,N], got {tuple(out.shape)}")
    return out


def _as_vec_field(value: Any, *, dim: int, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[-1] != dim:
        raise ValueError(f"Expected vector field [N,{dim}] or [E,N,{dim}], got {tuple(out.shape)}")
    return out


def normalize(x: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = q.unbind(dim=-1)
    rw, rx, ry, rz = r.unbind(dim=-1)
    return torch.stack(
        (
            qw * rw - qx * rx - qy * ry - qz * rz,
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec, v = torch.broadcast_tensors(q[..., 1:], v)
    q_w = q[..., :1].expand(q_vec.shape[:-1] + (1,))
    cross = 2.0 * torch.cross(q_vec, v, dim=-1)
    # Keep the same operation order as fk_backend_compare.heft_batch.  The two
    # common quaternion-vector formulas are algebraically identical, but their
    # float32 round-off is observably different after a deep FK chain.
    return v + q_w * cross + torch.cross(q_vec, cross, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec, v = torch.broadcast_tensors(q[..., 1:], v)
    q_w = q[..., :1].expand(q_vec.shape[:-1] + (1,))
    cross = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v - q_w * cross + torch.cross(q_vec, cross, dim=-1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half_angle = angle * 0.5
    # MJCF joint axes are unit vectors.  HEFT's authoritative offline backend
    # does not renormalize this intermediate quaternion; the composed body
    # quaternion is normalized once below in the FK recursion.
    return torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            axis * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )


def finite_diff_torch(x: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    x_t = x.movedim(dim, 0)
    vel = torch.zeros_like(x_t)
    if fps <= 0.0 or x_t.shape[0] < 2:
        return vel.movedim(0, dim)
    vel[1:-1] = (x_t[2:] - x_t[:-2]) * (fps / 2.0)
    vel[0] = (x_t[1] - x_t[0]) * fps
    vel[-1] = (x_t[-1] - x_t[-2]) * fps
    return vel.movedim(0, dim)


def angvel_from_quat_wxyz_torch(quat_wxyz: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    quat_t = normalize(quat_wxyz.movedim(dim, 0))
    if fps <= 0.0 or quat_t.shape[0] < 2:
        shape = quat_t.shape[:-1] + (3,)
        return torch.zeros(shape, dtype=quat_t.dtype, device=quat_t.device).movedim(0, dim)

    flat = quat_t.reshape(quat_t.shape[0], -1, 4)
    dots = (flat[1:] * flat[:-1]).sum(dim=-1)
    signs = torch.where(dots < 0.0, -torch.ones_like(dots), torch.ones_like(dots))
    signs = torch.cat([torch.ones_like(signs[:1]), signs], dim=0)
    flat = flat * torch.cumprod(signs, dim=0).unsqueeze(-1)

    qdot = torch.zeros_like(flat)
    qdot[1:-1] = (flat[2:] - flat[:-2]) * (fps / 2.0)
    qdot[0] = (flat[1] - flat[0]) * fps
    qdot[-1] = (flat[-1] - flat[-2]) * fps

    omega = 2.0 * quat_mul(qdot, quat_conjugate(flat))[..., 1:]
    return omega.reshape(quat_t.shape[:-1] + (3,)).movedim(0, dim)


def smooth_avg5_torch(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_t = x.movedim(dim, 0)
    if x_t.shape[0] == 0:
        return x
    time = torch.arange(x_t.shape[0], device=x.device)
    last = x_t.shape[0] - 1
    total = torch.zeros_like(x_t)
    for offset in (-2, -1, 0, 1, 2):
        total = total + x_t[(time + offset).clamp(0, last)]
    return (total * 0.2).movedim(0, dim)


def joint_vel_from_joint_pos_torch(
    joint_pos: torch.Tensor,
    fps: float,
    *,
    dim: int = 0,
) -> torch.Tensor:
    """Rebuild reference joint velocity with the SP convention.

    The source repository derives this field from reference joint positions using
    centered finite differences and a replicated-boundary five-frame average,
    rather than trusting the ``joint_vel`` array stored in an NPZ file.
    """
    return smooth_avg5_torch(finite_diff_torch(joint_pos, fps, dim=dim), dim=dim)


def _actor_support_time_gather(
    value_support: torch.Tensor,
    support_center_steps: torch.Tensor,
    absolute_steps: torch.Tensor,
    *,
    support_start: int,
) -> torch.Tensor:
    """Gather absolute motion times from a compact, centered support window.

    ``value_support`` has shape ``[E, R, S, ...]`` and ``absolute_steps`` has
    shape ``[E, R, ...]``.  Keeping the motion time itself explicit is what lets
    the online path reproduce the offline sequence-boundary rules without
    storing any derived velocity arrays.
    """
    if value_support.ndim < 3:
        raise ValueError(f"Expected compact support [E,R,S,...], got {tuple(value_support.shape)}")
    if support_center_steps.ndim != 2 or value_support.shape[:2] != tuple(
        support_center_steps.shape
    ):
        raise ValueError(
            "Support center shape must match the first two value dimensions; "
            f"value={tuple(value_support.shape)}, centers={tuple(support_center_steps.shape)}"
        )
    if absolute_steps.shape[:2] != support_center_steps.shape:
        raise ValueError(
            "Requested time prefix must match support centers; "
            f"times={tuple(absolute_steps.shape)}, centers={tuple(support_center_steps.shape)}"
        )

    envs, references = support_center_steps.shape
    request_shape = absolute_steps.shape[2:]
    flat_steps = absolute_steps.reshape(envs, references, -1)
    local_indices = flat_steps - support_center_steps.unsqueeze(-1) - int(support_start)
    tail_shape = value_support.shape[3:]
    gather_index = local_indices.reshape(
        (envs, references, flat_steps.shape[-1]) + (1,) * len(tail_shape)
    ).expand((envs, references, flat_steps.shape[-1]) + tail_shape)
    gathered = torch.gather(value_support, dim=2, index=gather_index)
    return gathered.reshape((envs, references) + request_shape + tail_shape)


def _actor_clamp_absolute_steps(
    absolute_steps: torch.Tensor, motion_lengths: torch.Tensor
) -> torch.Tensor:
    last_steps = motion_lengths - 1
    if absolute_steps.ndim > last_steps.ndim:
        last_steps = last_steps.reshape(
            last_steps.shape + (1,) * (absolute_steps.ndim - last_steps.ndim)
        )
    return torch.minimum(torch.clamp_min(absolute_steps, 0), last_steps)


def actor_gather_from_support_torch(
    value_support: torch.Tensor,
    support_center_steps: torch.Tensor,
    motion_lengths: torch.Tensor,
    center_offsets: Sequence[int],
    *,
    support_start: int = -5,
) -> torch.Tensor:
    """Gather values at clamped offsets around each effective target frame."""
    offsets = torch.as_tensor(
        tuple(int(offset) for offset in center_offsets),
        device=support_center_steps.device,
        dtype=torch.long,
    )
    absolute_steps = _actor_clamp_absolute_steps(
        support_center_steps.unsqueeze(-1) + offsets, motion_lengths
    )
    return _actor_support_time_gather(
        value_support,
        support_center_steps,
        absolute_steps,
        support_start=support_start,
    )


def _actor_finite_difference_at_steps_torch(
    value_support: torch.Tensor,
    support_center_steps: torch.Tensor,
    motion_lengths: torch.Tensor,
    absolute_steps: torch.Tensor,
    fps: float,
    *,
    support_start: int,
    quaternion: bool,
    quaternion_is_pre_normalized: bool,
) -> torch.Tensor:
    """Evaluate HEFT's length-aware finite difference at arbitrary times."""
    if fps <= 0.0:
        raise ValueError(f"FPS must be positive, got {fps}")
    absolute_steps = _actor_clamp_absolute_steps(absolute_steps, motion_lengths)
    last_steps = motion_lengths - 1
    expanded_last = last_steps.reshape(
        last_steps.shape + (1,) * (absolute_steps.ndim - last_steps.ndim)
    )
    previous_steps = torch.maximum(absolute_steps - 1, torch.zeros_like(absolute_steps))
    following_steps = torch.minimum(absolute_steps + 1, expanded_last)
    previous = _actor_support_time_gather(
        value_support,
        support_center_steps,
        previous_steps,
        support_start=support_start,
    )
    following = _actor_support_time_gather(
        value_support,
        support_center_steps,
        following_steps,
        support_start=support_start,
    )

    if quaternion:
        current = _actor_support_time_gather(
            value_support,
            support_center_steps,
            absolute_steps,
            support_start=support_start,
        )
        # Local sign alignment is equivalent to HEFT's full-sequence continuity
        # pass for the three samples involved in a finite difference.  Body-local
        # FK quaternions still need HEFT's normalization here.  In contrast, the
        # compact qpos root quaternion is the already-normalized pelvis output
        # stored by the offline pipeline; normalizing it again changes float32
        # values and therefore the derived angular velocity.
        if not quaternion_is_pre_normalized:
            previous = normalize(previous, eps=1.0e-6)
            current = normalize(current, eps=1.0e-6)
            following = normalize(following, eps=1.0e-6)
        previous = previous * torch.where(
            (previous * current).sum(dim=-1, keepdim=True) < 0.0,
            -torch.ones_like(previous[..., :1]),
            torch.ones_like(previous[..., :1]),
        )
        following = following * torch.where(
            (following * current).sum(dim=-1, keepdim=True) < 0.0,
            -torch.ones_like(following[..., :1]),
            torch.ones_like(following[..., :1]),
        )

    delta = following - previous
    endpoint = (absolute_steps == 0) | (absolute_steps == expanded_last)
    scale = torch.where(
        endpoint,
        torch.full_like(absolute_steps, fps, dtype=value_support.dtype),
        torch.full_like(absolute_steps, fps * 0.5, dtype=value_support.dtype),
    )
    scale = scale.reshape(scale.shape + (1,) * (delta.ndim - scale.ndim))
    derivative = delta * scale
    if not quaternion:
        return derivative
    return 2.0 * quat_mul(derivative, quat_conjugate(current))[..., 1:]


def actor_finite_difference_from_support_torch(
    value_support: torch.Tensor,
    support_center_steps: torch.Tensor,
    motion_lengths: torch.Tensor,
    fps: float,
    center_offsets: Sequence[int],
    *,
    support_start: int = -5,
    quaternion: bool = False,
    quaternion_is_pre_normalized: bool = False,
) -> torch.Tensor:
    """Evaluate offline finite differences from an online compact support."""
    offsets = torch.as_tensor(
        tuple(int(offset) for offset in center_offsets),
        device=support_center_steps.device,
        dtype=torch.long,
    )
    absolute_steps = _actor_clamp_absolute_steps(
        support_center_steps.unsqueeze(-1) + offsets, motion_lengths
    )
    return _actor_finite_difference_at_steps_torch(
        value_support,
        support_center_steps,
        motion_lengths,
        absolute_steps,
        fps,
        support_start=support_start,
        quaternion=quaternion,
        quaternion_is_pre_normalized=quaternion_is_pre_normalized,
    )


def actor_smoothed_finite_difference_from_support_torch(
    value_support: torch.Tensor,
    support_center_steps: torch.Tensor,
    motion_lengths: torch.Tensor,
    fps: float,
    center_offsets: Sequence[int],
    *,
    support_start: int = -5,
    quaternion: bool = False,
    quaternion_is_pre_normalized: bool = False,
) -> torch.Tensor:
    """Reproduce offline finite-difference + replicate-padded AVG5 exactly.

    The target offset is clamped first; the five smoothing samples are then
    taken around that effective frame.  This order matters for queries that
    reach either end of a motion.
    """
    offsets = torch.as_tensor(
        tuple(int(offset) for offset in center_offsets),
        device=support_center_steps.device,
        dtype=torch.long,
    )
    center_steps = _actor_clamp_absolute_steps(
        support_center_steps.unsqueeze(-1) + offsets, motion_lengths
    )
    smooth_offsets = torch.arange(-2, 3, device=support_center_steps.device, dtype=torch.long)
    raw_steps = _actor_clamp_absolute_steps(
        center_steps.unsqueeze(-1) + smooth_offsets, motion_lengths
    )
    raw = _actor_finite_difference_at_steps_torch(
        value_support,
        support_center_steps,
        motion_lengths,
        raw_steps,
        fps,
        support_start=support_start,
        quaternion=quaternion,
        quaternion_is_pre_normalized=quaternion_is_pre_normalized,
    )
    # Preserve HEFT's left-to-right accumulation order rather than relying on a
    # reduction kernel whose floating-point association may differ.
    total = torch.zeros_like(raw.select(dim=3, index=0))
    for index in range(5):
        total = total + raw.select(dim=3, index=index)
    return total * 0.2


def actor_angvel_from_quat_torch(
    quat_wxyz: torch.Tensor,
    fps: float,
    dim: int,
    *,
    quaternion_is_pre_normalized: bool = False,
) -> torch.Tensor:
    """Reproduce the SPV5+ Actor quaternion-velocity recipe exactly.

    The sequential sign alignment is intentionally written as a Python loop.
    Besides matching the Actor's historical numerics, this remains compatible
    with the legacy ONNX exporter, which cannot export ``cumprod`` here.
    """
    quat_t = quat_wxyz.movedim(dim, 0)
    if not quaternion_is_pre_normalized:
        quat_t = normalize(quat_t)
    if quat_t.shape[0] == 0:
        shape = quat_t.shape[:-1] + (3,)
        return torch.zeros(shape, dtype=quat_t.dtype, device=quat_t.device).movedim(0, dim)
    aligned = [quat_t[0]]
    for index in range(1, quat_t.shape[0]):
        current = quat_t[index]
        sign = torch.where(
            (current * aligned[-1]).sum(dim=-1, keepdim=True) < 0.0,
            -torch.ones_like(current[..., :1]),
            torch.ones_like(current[..., :1]),
        )
        aligned.append(current * sign)
    continuous = torch.stack(aligned, dim=0)
    if fps <= 0.0 or continuous.shape[0] < 2:
        shape = continuous.shape[:-1] + (3,)
        return torch.zeros(shape, dtype=continuous.dtype, device=continuous.device).movedim(0, dim)
    qdot = torch.zeros_like(continuous)
    qdot[1:-1] = (continuous[2:] - continuous[:-2]) * (fps / 2.0)
    qdot[0] = (continuous[1] - continuous[0]) * fps
    qdot[-1] = (continuous[-1] - continuous[-2]) * fps
    omega = 2.0 * quat_mul(qdot, quat_conjugate(continuous))[..., 1:]
    return omega.movedim(0, dim)


@dataclass(frozen=True)
class MotionFKResult:
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor


class MotionFKHelper:
    def __init__(
        self,
        *,
        device: torch.device,
        base_body_id: int,
        tree_body_ids: torch.Tensor,
        parent_local_idx: torch.Tensor,
        body_pos0: torch.Tensor,
        body_quat0: torch.Tensor,
        joint_types: torch.Tensor,
        joint_pos_local: torch.Tensor,
        joint_axis_local: torch.Tensor,
        joint_dataset_idx: torch.Tensor,
        output_local_idx: torch.Tensor,
        output_body_names: list[str],
    ):
        self.device = device
        self.base_body_id = int(base_body_id)
        self.tree_body_ids = tree_body_ids
        self.parent_local_idx = parent_local_idx
        self.body_pos0 = body_pos0.to(dtype=torch.float32, device=device)
        self.body_quat0 = normalize(body_quat0.to(dtype=torch.float32, device=device))
        self.joint_types = joint_types
        self.joint_pos_local = joint_pos_local.to(dtype=torch.float32, device=device)
        self.joint_axis_local = normalize(joint_axis_local.to(dtype=torch.float32, device=device))
        self.joint_dataset_idx = joint_dataset_idx
        self.output_local_idx = output_local_idx
        self.output_body_names = output_body_names
        self.base_local_idx = int(
            (self.tree_body_ids == self.base_body_id).nonzero(as_tuple=False)[0].item()
        )
        self._tree_size = int(self.tree_body_ids.numel())
        self._body_count = len(self.output_body_names)
        self._parent_local_idx_cpu = self.parent_local_idx.detach().cpu().tolist()
        self._joint_types_cpu = self.joint_types.detach().cpu().tolist()
        self._valid_output_idx = (self.output_local_idx >= 0).nonzero(as_tuple=False).squeeze(-1)
        self._valid_output_local_idx = self.output_local_idx[self._valid_output_idx]
        self._depth_groups = self._build_depth_groups()

    @classmethod
    def from_mjlab_asset(
        cls,
        *,
        asset: Any,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str],
    ) -> MotionFKHelper:
        device = torch.device(asset.data.device)
        model = asset.data.model

        body_name_to_id: dict[str, int] = {}
        for index, name in enumerate(asset.body_names):
            gid = int(asset.indexing.body_ids[index].item())
            body_name_to_id[_basename(name)] = gid

        joint_id_to_name: dict[int, str] = {}
        for index, name in enumerate(asset.joint_names):
            gid = int(asset.indexing.joint_ids[index].item())
            joint_id_to_name[gid] = _basename(name)

        base_body_name = _basename(asset.body_names[0])
        return cls._build(
            model=model,
            body_name_to_id=body_name_to_id,
            joint_id_to_name=joint_id_to_name,
            dataset_joint_names=dataset_joint_names,
            output_body_names=list(output_body_names),
            base_body_name=base_body_name,
            device=device,
        )

    @classmethod
    def from_mjcf_path(
        cls,
        *,
        xml_path: str | Path,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str],
        base_body_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> MotionFKHelper:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        return cls.from_mujoco_model(
            model=model,
            dataset_joint_names=dataset_joint_names,
            output_body_names=output_body_names,
            base_body_name=base_body_name,
            device=device,
        )

    @classmethod
    def from_mujoco_model(
        cls,
        *,
        model: Any,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str],
        base_body_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> MotionFKHelper:
        import mujoco

        torch_device = torch.device(device)
        body_name_to_id: dict[str, int] = {}
        ordered_body_names: list[str] = []
        for body_id in range(1, int(model.nbody)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                raise ValueError(f"Unnamed body id={body_id}")
            short_name = _basename(name)
            body_name_to_id[short_name] = body_id
            ordered_body_names.append(short_name)

        joint_id_to_name: dict[int, str] = {}
        free_base_body_id: int | None = None
        for joint_id in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                joint_id_to_name[joint_id] = _basename(name)
            elif int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"Unnamed actuated joint id={joint_id}")
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                free_base_body_id = int(model.jnt_bodyid[joint_id])

        if base_body_name is None:
            if free_base_body_id is not None and free_base_body_id > 0:
                base_body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, free_base_body_id
                )
            elif ordered_body_names:
                base_body_name = ordered_body_names[0]
        if base_body_name is None:
            raise ValueError("Could not infer a base body from the MuJoCo model")

        return cls._build(
            model=model,
            body_name_to_id=body_name_to_id,
            joint_id_to_name=joint_id_to_name,
            dataset_joint_names=dataset_joint_names,
            output_body_names=list(output_body_names),
            base_body_name=_basename(base_body_name),
            device=torch_device,
        )

    @classmethod
    def _build(
        cls,
        *,
        model: Any,
        body_name_to_id: dict[str, int],
        joint_id_to_name: dict[int, str],
        dataset_joint_names: Sequence[str],
        output_body_names: list[str],
        base_body_name: str,
        device: torch.device,
    ) -> MotionFKHelper:
        body_parentid = _as_scalar_1d(model.body_parentid, device=device).to(torch.long)
        body_jntnum = _as_scalar_1d(model.body_jntnum, device=device).to(torch.long)
        body_jntadr = _as_scalar_1d(model.body_jntadr, device=device).to(torch.long)
        body_pos_all = _as_vec_field(model.body_pos, dim=3, device=device)
        body_quat_all = _as_vec_field(model.body_quat, dim=4, device=device)
        jnt_type = _as_scalar_1d(model.jnt_type, device=device).to(torch.long)
        jnt_pos = _as_vec_field(model.jnt_pos, dim=3, device=device)
        jnt_axis = _as_vec_field(model.jnt_axis, dim=3, device=device)

        if base_body_name not in body_name_to_id:
            raise ValueError(f"Base body '{base_body_name}' not found in model")
        base_body_id = int(body_name_to_id[base_body_name])

        requested_ids: list[int] = []
        for body_name in output_body_names:
            if body_name not in body_name_to_id:
                raise ValueError(f"Output body '{body_name}' not found in model")
            requested_ids.append(int(body_name_to_id[body_name]))

        selected: set[int] = {base_body_id}
        for body_id in requested_ids:
            cur = body_id
            while True:
                selected.add(cur)
                if cur == base_body_id:
                    break
                cur = int(body_parentid[cur].item())
                if cur < 0:
                    raise RuntimeError(
                        f"Cannot trace body id={body_id} back to base '{base_body_name}'"
                    )

        children_by_gid: dict[int, list[int]] = {body_id: [] for body_id in selected}
        for body_id in selected:
            if body_id == base_body_id:
                continue
            parent_id = int(body_parentid[body_id].item())
            if parent_id in selected:
                children_by_gid[parent_id].append(body_id)
        for children in children_by_gid.values():
            children.sort()

        order_gid: list[int] = []

        def _dfs(body_id: int) -> None:
            order_gid.append(body_id)
            for child_id in children_by_gid.get(body_id, []):
                _dfs(child_id)

        _dfs(base_body_id)
        gid_to_local = {gid: idx for idx, gid in enumerate(order_gid)}
        joint_name_to_dataset_idx = {name: idx for idx, name in enumerate(dataset_joint_names)}
        output_name_to_index = {name: idx for idx, name in enumerate(output_body_names)}
        id_to_body_name = {value: key for key, value in body_name_to_id.items()}

        tree_body_ids = torch.tensor(order_gid, device=device, dtype=torch.long)
        parent_local_idx = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        body_pos0 = torch.empty((len(order_gid), 3), device=device, dtype=body_pos_all.dtype)
        body_quat0 = torch.empty((len(order_gid), 4), device=device, dtype=body_quat_all.dtype)
        joint_types = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        joint_pos_local = torch.zeros((len(order_gid), 3), device=device, dtype=jnt_pos.dtype)
        joint_axis_local = torch.zeros((len(order_gid), 3), device=device, dtype=jnt_axis.dtype)
        joint_dataset_idx = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        output_local_idx = torch.full(
            (len(output_body_names),), -1, device=device, dtype=torch.long
        )

        for local_idx, body_id in enumerate(order_gid):
            parent_id = int(body_parentid[body_id].item())
            parent_local_idx[local_idx] = (
                gid_to_local[parent_id] if parent_id in gid_to_local else -1
            )
            body_pos0[local_idx] = body_pos_all[body_id]
            body_quat0[local_idx] = body_quat_all[body_id]

            body_name = id_to_body_name[body_id]
            if body_name in output_name_to_index:
                output_local_idx[output_name_to_index[body_name]] = local_idx

            joint_count = int(body_jntnum[body_id].item())
            if joint_count > 1:
                raise NotImplementedError(
                    f"Body '{body_name}' has {joint_count} joints; only <=1 joint/body is supported."
                )
            if joint_count == 0:
                continue
            joint_id = int(body_jntadr[body_id].item())
            joint_type = int(jnt_type[joint_id].item())
            if body_id == base_body_id and joint_type == 0:
                continue
            if joint_type not in (2, 3):
                raise NotImplementedError(
                    f"Joint '{joint_id_to_name[joint_id]}' type={joint_type} unsupported."
                )

            joint_name = joint_id_to_name[joint_id]
            if joint_name not in joint_name_to_dataset_idx:
                raise ValueError(f"Joint '{joint_name}' missing from dataset_joint_names")
            joint_types[local_idx] = joint_type
            joint_pos_local[local_idx] = jnt_pos[joint_id]
            joint_axis_local[local_idx] = jnt_axis[joint_id]
            joint_dataset_idx[local_idx] = joint_name_to_dataset_idx[joint_name]

        if (output_local_idx < 0).any():
            missing = [
                output_body_names[i]
                for i in (output_local_idx < 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            ]
            raise ValueError(f"Failed to resolve requested output bodies: {missing}")

        return cls(
            device=device,
            base_body_id=base_body_id,
            tree_body_ids=tree_body_ids,
            parent_local_idx=parent_local_idx,
            body_pos0=body_pos0,
            body_quat0=body_quat0,
            joint_types=joint_types,
            joint_pos_local=joint_pos_local,
            joint_axis_local=joint_axis_local,
            joint_dataset_idx=joint_dataset_idx,
            output_local_idx=output_local_idx,
            output_body_names=output_body_names,
        )

    def _make_group(self, local_ids: list[int]):
        if len(local_ids) == 0:
            return None
        local_idx = torch.tensor(local_ids, device=self.device, dtype=torch.long)
        return {
            "local_idx": local_idx,
            "parent_idx": self.parent_local_idx.index_select(0, local_idx),
            "pos0": self.body_pos0.index_select(0, local_idx),
            "quat0": self.body_quat0.index_select(0, local_idx),
            "joint_dataset_idx": self.joint_dataset_idx.index_select(0, local_idx),
            "joint_pos_local": self.joint_pos_local.index_select(0, local_idx),
            "joint_axis_local": self.joint_axis_local.index_select(0, local_idx),
        }

    def _build_depth_groups(self):
        depths = [0] * self._tree_size
        max_depth = 0
        for local_idx in range(self._tree_size):
            parent_idx = self._parent_local_idx_cpu[local_idx]
            if parent_idx >= 0:
                depths[local_idx] = depths[parent_idx] + 1
                max_depth = max(max_depth, depths[local_idx])

        groups = []
        for depth in range(1, max_depth + 1):
            fixed_ids = []
            slide_ids = []
            hinge_ids = []
            for local_idx, node_depth in enumerate(depths):
                if node_depth != depth:
                    continue
                joint_type = self._joint_types_cpu[local_idx]
                if joint_type < 0:
                    fixed_ids.append(local_idx)
                elif joint_type == 2:
                    slide_ids.append(local_idx)
                elif joint_type == 3:
                    hinge_ids.append(local_idx)
                else:
                    raise RuntimeError(f"Unsupported joint type {joint_type} in FK depth grouping.")
            groups.append(
                {
                    "fixed": self._make_group(fixed_ids),
                    "slide": self._make_group(slide_ids),
                    "hinge": self._make_group(hinge_ids),
                }
            )
        return groups

    def body_pose(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = joint_pos.to(device=self.device, dtype=torch.float32)
        prefix = joint_pos.shape[:-1]
        flat_count = math.prod(prefix) if len(prefix) > 0 else 1
        joint_pos_f = joint_pos.reshape(flat_count, joint_pos.shape[-1])

        tree_pos_b = torch.zeros(
            (flat_count, self._tree_size, 3), device=self.device, dtype=torch.float32
        )
        tree_quat_b = torch.zeros(
            (flat_count, self._tree_size, 4), device=self.device, dtype=torch.float32
        )
        tree_quat_b[:, self.base_local_idx, 0] = 1.0

        for depth_group in self._depth_groups:
            fixed_group = depth_group["fixed"]
            if fixed_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, fixed_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, fixed_group["parent_idx"])
                rel_quat = fixed_group["quat0"].unsqueeze(0)
                rel_pos = fixed_group["pos0"].unsqueeze(0)
                tree_quat_b.index_copy_(
                    1,
                    fixed_group["local_idx"],
                    normalize(quat_mul(parent_quat_b, rel_quat)),
                )
                tree_pos_b.index_copy_(
                    1,
                    fixed_group["local_idx"],
                    parent_pos_b + quat_apply(parent_quat_b, rel_pos),
                )

            slide_group = depth_group["slide"]
            if slide_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, slide_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, slide_group["parent_idx"])
                quat0 = slide_group["quat0"].unsqueeze(0)
                pos0 = slide_group["pos0"].unsqueeze(0)
                axis_local = slide_group["joint_axis_local"].unsqueeze(0)
                joint_value = joint_pos_f.index_select(1, slide_group["joint_dataset_idx"])
                axis_parent = quat_apply(quat0, axis_local)
                rel_pos = pos0 + axis_parent * joint_value.unsqueeze(-1)
                tree_quat_b.index_copy_(
                    1,
                    slide_group["local_idx"],
                    normalize(quat_mul(parent_quat_b, quat0)),
                )
                tree_pos_b.index_copy_(
                    1,
                    slide_group["local_idx"],
                    parent_pos_b + quat_apply(parent_quat_b, rel_pos),
                )

            hinge_group = depth_group["hinge"]
            if hinge_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, hinge_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, hinge_group["parent_idx"])
                quat0 = hinge_group["quat0"].unsqueeze(0)
                pos0 = hinge_group["pos0"].unsqueeze(0)
                axis_local = hinge_group["joint_axis_local"].unsqueeze(0)
                anchor_local = hinge_group["joint_pos_local"].unsqueeze(0)
                joint_value = joint_pos_f.index_select(1, hinge_group["joint_dataset_idx"])
                joint_quat = quat_from_angle_axis(joint_value, axis_local)
                rel_quat = quat_mul(quat0, joint_quat)
                rel_pos = pos0 + quat_apply(
                    quat0, anchor_local - quat_apply(joint_quat, anchor_local)
                )
                tree_quat_b.index_copy_(
                    1,
                    hinge_group["local_idx"],
                    normalize(quat_mul(parent_quat_b, rel_quat)),
                )
                tree_pos_b.index_copy_(
                    1,
                    hinge_group["local_idx"],
                    parent_pos_b + quat_apply(parent_quat_b, rel_pos),
                )

        body_pos_b = torch.zeros(
            (flat_count, self._body_count, 3), device=self.device, dtype=torch.float32
        )
        body_quat_b = torch.zeros(
            (flat_count, self._body_count, 4), device=self.device, dtype=torch.float32
        )
        body_quat_b[..., 0] = 1.0
        if self._valid_output_idx.numel() > 0:
            body_pos_b.index_copy_(
                1,
                self._valid_output_idx,
                tree_pos_b.index_select(1, self._valid_output_local_idx),
            )
            body_quat_b.index_copy_(
                1,
                self._valid_output_idx,
                tree_quat_b.index_select(1, self._valid_output_local_idx),
            )

        return (
            body_pos_b.reshape(prefix + (self._body_count, 3)),
            body_quat_b.reshape(prefix + (self._body_count, 4)),
        )

    def body_kinematics(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run analytic FK from measured ``q``, ``dq``, and root gyro.

        Positions and rotations are root-relative and root-frame expressed.  Linear
        velocity subtracts root translation but includes the velocity induced by
        root angular motion.  Angular velocity is the absolute body angular
        velocity expressed in the root frame; callers that need the SPV4
        root-relative convention subtract ``root_ang_vel_b`` after applying any
        semantic point offsets.
        """
        joint_pos = joint_pos.to(device=self.device, dtype=torch.float32)
        joint_vel = joint_vel.to(device=self.device, dtype=torch.float32)
        root_ang_vel_b = root_ang_vel_b.to(device=self.device, dtype=torch.float32)
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                "joint_vel must have the same shape as joint_pos, got "
                f"{tuple(joint_vel.shape)} and {tuple(joint_pos.shape)}"
            )
        prefix = joint_pos.shape[:-1]
        if root_ang_vel_b.shape != prefix + (3,):
            raise ValueError(
                "root_ang_vel_b must match the joint-state prefix, got "
                f"{tuple(root_ang_vel_b.shape)} for {tuple(joint_pos.shape)}"
            )

        flat_count = math.prod(prefix) if len(prefix) > 0 else 1
        joint_pos_f = joint_pos.reshape(flat_count, joint_pos.shape[-1])
        joint_vel_f = joint_vel.reshape(flat_count, joint_vel.shape[-1])
        root_ang_vel_f = root_ang_vel_b.reshape(flat_count, 3)

        tree_pos_b = torch.zeros(
            (flat_count, self._tree_size, 3),
            device=self.device,
            dtype=torch.float32,
        )
        tree_quat_b = torch.zeros(
            (flat_count, self._tree_size, 4),
            device=self.device,
            dtype=torch.float32,
        )
        tree_lin_vel_b = torch.zeros_like(tree_pos_b)
        tree_ang_vel_b = torch.zeros_like(tree_pos_b)
        tree_quat_b[:, self.base_local_idx, 0] = 1.0
        tree_ang_vel_b[:, self.base_local_idx] = root_ang_vel_f

        for depth_group in self._depth_groups:
            fixed_group = depth_group["fixed"]
            if fixed_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, fixed_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, fixed_group["parent_idx"])
                parent_lin_vel_b = tree_lin_vel_b.index_select(1, fixed_group["parent_idx"])
                parent_ang_vel_b = tree_ang_vel_b.index_select(1, fixed_group["parent_idx"])
                rel_quat = fixed_group["quat0"].unsqueeze(0)
                rel_pos_parent = fixed_group["pos0"].unsqueeze(0)
                rel_pos_b = quat_apply(parent_quat_b, rel_pos_parent)
                tree_quat_b.index_copy_(
                    1,
                    fixed_group["local_idx"],
                    normalize(quat_mul(parent_quat_b, rel_quat)),
                )
                tree_pos_b.index_copy_(1, fixed_group["local_idx"], parent_pos_b + rel_pos_b)
                tree_lin_vel_b.index_copy_(
                    1,
                    fixed_group["local_idx"],
                    parent_lin_vel_b + torch.linalg.cross(parent_ang_vel_b, rel_pos_b, dim=-1),
                )
                tree_ang_vel_b.index_copy_(1, fixed_group["local_idx"], parent_ang_vel_b)

            slide_group = depth_group["slide"]
            if slide_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, slide_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, slide_group["parent_idx"])
                parent_lin_vel_b = tree_lin_vel_b.index_select(1, slide_group["parent_idx"])
                parent_ang_vel_b = tree_ang_vel_b.index_select(1, slide_group["parent_idx"])
                quat0 = slide_group["quat0"].unsqueeze(0)
                pos0 = slide_group["pos0"].unsqueeze(0)
                axis_local = slide_group["joint_axis_local"].unsqueeze(0)
                joint_value = joint_pos_f.index_select(1, slide_group["joint_dataset_idx"])
                joint_rate = joint_vel_f.index_select(1, slide_group["joint_dataset_idx"])
                axis_parent = quat_apply(quat0, axis_local)
                rel_pos_parent = pos0 + axis_parent * joint_value.unsqueeze(-1)
                rel_pos_b = quat_apply(parent_quat_b, rel_pos_parent)
                axis_b = quat_apply(parent_quat_b, axis_parent)
                tree_quat_b.index_copy_(
                    1,
                    slide_group["local_idx"],
                    normalize(quat_mul(parent_quat_b, quat0)),
                )
                tree_pos_b.index_copy_(1, slide_group["local_idx"], parent_pos_b + rel_pos_b)
                tree_lin_vel_b.index_copy_(
                    1,
                    slide_group["local_idx"],
                    parent_lin_vel_b
                    + torch.linalg.cross(parent_ang_vel_b, rel_pos_b, dim=-1)
                    + axis_b * joint_rate.unsqueeze(-1),
                )
                tree_ang_vel_b.index_copy_(1, slide_group["local_idx"], parent_ang_vel_b)

            hinge_group = depth_group["hinge"]
            if hinge_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, hinge_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, hinge_group["parent_idx"])
                parent_lin_vel_b = tree_lin_vel_b.index_select(1, hinge_group["parent_idx"])
                parent_ang_vel_b = tree_ang_vel_b.index_select(1, hinge_group["parent_idx"])
                quat0 = hinge_group["quat0"].unsqueeze(0)
                pos0 = hinge_group["pos0"].unsqueeze(0)
                axis_local = hinge_group["joint_axis_local"].unsqueeze(0)
                anchor_local = hinge_group["joint_pos_local"].unsqueeze(0)
                joint_value = joint_pos_f.index_select(1, hinge_group["joint_dataset_idx"])
                joint_rate = joint_vel_f.index_select(1, hinge_group["joint_dataset_idx"])
                joint_quat = quat_from_angle_axis(joint_value, axis_local)
                rel_quat = quat_mul(quat0, joint_quat)
                rotated_anchor_parent = quat_apply(quat0, quat_apply(joint_quat, anchor_local))
                rel_pos_parent = pos0 + quat_apply(
                    quat0, anchor_local - quat_apply(joint_quat, anchor_local)
                )
                rel_pos_b = quat_apply(parent_quat_b, rel_pos_parent)
                axis_parent = quat_apply(quat0, axis_local)
                joint_ang_vel_b = quat_apply(parent_quat_b, axis_parent) * joint_rate.unsqueeze(-1)
                joint_lever_b = -quat_apply(parent_quat_b, rotated_anchor_parent)
                tree_quat_b.index_copy_(
                    1,
                    hinge_group["local_idx"],
                    normalize(quat_mul(parent_quat_b, rel_quat)),
                )
                tree_pos_b.index_copy_(1, hinge_group["local_idx"], parent_pos_b + rel_pos_b)
                tree_lin_vel_b.index_copy_(
                    1,
                    hinge_group["local_idx"],
                    parent_lin_vel_b
                    + torch.linalg.cross(parent_ang_vel_b, rel_pos_b, dim=-1)
                    + torch.linalg.cross(joint_ang_vel_b, joint_lever_b, dim=-1),
                )
                tree_ang_vel_b.index_copy_(
                    1,
                    hinge_group["local_idx"],
                    parent_ang_vel_b + joint_ang_vel_b,
                )

        output_shape = prefix + (self._body_count,)
        body_pos_b = torch.zeros(output_shape + (3,), device=self.device, dtype=torch.float32)
        body_quat_b = torch.zeros(output_shape + (4,), device=self.device, dtype=torch.float32)
        body_lin_vel_b = torch.zeros_like(body_pos_b)
        body_ang_vel_b = torch.zeros_like(body_pos_b)
        body_quat_b[..., 0] = 1.0
        if self._valid_output_idx.numel() > 0:
            flat_pos = body_pos_b.reshape(flat_count, self._body_count, 3)
            flat_quat = body_quat_b.reshape(flat_count, self._body_count, 4)
            flat_lin_vel = body_lin_vel_b.reshape(flat_count, self._body_count, 3)
            flat_ang_vel = body_ang_vel_b.reshape(flat_count, self._body_count, 3)
            flat_pos.index_copy_(
                1,
                self._valid_output_idx,
                tree_pos_b.index_select(1, self._valid_output_local_idx),
            )
            flat_quat.index_copy_(
                1,
                self._valid_output_idx,
                tree_quat_b.index_select(1, self._valid_output_local_idx),
            )
            flat_lin_vel.index_copy_(
                1,
                self._valid_output_idx,
                tree_lin_vel_b.index_select(1, self._valid_output_local_idx),
            )
            flat_ang_vel.index_copy_(
                1,
                self._valid_output_idx,
                tree_ang_vel_b.index_select(1, self._valid_output_local_idx),
            )
        return body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b

    def expand_motion(
        self,
        *,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        joint_pos: torch.Tensor,
        fps: float,
    ) -> MotionFKResult:
        root_pos_w = root_pos_w.to(device=self.device, dtype=torch.float32)
        root_quat_w = normalize(root_quat_w.to(device=self.device, dtype=torch.float32))
        joint_pos = joint_pos.to(device=self.device, dtype=torch.float32)

        body_pos_b, body_quat_b = self.body_pose(joint_pos)
        # Match ``fk_backend_compare.heft_batch.expand_pos36`` explicitly.  The FK
        # recursion already keeps these quaternions close to unit length, but the
        # offline contract normalizes once more before pose composition and angular
        # finite differences.
        body_quat_b = normalize(body_quat_b, eps=1.0e-6)
        body_pos_w = quat_apply(root_quat_w.unsqueeze(-2), body_pos_b) + root_pos_w.unsqueeze(-2)
        body_quat_w = normalize(quat_mul(root_quat_w.unsqueeze(-2), body_quat_b))

        root_lin_vel_w = smooth_avg5_torch(finite_diff_torch(root_pos_w, fps, dim=0), dim=0)
        root_ang_vel_w = smooth_avg5_torch(
            angvel_from_quat_wxyz_torch(root_quat_w, fps, dim=0), dim=0
        )
        root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        body_lin_vel_b = smooth_avg5_torch(
            finite_diff_torch(body_pos_b, fps, dim=0)
            + torch.cross(
                root_ang_vel_b.unsqueeze(-2).expand_as(body_pos_b),
                body_pos_b,
                dim=-1,
            ),
            dim=0,
        )
        body_ang_vel_b = smooth_avg5_torch(
            angvel_from_quat_wxyz_torch(body_quat_b, fps, dim=0), dim=0
        )
        body_lin_vel_w = quat_apply(
            root_quat_w.unsqueeze(-2), body_lin_vel_b
        ) + root_lin_vel_w.unsqueeze(-2)
        body_ang_vel_w = quat_apply(
            root_quat_w.unsqueeze(-2), body_ang_vel_b
        ) + root_ang_vel_w.unsqueeze(-2)
        return MotionFKResult(
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
        )


def actor_body_kinematics_torch(
    helper: MotionFKHelper,
    joint_pos: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    fps: float,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact body-pose/velocity path used by the SPV5+ Actor.

    Inputs cover the seven-frame ``[-3,+3]`` support needed for a current-frame
    result.  All returned quantities are expressed relative to the reference
    root and in its local frame.  This deliberately uses pose finite differences
    instead of analytic Jacobian kinematics because that is the trained Actor's
    numerical contract.
    """
    body_pos_b, body_quat_b = helper.body_pose(joint_pos)
    body_lin_vel_b, body_ang_vel_b = actor_body_velocity_from_pose_torch(
        body_pos_b,
        body_quat_b,
        root_ang_vel_b,
        fps,
        dim=dim,
    )
    return body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b


def actor_body_velocity_from_pose_torch(
    body_pos_b: torch.Tensor,
    body_quat_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    fps: float,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Actor's exact finite-difference recipe to precomputed FK poses.

    Keeping this operation separate lets the task reuse pose-only motion tiles
    across control steps without changing the velocity arithmetic used by the
    historical Actor path.
    """
    body_lin_vel_b = smooth_avg5_torch(
        finite_diff_torch(body_pos_b, fps, dim=dim)
        + torch.linalg.cross(
            root_ang_vel_b.unsqueeze(-2).expand_as(body_pos_b),
            body_pos_b,
            dim=-1,
        ),
        dim=dim,
    )
    body_ang_vel_b = smooth_avg5_torch(
        actor_angvel_from_quat_torch(body_quat_b, fps, dim=dim), dim=dim
    )
    return body_lin_vel_b, body_ang_vel_b


def actor_body_velocity_from_compact_support_torch(
    body_pos_b_support: torch.Tensor,
    body_quat_b_support: torch.Tensor,
    root_ang_vel_b_smooth: torch.Tensor,
    support_center_steps: torch.Tensor,
    motion_lengths: torch.Tensor,
    fps: float,
    *,
    support_start: int = -5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute current body velocity with the authoritative offline semantics.

    The compact pose support is centered on each requested frame.  A symmetric
    ``[-5,+5]`` window is sufficient for the nested recipe used by HEFT:
    root angular finite difference -> AVG5 -> body decomposition -> AVG5.
    No precomputed velocity tensor is needed.
    """
    body_offsets = (-2, -1, 0, 1, 2)
    body_pos_b = actor_gather_from_support_torch(
        body_pos_b_support,
        support_center_steps,
        motion_lengths,
        body_offsets,
        support_start=support_start,
    )
    body_pos_b_derivative = actor_finite_difference_from_support_torch(
        body_pos_b_support,
        support_center_steps,
        motion_lengths,
        fps,
        body_offsets,
        support_start=support_start,
    )
    body_ang_vel_b_raw = actor_finite_difference_from_support_torch(
        body_quat_b_support,
        support_center_steps,
        motion_lengths,
        fps,
        body_offsets,
        support_start=support_start,
        quaternion=True,
    )
    body_lin_before_smooth = body_pos_b_derivative + torch.cross(
        root_ang_vel_b_smooth.unsqueeze(-2), body_pos_b, dim=-1
    )

    body_lin_vel_b = torch.zeros_like(body_lin_before_smooth[:, :, 0])
    body_ang_vel_b = torch.zeros_like(body_ang_vel_b_raw[:, :, 0])
    for index in range(5):
        body_lin_vel_b = body_lin_vel_b + body_lin_before_smooth[:, :, index]
        body_ang_vel_b = body_ang_vel_b + body_ang_vel_b_raw[:, :, index]
    return body_lin_vel_b * 0.2, body_ang_vel_b * 0.2
