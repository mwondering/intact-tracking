"""Train a residual tracker policy through a frozen latent-conditioned simulator model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.distributed import DistributedContext
from intact_tracking.model_gradient_residual import (
    ModelGradientLossConfig,
    ModelGradientResidualPolicy,
    PredictorCausalHistory,
    load_frozen_forward_predictor_checkpoint,
    model_gradient_loss,
)
from intact_tracking.rollout import FixedDRRolloutConfig, FixedDRTrackerRollout
from intact_tracking.rollout.mjlab_adapter import _forward_predictor_snapshot, _sha256
from intact_tracking.rollout.online import TRACKING_ERROR_NAMES
from intact_tracking.wandb_logger import WandbLogger

SPV52A_TASK_ID = "SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward"
T = TypeVar("T")

_REAL_TRACKING_ERROR_FIELDS = {
    "real_root_position_error_m": "error_anchor_pos",
    "real_root_orientation_error_rad": "error_anchor_rot",
    "real_root_linear_velocity_error_mps": "error_anchor_lin_vel",
    "real_root_angular_velocity_error_radps": "error_anchor_ang_vel",
    "real_joint_position_error_l2_rad": "error_joint_pos",
    "real_joint_velocity_error_l2_radps": "error_joint_vel",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint", required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-path")
    motion.add_argument("--motion-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", default=SPV52A_TASK_ID)
    parser.add_argument("--resume")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--gradient-steps-per-update", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--residual-hidden-dims", type=int, nargs="+", default=(512, 256, 128))
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--horizon-discount", type=float, default=0.9)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--root-position-weight", type=float, default=1.0)
    parser.add_argument("--root-orientation-weight", type=float, default=1.0)
    parser.add_argument("--root-linear-velocity-weight", type=float, default=0.25)
    parser.add_argument("--root-angular-velocity-weight", type=float, default=0.25)
    parser.add_argument("--joint-position-weight", type=float, default=1.0)
    parser.add_argument("--joint-velocity-weight", type=float, default=0.25)
    parser.add_argument(
        "--nominal-fraction",
        type=float,
        default=0.0,
        help="Fraction of live worlds restored to nominal physics; default trains on DR worlds.",
    )
    parser.add_argument(
        "--randomize-initial-episode-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--context-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="intact-model-gradient-residual")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "num_envs",
        "updates",
        "gradient_steps_per_update",
        "batch_size",
        "micro_batch_size",
        "probe_batch_size",
        "log_interval",
    )
    invalid = {name: getattr(args, name) for name in positive if getattr(args, name) < 1}
    if invalid:
        raise ValueError(f"Model-gradient residual arguments must be positive: {invalid}")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.micro_batch_size > args.batch_size:
        raise ValueError("micro-batch-size must not exceed batch-size")
    if not args.residual_hidden_dims or any(width < 1 for width in args.residual_hidden_dims):
        raise ValueError("residual-hidden-dims must contain positive widths")
    for name in ("learning_rate", "residual_scale", "huber_delta"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.nominal_fraction <= 1.0:
        raise ValueError("nominal-fraction must be in [0,1]")
    if not 0.0 < args.horizon_discount <= 1.0:
        raise ValueError("horizon-discount must be in (0,1]")
    for name in (
        "weight_decay",
        "residual_weight",
        "smoothness_weight",
        "root_position_weight",
        "root_orientation_weight",
        "root_linear_velocity_weight",
        "root_angular_velocity_weight",
        "joint_position_weight",
        "joint_velocity_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _main_process_call(distributed: DistributedContext, action: Callable[[], T]) -> T:
    payload: dict[str, Any] | None = None
    if distributed.is_main:
        try:
            payload = {"ok": True, "value": action()}
        except BaseException as error:
            payload = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    payload = distributed.broadcast_object(payload)
    if payload is None or not payload["ok"]:
        details = payload or {"error_type": "RuntimeError", "error": "missing result"}
        raise RuntimeError(
            f"Rank-zero operation failed ({details['error_type']}): {details['error']}"
        )
    return payload["value"]


def _prepare_paths(args: argparse.Namespace) -> dict[str, str]:
    tracker = Path(args.tracker_checkpoint).expanduser().resolve()
    forward = Path(args.forward_checkpoint).expanduser().resolve()
    if not tracker.is_file():
        raise FileNotFoundError(tracker)
    if not forward.is_file():
        raise FileNotFoundError(forward)
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and ((output / "last.pt").exists() or (output / "run_config.json").exists()):
        raise FileExistsError(f"Refusing to overwrite an existing run in {output}")
    return {
        "tracker": str(tracker),
        "forward": str(forward),
        "resume": "" if resume is None else str(resume),
        "output": str(output),
        "tracker_sha256": _sha256(tracker),
    }


def _collect_five_step_batch(
    rollout: FixedDRTrackerRollout,
    history: PredictorCausalHistory,
    policy: ModelGradientResidualPolicy,
    current: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    start = history.snapshot(
        current["robot_state"],
        current["foot"],
        current["contact_force"],
        current["contact_binary"],
    )
    predictor_inputs = {name: value.detach() for name, value in start.items() if name != "latent"}
    features: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    tracker_actions: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    valid_steps: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    tracking_errors: list[torch.Tensor] = []
    cumulative_valid = torch.ones(rollout.num_envs, dtype=torch.bool, device=rollout.env.device)
    current_view = start

    policy.eval()
    for _ in range(5):
        latent = current_view["latent"]

        def residual_action_fn(
            tracker_features: torch.Tensor,
            tracker_action: torch.Tensor,
            latent_value: torch.Tensor = latent,
        ) -> torch.Tensor:
            del tracker_action
            return policy(tracker_features, latent_value)

        transition = rollout.step(residual_action_fn=residual_action_fn)
        boundary = transition["reset_boundary"].bool()
        cumulative_valid = cumulative_valid & ~boundary
        features.append(transition["policy_observation"].detach())
        latents.append(latent.detach())
        tracker_actions.append(transition["tracker_action"].detach())
        references.append(transition["next_reference_state"].detach())
        valid_steps.append(cumulative_valid.clone())
        rewards.append(transition["reward"].detach())
        tracking_errors.append(transition["tracking_error"].detach())
        history.append(
            transition["robot_state"],
            transition["joint_target"],
            transition["foot"],
            transition["contact_force"],
            transition["contact_binary"],
            boundary,
        )
        current = {
            "robot_state": transition["next_robot_state"].detach(),
            "foot": transition["next_foot"].detach(),
            "contact_force": transition["next_contact_force"].detach(),
            "contact_binary": transition["next_contact_binary"].detach(),
        }
        current_view = history.snapshot(
            current["robot_state"],
            current["foot"],
            current["contact_force"],
            current["contact_binary"],
        )

    return (
        {
            "predictor_inputs": predictor_inputs,
            "tracker_features": torch.stack(features, dim=1),
            "latent_sequence": torch.stack(latents, dim=1),
            "tracker_actions": torch.stack(tracker_actions, dim=1),
            "reference_states": torch.stack(references, dim=1),
            "valid": torch.stack(valid_steps, dim=1),
            "env_ids": torch.arange(rollout.num_envs, device=rollout.env.device),
            "reward": torch.stack(rewards, dim=1),
            "tracking_error": torch.stack(tracking_errors, dim=1),
        },
        current,
    )


def _sample_indices(num_envs: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size <= num_envs:
        return torch.randperm(num_envs, device=device)[:batch_size]
    return torch.randint(num_envs, (batch_size,), device=device)


def _slice_batch(batch: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    return {
        "predictor_inputs": {
            name: value.index_select(0, indices)
            for name, value in batch["predictor_inputs"].items()
        },
        **{
            name: value.index_select(0, indices)
            for name, value in batch.items()
            if name != "predictor_inputs"
        },
    }


def _gradient_norm(module: nn.Module) -> torch.Tensor:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        return torch.zeros((), device=next(module.parameters()).device)
    return torch.stack(squares).sum().sqrt()


def _real_tracking_error_metrics(value: torch.Tensor) -> dict[str, float]:
    """Preserve physical units instead of averaging incompatible tracking errors."""

    if value.ndim < 2 or value.size(-1) != len(TRACKING_ERROR_NAMES):
        raise ValueError(
            "Tracking error must end in the eight FixedDR tracker metrics, got "
            f"{tuple(value.shape)}"
        )
    indices = {name: index for index, name in enumerate(TRACKING_ERROR_NAMES)}
    return {
        output_name: float(value[..., indices[source_name]].float().mean())
        for output_name, source_name in _REAL_TRACKING_ERROR_FIELDS.items()
    }


class _ZeroResidual(nn.Module):
    def forward(self, tracker_features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        del latent
        return tracker_features.new_zeros((*tracker_features.shape[:-1], 29))


def _objective(
    policy: nn.Module,
    batch: Mapping[str, Any],
    rollout: FixedDRTrackerRollout,
    checkpoint,
    loss_config: ModelGradientLossConfig,
) -> dict[str, torch.Tensor]:
    assert rollout.predictor_action_transform is not None
    return model_gradient_loss(
        policy=policy,  # type: ignore[arg-type]
        predictor_inputs=batch["predictor_inputs"],
        tracker_features=batch["tracker_features"],
        latent_sequence=batch["latent_sequence"],
        tracker_actions=batch["tracker_actions"],
        reference_states=batch["reference_states"],
        valid=batch["valid"],
        env_ids=batch["env_ids"],
        action_transform=rollout.predictor_action_transform,
        checkpoint=checkpoint,
        loss_config=loss_config,
        action_clip=rollout.action_clip,
    )


def _optimize(
    *,
    training_policy: nn.Module,
    raw_policy: ModelGradientResidualPolicy,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    rollout: FixedDRTrackerRollout,
    checkpoint,
    loss_config: ModelGradientLossConfig,
    batch_size: int,
    micro_batch_size: int,
    amp_enabled: bool,
) -> dict[str, float]:
    indices = _sample_indices(rollout.num_envs, batch_size, rollout.env.device)
    micro_batches = math.ceil(batch_size / micro_batch_size)
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "tracking_loss": 0.0,
        "residual_penalty": 0.0,
        "smoothness_loss": 0.0,
        "residual_rms": 0.0,
    }
    training_policy.train()
    for micro_index, start in enumerate(range(0, batch_size, micro_batch_size)):
        micro_indices = indices[start : start + micro_batch_size]
        micro_weight = micro_indices.numel() / batch_size
        micro = _slice_batch(batch, micro_indices)
        sync = (
            training_policy.no_sync()  # type: ignore[union-attr]
            if isinstance(training_policy, DistributedDataParallel)
            and micro_index + 1 < micro_batches
            else nullcontext()
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp_enabled
            else nullcontext()
        )
        with sync, autocast:
            output = _objective(training_policy, micro, rollout, checkpoint, loss_config)
            (output["loss"] * micro_weight).backward()
        for name in totals:
            totals[name] += float(output[name].detach()) * micro_weight
    norm = _gradient_norm(raw_policy)
    optimizer.step()
    totals["gradient_norm"] = float(norm)
    return totals


@torch.no_grad()
def _probe(
    *,
    policy: ModelGradientResidualPolicy,
    batch: Mapping[str, Any],
    rollout: FixedDRTrackerRollout,
    checkpoint,
    loss_config: ModelGradientLossConfig,
    probe_batch_size: int,
    amp_enabled: bool,
) -> dict[str, float]:
    count = min(probe_batch_size, rollout.num_envs)
    indices = torch.arange(count, device=rollout.env.device)
    probe = _slice_batch(batch, indices)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
    )
    policy.eval()
    with autocast:
        normal = _objective(policy, probe, rollout, checkpoint, loss_config)
        base = _objective(_ZeroResidual(), probe, rollout, checkpoint, loss_config)
        shuffled_probe = dict(probe)
        shuffled_probe["latent_sequence"] = probe["latent_sequence"].roll(1, dims=0)
        shuffled = _objective(policy, shuffled_probe, rollout, checkpoint, loss_config)
        normal_residual = policy(probe["tracker_features"], probe["latent_sequence"])
        shuffled_residual = policy(probe["tracker_features"], shuffled_probe["latent_sequence"])
    tracking = normal["tracking_loss"].float()
    return {
        "predicted_tracking_loss": float(tracking),
        "base_predicted_tracking_loss": float(base["tracking_loss"]),
        "predicted_improvement": float(base["tracking_loss"].float() - tracking),
        "latent_shuffle_loss_ratio": float(
            shuffled["tracking_loss"].float() / tracking.clamp_min(1.0e-8)
        ),
        "latent_policy_sensitivity_rms": float(
            (normal_residual - shuffled_residual).float().square().mean().sqrt()
        ),
        "residual_rms": float(normal_residual.float().square().mean().sqrt()),
        "valid_horizon_fraction": float(probe["valid"].float().mean()),
    }


def _save_checkpoint(
    *,
    distributed: DistributedContext,
    output_dir: Path,
    update: int,
    optimizer_steps: int,
    policy: ModelGradientResidualPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_config: Mapping[str, Any],
    history: list[dict[str, Any]],
    numbered: bool,
) -> None:
    state = {
        "architecture_version": "model_gradient_residual_policy_v1",
        "update": int(update),
        "optimizer_steps": int(optimizer_steps),
        "residual_policy": policy.state_dict(),
        "residual_mlp": policy.residual_mlp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "run_config": dict(run_config),
        "history": history,
    }

    def save() -> bool:
        if numbered:
            target = output_dir / f"update_{update:06d}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            temporary.replace(target)
        target = output_dir / "last.pt"
        temporary = output_dir / "last.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(target)
        return True

    _main_process_call(distributed, save)


def _run(args: argparse.Namespace, distributed: DistributedContext) -> Path:
    paths = _main_process_call(distributed, lambda: _prepare_paths(args))
    output_dir = Path(paths["output"])
    tracker_path = Path(paths["tracker"])
    forward_path = Path(paths["forward"])
    resume_path = Path(paths["resume"]) if paths["resume"] else None
    tracker_sha256 = paths["tracker_sha256"]
    device = distributed.device
    rank_seed = args.seed + distributed.rank
    _seed_everything(rank_seed)
    amp_enabled = device.type == "cuda" and args.amp_dtype == "bfloat16"
    if amp_enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp-dtype=bfloat16 requires a CUDA device with BF16 support")

    checkpoint = load_frozen_forward_predictor_checkpoint(
        forward_path,
        device=device,
        expected_tracker_sha256=tracker_sha256,
    )
    rollout = FixedDRTrackerRollout(
        FixedDRRolloutConfig(
            checkpoint_file=str(tracker_path),
            motion_path=args.motion_path,
            motion_file=args.motion_file,
            task_id=args.task_id,
            num_envs=args.num_envs,
            device=str(device),
            seed=rank_seed,
            dynamics_seed=rank_seed,
            world_id_offset=distributed.rank * args.num_envs,
            stochastic_policy=False,
            randomize_initial_episode_phase=args.randomize_initial_episode_phase,
            nominal_fraction=args.nominal_fraction,
        )
    )
    if rollout.checkpoint_task_id != SPV52A_TASK_ID:
        rollout.close()
        raise ValueError(
            f"Model-gradient residual baseline is fixed to {SPV52A_TASK_ID!r}, got "
            f"{rollout.checkpoint_task_id!r}"
        )
    if rollout.predictor_action_transform is None:
        error = rollout.predictor_action_transform_error or "unknown action-chain error"
        rollout.close()
        raise RuntimeError(
            "Model-gradient training requires a differentiable memoryless action transform: "
            f"{error}"
        )

    _seed_everything(rank_seed)
    raw_policy = ModelGradientResidualPolicy(
        rollout.policy_observation_dim,
        checkpoint.config.dynamics_latent_dim,
        hidden_dims=tuple(args.residual_hidden_dims),
        residual_scale=args.residual_scale,
    ).to(device)
    training_policy: nn.Module
    if distributed.enabled:
        ddp_options: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_options.update(device_ids=[device.index], output_device=device.index)
        training_policy = DistributedDataParallel(raw_policy, **ddp_options)
    else:
        training_policy = raw_policy
    optimizer = torch.optim.AdamW(
        raw_policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.updates * args.gradient_steps_per_update,
    )
    start_update = 0
    optimizer_steps = 0
    saved_history: list[dict[str, Any]] = []
    if resume_path is not None:
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("architecture_version") != "model_gradient_residual_policy_v1":
            raise ValueError("Resume checkpoint is not a model-gradient residual policy")
        raw_policy.load_state_dict(resume["residual_policy"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_update = int(resume["update"])
        optimizer_steps = int(resume["optimizer_steps"])
        saved_history = list(resume.get("history", []))

    loss_config = ModelGradientLossConfig(
        horizon_discount=args.horizon_discount,
        huber_delta=args.huber_delta,
        residual_weight=args.residual_weight,
        smoothness_weight=args.smoothness_weight,
        root_position_weight=args.root_position_weight,
        root_orientation_weight=args.root_orientation_weight,
        root_linear_velocity_weight=args.root_linear_velocity_weight,
        root_angular_velocity_weight=args.root_angular_velocity_weight,
        joint_position_weight=args.joint_position_weight,
        joint_velocity_weight=args.joint_velocity_weight,
    )
    run_config = {
        "method": "five-step frozen-predictor model-gradient residual policy",
        "architecture_version": "model_gradient_residual_policy_v1",
        "arguments": vars(args),
        "tracker": {
            "path": str(tracker_path),
            "sha256": tracker_sha256,
            "frozen": True,
            "policy_feature_dim": rollout.policy_observation_dim,
        },
        "forward_predictor": {
            "path": str(forward_path),
            "sha256": checkpoint.sha256,
            "architecture_version": checkpoint.config.architecture_version,
            "frozen": True,
            "context_encoder_frozen": True,
            "latent_dim": checkpoint.config.dynamics_latent_dim,
        },
        "loss": asdict(loss_config),
        "global_num_envs": args.num_envs * distributed.world_size,
        "num_envs_per_rank": args.num_envs,
        "action_contract": rollout.predictor_action_transform.contract,
        "rollout_contract": (
            "five real SPV5-2A observations provide detached tracker features and history-only "
            "latents; their recomputed residual sequence is transformed to physical PD targets "
            "and recursively rolled through the frozen Forward Predictor. Future residual inputs "
            "are teacher-forced real tracker features, while predicted simulator state recurs."
        ),
        "disturbance_contract": "step/interval random pushes are removed; startup DR remains fixed",
        "gradient_contract": (
            "only residual_mlp is trainable; gradients pass through action transform and all five "
            "frozen predictor steps, never through MuJoCo, tracker, or Context Encoder"
        ),
    }
    _main_process_call(
        distributed,
        lambda: (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n"
        ),
    )

    wandb_logger: WandbLogger | None = None

    def initialize_wandb() -> bool:
        nonlocal wandb_logger
        wandb_logger = WandbLogger(
            enabled=args.wandb,
            is_main=True,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=tuple(args.wandb_tag),
            mode=args.wandb_mode,
            output_dir=output_dir,
            config=run_config,
        )
        return True

    history_buffer = PredictorCausalHistory(
        checkpoint,
        num_envs=rollout.num_envs,
        device=device,
        use_bfloat16=args.context_bfloat16,
    )
    current = {
        name: value.detach()
        for name, value in _forward_predictor_snapshot(rollout.env).items()
        if name in {"robot_state", "foot", "contact_force", "contact_binary"}
    }
    metrics_path = output_dir / "metrics.jsonl"
    completed = False
    try:
        _main_process_call(distributed, initialize_wandb)
        if wandb_logger is None:
            wandb_logger = WandbLogger(
                enabled=False,
                is_main=False,
                project=args.wandb_project,
                output_dir=output_dir,
                config=run_config,
            )
        if distributed.is_main:
            print(
                json.dumps({"event": "model_gradient_residual_contract", **run_config}), flush=True
            )
        for update in range(start_update + 1, args.updates + 1):
            collect_start = time.monotonic()
            batch, current = _collect_five_step_batch(
                rollout,
                history_buffer,
                raw_policy,
                current,
            )
            collect_seconds = time.monotonic() - collect_start
            optimization: list[dict[str, float]] = []
            optimize_start = time.monotonic()
            for _ in range(args.gradient_steps_per_update):
                optimization.append(
                    _optimize(
                        training_policy=training_policy,
                        raw_policy=raw_policy,
                        optimizer=optimizer,
                        batch=batch,
                        rollout=rollout,
                        checkpoint=checkpoint,
                        loss_config=loss_config,
                        batch_size=args.batch_size,
                        micro_batch_size=args.micro_batch_size,
                        amp_enabled=amp_enabled,
                    )
                )
                scheduler.step()
                optimizer_steps += 1
            optimize_seconds = time.monotonic() - optimize_start

            if update == 1 or update % args.log_interval == 0:
                probe = _probe(
                    policy=raw_policy,
                    batch=batch,
                    rollout=rollout,
                    checkpoint=checkpoint,
                    loss_config=loss_config,
                    probe_batch_size=args.probe_batch_size,
                    amp_enabled=amp_enabled,
                )
                local = {
                    **{
                        name: sum(item[name] for item in optimization) / len(optimization)
                        for name in optimization[0]
                    },
                    **probe,
                    "real_reward_mean": float(batch["reward"].float().mean()),
                    **_real_tracking_error_metrics(batch["tracking_error"]),
                    **history_buffer.metrics,
                    "collect_seconds": collect_seconds,
                    "optimize_seconds": optimize_seconds,
                }
                metrics = distributed.mean_scalars(local)
                counts = distributed.sum_integers(
                    {
                        "transitions": rollout.transitions,
                        "environments_reset": rollout.environments_reset,
                    }
                )
                record = {
                    "update": update,
                    "optimizer_steps": optimizer_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **counts,
                    "train": metrics,
                }
                if distributed.is_main:
                    saved_history.append(record)
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    wandb_logger.log(
                        {
                            "update": update,
                            "optimizer_steps": optimizer_steps,
                            "optimization/learning_rate": record["learning_rate"],
                            "rollout/transitions": counts["transitions"],
                            **{f"train/{name}": value for name, value in metrics.items()},
                        },
                        step=update,
                    )
                    print(json.dumps(record, sort_keys=True), flush=True)
            if args.checkpoint_interval and update % args.checkpoint_interval == 0:
                _save_checkpoint(
                    distributed=distributed,
                    output_dir=output_dir,
                    update=update,
                    optimizer_steps=optimizer_steps,
                    policy=raw_policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    run_config=run_config,
                    history=saved_history,
                    numbered=True,
                )

        _save_checkpoint(
            distributed=distributed,
            output_dir=output_dir,
            update=args.updates,
            optimizer_steps=optimizer_steps,
            policy=raw_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            run_config=run_config,
            history=saved_history,
            numbered=False,
        )
        completed = True
        return output_dir / "last.pt"
    finally:
        if wandb_logger is not None:
            wandb_logger.finish(exit_code=0 if completed else 1)
        rollout.close()


def run(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/intact-matplotlib")
    torch.set_float32_matmul_precision("high")
    distributed = DistributedContext.initialize(
        requested_device=args.device,
        requested_backend=args.distributed_backend,
    )
    if distributed.device.type == "cuda":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(distributed.local_rank)
    try:
        return _run(args, distributed)
    finally:
        distributed.close()


def main() -> None:
    checkpoint = run(build_parser().parse_args())
    if int(os.environ.get("RANK", "0")) == 0:
        print(checkpoint)


if __name__ == "__main__":
    main()
