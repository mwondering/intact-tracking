"""Train a contrastive dynamics-context predictor from tracker rollouts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.data import (
    ForwardPredictorNormalizationStats,
    ForwardPredictorReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.forward_predictor import (
    ForwardDynamicsTransformer,
    ForwardPredictorConfig,
)
from intact_tracking.forward_predictor_objective import (
    DEFAULT_RECURSIVE_WEIGHT,
    ForwardPredictorLossConfig,
    ForwardPredictorObjective,
)
from intact_tracking.rollout import FixedDRRolloutConfig, FixedDRTrackerRollout
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.wandb_logger import WandbLogger

T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-path")
    motion.add_argument("--motion-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument(
        "--randomize-initial-episode-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--nominal-fraction",
        type=float,
        default=0.0,
        help="Fraction of vector worlds restored to nominal physics; the default keeps all DR.",
    )

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-warmup-steps", type=int, default=10_000)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--rollout-steps-per-update", type=int, default=5)
    parser.add_argument("--gradient-steps-per-update", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4_096,
        help="Effective per-rank batch accumulated before each optimizer step.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=512,
        help="Maximum per-rank batch processed by one forward/backward pass.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="CUDA training precision; bfloat16 keeps model parameters in float32.",
    )
    parser.add_argument("--replay-capacity", type=int, default=262_144)
    parser.add_argument(
        "--replay-sampling",
        choices=("motion_balanced", "uniform"),
        default="motion_balanced",
    )
    parser.add_argument("--fixed-probe-batch-size", type=int, default=512)
    parser.add_argument(
        "--fixed-batch-overfit",
        action="store_true",
        help="Freeze one replay batch after warmup and optimize only that batch.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--warmup-log-interval", type=int, default=10)
    parser.add_argument("--metric-window", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--model-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--context-history-steps", type=int, default=100)
    parser.add_argument(
        "--dynamics-classes",
        type=int,
        default=128,
        help="Fixed startup-DR prototypes repeated in each synchronized motion family.",
    )
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--dynamics-latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--recursive-weight",
        type=float,
        default=DEFAULT_RECURSIVE_WEIGHT,
        help="Constant five-step recursive-loss weight used from the first optimizer step.",
    )

    parser.add_argument("--root-position-weight", type=float, default=1.0)
    parser.add_argument("--root-orientation-weight", type=float, default=1.0)
    parser.add_argument("--root-linear-velocity-weight", type=float, default=1.0)
    parser.add_argument("--root-angular-velocity-weight", type=float, default=1.0)
    parser.add_argument("--joint-position-weight", type=float, default=1.0)
    parser.add_argument("--joint-velocity-weight", type=float, default=1.0)
    parser.add_argument("--foot-weight", type=float, default=1.0)
    parser.add_argument("--contact-force-weight", type=float, default=1.0)
    parser.add_argument("--contact-binary-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.01)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--contrastive-negative-distance",
        type=float,
        default=1.25,
        help=(
            "Minimum RMS distance between normalized dynamics parameters for two "
            "different worlds to be treated as negatives."
        ),
    )
    parser.add_argument(
        "--contrastive-hard-negative-count",
        type=int,
        default=255,
        help="Maximum theta-far negatives; exact motion/phase cohorts are selected first.",
    )
    parser.add_argument(
        "--contrastive-phase-distance-scale",
        type=float,
        default=50.0,
        help=(
            "Motion-step difference equivalent to one unit of normalized state/action "
            "distance during hard-negative ranking."
        ),
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project", default="intact-forward-predictor")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "num_envs",
        "warmup_steps",
        "max_warmup_steps",
        "updates",
        "rollout_steps_per_update",
        "gradient_steps_per_update",
        "batch_size",
        "micro_batch_size",
        "replay_capacity",
        "fixed_probe_batch_size",
        "log_interval",
        "warmup_log_interval",
        "metric_window",
        "history_steps",
        "context_history_steps",
        "dynamics_classes",
        "transformer_dim",
        "transformer_depth",
        "transformer_heads",
        "context_dim",
        "context_depth",
        "context_heads",
        "dynamics_latent_dim",
        "contrastive_hard_negative_count",
    )
    invalid = {name: getattr(args, name) for name in positive if getattr(args, name) < 1}
    if invalid:
        raise ValueError(f"Forward Predictor arguments must be positive: {invalid}")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.max_warmup_steps < args.warmup_steps:
        raise ValueError("max-warmup-steps must be at least warmup-steps")
    if args.replay_capacity < max(args.batch_size, args.fixed_probe_batch_size):
        raise ValueError("replay-capacity must fit both training and fixed-probe batches")
    if args.micro_batch_size > args.batch_size:
        raise ValueError("micro-batch-size must not exceed effective batch-size")
    if args.rollout_steps_per_update != 5:
        raise ValueError("Forward Predictor collection requires rollout-steps-per-update=5")
    if args.history_steps != 10:
        raise ValueError("Forward Predictor history-steps is fixed to ten")
    if args.context_history_steps < args.history_steps:
        raise ValueError("context-history-steps must be at least history-steps")
    if args.num_envs % args.dynamics_classes:
        raise ValueError("dynamics-classes must divide num-envs")
    if args.nominal_fraction:
        raise ValueError("Fixed dynamics classes require --nominal-fraction=0")
    if not 0.0 <= args.nominal_fraction <= 1.0:
        raise ValueError("nominal-fraction must be in [0, 1]")
    for name in (
        "model_learning_rate",
        "huber_delta",
        "contrastive_temperature",
        "contrastive_phase_distance_scale",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    for name in (
        "weight_decay",
        "root_position_weight",
        "root_orientation_weight",
        "root_linear_velocity_weight",
        "root_angular_velocity_weight",
        "joint_position_weight",
        "joint_velocity_weight",
        "foot_weight",
        "contact_force_weight",
        "contact_binary_weight",
        "contrastive_weight",
        "contrastive_negative_distance",
        "recursive_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if args.contrastive_weight > 0.0:
        if args.num_envs < 2:
            raise ValueError("Contrastive training requires at least two vector worlds")
        if args.nominal_fraction == 1.0:
            raise ValueError(
                "Contrastive training needs randomized dynamics; use --contrastive-weight=0 "
                "for a 100% nominal rollout"
            )
        invalid_blocks = {
            "batch_size": args.batch_size % args.micro_batch_size,
            "fixed_probe_batch_size": args.fixed_probe_batch_size % args.micro_batch_size,
            "micro_batch_size/dynamics_classes": (args.micro_batch_size % args.dynamics_classes),
        }
        invalid_blocks = {name: value for name, value in invalid_blocks.items() if value}
        if invalid_blocks or args.micro_batch_size // args.dynamics_classes < 2:
            raise ValueError(
                "Contrastive training requires complete micro-batches with at least two "
                f"views of every dynamics class: {invalid_blocks}"
            )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() or (output / "run_config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run in {output}")
    checkpoint = Path(args.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "output_dir": str(output),
        "checkpoint_path": str(checkpoint),
        "tracker_sha256": _sha256(checkpoint),
    }


def _global_normalization(
    distributed: DistributedContext,
    replay: ForwardPredictorReplayBuffer,
    global_world_ids: tuple[int, ...],
) -> ForwardPredictorNormalizationStats:
    packed = replay.normalizer.packed_statistics(distributed.device)
    distributed.all_reduce_sum(packed)
    return replay.normalizer.snapshot_from_packed(packed, global_world_ids)


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.nn.utils.get_total_norm(gradients, norm_type=2.0, foreach=True))


def _scalar_tensors_to_floats(values: dict[str, torch.Tensor]) -> dict[str, float]:
    """Transfer scalar diagnostics to the host with one synchronization."""

    scalars = [(name, value) for name, value in values.items() if value.numel() == 1]
    if not scalars:
        return {}
    packed = torch.stack([value.detach().float().reshape(()) for _, value in scalars])
    host_values = packed.cpu().tolist()
    return {name: float(value) for (name, _), value in zip(scalars, host_values, strict=True)}


_BATCH_FIELDS = frozenset(
    {
        "state",
        "action",
        "history_state",
        "history_action",
        "foot",
        "history_foot",
        "contact_force",
        "contact_binary",
        "history_contact_force",
        "history_contact_binary",
        "history_valid",
        "world_id",
        "episode_id",
        "episode_step",
        "motion_id",
        "motion_step",
        "dynamics_id",
        "motion_group_id",
        "cohort_id",
        "context_full",
        "privileged_dynamics",
    }
)


def _slice_predictor_batch(
    batch: dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    """Slice sample fields while sharing the rank-global normalization tensors."""

    return {
        name: value[start:stop] if name in _BATCH_FIELDS else value for name, value in batch.items()
    }


def _loss_weight_payload(config: ForwardPredictorLossConfig) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(config).items()}


def _wandb_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "update": record["update"],
        "optimizer_steps": record["optimizer_steps"],
        "optimization/learning_rate_model": record["learning_rate_model"],
        "optimization/gradient_norm": record["gradient_norm"],
        "replay/size": record["replay_size"],
        "replay/samples_generated": record["samples_generated"],
        "replay/new_samples_generated": record["new_samples_generated"],
        "replay/storage_bytes": record["replay_storage_bytes"],
        "rollout/transitions": record["transitions"],
        "rollout/environments_reset": record["environments_reset"],
        "rollout/environments_reset_delta": record["new_environments_reset"],
        "rollout/reset_events_delta": record["new_reset_events"],
        "rollout/reset_fraction": record["reset_fraction"],
    }
    payload.update({f"train/{name}": value for name, value in record["train"].items()})
    payload.update(
        {
            f"optimization_train/{name}": value
            for name, value in record["optimization_train"].items()
        }
    )
    payload.update({f"fixed_probe/{name}": value for name, value in record["fixed_probe"].items()})
    return payload


def _save_checkpoint(
    *,
    distributed: DistributedContext,
    output_dir: Path,
    update: int,
    optimizer_steps: int,
    model: ForwardDynamicsTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: ForwardPredictorConfig,
    loss_config: ForwardPredictorLossConfig,
    normalization: ForwardPredictorNormalizationStats,
    history: list[dict[str, Any]],
    rollout: FixedDRTrackerRollout,
    tracker_sha256: str,
    wandb_logger: WandbLogger,
    numbered: bool,
) -> None:
    state = {
        "architecture_version": model_config.architecture_version,
        "update": update,
        "optimizer_steps": optimizer_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "normalization": asdict(normalization),
        "privileged_dynamics": {
            "names": list(rollout.privileged_dynamics_names),
            "ignored_startup_events": list(rollout.ignored_privileged_startup_events),
            "inference_contract": (
                "history_only; simulator parameters are used only to mine theta-far training "
                "negatives and are never model inputs or prediction targets"
            ),
        },
        "tracker": {
            "checkpoint_path": str(rollout.checkpoint_path),
            "checkpoint_sha256": tracker_sha256,
            "task_id": rollout.checkpoint_task_id,
            "frozen": True,
        },
        "wandb": {"id": wandb_logger.id, "url": wandb_logger.url},
    }

    def save() -> None:
        normalization.to_json(output_dir / "normalization.json")
        _write_json(output_dir / "history.json", history)
        if numbered:
            target = output_dir / f"update_{update:06d}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            temporary.replace(target)
        target = output_dir / "last.pt"
        temporary = output_dir / "last.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(target)

    _main_process_call(distributed, save)


def _run(args: argparse.Namespace, distributed: DistributedContext) -> Path:
    rank_seed = args.seed + distributed.rank
    _seed_everything(rank_seed)
    torch.set_float32_matmul_precision("high")
    paths = _main_process_call(distributed, lambda: _prepare_paths(args))
    output_dir = Path(paths["output_dir"])
    checkpoint_path = Path(paths["checkpoint_path"])
    tracker_sha256 = paths["tracker_sha256"]
    device = distributed.device
    amp_enabled = device.type == "cuda" and args.amp_dtype == "bfloat16"
    if amp_enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp-dtype=bfloat16 requires a CUDA device with BF16 support")
    world_id_offset = distributed.rank * args.num_envs
    global_world_ids = tuple(range(args.num_envs * distributed.world_size))
    dimensions = RolloutDimensions()

    rollout = FixedDRTrackerRollout(
        FixedDRRolloutConfig(
            checkpoint_file=str(checkpoint_path),
            motion_path=args.motion_path,
            motion_file=args.motion_file,
            task_id=args.task_id,
            num_envs=args.num_envs,
            device=str(device),
            seed=rank_seed,
            dynamics_seed=args.seed,
            world_id_offset=world_id_offset,
            stochastic_policy=args.stochastic_policy,
            randomize_initial_episode_phase=args.randomize_initial_episode_phase,
            nominal_fraction=args.nominal_fraction,
            dynamics_classes=args.dynamics_classes,
        )
    )
    prototype_hashes = distributed.all_gather_object(rollout.dynamics_prototype_sha256)
    if len(set(prototype_hashes)) != 1:
        rollout.close()
        raise RuntimeError("DDP ranks constructed different fixed dynamics prototypes")
    if rollout.predictor_action_transform is None:
        error = rollout.predictor_action_transform_error or "unknown action-chain error"
        rollout.close()
        raise RuntimeError(
            "Forward Predictor requires an external memoryless policy-action to physical "
            f"PD-target transform: {error}"
        )
    replay = ForwardPredictorReplayBuffer(
        num_worlds=args.num_envs,
        dimensions=dimensions,
        privileged_dim=len(rollout.privileged_dynamics_names),
        capacity=args.replay_capacity,
        history_steps=args.history_steps,
        context_history_steps=args.context_history_steps,
        dynamics_classes=args.dynamics_classes,
        sampling_mode=args.replay_sampling,
        seed=rank_seed,
        world_id_offset=world_id_offset,
        device=device,
    )
    # Simulator construction intentionally uses a rank-independent dynamics
    # seed. Restore the rank seed before model initialization and replay draws.
    _seed_everything(rank_seed)
    model_config = ForwardPredictorConfig(
        state_dim=dimensions.robot_state,
        action_dim=dimensions.action,
        history_steps=args.history_steps,
        context_history_steps=args.context_history_steps,
        transformer_dim=args.transformer_dim,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        context_dim=args.context_dim,
        context_depth=args.context_depth,
        context_heads=args.context_heads,
        dynamics_latent_dim=args.dynamics_latent_dim,
        dropout=args.dropout,
    )
    loss_config = ForwardPredictorLossConfig(
        root_position_weight=args.root_position_weight,
        root_orientation_weight=args.root_orientation_weight,
        root_linear_velocity_weight=args.root_linear_velocity_weight,
        root_angular_velocity_weight=args.root_angular_velocity_weight,
        joint_position_weight=args.joint_position_weight,
        joint_velocity_weight=args.joint_velocity_weight,
        foot_weight=args.foot_weight,
        contact_force_weight=args.contact_force_weight,
        contact_binary_weight=args.contact_binary_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_negative_distance=args.contrastive_negative_distance,
        contrastive_hard_negative_count=args.contrastive_hard_negative_count,
        contrastive_phase_distance_scale=args.contrastive_phase_distance_scale,
        huber_delta=args.huber_delta,
    )
    contrastive_enabled = loss_config.contrastive_weight > 0.0
    model = ForwardDynamicsTransformer(model_config).to(device)
    objective = ForwardPredictorObjective(model, loss_config)
    training_module: torch.nn.Module
    if distributed.enabled:
        ddp_options: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_options.update(device_ids=[device.index], output_device=device.index)
        training_module = DistributedDataParallel(objective, **ddp_options)
    else:
        training_module = objective

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.model_learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    optimizer_steps_target = args.updates * args.gradient_steps_per_update
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_steps_target,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_config = {
        "method": "100-frame grouped-dynamics Context Forward Predictor v10",
        "architecture": {
            "controller": "frozen tracker",
            "physics": (
                "128 fixed startup-DR prototypes repeated across synchronized motion groups"
            ),
            "input": (
                "ten historical and one current predictor token; each contains 71-D robot state, "
                "8-D simulator foot height/velocity, 6-D contact force, 2-D contact "
                "state and the external 29-D physical PD joint target"
            ),
            "transition": (
                "shared causal Transformer predicts normalized 70-D robot-state delta, "
                "normalized 8-D next foot state, normalized 6-D next contact force and "
                "2-D next-contact logits, conditioned on a history-inferred dynamics latent"
            ),
            "context": (
                "a separate encoder maps 100 completed (state, physical PD target, next-state) "
                "interactions to z; same-dynamics contexts are positives; each motion/phase "
                "cohort supplies one context from every fixed dynamics class as hard negatives; "
                "reset-padded contexts train prediction but are excluded from contrastive pairs; "
                "there is no theta encoder or theta decoder"
            ),
            "rollout": (
                "predicted robot/foot/contact state is recursively fed back for five targets; "
                "the training/model hot path contains no articulated foot FK"
            ),
            "excluded": ["residual_policy", "backward"],
            "normalization": (
                "robot state, physical target, simulator foot, contact force, robot delta and "
                "training-only theta mining statistics frozen immediately after warmup"
            ),
        },
        "arguments": vars(args),
        "model": asdict(model_config),
        "model_parameters": parameter_count,
        "loss": asdict(loss_config),
        "rollout": rollout.metadata,
        "tracker_checkpoint_sha256": tracker_sha256,
        "mjlab_version": importlib.metadata.version("mjlab"),
        "distributed": {
            "enabled": distributed.enabled,
            "world_size": distributed.world_size,
            "rank_seed": "seed + rank",
            "effective_batch_size_global": args.batch_size * distributed.world_size,
        },
        "optimization": {
            "effective_batch_size_per_rank": args.batch_size,
            "micro_batch_size_per_rank": args.micro_batch_size,
            "gradient_accumulation": True,
            "gradient_clipping": False,
            "amp_dtype": "bfloat16" if amp_enabled else "float32",
            "fused_adamw": device.type == "cuda",
            "diagnostics_interval": args.log_interval,
        },
        "replay": {
            "estimated_storage_bytes_per_rank": replay.estimated_storage_bytes,
            "horizon": 5,
            "history_steps": args.history_steps,
            "context_history_steps": args.context_history_steps,
            "dynamics_classes": args.dynamics_classes,
            "history_storage": "time archive reconstructed at sample time",
            "sampling": args.replay_sampling,
            "dynamics_paired_batches": args.contrastive_weight > 0.0,
            "positive_pair_preference": "same dynamics across motion/phase cohorts",
            "negative_pair_preference": "theta-far exact motion/phase cohort first",
        },
        "objective_weights": {
            "teacher_forced_weight": 1.0,
            "recursive_weight": args.recursive_weight,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_temperature": args.contrastive_temperature,
            "contrastive_negative_distance": args.contrastive_negative_distance,
            "contrastive_hard_negative_count": args.contrastive_hard_negative_count,
            "contrastive_phase_distance_scale": args.contrastive_phase_distance_scale,
        },
    }
    _main_process_call(distributed, lambda: _write_json(output_dir / "run_config.json", run_config))
    if distributed.is_main:
        print(
            json.dumps(
                {
                    "event": "forward_predictor_contract",
                    "model_parameters": parameter_count,
                    "loss_weights": _loss_weight_payload(loss_config),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    wandb_logger: WandbLogger | None = None

    def initialize_wandb_on_main() -> bool:
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

    try:
        _main_process_call(distributed, initialize_wandb_on_main)
        if wandb_logger is None:
            wandb_logger = WandbLogger(
                enabled=False,
                is_main=False,
                project=args.wandb_project,
                output_dir=output_dir,
                config=run_config,
            )
    except BaseException:
        rollout.close()
        raise

    history: list[dict[str, Any]] = []
    rolling_loss: deque[dict[str, float]] = deque(maxlen=args.metric_window)
    metrics_path = output_dir / "metrics.jsonl"
    optimizer_steps = 0
    completed = False
    try:
        warmup_started = time.monotonic()
        while True:
            required_batch = max(args.batch_size, args.fixed_probe_batch_size)
            ready = rollout.collector_step >= args.warmup_steps and len(replay) >= required_batch
            if ready and contrastive_enabled:
                ready = replay.can_sample_dynamics_pairs(
                    required_batch,
                    contrastive_block_size=args.micro_batch_size,
                )
            if distributed.all_true(ready):
                break
            if rollout.collector_step >= args.max_warmup_steps:
                raise RuntimeError(
                    "Forward Predictor warmup exhausted before every rank had a full batch: "
                    f"steps={rollout.collector_step}, replay={len(replay)}"
                )
            replay.add_step(rollout.step(predictor_only=True))
            if distributed.is_main and (
                rollout.collector_step == 1
                or rollout.collector_step % args.warmup_log_interval == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "forward_predictor_warmup",
                            "env_steps": rollout.collector_step,
                            "replay_size": len(replay),
                            "samples_generated": replay.total_samples_generated,
                            "elapsed_seconds": time.monotonic() - warmup_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        normalization = _global_normalization(distributed, replay, global_world_ids)
        replay.normalizer.freeze()
        fixed_probe_batch = replay.sample_batch(
            args.fixed_probe_batch_size,
            normalization,
            paired_dynamics=contrastive_enabled,
            contrastive_block_size=args.micro_batch_size,
        )
        fixed_train_batch = (
            replay.sample_batch(
                args.batch_size,
                normalization,
                paired_dynamics=contrastive_enabled,
                contrastive_block_size=args.micro_batch_size,
            )
            if args.fixed_batch_overfit
            else None
        )
        if distributed.is_main:
            print(
                json.dumps(
                    {
                        "event": "forward_predictor_normalization_frozen",
                        "collector_step": rollout.collector_step,
                        "fixed_batch_overfit": args.fixed_batch_overfit,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        previous_samples_generated = 0
        previous_environments_reset = 0
        previous_reset_events = 0
        previous_transitions = 0

        for update in range(1, args.updates + 1):
            if update > 1 and not args.fixed_batch_overfit:
                model.eval()
                for _ in range(args.rollout_steps_per_update):
                    replay.add_step(rollout.step(predictor_only=True))

            training_module.train()
            step_losses: list[dict[str, torch.Tensor]] = []
            gradient_norms: list[float] = []
            for _ in range(args.gradient_steps_per_update):
                train_batch = (
                    fixed_train_batch
                    if fixed_train_batch is not None
                    else replay.sample_batch(
                        args.batch_size,
                        normalization,
                        paired_dynamics=contrastive_enabled,
                        contrastive_block_size=args.micro_batch_size,
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated_losses: dict[str, torch.Tensor] = {}
                micro_starts = range(0, args.batch_size, args.micro_batch_size)
                for start in micro_starts:
                    stop = min(start + args.micro_batch_size, args.batch_size)
                    micro_batch = _slice_predictor_batch(train_batch, start, stop)
                    fraction = (stop - start) / args.batch_size
                    final_micro_batch = stop == args.batch_size
                    sync_context = (
                        training_module.no_sync()
                        if isinstance(training_module, DistributedDataParallel)
                        and not final_micro_batch
                        else nullcontext()
                    )
                    autocast_context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if amp_enabled
                        else nullcontext()
                    )
                    with sync_context, autocast_context:
                        model_output = training_module(
                            micro_batch,
                            recursive_weight=args.recursive_weight,
                            compute_metrics=False,
                            validate_batch=False,
                        )
                        (model_output["loss"] * fraction).backward()
                    for name in (
                        "loss",
                        "teacher_loss",
                        "recursive_loss",
                        "contrastive_loss",
                    ):
                        weighted = model_output[name].detach().float() * fraction
                        accumulated_losses[name] = (
                            accumulated_losses.get(name, torch.zeros_like(weighted)) + weighted
                        )

                gradient_norm = _gradient_norm(parameters)
                if not np.isfinite(gradient_norm):
                    raise RuntimeError("Forward Predictor produced a non-finite gradient norm")
                gradient_norms.append(gradient_norm)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                step_losses.append(accumulated_losses)

            should_report = update == 1 or update % args.log_interval == 0 or update == args.updates
            if should_report:
                local_optimization_train = _scalar_tensors_to_floats(
                    {
                        name: torch.stack([item[name] for item in step_losses]).mean()
                        for name in step_losses[0]
                    }
                )
                optimization_train = distributed.mean_scalars(local_optimization_train)
                training_module.eval()
                diagnostic_batch = _slice_predictor_batch(
                    train_batch,
                    0,
                    min(args.fixed_probe_batch_size, args.batch_size),
                )
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if amp_enabled
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    local_train = _scalar_tensors_to_floats(
                        objective(
                            diagnostic_batch,
                            recursive_weight=args.recursive_weight,
                            validate_batch=False,
                        )
                    )
                    local_probe = _scalar_tensors_to_floats(
                        objective(
                            fixed_probe_batch,
                            recursive_weight=args.recursive_weight,
                            validate_batch=False,
                        )
                    )
                local_train["sample_motion_count"] = float(
                    torch.unique(diagnostic_batch["motion_id"]).numel()
                )
                local_probe["sample_motion_count"] = float(
                    torch.unique(fixed_probe_batch["motion_id"]).numel()
                )
                train = distributed.mean_scalars(local_train)
                fixed_probe = distributed.mean_scalars(local_probe)
                counts = distributed.sum_integers(
                    {
                        "transitions": rollout.transitions,
                        "replay_size": len(replay),
                        "replay_storage_bytes": replay.storage_bytes,
                        "samples_generated": replay.total_samples_generated,
                        "reset_events": rollout.reset_events,
                        "environments_reset": rollout.environments_reset,
                    }
                )
                new_samples = counts["samples_generated"] - previous_samples_generated
                previous_samples_generated = counts["samples_generated"]
                new_environments_reset = counts["environments_reset"] - previous_environments_reset
                previous_environments_reset = counts["environments_reset"]
                new_reset_events = counts["reset_events"] - previous_reset_events
                previous_reset_events = counts["reset_events"]
                new_transitions = counts["transitions"] - previous_transitions
                previous_transitions = counts["transitions"]
                gradient_summary = distributed.mean_scalars(
                    {"gradient_norm": sum(gradient_norms) / len(gradient_norms)}
                )
                record = {
                    "update": update,
                    "optimizer_steps": optimizer_steps,
                    **counts,
                    "new_samples_generated": new_samples,
                    "new_environments_reset": new_environments_reset,
                    "new_reset_events": new_reset_events,
                    "reset_fraction": new_environments_reset / max(new_transitions, 1),
                    "learning_rate_model": optimizer.param_groups[0]["lr"],
                    "normalization_frozen": True,
                    "fixed_batch_overfit": args.fixed_batch_overfit,
                    "loss_weights": _loss_weight_payload(loss_config),
                    **gradient_summary,
                    "optimization_train": optimization_train,
                    "train": train,
                    "fixed_probe": fixed_probe,
                }
                if distributed.is_main:
                    rolling_loss.append(train)
                    record["window"] = {
                        name: {
                            "mean": float(np.mean([item[name] for item in rolling_loss])),
                            "std": float(np.std([item[name] for item in rolling_loss])),
                        }
                        for name in train
                    }
                    history.append(record)
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    wandb_logger.log(_wandb_payload(record), step=update)
                    print(json.dumps(record, sort_keys=True), flush=True)

            if args.checkpoint_interval and update % args.checkpoint_interval == 0:
                _save_checkpoint(
                    distributed=distributed,
                    output_dir=output_dir,
                    update=update,
                    optimizer_steps=optimizer_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_config=model_config,
                    loss_config=loss_config,
                    normalization=normalization,
                    history=history,
                    rollout=rollout,
                    tracker_sha256=tracker_sha256,
                    wandb_logger=wandb_logger,
                    numbered=True,
                )

        _save_checkpoint(
            distributed=distributed,
            output_dir=output_dir,
            update=args.updates,
            optimizer_steps=optimizer_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            loss_config=loss_config,
            normalization=normalization,
            history=history,
            rollout=rollout,
            tracker_sha256=tracker_sha256,
            wandb_logger=wandb_logger,
            numbered=False,
        )
        distributed.barrier()
        completed = True
        return output_dir / "last.pt"
    finally:
        wandb_logger.finish(exit_code=0 if completed else 1)
        rollout.close()


def run(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    distributed = DistributedContext.initialize(
        requested_device=args.device,
        requested_backend=args.distributed_backend,
    )
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
