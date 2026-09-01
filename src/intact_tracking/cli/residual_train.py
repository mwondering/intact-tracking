"""Train a context-conditioned residual policy through a learned five-step model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.data import (
    ResidualNormalizationStats,
    ResidualOnlineReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.residual_control import ResidualTrunkController
from intact_tracking.residual_model import ResidualTrackingConfig, ResidualTrackingModel
from intact_tracking.residual_objective import ResidualLossConfig, ResidualTrainingObjective
from intact_tracking.rollout import (
    FixedDRRolloutConfig,
    FixedDRTrackerRollout,
    NominalPairRollout,
    NominalPairRolloutConfig,
)
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.rollout.online import TRACKING_ERROR_NAMES
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
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument(
        "--randomize-initial-episode-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Desynchronize timeout resets across vector environments without dropping data.",
    )
    parser.add_argument(
        "--nominal-rollout-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of online vector slots restored to compiled nominal physics. "
            "The default creates an exact half nominal / half fixed-DR rollout."
        ),
    )

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-warmup-steps", type=int, default=10_000)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--rollout-steps-per-update", type=int, default=5)
    parser.add_argument("--gradient-steps-per-update", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=8192)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--warmup-log-interval", type=int, default=10)
    parser.add_argument("--metric-window", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--nominal-pair-batch-size",
        type=int,
        default=None,
        help=(
            "Source samples per model batch replayed in a lightweight no-DR simulator; "
            "defaults to the full batch-size and 0 disables."
        ),
    )
    parser.add_argument(
        "--nominal-restore-atol",
        type=float,
        default=1.0e-5,
        help=(
            "Hard tolerance for immediate nominal state restore and warning threshold "
            "for repeated five-step trajectories."
        ),
    )

    parser.add_argument("--model-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--policy-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--forward-depth", type=int, default=2)
    parser.add_argument("--backward-depth", type=int, default=3)
    parser.add_argument("--policy-depth", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=0.25)

    parser.add_argument("--forward-weight", type=float, default=2.0)
    parser.add_argument("--backward-weight", type=float, default=2.0)
    parser.add_argument("--nominal-pair-weight", type=float, default=1.0)
    parser.add_argument("--nominal-effect-weight", type=float, default=1.0)
    parser.add_argument("--nominal-consistency-weight", type=float, default=1.0)
    parser.add_argument("--tracking-weight", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=0.2)
    parser.add_argument("--residual-smooth-weight", type=float, default=1.0e-3)
    parser.add_argument("--root-position-weight", type=float, default=5.0)
    parser.add_argument("--root-orientation-weight", type=float, default=2.0)
    parser.add_argument("--joint-position-weight", type=float, default=1.0)

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload key losses, tracking errors, gradients, and replay statistics.",
    )
    parser.add_argument("--wandb-project", default="intact-residual-tracking")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.nominal_pair_batch_size is None:
        args.nominal_pair_batch_size = args.batch_size
    positive = (
        "num_envs",
        "warmup_steps",
        "max_warmup_steps",
        "updates",
        "rollout_steps_per_update",
        "gradient_steps_per_update",
        "batch_size",
        "replay_capacity",
        "sample_stride",
        "log_interval",
        "warmup_log_interval",
        "metric_window",
        "context_dim",
        "context_depth",
        "context_heads",
        "hidden_dim",
        "forward_depth",
        "backward_depth",
        "policy_depth",
    )
    invalid = {name: getattr(args, name) for name in positive if getattr(args, name) < 1}
    if invalid:
        raise ValueError(f"Residual training arguments must be positive: {invalid}")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.max_warmup_steps < args.warmup_steps:
        raise ValueError("max-warmup-steps must be at least warmup-steps")
    if args.replay_capacity < args.batch_size:
        raise ValueError("replay-capacity must be at least batch-size")
    if args.nominal_pair_batch_size < 0:
        raise ValueError("nominal-pair-batch-size must be non-negative")
    if args.nominal_pair_batch_size > args.batch_size:
        raise ValueError("nominal-pair-batch-size cannot exceed batch-size")
    if args.nominal_restore_atol <= 0.0:
        raise ValueError("nominal-restore-atol must be positive")
    if not 0.0 < args.nominal_rollout_fraction < 1.0:
        raise ValueError("nominal-rollout-fraction must be strictly between zero and one")
    nominal_worlds = args.num_envs * args.nominal_rollout_fraction
    if abs(nominal_worlds - round(nominal_worlds)) > 1.0e-8:
        raise ValueError("num-envs * nominal-rollout-fraction must be an integer")
    if args.rollout_steps_per_update != 5:
        raise ValueError("action-trunk residual training requires rollout-steps-per-update=5")
    if args.sample_stride != 1:
        raise ValueError(
            "action-trunk residual replay already emits non-overlapping windows and "
            "requires sample-stride=1"
        )
    for name in (
        "model_learning_rate",
        "policy_learning_rate",
        "residual_scale",
        "gradient_clip",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "forward_weight",
        "backward_weight",
        "nominal_pair_weight",
        "nominal_effect_weight",
        "nominal_consistency_weight",
        "tracking_weight",
        "residual_l2_weight",
        "residual_smooth_weight",
        "root_position_weight",
        "root_orientation_weight",
        "joint_position_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _main_process_call(
    distributed: DistributedContext,
    action: Callable[[], T],
) -> T:
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
    replay: ResidualOnlineReplayBuffer,
    global_world_ids: tuple[int, ...],
) -> ResidualNormalizationStats:
    packed = replay.normalizer.packed_statistics(distributed.device)
    distributed.all_reduce_sum(packed)
    return replay.normalizer.snapshot_from_packed(packed, global_world_ids)


class _TrackingAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.total = torch.zeros(len(TRACKING_ERROR_NAMES), dtype=torch.float64, device=device)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, batch: dict[str, torch.Tensor]) -> None:
        values = batch.get("tracking_error")
        if values is None:
            return
        valid = ~batch["reset_boundary"]
        if valid.any():
            selected = values[valid].detach().double()
            self.total.add_(selected.sum(dim=0))
            self.count.add_(selected.size(0))

    def global_mean(self, distributed: DistributedContext) -> dict[str, float]:
        packed = torch.cat((self.total, self.count[None])).clone()
        distributed.all_reduce_sum(packed)
        count = packed[-1].clamp_min(1.0)
        return {
            name: float(value)
            for name, value in zip(
                TRACKING_ERROR_NAMES, (packed[:-1] / count).tolist(), strict=True
            )
        }

    def reset(self) -> None:
        self.total.zero_()
        self.count.zero_()


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).norm(2))


def _tracking_comparison(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in TRACKING_ERROR_NAMES:
        denominator = max(baseline[name], 1.0e-8)
        result[f"{name}_ratio_to_tracker"] = current[name] / denominator
        result[f"{name}_relative_improvement"] = (baseline[name] - current[name]) / denominator
    return result


def _loss_weight_payload(config: ResidualLossConfig) -> dict[str, dict[str, float]]:
    """Expose every multiplicative loss weight in a self-describing log payload."""
    return {
        "objective_terms": {
            "forward_loss": config.forward_weight,
            "backward_loss": config.backward_weight,
            "nominal_pair_loss": config.nominal_pair_weight,
            "nominal_effect_within_pair": config.nominal_effect_weight,
            "nominal_consistency_within_pair": config.nominal_consistency_weight,
            "tracking_loss": config.tracking_weight,
            "residual_l2": config.residual_l2_weight,
            "residual_smooth": config.residual_smooth_weight,
        },
        "pose_terms_shared_by_forward_and_tracking": {
            "root_position": config.root_position_weight,
            "root_orientation": config.root_orientation_weight,
            "joint_position": config.joint_position_weight,
        },
    }


def _wandb_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "update": record["update"],
        "optimizer_steps": record["optimizer_steps"],
        "optimization/policy_optimizer_steps": record.get(
            "policy_optimizer_steps", record["optimizer_steps"]
        ),
        "optimization/learning_rate_model": record["learning_rate_model"],
        "optimization/learning_rate_policy": record["learning_rate_policy"],
        "optimization/gradient_norm": record["gradient_norm"],
        "optimization/model_gradient_norm": record["model_gradient_norm"],
        "optimization/policy_gradient_norm": record["policy_gradient_norm"],
        "replay/size": record["replay_size"],
        "replay/samples_generated": record["samples_generated"],
        "replay/nominal_samples_available": record.get("nominal_samples_available", 0),
        "replay/new_samples_generated": record["new_samples_generated"],
        "replay/storage_bytes": record["replay_storage_bytes"],
        "rollout/transitions": record["transitions"],
        "rollout/environments_reset": record["environments_reset"],
        "rollout/environments_reset_delta": record["new_environments_reset"],
        "rollout/reset_events_delta": record["new_reset_events"],
        "rollout/reset_fraction": record["reset_fraction"],
        "rollout/residual_trunks_generated": record.get("residual_trunks_generated", 0),
        "rollout/residual_trunks_invalidated": record.get("residual_trunks_invalidated", 0),
    }
    payload.update({f"train/{name}": value for name, value in record["train"].items()})
    payload.update(
        {f"tracking/rollout_{name}": value for name, value in record["tracking"].items()}
    )
    payload.update(
        {
            f"tracking/tracker_baseline_{name}": value
            for name, value in record["tracking_baseline"].items()
        }
    )
    payload.update(
        {f"tracking/{name}": value for name, value in record["tracking_comparison"].items()}
    )
    return payload


def _save_checkpoint(
    *,
    distributed: DistributedContext,
    output_dir: Path,
    update: int,
    optimizer_steps: int,
    policy_optimizer_steps: int,
    model: ResidualTrackingModel,
    model_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    model_scheduler: torch.optim.lr_scheduler.LRScheduler,
    policy_scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: ResidualTrackingConfig,
    loss_config: ResidualLossConfig,
    normalization: ResidualNormalizationStats,
    baseline: dict[str, float],
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
        "policy_optimizer_steps": policy_optimizer_steps,
        "model": model.state_dict(),
        "optimizer": {
            "model": model_optimizer.state_dict(),
            "policy": policy_optimizer.state_dict(),
        },
        "scheduler": {
            "model": model_scheduler.state_dict(),
            "policy": policy_scheduler.state_dict(),
        },
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "normalization": asdict(normalization),
        "tracking_baseline": baseline,
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
    world_id_offset = distributed.rank * args.num_envs
    dimensions = RolloutDimensions()
    global_world_ids = tuple(range(args.num_envs * distributed.world_size))

    rollout = FixedDRTrackerRollout(
        FixedDRRolloutConfig(
            checkpoint_file=str(checkpoint_path),
            motion_path=args.motion_path,
            motion_file=args.motion_file,
            task_id=args.task_id,
            num_envs=args.num_envs,
            device=str(device),
            seed=rank_seed,
            world_id_offset=world_id_offset,
            stochastic_policy=args.stochastic_policy,
            randomize_initial_episode_phase=args.randomize_initial_episode_phase,
            nominal_fraction=args.nominal_rollout_fraction,
        )
    )
    nominal_rollout: NominalPairRollout | None = None
    if args.nominal_pair_batch_size:
        try:
            nominal_rollout = NominalPairRollout(
                NominalPairRolloutConfig(
                    checkpoint_file=str(checkpoint_path),
                    motion_path=args.motion_path,
                    motion_file=args.motion_file,
                    task_id=args.task_id,
                    num_envs=args.nominal_pair_batch_size,
                    device=str(device),
                    seed=rank_seed + 100_000,
                    horizon=5,
                    restore_atol=args.nominal_restore_atol,
                    failure_log_file=str(
                        output_dir
                        / f"nominal_repeat_failures_rank_{distributed.rank}.jsonl"
                    ),
                )
            )
        except BaseException:
            rollout.close()
            raise
    replay = ResidualOnlineReplayBuffer(
        num_worlds=args.num_envs,
        policy_observation_dim=rollout.policy_observation_dim,
        context_latent_dim=args.context_dim,
        dimensions=dimensions,
        horizon=5,
        context_chunk_steps=5,
        sample_stride=args.sample_stride,
        context_tokens=16,
        capacity=args.replay_capacity,
        seed=rank_seed,
        world_id_offset=world_id_offset,
        device=device,
    )
    model_config = ResidualTrackingConfig(
        policy_observation_dim=rollout.policy_observation_dim,
        proprio_dim=dimensions.proprio,
        action_dim=dimensions.action,
        state_dim=dimensions.robot_state,
        horizon=5,
        context_chunk_steps=5,
        context_tokens=16,
        context_dim=args.context_dim,
        context_depth=args.context_depth,
        context_heads=args.context_heads,
        hidden_dim=args.hidden_dim,
        forward_depth=args.forward_depth,
        backward_depth=args.backward_depth,
        policy_depth=args.policy_depth,
        residual_scale=args.residual_scale,
    )
    loss_config = ResidualLossConfig(
        forward_weight=args.forward_weight,
        backward_weight=args.backward_weight,
        nominal_pair_weight=args.nominal_pair_weight,
        nominal_effect_weight=args.nominal_effect_weight,
        nominal_consistency_weight=args.nominal_consistency_weight,
        tracking_weight=args.tracking_weight,
        residual_l2_weight=args.residual_l2_weight,
        residual_smooth_weight=args.residual_smooth_weight,
        root_position_weight=args.root_position_weight,
        root_orientation_weight=args.root_orientation_weight,
        joint_position_weight=args.joint_position_weight,
        action_clip=rollout.action_clip,
    )
    loss_weights = _loss_weight_payload(loss_config)
    model = ResidualTrackingModel(model_config).to(device)
    objective = ResidualTrainingObjective(model, loss_config)
    training_module: torch.nn.Module
    if distributed.enabled:
        ddp_options: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": True,
        }
        if device.type == "cuda":
            ddp_options.update(device_ids=[device.index], output_device=device.index)
        training_module = DistributedDataParallel(objective, **ddp_options)
    else:
        training_module = objective

    model_parameters = [
        *model.context_encoder.parameters(),
        *model.forward_predictor.parameters(),
        *model.backward_predictor.parameters(),
    ]
    policy_parameters = list(model.residual_policy.parameters())
    model_optimizer = torch.optim.AdamW(
        model_parameters,
        lr=args.model_learning_rate,
        weight_decay=args.weight_decay,
    )
    policy_optimizer = torch.optim.AdamW(
        policy_parameters,
        lr=args.policy_learning_rate,
        weight_decay=args.weight_decay,
    )
    model_optimizer_steps_target = args.updates * args.gradient_steps_per_update
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, T_max=model_optimizer_steps_target
    )
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer, T_max=args.updates
    )
    run_config = {
        "method": "context-conditioned residual tracking",
        "architecture": {
            "action": "clip(per-step frozen tracker action + cached residual trunk slot)",
            "policy_update": "one current tracker feature -> five residual actions -> frozen-parameter differentiable Forward -> five reconstructed reference pose losses",
            "execution": "generate one five-action residual trunk; execute one slot per real simulator step; invalidate only reset worlds",
            "gradient_routes": {
                "forward_loss": ["context_encoder", "forward_predictor"],
                "backward_loss": ["context_encoder", "backward_predictor"],
                "nominal_pair_loss": ["context_encoder", "forward_predictor"],
                "tracking_loss": ["residual_policy"],
            },
            "optimization_schedule": "model replay updates followed by one recent on-policy trunk update",
            "policy_reference_offsets": {
                "tracker_latent": [0, 1, 2, 3, 4],
                "tracking_targets": [1, 2, 3, 4, 5],
                "explicit_t_plus_5_in_tracker_latent": False,
            },
            "context": "16 x [proprio_before, five total commands, proprio_after]",
            "forward": "causal GRU over five action prefixes; predicts five non-chained pose deltas from the current state",
            "nominal_pair": (
                "collect real interaction contexts from half nominal and half fixed-DR "
                "online worlds; restore every sampled state into separate no-DR physics "
                "and replay the same five actions; train source-context and independently "
                "sampled nominal-context predictions, their DR effect, and nominal-context "
                "consistency"
            ),
            "state": "Forward input and Backward state retain root pose/velocity + joint position/velocity; Forward output and Tracking loss use pose only",
        },
        "arguments": vars(args),
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "loss_weights": loss_weights,
        "rollout": rollout.metadata,
        "nominal_pair_rollout": (
            nominal_rollout.metadata if nominal_rollout is not None else {"enabled": False}
        ),
        "tracker_checkpoint_sha256": tracker_sha256,
        "mjlab_version": importlib.metadata.version("mjlab"),
        "distributed": {
            "enabled": distributed.enabled,
            "world_size": distributed.world_size,
            "rank_seed": "seed + rank",
        },
        "replay": {
            "estimated_storage_bytes_per_rank": replay.estimated_storage_bytes,
            "minimum_steps": replay.minimum_steps,
            "horizon": 5,
            "context_tokens": 16,
            "context_chunk_steps": 5,
        },
    }
    _main_process_call(distributed, lambda: _write_json(output_dir / "run_config.json", run_config))
    if distributed.is_main:
        print(
            json.dumps({"event": "loss_weights", "loss_weights": loss_weights}, sort_keys=True),
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
        if nominal_rollout is not None:
            nominal_rollout.close()
        raise

    history: list[dict[str, Any]] = []
    rolling_loss: deque[dict[str, float]] = deque(maxlen=args.metric_window)
    metrics_path = output_dir / "metrics.jsonl"
    tracker_metrics = _TrackingAccumulator(device)
    update_metrics = _TrackingAccumulator(device)
    optimizer_steps = 0
    policy_optimizer_steps = 0
    normalization: ResidualNormalizationStats | None = None
    completed = False
    try:
        warmup_started = time.monotonic()
        while True:
            nominal_ready = (
                nominal_rollout is None or replay.nominal_sample_count >= 2
            )
            ready = (
                rollout.collector_step >= args.warmup_steps
                and len(replay) >= args.batch_size
                and nominal_ready
            )
            if distributed.all_true(ready):
                break
            if rollout.collector_step >= args.max_warmup_steps:
                raise RuntimeError(
                    "Residual warmup exhausted before every rank had a full causal batch: "
                    f"steps={rollout.collector_step}, replay={len(replay)}"
                )
            batch = rollout.step()
            # The zero-initialized warmup controller is equivalent to a trunk
            # beginning at every episode step divisible by five.
            batch["residual_trunk_step"] = batch["episode_step"].remainder(5)
            batch["residual_world"] = torch.zeros(
                args.num_envs, model_config.context_dim, device=device
            )
            replay.add_step(batch)
            tracker_metrics.add(batch)
            if distributed.is_main and (
                rollout.collector_step == 1
                or rollout.collector_step % args.warmup_log_interval == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "residual_warmup",
                            "env_steps": rollout.collector_step,
                            "replay_size": len(replay),
                            "nominal_replay_size": replay.nominal_sample_count,
                            "samples_generated": replay.total_samples_generated,
                            "elapsed_seconds": time.monotonic() - warmup_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        baseline = tracker_metrics.global_mean(distributed)
        update_metrics.total.copy_(tracker_metrics.total)
        update_metrics.count.copy_(tracker_metrics.count)
        tracker_metrics.reset()
        normalization = _global_normalization(distributed, replay, global_world_ids)
        controller_normalization = normalization

        def controller_context(
            env_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return replay.latest_context(controller_normalization, env_ids)

        trunk_controller = ResidualTrunkController(
            model,
            num_worlds=args.num_envs,
            context_provider=controller_context,
            device=device,
        )
        recent_policy_samples = min(replay.total_samples_generated, replay.capacity)
        previous_samples_generated = 0
        previous_environments_reset = 0
        previous_reset_events = 0
        previous_transitions = 0

        for update in range(1, args.updates + 1):
            if update > 1:
                update_metrics.reset()
                model.eval()
                assert normalization is not None
                controller_normalization = normalization

                for _ in range(args.rollout_steps_per_update):
                    batch = rollout.step(trunk_controller)
                    batch["residual_trunk_step"] = trunk_controller.last_step.clone()
                    batch["residual_world"] = trunk_controller.last_world.clone()
                    recent_policy_samples += replay.add_step(batch)
                    update_metrics.add(batch)
                    # The boundary action belongs to the old episode.  Only its
                    # unconsumed suffix is discarded; the next call starts at
                    # slot zero from the simulator's post-reset observation.
                    trunk_controller.invalidate(batch["reset_boundary"])

            normalization = _global_normalization(distributed, replay, global_world_ids)
            training_module.train()
            model_step_metrics: list[dict[str, float]] = []
            model_gradient_norms: list[float] = []
            for _ in range(args.gradient_steps_per_update):
                train_batch = replay.sample_batch(
                    args.batch_size,
                    normalization,
                    include_nominal_context=nominal_rollout is not None,
                )
                nominal_metrics: dict[str, float] = {}
                if nominal_rollout is not None:
                    pair_count = nominal_rollout.num_envs
                    state_mean = train_batch["state_mean"]
                    state_std = train_batch["state_std"]
                    action_mean = train_batch["action_mean"]
                    action_std = train_batch["action_std"]
                    physical_state = (
                        train_batch["state"][:pair_count, 0] * state_std + state_mean
                    )
                    physical_previous_action = (
                        train_batch["previous_action"][:pair_count] * action_std + action_mean
                    )
                    physical_actions = (
                        train_batch["action"][:pair_count] * action_std + action_mean
                    )
                    with torch.inference_mode():
                        nominal_state, nominal_metrics = nominal_rollout.rollout(
                            physical_state,
                            physical_previous_action,
                            physical_actions,
                            motion_ids=train_batch["motion_id"][:pair_count],
                            motion_steps=train_batch["motion_step"][:pair_count],
                            motion_files=rollout.motion_files,
                        )
                    train_batch["nominal_state"] = (
                        nominal_state - state_mean
                    ) / state_std
                model_optimizer.zero_grad(set_to_none=True)
                model_output = training_module(train_batch, phase="model")
                if not isinstance(model_output, dict) or not torch.isfinite(model_output["loss"]):
                    raise RuntimeError(
                        f"Non-finite residual model objective at step {optimizer_steps + 1}"
                    )
                model_output["loss"].backward()
                model_gradient_norms.append(_gradient_norm(model_parameters))
                model_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model_parameters, args.gradient_clip
                )
                if not torch.isfinite(model_gradient_norm):
                    raise RuntimeError("Residual model produced a non-finite gradient norm")
                model_optimizer.step()
                model_scheduler.step()
                optimizer_steps += 1
                model_step_metrics.append(
                    {
                        name: float(value.detach())
                        for name, value in model_output.items()
                        if value.numel() == 1
                    }
                    | nominal_metrics
                )

            if not distributed.all_true(recent_policy_samples > 0):
                raise RuntimeError(
                    "No complete current-policy residual trunk is available on every rank"
                )
            policy_batch = replay.sample_recent_batch(
                args.batch_size,
                normalization,
                recent_count=recent_policy_samples,
            )
            policy_optimizer.zero_grad(set_to_none=True)
            policy_output = training_module(policy_batch, phase="policy")
            if not isinstance(policy_output, dict) or not torch.isfinite(policy_output["loss"]):
                raise RuntimeError(
                    f"Non-finite residual policy objective at step {policy_optimizer_steps + 1}"
                )
            policy_output["loss"].backward()
            policy_gradient_norm = _gradient_norm(policy_parameters)
            clipped_policy_gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy_parameters, args.gradient_clip
            )
            if not torch.isfinite(clipped_policy_gradient_norm):
                raise RuntimeError("Residual policy produced a non-finite gradient norm")
            policy_optimizer.step()
            policy_scheduler.step()
            policy_optimizer_steps += 1

            local_model_train = {
                name: sum(item[name] for item in model_step_metrics) / len(model_step_metrics)
                for name in model_step_metrics[0]
            }
            local_policy_train = {
                name: float(value.detach())
                for name, value in policy_output.items()
                if value.numel() == 1
            }
            train = distributed.mean_scalars(
                {
                    **{name: value for name, value in local_model_train.items() if name != "loss"},
                    **{name: value for name, value in local_policy_train.items() if name != "loss"},
                    "loss": local_model_train["loss"] + local_policy_train["loss"],
                    "policy_update_applied": 1.0,
                }
            )
            # No future transition may finish a trunk produced by the old
            # policy parameters after this optimizer step.
            trunk_controller.invalidate_all()
            recent_policy_samples = 0
            current_tracking = update_metrics.global_mean(distributed)
            tracking_comparison = _tracking_comparison(current_tracking, baseline)
            counts = distributed.sum_integers(
                {
                    "transitions": rollout.transitions,
                    "replay_size": len(replay),
                    "replay_storage_bytes": replay.storage_bytes,
                    "samples_generated": replay.total_samples_generated,
                    "nominal_samples_available": replay.nominal_sample_count,
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
                {
                    "model_gradient_norm": sum(model_gradient_norms) / len(model_gradient_norms),
                    "policy_gradient_norm": policy_gradient_norm,
                }
            )
            gradient_summary["gradient_norm"] = (
                gradient_summary["model_gradient_norm"] ** 2
                + gradient_summary["policy_gradient_norm"] ** 2
            ) ** 0.5
            record = {
                "update": update,
                "optimizer_steps": optimizer_steps,
                "policy_optimizer_steps": policy_optimizer_steps,
                **counts,
                "new_samples_generated": new_samples,
                "new_environments_reset": new_environments_reset,
                "new_reset_events": new_reset_events,
                "reset_fraction": new_environments_reset / max(new_transitions, 1),
                "learning_rate_model": model_optimizer.param_groups[0]["lr"],
                "learning_rate_policy": policy_optimizer.param_groups[0]["lr"],
                "residual_trunks_generated": trunk_controller.trunks_generated,
                "residual_trunks_invalidated": trunk_controller.trunks_invalidated,
                "loss_weights": loss_weights,
                **gradient_summary,
                "train": train,
                "tracking": current_tracking,
                "tracking_baseline": baseline,
                "tracking_comparison": tracking_comparison,
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
                if update == 1 or update % args.log_interval == 0 or update == args.updates:
                    print(json.dumps(record, sort_keys=True), flush=True)

            if args.checkpoint_interval and update % args.checkpoint_interval == 0:
                _save_checkpoint(
                    distributed=distributed,
                    output_dir=output_dir,
                    update=update,
                    optimizer_steps=optimizer_steps,
                    policy_optimizer_steps=policy_optimizer_steps,
                    model=model,
                    model_optimizer=model_optimizer,
                    policy_optimizer=policy_optimizer,
                    model_scheduler=model_scheduler,
                    policy_scheduler=policy_scheduler,
                    model_config=model_config,
                    loss_config=loss_config,
                    normalization=normalization,
                    baseline=baseline,
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
            policy_optimizer_steps=policy_optimizer_steps,
            model=model,
            model_optimizer=model_optimizer,
            policy_optimizer=policy_optimizer,
            model_scheduler=model_scheduler,
            policy_scheduler=policy_scheduler,
            model_config=model_config,
            loss_config=loss_config,
            normalization=normalization,
            baseline=baseline,
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
        if nominal_rollout is not None:
            nominal_rollout.close()
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
