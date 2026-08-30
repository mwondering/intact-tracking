"""Train INTACT online from frozen trackers and live fixed-DR MJLab worlds."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.data import (
    NormalizationStats,
    OnlineReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.model import SIGReg, TrackingINTACT, TrackingINTACTConfig
from intact_tracking.objective import INTACTLossConfig, TrackingINTACTObjective
from intact_tracking.rollout import FixedDRRolloutConfig, FixedDRTrackerRollout
from intact_tracking.rollout.mjlab_adapter import _sha256

T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-path")
    motion.add_argument("--motion-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Vector environments per process/GPU (global count is WORLD_SIZE times this).",
    )
    parser.add_argument(
        "--device",
        help="Single-process device only; torchrun assigns cuda:LOCAL_RANK automatically.",
    )
    parser.add_argument(
        "--distributed-backend",
        choices=("nccl", "gloo"),
        help="Optional torchrun backend override (default: NCCL on CUDA, Gloo on CPU).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=120,
        help="Minimum vector-environment steps per rank before the first update.",
    )
    parser.add_argument(
        "--max-warmup-steps",
        type=int,
        default=10_000,
        help="Fail if every rank lacks a full local replay batch by this step.",
    )
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--rollout-steps-per-update", type=int, default=5)
    parser.add_argument("--gradient-steps-per-update", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per rank (global DDP batch is WORLD_SIZE times this).",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=8192,
        help="Causal replay sample capacity per rank.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--warmup-log-interval",
        type=int,
        default=10,
        help="Print replay/reset progress every N warmup vector steps.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--forward-history", type=int, default=3)
    parser.add_argument("--forward-depth", type=int, default=6)
    parser.add_argument("--forward-heads", type=int, default=8)
    parser.add_argument("--actor-hidden-dim", type=int, default=1024)
    parser.add_argument("--actor-depth", type=int, default=3)
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--forward-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--physical-weight", type=float, default=0.1)
    parser.add_argument("--goal-weight", type=float, default=0.05)
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _scalar_metrics(output: dict[str, torch.Tensor]) -> dict[str, float]:
    names = (
        "loss",
        "forward_loss",
        "sigreg_loss",
        "action_loss",
        "physical_nll",
        "goal_nll",
        "physical_mae",
        "goal_mae",
    )
    return {name: float(output[name].detach()) for name in names}


def _average(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {name: sum(item[name] for item in metrics) / len(metrics) for name in metrics[0]}


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "num_envs",
        "warmup_steps",
        "max_warmup_steps",
        "updates",
        "rollout_steps_per_update",
        "gradient_steps_per_update",
        "batch_size",
        "replay_capacity",
        "log_interval",
        "warmup_log_interval",
        "block_size",
        "horizon",
        "embed_dim",
        "encoder_hidden_dim",
        "context_depth",
        "context_heads",
        "forward_history",
        "forward_depth",
        "forward_heads",
        "actor_hidden_dim",
        "actor_depth",
        "sigreg_projections",
    )
    invalid = {name: getattr(args, name) for name in positive if getattr(args, name) < 1}
    if invalid:
        raise ValueError(f"Online training arguments must be positive: {invalid}")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.max_warmup_steps < args.warmup_steps:
        raise ValueError("max-warmup-steps must be at least warmup-steps")
    if args.replay_capacity < args.batch_size:
        raise ValueError("replay-capacity must be at least batch-size")
    for name in (
        "learning_rate",
        "forward_weight",
        "sigreg_weight",
        "physical_weight",
        "goal_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if args.learning_rate == 0:
        raise ValueError("learning-rate must be positive")
    if args.gradient_clip <= 0:
        raise ValueError("gradient-clip must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _main_process_call(
    distributed: DistributedContext,
    action: Callable[[], T],
) -> T:
    """Run filesystem work on rank 0 and propagate its result or failure."""
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
        raise RuntimeError(f"Rank-0 operation failed ({details['error_type']}): {details['error']}")
    return payload["value"]


def _prepare_paths(args: argparse.Namespace) -> dict[str, str]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() or (output_dir / "run_config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run in {output_dir}")
    checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "tracker_sha256": _sha256(checkpoint_path),
    }


def _global_normalization(
    distributed: DistributedContext,
    replay: OnlineReplayBuffer,
    global_num_envs: int,
) -> NormalizationStats:
    packed = replay.normalizer.packed_statistics(distributed.device)
    distributed.all_reduce_sum(packed)
    return replay.normalizer.snapshot_from_packed(
        packed,
        tuple(range(global_num_envs)),
    )


def _aggregate_online_counts(
    distributed: DistributedContext,
    replay: OnlineReplayBuffer,
    rollout: FixedDRTrackerRollout,
) -> dict[str, int]:
    totals = distributed.sum_integers(
        {
            "rank_env_steps": rollout.collector_step,
            "transitions": rollout.transitions,
            "replay_size": len(replay),
            "samples_generated": replay.total_samples_generated,
            "reset_events": rollout.reset_events,
            "environments_reset": rollout.environments_reset,
            "synchronous_resets": rollout.synchronous_resets,
            "dr_invariance_checks": rollout.dr_invariance_checks,
            "motions_seen": len(rollout.motion_ids_seen),
        }
    )
    totals["env_steps"] = totals.pop("rank_env_steps") // distributed.world_size
    return totals


def _checkpoint_online_state(
    distributed: DistributedContext,
    replay: OnlineReplayBuffer,
    rollout: FixedDRTrackerRollout,
) -> dict[str, Any]:
    rank_states = distributed.all_gather_object(
        {
            "rank": distributed.rank,
            "device": str(distributed.device),
            "world_ids": list(replay.world_ids),
            "env_steps": rollout.collector_step,
            "transitions": rollout.transitions,
            "replay_size": len(replay),
            "samples_generated": replay.total_samples_generated,
            "reset_events": rollout.reset_events,
            "environments_reset": rollout.environments_reset,
            "synchronous_resets": rollout.synchronous_resets,
            "dr_invariance_checks": rollout.dr_invariance_checks,
            "motion_ids_seen": sorted(rollout.motion_ids_seen),
        }
    )
    counts = _aggregate_online_counts(distributed, replay, rollout)
    return {
        **counts,
        "ranks": rank_states,
        "domain_randomization_contract": rollout.metadata["domain_randomization_contract"],
    }


def _checkpoint_state(
    *,
    update: int,
    optimizer_steps: int,
    model: TrackingINTACT,
    sigreg: SIGReg,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: TrackingINTACTConfig,
    loss_config: INTACTLossConfig,
    normalization: NormalizationStats,
    online_state: dict[str, Any],
    rollout: FixedDRTrackerRollout,
    tracker_sha256: str,
    distributed: DistributedContext,
    batch_size_per_rank: int,
) -> dict[str, Any]:
    return {
        "update": update,
        "optimizer_steps": optimizer_steps,
        "model": model.state_dict(),
        "sigreg": sigreg.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "normalization": asdict(normalization),
        "distributed": {
            "enabled": distributed.enabled,
            "backend": distributed.backend,
            "world_size": distributed.world_size,
            "batch_size_per_rank": batch_size_per_rank,
            "global_batch_size": batch_size_per_rank * distributed.world_size,
        },
        "tracker": {
            "checkpoint_path": str(rollout.checkpoint_path),
            "checkpoint_sha256": tracker_sha256,
            "task_id": rollout.checkpoint_task_id,
            "frozen": True,
        },
        "online_state": online_state,
    }


def _save_checkpoint(
    output_dir: Path,
    state: dict[str, Any],
    normalization: NormalizationStats,
    history: list[dict[str, Any]],
    *,
    numbered: bool,
) -> None:
    normalization.to_json(output_dir / "normalization.json")
    _write_json(output_dir / "history.json", history)
    if numbered:
        target = output_dir / f"update_{state['update']:06d}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(target)
    target = output_dir / "last.pt"
    temporary = output_dir / "last.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(target)


def _save_current_checkpoint(
    *,
    distributed: DistributedContext,
    output_dir: Path,
    update: int,
    optimizer_steps: int,
    model: TrackingINTACT,
    sigreg: SIGReg,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: TrackingINTACTConfig,
    loss_config: INTACTLossConfig,
    normalization: NormalizationStats,
    replay: OnlineReplayBuffer,
    rollout: FixedDRTrackerRollout,
    tracker_sha256: str,
    history: list[dict[str, Any]],
    batch_size_per_rank: int,
    numbered: bool,
) -> None:
    online_state = _checkpoint_online_state(distributed, replay, rollout)

    def save() -> None:
        state = _checkpoint_state(
            update=update,
            optimizer_steps=optimizer_steps,
            model=model,
            sigreg=sigreg,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            loss_config=loss_config,
            normalization=normalization,
            online_state=online_state,
            rollout=rollout,
            tracker_sha256=tracker_sha256,
            distributed=distributed,
            batch_size_per_rank=batch_size_per_rank,
        )
        _save_checkpoint(
            output_dir,
            state,
            normalization,
            history,
            numbered=numbered,
        )

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
    dimensions = RolloutDimensions()
    global_num_envs = args.num_envs * distributed.world_size
    global_batch_size = args.batch_size * distributed.world_size
    world_id_offset = distributed.rank * args.num_envs

    model_config = TrackingINTACTConfig(
        observation_dim=dimensions.observation,
        proprio_dim=dimensions.proprio,
        action_dim=dimensions.action,
        action_block_size=args.block_size,
        context_tokens=16,
        embed_dim=args.embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        context_depth=args.context_depth,
        context_heads=args.context_heads,
        forward_history=args.forward_history,
        forward_depth=args.forward_depth,
        forward_heads=args.forward_heads,
        forward_mlp_dim=4 * args.embed_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_depth=args.actor_depth,
    )
    loss_config = INTACTLossConfig(
        forward_weight=args.forward_weight,
        sigreg_weight=args.sigreg_weight,
        physical_weight=args.physical_weight,
        goal_weight=args.goal_weight,
    )
    rollout_config = FixedDRRolloutConfig(
        checkpoint_file=str(checkpoint_path),
        motion_path=args.motion_path,
        motion_file=args.motion_file,
        task_id=args.task_id,
        num_envs=args.num_envs,
        device=str(device),
        seed=rank_seed,
        world_id_offset=world_id_offset,
        stochastic_policy=args.stochastic_policy,
    )
    replay = OnlineReplayBuffer(
        num_worlds=args.num_envs,
        dimensions=dimensions,
        block_size=args.block_size,
        horizon=args.horizon,
        context_tokens=16,
        capacity=args.replay_capacity,
        seed=rank_seed,
        world_id_offset=world_id_offset,
    )

    startup_started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "startup",
                "stage": "creating_rollout",
                "rank": distributed.rank,
                "device": str(device),
                "num_envs_per_rank": args.num_envs,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    rollout = FixedDRTrackerRollout(rollout_config)
    print(
        json.dumps(
            {
                "event": "startup",
                "stage": "rollout_ready",
                "rank": distributed.rank,
                "elapsed_seconds": time.monotonic() - startup_started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        print(
            json.dumps(
                {
                    "event": "startup",
                    "stage": "waiting_for_all_ranks",
                    "rank": distributed.rank,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        rollout_metadata = distributed.all_gather_object(rollout.metadata)
        run_config = {
            "method": "INTACT-online",
            "mode": "pure online rollout-and-update",
            "training_architecture": {
                "forward": "LeWM-style causal Forward Predictor",
                "physical_intent": "attached z[t+1] - z[t]",
                "goal_intent": "stop-gradient z_goal - z[t]",
                "intent_actor": "one shared four-slot Gaussian actor",
                "context_tokens": 16,
                "context_injection": "shared latent FiLM; no fifth actor slot",
                "distributed": "full-model synchronous data parallelism",
            },
            "arguments": vars(args),
            "model": asdict(model_config),
            "loss": asdict(loss_config),
            "rollout": rollout_metadata[0],
            "rollout_ranks": rollout_metadata,
            "distributed": {
                "enabled": distributed.enabled,
                "backend": distributed.backend,
                "world_size": distributed.world_size,
                "launcher": "torchrun" if distributed.enabled else "single process",
                "num_envs_per_rank": args.num_envs,
                "global_num_envs": global_num_envs,
                "batch_size_per_rank": args.batch_size,
                "global_batch_size": global_batch_size,
                "replay_capacity_per_rank": args.replay_capacity,
                "global_replay_capacity": args.replay_capacity * distributed.world_size,
                "rank_seed": "seed + global_rank",
                "world_id": "global_rank * num_envs + local_env_id",
                "gradient_reduction": "DDP mean over all ranks",
                "normalization": "globally summed sufficient statistics",
                "checkpoint_owner": "rank 0",
            },
            "tracker_checkpoint_sha256": tracker_sha256,
            "mjlab_version": importlib.metadata.version("mjlab"),
            "replay": {
                "storage": "rank-local in-memory raw transitions and causal samples",
                "capacity_per_rank": args.replay_capacity,
                "minimum_full_context_steps_per_world": replay.minimum_steps,
                "normalization": "global running statistics over every live rank stream",
                "context_scope": "same fixed-DR world only; never crosses ranks",
                "validation_split": None,
            },
        }
        _main_process_call(
            distributed,
            lambda: _write_json(output_dir / "run_config.json", run_config),
        )

        warmup_started = time.monotonic()
        if distributed.is_main:
            print(
                json.dumps(
                    {
                        "event": "warmup_start",
                        "minimum_steps": replay.minimum_steps,
                        "requested_warmup_steps": args.warmup_steps,
                        "batch_size_per_rank": args.batch_size,
                        "num_envs_per_rank": args.num_envs,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        while True:
            locally_ready = (
                rollout.collector_step >= args.warmup_steps and len(replay) >= args.batch_size
            )
            if distributed.all_true(locally_ready):
                break
            if rollout.collector_step >= args.max_warmup_steps:
                replay_sizes = distributed.all_gather_object(len(replay))
                raise RuntimeError(
                    "Online warmup reached max-warmup-steps before every rank had a "
                    f"full local batch: steps={rollout.collector_step}, "
                    f"replay_sizes={replay_sizes}, batch_size_per_rank={args.batch_size}. "
                    "Motions may reset before the causal query is complete."
                )
            replay.add_step(rollout.step())
            if (
                rollout.collector_step == 1
                or rollout.collector_step % args.warmup_log_interval == 0
            ):
                rank_progress = distributed.all_gather_object(
                    {
                        "rank": distributed.rank,
                        "env_steps": rollout.collector_step,
                        "replay_size": len(replay),
                        "samples_generated": replay.total_samples_generated,
                        "reset_events": rollout.reset_events,
                        "environments_reset": rollout.environments_reset,
                    }
                )
                if distributed.is_main:
                    elapsed = max(time.monotonic() - warmup_started, 1e-9)
                    print(
                        json.dumps(
                            {
                                "event": "warmup_progress",
                                "env_steps": rollout.collector_step,
                                "elapsed_seconds": elapsed,
                                "vector_steps_per_second": rollout.collector_step / elapsed,
                                "replay_size_min": min(
                                    item["replay_size"] for item in rank_progress
                                ),
                                "replay_size_max": max(
                                    item["replay_size"] for item in rank_progress
                                ),
                                "samples_generated_global": sum(
                                    item["samples_generated"] for item in rank_progress
                                ),
                                "reset_events_global": sum(
                                    item["reset_events"] for item in rank_progress
                                ),
                                "environments_reset_global": sum(
                                    item["environments_reset"] for item in rank_progress
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        final_warmup_progress = distributed.all_gather_object(
            {
                "replay_size": len(replay),
                "samples_generated": replay.total_samples_generated,
            }
        )
        if distributed.is_main:
            print(
                json.dumps(
                    {
                        "event": "warmup_complete",
                        "env_steps": rollout.collector_step,
                        "elapsed_seconds": time.monotonic() - warmup_started,
                        "replay_size_min": min(
                            item["replay_size"] for item in final_warmup_progress
                        ),
                        "replay_size_max": max(
                            item["replay_size"] for item in final_warmup_progress
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        model = TrackingINTACT(model_config).to(device)
        sigreg = SIGReg(num_proj=args.sigreg_projections).to(device)
        objective = TrackingINTACTObjective(
            model,
            loss_config=loss_config,
            sigreg=sigreg,
        )
        training_module: torch.nn.Module
        if distributed.enabled:
            ddp_options: dict[str, Any] = {"broadcast_buffers": False}
            if device.type == "cuda":
                ddp_options.update(
                    device_ids=[device.index],
                    output_device=device.index,
                )
            training_module = DistributedDataParallel(objective, **ddp_options)
        else:
            training_module = objective

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        optimizer_steps_target = args.updates * args.gradient_steps_per_update
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimizer_steps_target
        )
        history: list[dict[str, Any]] = []
        metrics_path = output_dir / "metrics.jsonl"
        optimizer_steps = 0
        normalization = _global_normalization(distributed, replay, global_num_envs)

        for update in range(1, args.updates + 1):
            if update > 1:
                for _ in range(args.rollout_steps_per_update):
                    replay.add_step(rollout.step())

            normalization = _global_normalization(distributed, replay, global_num_envs)
            training_module.train()
            train_metrics: list[dict[str, float]] = []
            gradient_norms: list[float] = []
            for _ in range(args.gradient_steps_per_update):
                batch = _to_device(
                    replay.sample_batch(args.batch_size, normalization=normalization),
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                output = training_module(batch)
                if not isinstance(output, dict):
                    raise TypeError("INTACT objective module must return a metric dictionary")
                if not torch.isfinite(output["loss"]):
                    raise RuntimeError(f"Non-finite loss at optimizer step {optimizer_steps + 1}")
                output["loss"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at optimizer step {optimizer_steps + 1}"
                    )
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                train_metrics.append(_scalar_metrics(output))
                gradient_norms.append(float(gradient_norm.detach()))

            global_train_metrics = distributed.mean_scalars(_average(train_metrics))
            global_gradient_norm = distributed.mean_scalars(
                {"gradient_norm": sum(gradient_norms) / len(gradient_norms)}
            )["gradient_norm"]
            counts = _aggregate_online_counts(distributed, replay, rollout)
            record = {
                "update": update,
                "optimizer_steps": optimizer_steps,
                **counts,
                "num_envs_per_rank": args.num_envs,
                "global_num_envs": global_num_envs,
                "batch_size_per_rank": args.batch_size,
                "global_batch_size": global_batch_size,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": global_gradient_norm,
                "train": global_train_metrics,
            }
            if distributed.is_main:
                history.append(record)
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if update == 1 or update % args.log_interval == 0 or update == args.updates:
                    print(json.dumps(record, sort_keys=True), flush=True)

            checkpoint_due = bool(args.checkpoint_interval) and (
                update % args.checkpoint_interval == 0
            )
            if checkpoint_due:
                _save_current_checkpoint(
                    distributed=distributed,
                    output_dir=output_dir,
                    update=update,
                    optimizer_steps=optimizer_steps,
                    model=model,
                    sigreg=sigreg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_config=model_config,
                    loss_config=loss_config,
                    normalization=normalization,
                    replay=replay,
                    rollout=rollout,
                    tracker_sha256=tracker_sha256,
                    history=history,
                    batch_size_per_rank=args.batch_size,
                    numbered=True,
                )

        _save_current_checkpoint(
            distributed=distributed,
            output_dir=output_dir,
            update=args.updates,
            optimizer_steps=optimizer_steps,
            model=model,
            sigreg=sigreg,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            loss_config=loss_config,
            normalization=normalization,
            replay=replay,
            rollout=rollout,
            tracker_sha256=tracker_sha256,
            history=history,
            batch_size_per_rank=args.batch_size,
            numbered=False,
        )
        distributed.barrier()
        return output_dir / "last.pt"
    finally:
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
