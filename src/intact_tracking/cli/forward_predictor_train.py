"""Train a recursive nominal Forward Predictor from frozen-tracker rollouts."""

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
    ForwardPredictorNormalizationStats,
    ForwardPredictorReplayBuffer,
    RolloutDimensions,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.forward_predictor import ForwardDynamicsMLP, ForwardPredictorConfig
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
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument(
        "--randomize-initial-episode-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-warmup-steps", type=int, default=10_000)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--rollout-steps-per-update", type=int, default=5)
    parser.add_argument("--gradient-steps-per-update", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2_048)
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
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--history-steps", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=1100)
    parser.add_argument("--residual-blocks", type=int, default=8)
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
        "replay_capacity",
        "fixed_probe_batch_size",
        "log_interval",
        "warmup_log_interval",
        "metric_window",
        "history_steps",
        "hidden_dim",
        "residual_blocks",
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
    if args.rollout_steps_per_update != 5:
        raise ValueError("Forward Predictor collection requires rollout-steps-per-update=5")
    if args.history_steps != 5:
        raise ValueError("Forward Predictor history-steps is fixed to five")
    for name in ("model_learning_rate", "gradient_clip", "huber_delta"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "root_position_weight",
        "root_orientation_weight",
        "root_linear_velocity_weight",
        "root_angular_velocity_weight",
        "joint_position_weight",
        "joint_velocity_weight",
        "recursive_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")


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
    values = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).norm(2))


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
    payload.update({f"fixed_probe/{name}": value for name, value in record["fixed_probe"].items()})
    return payload


def _save_checkpoint(
    *,
    distributed: DistributedContext,
    output_dir: Path,
    update: int,
    optimizer_steps: int,
    model: ForwardDynamicsMLP,
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
            world_id_offset=world_id_offset,
            stochastic_policy=args.stochastic_policy,
            randomize_initial_episode_phase=args.randomize_initial_episode_phase,
            nominal_fraction=1.0,
        )
    )
    replay = ForwardPredictorReplayBuffer(
        num_worlds=args.num_envs,
        dimensions=dimensions,
        capacity=args.replay_capacity,
        history_steps=args.history_steps,
        sampling_mode=args.replay_sampling,
        seed=rank_seed,
        world_id_offset=world_id_offset,
        device=device,
    )
    model_config = ForwardPredictorConfig(
        state_dim=dimensions.robot_state,
        action_dim=dimensions.action,
        history_steps=args.history_steps,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
    )
    loss_config = ForwardPredictorLossConfig(
        root_position_weight=args.root_position_weight,
        root_orientation_weight=args.root_orientation_weight,
        root_linear_velocity_weight=args.root_linear_velocity_weight,
        root_angular_velocity_weight=args.root_angular_velocity_weight,
        joint_position_weight=args.joint_position_weight,
        joint_velocity_weight=args.joint_velocity_weight,
        huber_delta=args.huber_delta,
    )
    model = ForwardDynamicsMLP(model_config).to(device)
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
    )
    optimizer_steps_target = args.updates * args.gradient_steps_per_update
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_steps_target,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_config = {
        "method": "nominal flat-history Forward Predictor v2",
        "architecture": {
            "controller": "frozen tracker",
            "physics": "100% compiled nominal dynamics; DR slots restored to defaults",
            "input": (
                "five flattened causal (71-D state, 29-D action, valid) history frames, "
                "current 71-D state, and current 29-D action"
            ),
            "transition": "shared residual MLP predicts normalized 70-D full-state delta",
            "rollout": "predicted state is recursively fed back for all five actions",
            "excluded": ["context_encoder", "transformer", "residual_policy", "backward"],
            "normalization": "state/action/delta statistics frozen immediately after warmup",
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
        },
        "replay": {
            "estimated_storage_bytes_per_rank": replay.estimated_storage_bytes,
            "horizon": 5,
            "history_steps": args.history_steps,
            "sampling": args.replay_sampling,
        },
        "objective_weights": {
            "teacher_forced_weight": 1.0,
            "recursive_weight": args.recursive_weight,
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
            ready = rollout.collector_step >= args.warmup_steps and len(replay) >= max(
                args.batch_size, args.fixed_probe_batch_size
            )
            if distributed.all_true(ready):
                break
            if rollout.collector_step >= args.max_warmup_steps:
                raise RuntimeError(
                    "Forward Predictor warmup exhausted before every rank had a full batch: "
                    f"steps={rollout.collector_step}, replay={len(replay)}"
                )
            replay.add_step(rollout.step())
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
        fixed_probe_batch = replay.sample_batch(args.fixed_probe_batch_size, normalization)
        fixed_train_batch = (
            replay.sample_batch(args.batch_size, normalization)
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
                    replay.add_step(rollout.step())

            training_module.train()
            step_metrics: list[dict[str, float]] = []
            gradient_norms: list[float] = []
            for _ in range(args.gradient_steps_per_update):
                train_batch = (
                    fixed_train_batch
                    if fixed_train_batch is not None
                    else replay.sample_batch(args.batch_size, normalization)
                )
                optimizer.zero_grad(set_to_none=True)
                model_output = training_module(
                    train_batch,
                    recursive_weight=args.recursive_weight,
                )
                if not isinstance(model_output, dict) or not torch.isfinite(model_output["loss"]):
                    raise RuntimeError(
                        f"Non-finite Forward Predictor objective at step {optimizer_steps + 1}"
                    )
                model_output["loss"].backward()
                gradient_norms.append(_gradient_norm(parameters))
                clipped_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
                if not torch.isfinite(clipped_norm):
                    raise RuntimeError("Forward Predictor produced a non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                metrics = {
                    name: float(value.detach())
                    for name, value in model_output.items()
                    if value.numel() == 1
                }
                metrics["sample_motion_count"] = float(
                    torch.unique(train_batch["motion_id"]).numel()
                )
                step_metrics.append(metrics)

            local_train = {
                name: sum(item[name] for item in step_metrics) / len(step_metrics)
                for name in step_metrics[0]
            }
            train = distributed.mean_scalars(local_train)
            model.eval()
            with torch.inference_mode():
                local_probe = {
                    name: float(value.detach())
                    for name, value in objective(
                        fixed_probe_batch,
                        recursive_weight=args.recursive_weight,
                    ).items()
                    if value.numel() == 1
                }
                local_probe["sample_motion_count"] = float(
                    torch.unique(fixed_probe_batch["motion_id"]).numel()
                )
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
                if update == 1 or update % args.log_interval == 0 or update == args.updates:
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
