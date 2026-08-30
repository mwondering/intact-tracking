"""Train INTACT online from a frozen tracker and live fixed-DR MJLab worlds."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from intact_tracking.data import OnlineReplayBuffer, RolloutDimensions
from intact_tracking.model import SIGReg, TrackingINTACT, TrackingINTACTConfig
from intact_tracking.objective import INTACTLossConfig, intact_objective
from intact_tracking.rollout import FixedDRRolloutConfig, FixedDRTrackerRollout
from intact_tracking.rollout.mjlab_adapter import _sha256


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=120,
        help="Minimum vector-environment steps before the first optimizer update.",
    )
    parser.add_argument(
        "--max-warmup-steps",
        type=int,
        default=10_000,
        help="Fail if reset boundaries prevent a full replay batch by this step.",
    )
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--rollout-steps-per-update", type=int, default=5)
    parser.add_argument("--gradient-steps-per-update", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=8192)
    parser.add_argument("--log-interval", type=int, default=10)
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
    replay: OnlineReplayBuffer,
    rollout: FixedDRTrackerRollout,
    tracker_sha256: str,
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
        "normalization": asdict(replay.normalization()),
        "tracker": {
            "checkpoint_path": str(rollout.checkpoint_path),
            "checkpoint_sha256": tracker_sha256,
            "task_id": rollout.checkpoint_task_id,
            "frozen": True,
        },
        "online_state": {
            "env_steps": rollout.collector_step,
            "transitions": rollout.transitions,
            "replay_size": len(replay),
            "samples_generated": replay.total_samples_generated,
            "synchronous_resets": rollout.synchronous_resets,
            "dr_invariance_checks": rollout.dr_invariance_checks,
            "motion_ids_seen": sorted(rollout.motion_ids_seen),
            "domain_randomization_contract": rollout.metadata["domain_randomization_contract"],
        },
    }


def _save_checkpoint(
    output_dir: Path,
    state: dict[str, Any],
    normalization: Any,
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


def run(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() or (output_dir / "run_config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run in {output_dir}")

    checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    tracker_sha256 = _sha256(checkpoint_path)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dimensions = RolloutDimensions()
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
        seed=args.seed,
        stochastic_policy=args.stochastic_policy,
    )
    replay = OnlineReplayBuffer(
        num_worlds=args.num_envs,
        dimensions=dimensions,
        block_size=args.block_size,
        horizon=args.horizon,
        context_tokens=16,
        capacity=args.replay_capacity,
        seed=args.seed,
    )

    rollout = FixedDRTrackerRollout(rollout_config)
    try:
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
            },
            "arguments": vars(args),
            "model": asdict(model_config),
            "loss": asdict(loss_config),
            "rollout": rollout.metadata,
            "tracker_checkpoint_sha256": tracker_sha256,
            "mjlab_version": importlib.metadata.version("mjlab"),
            "replay": {
                "storage": "in-memory raw transitions and causal samples",
                "capacity": args.replay_capacity,
                "minimum_full_context_steps_per_world": replay.minimum_steps,
                "normalization": "running statistics over the live stream",
                "validation_split": None,
            },
        }
        _write_json(output_dir / "run_config.json", run_config)

        while rollout.collector_step < args.warmup_steps or len(replay) < args.batch_size:
            replay.add_step(rollout.step())
            if rollout.collector_step >= args.max_warmup_steps and len(replay) < args.batch_size:
                raise RuntimeError(
                    "Online warmup reached max-warmup-steps without a full replay batch: "
                    f"steps={rollout.collector_step}, replay={len(replay)}, "
                    f"batch_size={args.batch_size}. Motions may reset before the "
                    f"{args.horizon * args.block_size}-step query is complete."
                )

        model = TrackingINTACT(model_config).to(device)
        sigreg = SIGReg(num_proj=args.sigreg_projections).to(device)
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

        for update in range(1, args.updates + 1):
            if update > 1:
                for _ in range(args.rollout_steps_per_update):
                    replay.add_step(rollout.step())

            model.train()
            train_metrics: list[dict[str, float]] = []
            gradient_norms: list[float] = []
            for _ in range(args.gradient_steps_per_update):
                batch = _to_device(replay.sample_batch(args.batch_size), device)
                optimizer.zero_grad(set_to_none=True)
                output = intact_objective(
                    model,
                    batch,
                    loss_config=loss_config,
                    sigreg=sigreg,
                )
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

            record = {
                "update": update,
                "optimizer_steps": optimizer_steps,
                "env_steps": rollout.collector_step,
                "transitions": rollout.transitions,
                "replay_size": len(replay),
                "samples_generated": replay.total_samples_generated,
                "synchronous_resets": rollout.synchronous_resets,
                "dr_invariance_checks": rollout.dr_invariance_checks,
                "motions_seen": len(rollout.motion_ids_seen),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": sum(gradient_norms) / len(gradient_norms),
                "train": _average(train_metrics),
            }
            history.append(record)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if update == 1 or update % args.log_interval == 0 or update == args.updates:
                print(json.dumps(record, sort_keys=True), flush=True)

            checkpoint_due = bool(args.checkpoint_interval) and (
                update % args.checkpoint_interval == 0
            )
            if checkpoint_due:
                state = _checkpoint_state(
                    update=update,
                    optimizer_steps=optimizer_steps,
                    model=model,
                    sigreg=sigreg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_config=model_config,
                    loss_config=loss_config,
                    replay=replay,
                    rollout=rollout,
                    tracker_sha256=tracker_sha256,
                )
                _save_checkpoint(
                    output_dir,
                    state,
                    replay.normalization(),
                    history,
                    numbered=True,
                )

        final_state = _checkpoint_state(
            update=args.updates,
            optimizer_steps=optimizer_steps,
            model=model,
            sigreg=sigreg,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            loss_config=loss_config,
            replay=replay,
            rollout=rollout,
            tracker_sha256=tracker_sha256,
        )
        _save_checkpoint(
            output_dir,
            final_state,
            replay.normalization(),
            history,
            numbered=False,
        )
        return output_dir / "last.pt"
    finally:
        rollout.close()


def main() -> None:
    checkpoint = run(build_parser().parse_args())
    print(checkpoint)


if __name__ == "__main__":
    main()
