"""Train nominal50 Memory350 with twice the attention depth and matched original controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from intact_tracking.data import (
    ForwardPredictorNormalizationStats,
    RolloutDimensions,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.memory350_scale_model import (
    Memory350ScalePredictor as ForwardDynamicsTransformer,
    Memory350ScaleConfig as ForwardPredictorConfig,
    parameter_counts,
)
from intact_tracking.memory350_scale_nominal_comparison import (
    PAIRED_UPDATES, reference_contract, reference_normalization, reference_probes,
)
from intact_tracking.memory350_replay import Memory350ReplayBuffer as ForwardPredictorReplayBuffer
from intact_tracking.memory350_objective import Memory350Objective as ForwardPredictorObjective
from intact_tracking.memory350_nominal_rollout import (
    NominalMemory350TrackerRollout as FixedDRTrackerRollout,
    NominalMemory350RolloutConfig as FixedDRRolloutConfig,
)
from intact_tracking.forward_predictor_objective import (
    DEFAULT_RECURSIVE_WEIGHT,
    ForwardPredictorLossConfig,
)
from intact_tracking.forward_predictor_schedule import extend_cosine_schedule
from intact_tracking.rollout import (
    NominalPairRollout,
    NominalPairRolloutConfig,
)
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.wandb_logger import WandbLogger

T = TypeVar("T")


def _validate_resume_loss_config(previous, actual):
    """Ordinary resume requires identical losses; explicit tuning entries override this hook."""
    if previous != actual:
        raise ValueError("Resume model/loss configuration changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-path")
    motion.add_argument("--motion-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--comparison-reference-dir", help="Memory350 run supplying matched normalization and held-out validation")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int, default=0)
    limb_profile = parser.add_mutually_exclusive_group()
    limb_profile.add_argument("--limb-payload-only", action="store_true")
    limb_profile.add_argument("--tracker-dr-plus-limb-payload", action="store_true",
                              help="Preserve the frozen tracker's full DR and add four independent U(0,4) kg loads")
    parser.add_argument("--validation-worlds", type=int, default=0,
                        help="Last N independently sampled worlds per rank are excluded from training replay")
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--resume", help="Restore optimizer/model/stats; simulator and replay restart")
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument(
        "--randomize-initial-episode-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--nominal-fraction",
        type=float,
        default=0.5,
        help="Batch-A fraction restored to nominal physics; fixed to one half for this task.",
    )
    parser.add_argument("--nominal-restore-atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--payload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a fixed per-world rigid hand payload on top of checkpoint startup DR.",
    )
    parser.add_argument("--payload-body-name", default="right_wrist_yaw_link")
    parser.add_argument(
        "--payload-mass-range-kg",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1.0, 3.0),
    )
    parser.add_argument(
        "--payload-position-body-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.12, 0.0, 0.0),
        help="Payload COM in the selected body frame.",
    )
    parser.add_argument(
        "--payload-size-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.10, 0.08, 0.08),
        help="Cuboid side lengths used to compute payload rotational inertia.",
    )

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-warmup-steps", type=int, default=10_000)
    parser.add_argument("--updates", type=int, default=10_000,
                        help="Update limit; with --until-user-stop, only the cosine decay horizon")
    parser.add_argument("--until-user-stop", action="store_true",
                        help="No update cap or automatic plateau stopping; stop only on a signal")
    parser.add_argument("--stop-after-updates", type=int,
                        help="Exact update cap, independent of the --updates cosine horizon")
    parser.add_argument("--continuation-min-learning-rate", type=float, default=1e-5)
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
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--model-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--context-history-steps", type=int, default=100)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--chunk-depth", type=int, default=2)
    parser.add_argument("--memory-depth", type=int, default=4)
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
    parser.add_argument("--representation-weight", type=float, default=0.01)
    parser.add_argument("--representation-relation-weight", type=float, default=1.0,
                        help="Multiplier on cross-world response geometry, inside representation-weight")
    parser.add_argument("--response-distance-scale", type=float, default=1.0)
    parser.add_argument(
        "--positive-offset-steps",
        type=int,
        default=5,
        help="Exact same-world context-window shift used as the local positive.",
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
    parser.add_argument("--bounded-smoke", action="store_true",
                        help="Implementation check only: allow up to three updates on a single motion")
    parser.set_defaults(num_envs=8192, validation_worlds=128, seed=717,
                        nominal_fraction=0.5, payload=False, tracker_dr_plus_limb_payload=True,
                        context_history_steps=50, context_depth=4, representation_relation_weight=2.0,
                        response_distance_scale=0.75, updates=8000, until_user_stop=True,
                        batch_size=1024, micro_batch_size=256, warmup_steps=500,
                        checkpoint_interval=250, validation_interval=100)
    return parser


def _limb_experiment(args: argparse.Namespace) -> bool:
    return args.limb_payload_only or getattr(args, "tracker_dr_plus_limb_payload", False)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        raise ValueError("stop-after-updates must be positive")
    if args.bounded_smoke:
        if not args.motion_file or args.updates > 3:
            raise ValueError("A bounded smoke must use a single motion and at most three updates")
        args.until_user_stop = False
    if args.context_history_steps != 50 or not args.tracker_dr_plus_limb_payload or args.limb_payload_only:
        raise ValueError("Memory350 scaling fixes short50 and frozen-tracker DR plus four-limb loads")
    if (args.chunk_depth, args.memory_depth, args.context_depth) != (2, 4, 4):
        raise ValueError("The encoder scale experiment fixes attention depths to 2/4/4")
    if not args.bounded_smoke and not args.comparison_reference_dir:
        raise ValueError("Formal scale training requires the original Memory350 reference directory")
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
        "history_steps",
        "context_history_steps",
        "transformer_dim",
        "transformer_depth",
        "transformer_heads",
        "context_dim",
        "context_depth",
        "context_heads",
        "dynamics_latent_dim",
        "positive_offset_steps",
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
    if _limb_experiment(args):
        if args.nominal_fraction != 0.5 or args.payload:
            raise ValueError("Nominal Memory350 requires --nominal-fraction 0.5 --no-payload")
        if args.validation_worlds % 2:
            raise ValueError("Validation worlds must split equally between nominal and DR")
        if not 0 < args.validation_worlds < args.num_envs:
            raise ValueError("Four-limb profiles require disjoint validation worlds")
    elif args.nominal_fraction != 0.5:
        raise ValueError("This task requires --nominal-fraction=0.5")
    if args.until_user_stop and not 0 < args.continuation_min_learning_rate <= args.model_learning_rate:
        raise ValueError("Manual stopping requires a positive continuation LR no larger than the initial LR")
    if args.num_envs % 2:
        raise ValueError("num-envs must be even for the 50/50 nominal/DR A batch")
    if args.positive_offset_steps != 5:
        raise ValueError("This task requires --positive-offset-steps=5")
    for name in (
        "model_learning_rate",
        "huber_delta",
        "response_distance_scale",
        "nominal_restore_atol",
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
        "representation_weight",
        "representation_relation_weight",
        "recursive_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if args.representation_weight > 0.0 and args.num_envs < 2:
        raise ValueError("Representation training requires at least two vector worlds")
    if args.payload:
        payload_min, payload_max = args.payload_mass_range_kg
        if not 0.0 < payload_min <= payload_max:
            raise ValueError("payload-mass-range-kg must satisfy 0 < MIN <= MAX")
        if not args.payload_body_name.strip():
            raise ValueError("payload-body-name must not be empty")
        if not all(np.isfinite(value) for value in args.payload_position_body_m):
            raise ValueError("payload-position-body-m values must be finite")
        if not all(np.isfinite(value) and value > 0.0 for value in args.payload_size_m):
            raise ValueError("payload-size-m values must be positive and finite")


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
    if not args.resume and ((output / "last.pt").exists() or (output / "run_config.json").exists()):
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
        "nominal_state",
        "action",
        "history_state",
        "history_action",
        "positive_current_state",
        "positive_history_state",
        "positive_history_action",
        "positive_history_valid",
        "positive_pair_valid",
        "foot",
        "history_foot",
        "contact_force",
        "contact_binary",
        "history_contact_force",
        "history_contact_binary",
        "history_valid",
        "world_id",
        "motion_id",
        "is_nominal",
        "context_full",
        "history_next_state", "positive_history_next_state",
        "memory_interactions", "memory_valid", "positive_memory_interactions", "positive_memory_valid",
    }
)

_CORE_PROBE_METRICS = (
    "one_step_nmse",
    "nominal_five_step_nmse",
    "dr_five_step_nmse",
    "latent_positive_cosine",
    "latent_response_correlation",
    "latent_shuffle_dr_error_ratio",
    "dr_counterfactual_rms",
    "nominal_counterfactual_rms",
)

_REPRESENTATION_PROBE_METRICS = (
    "representation_positive_loss", "representation_relation_loss", "latent_relation_pairs",
    "latent_distance_mean", "latent_target_distance_mean",
    "latent_distance_batch_p10", "latent_distance_batch_p50", "latent_distance_batch_p90",
    "latent_target_distance_batch_p10", "latent_target_distance_batch_p50",
    "latent_target_distance_batch_p90",
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


def _collect_counterfactual_block(
    rollout: FixedDRTrackerRollout,
    nominal_rollout: NominalPairRollout,
    replay: ForwardPredictorReplayBuffer,
) -> dict[str, float]:
    """Collect broad A data and one exactly action-matched nominal B rollout."""

    batches = [rollout.step(predictor_only=True) for _ in range(5)]
    joint_targets = torch.stack([batch["joint_target"] for batch in batches], dim=1)
    with torch.inference_mode():
        nominal_states, diagnostics = nominal_rollout.rollout_joint_targets(
            batches[0]["robot_state"],
            joint_targets,
            motion_ids=batches[0]["motion_id"],
            motion_steps=batches[0]["motion_step"],
            motion_files=rollout.motion_files,
        )
    for index, batch in enumerate(batches):
        batch["nominal_next_robot_state"] = nominal_states[:, index]
        replay.add_step(batch)
    return diagnostics


def _loss_weight_payload(config: ForwardPredictorLossConfig) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(config).items()}


def _wandb_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "update": record["update"],
        "optimizer_steps": record["optimizer_steps"],
        "optimization/learning_rate_model": record["learning_rate_model"],
        "replay/size": record["replay_size"],
        "replay/samples_generated": record["samples_generated"],
        "rollout/transitions": record["transitions"],
    }
    payload.update(
        {
            f"optimization_train/{name}": value
            for name, value in record["optimization_train"].items()
        }
    )
    payload.update({f"fixed_probe/{name}": value for name, value in record["fixed_probe"].items()})
    payload.update({f"collector_memory/{name}": value for name, value in record.get("collector_memory", {}).items()})
    payload.update({f"matched_comparison/{name}": value for name, value in record.get("matched_comparison", {}).items()})
    payload["model/long_memory_enabled"] = True
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
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    model_digests = distributed.all_gather_object(digest.hexdigest())
    if len(set(model_digests)) != 1:
        raise RuntimeError("DDP ranks disagree on the trained hierarchical model")
    state = {
        "supervision_horizons": {"predictor": 5,
                                 "response_label": getattr(rollout, "response_label_horizon", 5)},
        "nominal_counterfactual_representation_supervision": True,
        "nominal_a_fraction": 0.5,
        "training_mixture_version": "memory350_nominal50_v1",
        "load_only_experiment": rollout.config.limb_payload_only,
        "dr_profile": rollout.payload_configuration.get("dr_profile"),
        "training_physics": rollout.payload_configuration,
        "architecture_version": model_config.architecture_version,
        "memory_contract": "short50_disjoint_chunk10_long30_v1",
        "long_memory_in_model": True,
        "encoder_scale": "attention_depth_2x_v1",
        "parameter_counts": parameter_counts(model),
        "episode_length_control_steps": 1000,
        "distributed_parameter_agreement": {"passed": True, "sha256_by_rank": model_digests},
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
            "prototype_sha256": rollout.dynamics_prototype_sha256,
            "inference_contract": (
                "history_only; simulator parameters are retained only as DR provenance. "
                "Representation supervision comes from the observed A-minus-nominal-B "
                f"{getattr(rollout, 'response_label_horizon', 5)}-step response, never from simulator parameters"
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
    training_worlds = args.num_envs - args.validation_worlds
    global_world_ids = tuple(rank * args.num_envs + i for rank in range(distributed.world_size)
                             for i in range(training_worlds))
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
            dynamics_seed=rank_seed,
            world_id_offset=world_id_offset,
            stochastic_policy=args.stochastic_policy,
            randomize_initial_episode_phase=args.randomize_initial_episode_phase,
            nominal_fraction=args.nominal_fraction,
            payload_enabled=args.payload,
            payload_body_name=args.payload_body_name,
            payload_mass_range_kg=tuple(args.payload_mass_range_kg),
            payload_position_body_m=tuple(args.payload_position_body_m),
            payload_size_m=tuple(args.payload_size_m),
            limb_payload_only=args.limb_payload_only,
            tracker_dr_plus_limb_payload=args.tracker_dr_plus_limb_payload,
        )
    )
    dataset = None
    if _limb_experiment(args):
        from intact_tracking.preview_protocol import dataset_identity
        files, dataset = dataset_identity(args.motion_file, args.motion_path)
        expected = files[distributed.rank::distributed.world_size] if len(files) > 1 else files
        if tuple(map(str, expected)) != rollout.motion_files:
            rollout.close()
            raise ValueError("Stage 1 shard differs from the complete requested dataset")
        dataset.update(rank_partition="sorted files [rank::world_size], union equals full catalog",
                       loaded_local_motion_count=len(rollout.motion_files),
                       loaded_local_frames=int(rollout.motion_command.motion.file_lengths.sum()))
        rank_audits = distributed.all_gather_object({
            "rank": distributed.rank, "local_rank": distributed.local_rank,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[distributed.local_rank],
            "num_envs": rollout.env.num_envs, "training_worlds": training_worlds,
            "validation_worlds": args.validation_worlds,
            "nominal_training_worlds": int(rollout.is_nominal[:training_worlds].sum()),
            "dr_training_worlds": int((~rollout.is_nominal[:training_worlds]).sum()),
            "nominal_validation_worlds": int(rollout.is_nominal[training_worlds:].sum()),
            "dr_validation_worlds": int((~rollout.is_nominal[training_worlds:]).sum()),
            "motion_count": len(rollout.motion_files),
            "loaded_frames": int(rollout.motion_command.motion.file_lengths.sum()),
            "motion_file_list_sha256": hashlib.sha256("\n".join(rollout.motion_files).encode()).hexdigest(),
            "episode_length_control_steps": rollout.env.max_episode_length,
            "dr_profile": rollout.payload_configuration["dr_profile"],
            "physics": rollout.payload_configuration["runtime_audit"],
        })
        dataset["runtime_audits_by_rank"] = rank_audits
        dataset["loaded_global_motion_count"] = sum(r["motion_count"] for r in rank_audits) if len(files) > 1 else 1
        dataset["loaded_global_frames"] = sum(r["loaded_frames"] for r in rank_audits) if len(files) > 1 else rank_audits[0]["loaded_frames"]
        if dataset["loaded_global_motion_count"] != len(files):
            raise RuntimeError("Distributed motion shards do not cover the requested catalog")
    if rollout.predictor_action_transform is None:
        error = rollout.predictor_action_transform_error or "unknown action-chain error"
        rollout.close()
        raise RuntimeError(
            "Forward Predictor requires an external memoryless policy-action to physical "
            f"PD-target transform: {error}"
        )
    try:
        nominal_rollout = NominalPairRollout(
            NominalPairRolloutConfig(
                checkpoint_file=str(checkpoint_path),
                motion_path=args.motion_path,
                motion_file=args.motion_file,
                task_id=args.task_id,
                num_envs=args.num_envs,
                device=str(device),
                seed=rank_seed + 100_000,
                horizon=5,
                restore_atol=args.nominal_restore_atol,
            )
        )
    except BaseException:
        rollout.close()
        raise
    replay = ForwardPredictorReplayBuffer(
        num_worlds=training_worlds,
        dimensions=dimensions,
        capacity=args.replay_capacity,
        history_steps=args.history_steps,
        context_history_steps=args.context_history_steps,
        positive_offset_steps=args.positive_offset_steps,
        sampling_mode=args.replay_sampling,
        seed=rank_seed,
        world_id_offset=world_id_offset,
        device=device,
    )
    validation_replay = None
    collector_replay = replay
    if args.validation_worlds:
        from intact_tracking.limb_context_validation import SplitWorldReplay
        validation_replay = ForwardPredictorReplayBuffer(
            num_worlds=args.validation_worlds, dimensions=dimensions,
            capacity=max(32768, args.fixed_probe_batch_size * 8),
            history_steps=args.history_steps, context_history_steps=args.context_history_steps,
            positive_offset_steps=args.positive_offset_steps, sampling_mode=args.replay_sampling,
            seed=rank_seed + 908177, world_id_offset=world_id_offset + training_worlds, device=device)
        collector_replay = SplitWorldReplay(replay, validation_replay)
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
        chunk_depth=args.chunk_depth,
        memory_depth=args.memory_depth,
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
        representation_weight=args.representation_weight,
        representation_relation_weight=args.representation_relation_weight,
        response_distance_scale=args.response_distance_scale,
        huber_delta=args.huber_delta,
    )
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
    resumed = None
    schedule_extension = None
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        if resumed.get("nominal_a_fraction") != 0.5:
            raise ValueError("Resume requires the nominal50 training distribution and its normalization")
        resumed_loss_config = asdict(ForwardPredictorLossConfig(**resumed["loss_config"]))
        if resumed["model_config"] != asdict(model_config):
            raise ValueError("Resume model/loss configuration changed")
        _validate_resume_loss_config(resumed_loss_config, asdict(loss_config))
        if resumed["tracker"]["checkpoint_sha256"] != tracker_sha256:
            raise ValueError("Resume tracker changed")
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        schedule_extension = extend_cosine_schedule(scheduler, optimizer_steps_target)
    if args.until_user_stop:
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"], args.continuation_min_learning_rate)
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        scheduler.continuation_min_learning_rate = args.continuation_min_learning_rate
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload_contract = (
        f"DR worlds additionally carry one fixed rigid payload on {args.payload_body_name}, "
        f"sampled from {args.payload_mass_range_kg[0]:g}-{args.payload_mass_range_kg[1]:g} kg"
        if args.payload
        else "the payload ablation is disabled"
    )
    run_config = {
        "method": (
            "payload nominal-counterfactual dynamics-context Forward Predictor v13"
            if args.payload
            else "nominal-counterfactual dynamics-context Forward Predictor v12 ablation"
        ),
        "architecture": {
            "controller": "frozen tracker",
            "physics": (
                "batch A is an independent-motion 50/50 mixture of compiled nominal and "
                f"fixed startup-DR worlds; {payload_contract}; batch B restores every A start "
                "state into nominal physics and replays the exact five physical PD targets"
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
                "interactions to z; the exact +/-5-frame window in the same A world/episode/motion "
                "is the invariant view, while cross-world latent geometry continuously matches "
                "the corresponding A-minus-B response geometry. There is no threshold, dynamics "
                "class, theta encoder or theta decoder"
            ),
            "rollout": (
                "predicted robot/foot/contact state is recursively fed back for five targets; "
                "the training/model hot path contains no articulated foot FK"
            ),
            "excluded": ["residual_policy", "backward"],
            "normalization": (
                "robot state, physical target, simulator foot, contact force and robot delta "
                "statistics frozen immediately after warmup; theta is not normalized or replayed"
            ),
        },
        "arguments": vars(args),
        "model": asdict(model_config),
        "model_parameters": parameter_count,
        "loss": asdict(loss_config),
        "batch_a_rollout": rollout.metadata,
        "batch_b_rollout": nominal_rollout.metadata,
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
            "positive_offset_steps": args.positive_offset_steps,
            "history_storage": "time archive reconstructed at sample time",
            "sampling": args.replay_sampling,
            "positive_pairs": (
                "same A world/episode/motion, exact +/-5-frame context-window shift; both "
                f"contexts must contain all {args.context_history_steps} frames"
            ),
            "response_pairs": (
                "cross-world broad-replay pairs; normalized latent distance continuously matches "
                "five-step A-minus-nominal-B response distance without a threshold"
            ),
            "predictor_sampling": (
                "broad ordinary replay; incomplete contexts remain eligible for prediction"
            ),
        },
        "objective_weights": {
            "teacher_forced_weight": 1.0,
            "recursive_weight": args.recursive_weight,
            "representation_weight": args.representation_weight,
            "representation_relation_weight": args.representation_relation_weight,
            "effective_positive_weight": args.representation_weight,
            "effective_relation_weight": args.representation_weight * args.representation_relation_weight,
            "response_distance_scale": args.response_distance_scale,
        },
        "reported_probe_metrics": list(_CORE_PROBE_METRICS + _REPRESENTATION_PROBE_METRICS),
        "representation_distance_diagnostics": (
            "Quantiles use eligible pairs in each rank's probe batch; distributed logs average "
            "rank-local quantiles and do not represent pooled global quantiles. Zero valid "
            "pairs produce zero diagnostics; inspect latent_relation_pairs alongside distances."
        ),
        "research_source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[3])): _sha256(p) for p in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "forward_predictor.py",
                Path(__file__).resolve().parents[1] / "forward_predictor_objective.py",
                Path(__file__).resolve().parents[1] / "forward_predictor_schedule.py",
                Path(__file__).resolve().parents[1] / "data/predictor_online.py",
                Path(__file__).resolve().parents[1] / "rollout/online.py",
                Path(__file__).resolve().parents[1] / "rollout/nominal.py")},
    }
    if _limb_experiment(args):
        run_config["dataset"] = dataset
        run_config["method"] = "four-limb independent uniform nominal-counterfactual context"
        run_config["architecture"]["physics"] = (
            "half A worlds compiled nominal, half independently sampled four-limb loads; "
            "B restores every exact A start into nominal physics and replays identical PD targets")
        if args.tracker_dr_plus_limb_payload:
            run_config["architecture"]["physics"] = (
                "A alternates compiled nominal slots (no payload/DR/pulses) with frozen-tracker DR "
                "plus independent hand/mid-shin U(0,4kg) and original force pulses; B restores each exact A start into "
                "nominal physics without pulses and replays identical physical PD targets")
        run_config["dr_profile"] = rollout.payload_configuration["dr_profile"]
        run_config["validation"] = {
            "worlds_per_rank": args.validation_worlds,
            "training_worlds_per_rank": training_worlds,
            "contract": "disjoint physical world IDs, independent loads/motions/starts/history; excluded from training and normalization",
            "selection": "minimum held-out five-step DR NMSE on broad contexts, including incomplete histories",
            "representation_probe": "held-out exact +/-5-step positive pairs with any usable short or long history",
            "interval": args.validation_interval,
            "convergence": "at least 2000 updates; <1% best NMSE improvement over 1000 updates and latent diagnostic ranges <0.1",
            "nominal_metrics": "separate nominal/DR held-out errors; each validation partition contains 50% nominal",
        }
    run_config["method"] = "nominal50-supervised Memory350 doubled context attention depth v1"
    run_config["parameter_counts"] = parameter_counts(model)
    run_config["architecture"]["context"] = (
        "same short50 + disjoint chunk10 x 30 and 64-dimensional latent; "
        "chunk/memory/context attention depths 2/4/4 at width 128, four heads; "
        "fresh predictor and retained context weights match the original initialization")
    run_config["memory_contract"] = {
        "short_steps": 50, "chunk_steps": 10, "long_chunks_in_model": 30, "long_summary_tokens": 1,
        "maximum_represented_interactions": 350, "cross_reset_model_history": True,
        "reset": "preserve completed contiguous chunks; discard incomplete tails; clear and pad short history",
        "collector": "unchanged Memory350 replay, memory rules and sample/positive-pair selection",
        "representation_eligibility": "identical reference-collector eligibility, including incomplete short history",
    }
    run_config["replay"]["positive_pairs"] = (
        "same physical A world/episode/motion, exact +/-5-frame shift; identical "
        "Memory350 sample eligibility; full short history is not required")
    run_config["episode_length_control_steps"] = 1000
    run_config["research_source_sha256"].update({
        str(p.relative_to(Path(__file__).resolve().parents[3])): _sha256(p)
        for p in [*sorted(Path(__file__).resolve().parents[1].glob("memory350_*.py")),
                  Path(__file__).resolve().parents[1] / "short50_comparison.py"]})
    run_config["training_control"] = {
        "mode": "until_user_stop" if args.until_user_stop else "automatic_budget_or_plateau",
        "maximum_updates": None if args.until_user_stop else args.updates,
        "automatic_early_stopping": not args.until_user_stop,
        "cosine_horizon_updates": args.updates,
        "continuation_min_learning_rate": args.continuation_min_learning_rate if args.until_user_stop else None,
        "stage2_requires_explicit_user_convergence_decision": args.until_user_stop,
    }
    if args.stop_after_updates is not None:
        run_config["training_control"].update(
            mode="fixed_update_budget", maximum_updates=args.stop_after_updates,
            automatic_early_stopping=False)
    if args.comparison_reference_dir:
        run_config["matched_control"] = reference_contract(args.comparison_reference_dir, run_config)
    if args.resume:
        old_config = json.loads((output_dir / "run_config.json").read_text())
        for name in ("num_envs", "validation_worlds", "seed", "batch_size", "gradient_steps_per_update",
                     "motion_path", "motion_file", "limb_payload_only", "nominal_fraction"):
            if old_config["arguments"][name] != vars(args)[name]:
                raise ValueError(f"Resume changed {name}")
        if old_config["arguments"].get("tracker_dr_plus_limb_payload", False) != args.tracker_dr_plus_limb_payload:
            raise ValueError("Resume changed the original tracker DR profile")
        if args.updates < old_config["arguments"]["updates"]:
            raise ValueError("Resume cannot reduce the training budget")
        run_config["resume_history"] = old_config.get("resume_history", []) + [{
            "checkpoint": str(Path(args.resume).resolve()), "completed_update": resumed["update"],
            "checkpoint_sha256": _sha256(Path(args.resume)),
            "optimizer_steps": resumed["optimizer_steps"],
            "previous_update_limit": None if old_config["arguments"].get("until_user_stop") else old_config["arguments"]["updates"],
            "update_limit": None if args.until_user_stop else args.updates,
            "schedule_extension": schedule_extension, "training_control": run_config["training_control"],
            "simulator_and_replay_restarted": True, "normalization_and_validation_preserved": True,
            "unix_time": time.time()}]

        def save_resume_config():
            initial = output_dir / "run_config.initial.json"
            if not initial.exists():
                _write_json(initial, old_config)
            _write_json(output_dir / "run_config.json", run_config)

        _main_process_call(distributed, save_resume_config)
        if distributed.is_main:
            print(json.dumps({"event": "forward_predictor_resume", **run_config["resume_history"][-1]}), flush=True)
    else:
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
        if wandb_logger.run is not None:
            wandb_logger.run.define_metric("memory/*", step_metric="update")
            wandb_logger.run.define_metric("optimization_train/*", step_metric="update")
            wandb_logger.run.define_metric("optimization_rank0/*", step_metric="update")
            _write_json(output_dir / "wandb_run.json", {"id": wandb_logger.id, "url": wandb_logger.url})
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
        nominal_rollout.close()
        rollout.close()
        raise

    history: list[dict[str, Any]] = (
        json.loads((output_dir / "history.json").read_text()) if args.resume else [])
    metrics_path = output_dir / "metrics.jsonl"
    optimizer_steps = resumed["optimizer_steps"] if resumed else 0
    first_update = resumed["update"] + 1 if resumed else 1
    stop_requested = False
    converged = False
    plateau_detected = False
    handlers = {}

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, stop)
    completed = False
    try:
        warmup_started = time.monotonic()
        while True:
            required_batch = max(args.batch_size, args.fixed_probe_batch_size)
            ready = rollout.collector_step >= args.warmup_steps and len(replay) >= required_batch
            if ready and args.representation_weight > 0.0:
                positive_batch = (
                    args.batch_size if args.fixed_batch_overfit else args.fixed_probe_batch_size
                )
                ready = replay.can_sample_positive_pairs(positive_batch)
            if ready and validation_replay is not None:
                ready = validation_replay.can_sample_positive_pairs(args.fixed_probe_batch_size)
            if distributed.all_true(ready):
                break
            if rollout.collector_step >= args.max_warmup_steps:
                raise RuntimeError(
                    "Forward Predictor warmup exhausted before every rank had a full batch: "
                    f"steps={rollout.collector_step}, replay={len(replay)}"
                )
            nominal_diagnostics = _collect_counterfactual_block(rollout, nominal_rollout, collector_replay)
            if distributed.is_main and (
                rollout.collector_step == 5
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
                            **nominal_diagnostics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        normalization = (ForwardPredictorNormalizationStats(**resumed["normalization"]) if resumed
                         else _global_normalization(distributed, replay, global_world_ids))
        if args.comparison_reference_dir:
            normalization = reference_normalization(args.comparison_reference_dir, global_world_ids)
        replay.normalizer.freeze()
        fixed_probe_batch = (validation_replay if validation_replay is not None else replay).sample_batch(
            args.fixed_probe_batch_size,
            normalization,
            positive_ready_only=args.representation_weight > 0.0,
        )
        broad_probe_batch = fixed_probe_batch
        if validation_replay is not None:
            broad_probe_batch = validation_replay.sample_batch(args.fixed_probe_batch_size, normalization)
            probe_file = output_dir / f"validation_rank_{distributed.rank}.pt"
            broad_file = output_dir / f"validation_broad_rank_{distributed.rank}.pt"
            if args.comparison_reference_dir and not args.resume:
                fixed_probe_batch, broad_probe_batch, reference_evidence = reference_probes(
                    args.comparison_reference_dir, output_dir, distributed.rank, args.num_envs,
                    args.validation_worlds, normalization, device)
                run_config["matched_control"]["validation_sha256_by_rank"] = distributed.all_gather_object(reference_evidence)
            elif args.resume:
                fixed_probe_batch = torch.load(probe_file, map_location=device, weights_only=False)
                broad_probe_batch = torch.load(broad_file, map_location=device, weights_only=False)
            else:
                torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in fixed_probe_batch.items()}, probe_file)
                torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in broad_probe_batch.items()}, broad_file)
            collector_replay.collect_validation = False
        run_config["nominal_pair_numerical_audit"] = nominal_rollout.metadata.get("repeat_diagnostics")
        _main_process_call(distributed, lambda: _write_json(output_dir / "run_config.json", run_config))
        fixed_train_batch = (
            replay.sample_batch(
                args.batch_size,
                normalization,
                positive_ready_only=args.representation_weight > 0.0,
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
        update = first_update - 1
        best_score = min((r["fixed_probe"]["dr_five_step_nmse"] for r in history), default=float("inf"))
        from intact_tracking.forward_predictor_schedule import training_update_indices, advance_predictor_schedule
        for update in training_update_indices(first_update, args.updates, args.until_user_stop):
            if update > 1 and not args.fixed_batch_overfit:
                model.eval()
                _collect_counterfactual_block(rollout, nominal_rollout, collector_replay)

            training_module.train()
            step_losses: list[dict[str, torch.Tensor]] = []
            for _ in range(args.gradient_steps_per_update):
                train_batch = (
                    fixed_train_batch
                    if fixed_train_batch is not None
                    else replay.sample_batch(args.batch_size, normalization)
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
                    for name in model_output:
                        weighted = model_output[name].detach().float() * fraction
                        accumulated_losses[name] = (
                            accumulated_losses.get(name, torch.zeros_like(weighted)) + weighted
                        )
                # Infinite max norm checks finiteness without clipping any finite gradient.
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
                accumulated_losses["gradient_norm"] = gradient_norm.detach().float()
                optimizer.step()
                advance_predictor_schedule(scheduler,
                    args.continuation_min_learning_rate if args.until_user_stop else None)
                optimizer_steps += 1
                step_losses.append(accumulated_losses)

            report_interval = args.validation_interval if validation_replay is not None else args.log_interval
            should_report = update == first_update or update % report_interval == 0 or update == args.updates
            improved = False
            if should_report:
                local_optimization_train = _scalar_tensors_to_floats(
                    {
                        name: torch.stack([item[name] for item in step_losses]).mean()
                        for name in step_losses[0]
                    }
                )
                optimization_train = distributed.mean_scalars(local_optimization_train)
                training_module.eval()
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if amp_enabled
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    probe_output = objective(
                        broad_probe_batch,
                        recursive_weight=args.recursive_weight,
                        validate_batch=False,
                    )
                    if validation_replay is not None:
                        full_output = objective(fixed_probe_batch, recursive_weight=args.recursive_weight,
                                                validate_batch=False)
                        for name in ("latent_positive_cosine", "latent_response_correlation", "latent_shuffle_dr_error_ratio",
                                     *_REPRESENTATION_PROBE_METRICS):
                            probe_output[name] = full_output[name]
                    local_probe = _scalar_tensors_to_floats({name: probe_output[name] for name in _CORE_PROBE_METRICS + _REPRESENTATION_PROBE_METRICS
                        if not _limb_experiment(args) or args.nominal_fraction > 0 or name not in ("nominal_five_step_nmse", "nominal_counterfactual_rms")})
                fixed_probe = distributed.mean_scalars(local_probe)
                matched_comparison = {}
                if args.comparison_reference_dir:
                    reference_text = (Path(args.comparison_reference_dir) / "metrics.jsonl").read_text()
                    # The reference trainer may be appending its next report.
                    # Only newline-terminated records belong to this snapshot.
                    reference_rows = [json.loads(line) for line in reference_text.split("\n")[:-1] if line.strip()]
                    reference_row = next((r for r in reference_rows if r["update"] == update), None)
                    if reference_row is not None:
                        ref_nmse = reference_row["fixed_probe"]["dr_five_step_nmse"]
                        matched_comparison = {
                            "reference_update": update, "memory350_dr_five_step_nmse": ref_nmse,
                            "encoder2x_dr_five_step_nmse": fixed_probe["dr_five_step_nmse"],
                            "encoder2x_to_memory350_nmse_ratio": fixed_probe["dr_five_step_nmse"] / ref_nmse,
                            "encoder2x_nmse_reduction_percent": 100 * (1 - fixed_probe["dr_five_step_nmse"] / ref_nmse),
                        }
                counts = distributed.sum_integers(
                    {
                        "transitions": rollout.transitions,
                        "replay_size": len(replay),
                        "samples_generated": replay.total_samples_generated,
                    }
                )
                memory_metrics = distributed.mean_scalars(replay.memory.metrics())
                for label, prefix in (("chunk_encoder", "context_encoder.chunk_encoder"),
                                      ("long_encoder", "context_encoder.memory_encoder"),
                                      ("final_encoder", "context_encoder.transformer")):
                    gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
                    if any(g is None for g in gradients):
                        raise RuntimeError(f"Missing training gradient in {label}")
                    memory_metrics["gradient_norm_" + label] = float(torch.stack([
                        g.detach().float().square().sum() for g in gradients]).sum().sqrt())
                if device.type == "cuda":
                    memory_metrics.update(distributed.mean_scalars({
                        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                    }))
                memory_metrics.update({
                    "batch_nominal_fraction": float(train_batch["is_nominal"].float().mean()),
                    "validation_nominal_fraction": float(broad_probe_batch["is_nominal"].float().mean()),
                    "batch_short_full_fraction": float(train_batch["history_valid"].all(1).float().mean()),
                    "batch_long_available_fraction": float(train_batch["memory_valid"].any(1).float().mean()),
                    "validation_long_available_fraction": float(broad_probe_batch["memory_valid"].any(1).float().mean()),
                    "validation_positive_pair_fraction": float(fixed_probe_batch["positive_pair_valid"].float().mean()),
                })
                record = {
                    "collector_memory": memory_metrics,
                    "model_long_memory_enabled": True,
                    "matched_comparison": matched_comparison,
                    "update": update,
                    "optimizer_steps": optimizer_steps,
                    **counts,
                    "learning_rate_model": optimizer.param_groups[0]["lr"],
                    "optimization_train": optimization_train,
                    "fixed_probe": fixed_probe,
                    "unix_time": time.time(),
                    "elapsed_seconds": time.monotonic() - warmup_started,
                    "validation_context_full_fraction": float(broad_probe_batch["context_full"].float().mean()),
                    "training_context_full_fraction": float(train_batch["context_full"].float().mean()),
                }
                improved = fixed_probe["dr_five_step_nmse"] < best_score
                best_score = min(best_score, fixed_probe["dr_five_step_nmse"])
                history.append(record)
                if _limb_experiment(args):
                    from intact_tracking.limb_context_validation import validation_plateau
                    plateau_detected = validation_plateau(history, update)
                    converged = plateau_detected and not args.until_user_stop and args.stop_after_updates is None
                    record["automatic_plateau_detected"] = plateau_detected
                if distributed.is_main:
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    wandb_logger.log(_wandb_payload(record), step=update)
                    print(json.dumps(record, sort_keys=True), flush=True)

            if distributed.is_main:
                _write_json(output_dir / "progress.json", {"completed_updates": update,
                    "optimizer_steps": optimizer_steps, "unix_time": time.time(),
                    "unbounded": args.until_user_stop and args.stop_after_updates is None,
                    "stop_after_updates": args.stop_after_updates})
                if not should_report and update % args.log_interval == 0:
                    light = {"update": update, "optimizer_steps": optimizer_steps,
                             "learning_rate_model": optimizer.param_groups[0]["lr"]}
                    for name in step_losses[0]:
                        light["optimization_rank0/" + name] = float(torch.stack([row[name] for row in step_losses]).mean())
                    wandb_logger.log(light, step=update)
                    with (output_dir / "training_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(light) + "\n")
            interrupted = not distributed.all_true(not stop_requested)
            reached_update_cap = args.stop_after_updates is not None and update >= args.stop_after_updates
            if (improved or update in PAIRED_UPDATES
                    or (args.checkpoint_interval and update % args.checkpoint_interval == 0)
                    or interrupted or converged or reached_update_cap):
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
                if improved:
                    _main_process_call(distributed, lambda: shutil.copy2(output_dir / "last.pt", output_dir / "best.pt"))
            if interrupted or converged or reached_update_cap:
                break

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
            numbered=False,
        )
        distributed.barrier()
        _main_process_call(distributed, lambda: _write_json(output_dir / "completion.json", {
            "completed_updates": update, "optimizer_steps": optimizer_steps, "converged": converged,
            "target_updates": args.stop_after_updates or (None if args.until_user_stop else args.updates),
            "stopping_mode": "fixed_update_budget" if args.stop_after_updates else ("until_user_stop" if args.until_user_stop else "automatic_budget_or_plateau"),
            "automatic_plateau_detected": plateau_detected,
            "stopped": interrupted if update >= first_update else False,
            "hit_cap": reached_update_cap or (not args.until_user_stop and update >= args.updates), "best_validation_dr_nmse": best_score,
            "selected_checkpoint": str(output_dir / "best.pt"), "unix_time": time.time()}))
        completed = True
        return output_dir / "last.pt"
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        wandb_logger.finish(exit_code=0 if completed else 1)
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
