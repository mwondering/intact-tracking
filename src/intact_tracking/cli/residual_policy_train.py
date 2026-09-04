"""Train SPV5-2A-compatible residual PPO with or without a frozen dynamics latent."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import DictConfig, OmegaConf

from intact_tracking.distributed import DistributedContext
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.residual_context import (
    ResidualLatentVecEnvWrapper,
    load_frozen_context_checkpoint,
)
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import (
    _clear_missing_motion_exclusions,
    _filter_disturbance_events,
    _sha256,
)

SPV52A_TASK_ID = "SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", required=True)
    parser.add_argument("--forward-checkpoint")
    parser.add_argument("--baseline", choices=("latent", "no-latent"), required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion-path")
    motion.add_argument("--motion-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", default=SPV52A_TASK_ID)
    parser.add_argument("--resume")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--residual-hidden-dims", type=int, nargs="+", default=(512, 256, 128))
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--actor-learning-rate", type=float)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument(
        "--context-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-disturbances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-enable checkpoint step/interval events such as random pushes.",
    )
    parser.add_argument(
        "--randomize-initial-episode-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-for-nan",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    parser.add_argument("--wandb-project", default="intact-residual-policy")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.baseline == "latent" and not args.forward_checkpoint:
        raise ValueError("The latent baseline requires --forward-checkpoint")
    if args.baseline == "no-latent" and args.forward_checkpoint:
        raise ValueError("The no-latent baseline must not receive --forward-checkpoint")
    positive = ("num_envs", "iterations", "num_steps_per_env", "save_interval")
    invalid = {name: getattr(args, name) for name in positive if getattr(args, name) < 1}
    if invalid:
        raise ValueError(f"Residual PPO arguments must be positive: {invalid}")
    if not args.residual_hidden_dims or any(width < 1 for width in args.residual_hidden_dims):
        raise ValueError("residual-hidden-dims must contain positive widths")
    if args.residual_scale <= 0.0:
        raise ValueError("residual-scale must be positive")
    for name in ("actor_learning_rate", "critic_learning_rate"):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolved_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    return copy.deepcopy(dict(value))


def _algorithm_configuration(
    source: Mapping[str, Any],
    *,
    actor_learning_rate: float | None,
    critic_learning_rate: float | None,
) -> dict[str, Any]:
    allowed = {
        "num_learning_epochs",
        "num_mini_batches",
        "clip_param",
        "gamma",
        "lam",
        "value_loss_coef",
        "entropy_coef",
        "learning_rate",
        "max_grad_norm",
        "optimizer",
        "use_clipped_value_loss",
        "schedule",
        "desired_kl",
        "normalize_advantage_per_mini_batch",
        "actor_learning_rate",
        "critic_learning_rate",
        "adaptive_critic_learning_rate",
    }
    result = {name: copy.deepcopy(value) for name, value in source.items() if name in allowed}
    source_actor_lr = float(result.get("actor_learning_rate", result.get("learning_rate", 1.0e-3)))
    result["actor_learning_rate"] = float(actor_learning_rate or source_actor_lr)
    result["critic_learning_rate"] = float(
        critic_learning_rate or result.get("critic_learning_rate", 5.0e-4)
    )
    result["learning_rate"] = result["actor_learning_rate"]
    result.setdefault("adaptive_critic_learning_rate", False)
    result.setdefault("optimizer", "adam")
    result["class_name"] = "intact_tracking.residual_policy:ResidualPPO"
    result["rnd_cfg"] = None
    result["symmetry_cfg"] = None
    return result


def _build_train_configuration(
    source_config: DictConfig,
    *,
    tracker_checkpoint: Path,
    tracker_actor_kwargs: Mapping[str, Any],
    tracker_obs_groups: Mapping[str, list[str]],
    baseline: str,
    dynamics_latent_dim: int,
    residual_hidden_dims: tuple[int, ...],
    residual_scale: float,
    iterations: int,
    num_steps_per_env: int,
    save_interval: int,
    seed: int,
    logger: str,
    wandb_project: str,
    actor_learning_rate: float | None,
    critic_learning_rate: float | None,
    check_for_nan: bool,
) -> dict[str, Any]:
    merged = OmegaConf.merge(source_config.agent, source_config.task.get("agent_overrides", {}))
    train = _resolved_mapping(merged)
    base_actor = _resolved_mapping(train["actor"])
    base_critic = _resolved_mapping(train["critic"])
    base_algorithm = _resolved_mapping(train["algorithm"])
    distribution_cfg = copy.deepcopy(base_actor.get("distribution_cfg"))
    train.update(
        {
            "num_steps_per_env": int(num_steps_per_env),
            "max_iterations": int(iterations),
            "save_interval": int(save_interval),
            "seed": int(seed),
            "logger": str(logger),
            "wandb_project": str(wandb_project),
            "run_name": None,
            "resume": False,
            "upload_model": False,
            "check_for_nan": bool(check_for_nan),
            "torch_compile_mode": None,
            "obs_groups": copy.deepcopy(dict(tracker_obs_groups)),
        }
    )
    train["actor"] = {
        "class_name": "intact_tracking.residual_policy:FrozenTrackerResidualActor",
        "tracker_checkpoint": str(tracker_checkpoint),
        "tracker_actor_kwargs": copy.deepcopy(dict(tracker_actor_kwargs)),
        "tracker_obs_groups": copy.deepcopy(dict(tracker_obs_groups)),
        "use_dynamics_latent": baseline == "latent",
        "dynamics_latent_dim": int(dynamics_latent_dim),
        "residual_hidden_dims": list(residual_hidden_dims),
        "residual_activation": "elu",
        "residual_scale": float(residual_scale),
        "distribution_cfg": distribution_cfg,
    }
    base_critic["class_name"] = "intact_tracking.residual_policy:WarmStartedHeftCritic"
    base_critic["initial_checkpoint"] = str(tracker_checkpoint)
    train["critic"] = base_critic
    train["algorithm"] = _algorithm_configuration(
        base_algorithm,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    )
    return train


def _checkpoint_configuration(
    source: DictConfig,
    train: Mapping[str, Any],
    residual_metadata: Mapping[str, Any],
) -> DictConfig:
    container = OmegaConf.to_container(source, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Tracker checkpoint configuration must resolve to a mapping")
    container["agent"] = copy.deepcopy(dict(train))
    task = container.get("task")
    if isinstance(task, dict):
        task["agent_overrides"] = {}
    container["residual_policy"] = copy.deepcopy(dict(residual_metadata))
    return OmegaConf.create(container)


def _prepare_output(
    output_dir: Path,
    *,
    resume: str | None,
    run_config: Mapping[str, Any],
    checkpoint_config: DictConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and any(
        path.exists()
        for path in (
            output_dir / "checkpoint_final.pt",
            output_dir / "run_config.json",
        )
    ):
        raise FileExistsError(f"Refusing to overwrite an existing residual run in {output_dir}")
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(run_config), indent=2, sort_keys=True) + "\n"
    )
    OmegaConf.save(checkpoint_config, output_dir / "config.yaml")


def _run(args: argparse.Namespace, distributed: DistributedContext) -> Path:
    device = distributed.device
    tracker_path = Path(args.tracker_checkpoint).expanduser().resolve()
    if not tracker_path.is_file():
        raise FileNotFoundError(tracker_path)
    forward_path = (
        Path(args.forward_checkpoint).expanduser().resolve() if args.forward_checkpoint else None
    )
    if forward_path is not None and not forward_path.is_file():
        raise FileNotFoundError(forward_path)
    resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    output_dir = Path(args.output_dir).expanduser().resolve()

    source_config = _load_saved_config(tracker_path)
    source_seed = int(source_config.get("seed", source_config.agent.get("seed", 42)))
    seed = source_seed if args.seed is None else int(args.seed)
    rank_seed = seed + distributed.rank
    _seed_everything(rank_seed)
    prepared = prepare_rollout(
        checkpoint_file=str(tracker_path),
        num_envs=args.num_envs,
        motion_path=args.motion_path,
        motion_file=args.motion_file,
        task_id=args.task_id,
    )
    if prepared.checkpoint_task_id != SPV52A_TASK_ID:
        raise ValueError(
            f"Residual PPO is fixed to {SPV52A_TASK_ID!r}, got {prepared.checkpoint_task_id!r}"
        )
    prepared.env.seed = rank_seed
    cleared_exclusions = _clear_missing_motion_exclusions(prepared.env)
    removed_disturbances = (
        [] if args.include_disturbances else _filter_disturbance_events(prepared.env)
    )
    tracker_sha256 = _sha256(tracker_path)

    context_checkpoint = None
    if forward_path is not None:
        context_checkpoint = load_frozen_context_checkpoint(
            forward_path,
            device=device,
            expected_tracker_sha256=tracker_sha256,
        )
    latent_dim = int(context_checkpoint.config.dynamics_latent_dim) if context_checkpoint else 0
    train_config = _build_train_configuration(
        source_config,
        tracker_checkpoint=tracker_path,
        tracker_actor_kwargs=prepared.actor_kwargs,
        tracker_obs_groups=prepared.obs_groups,
        baseline=args.baseline,
        dynamics_latent_dim=latent_dim,
        residual_hidden_dims=tuple(args.residual_hidden_dims),
        residual_scale=args.residual_scale,
        iterations=args.iterations,
        num_steps_per_env=args.num_steps_per_env,
        save_interval=args.save_interval,
        seed=seed,
        logger=args.logger,
        wandb_project=args.wandb_project,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        check_for_nan=args.check_for_nan,
    )
    residual_metadata = {
        "version": "spv52a_frozen_tracker_residual_v1",
        "baseline": args.baseline,
        "tracker_checkpoint": str(tracker_path),
        "tracker_sha256": tracker_sha256,
        "tracker_frozen": True,
        "critic_warm_started": True,
        "forward_checkpoint": str(forward_path) if forward_path else None,
        "forward_checkpoint_sha256": context_checkpoint.sha256 if context_checkpoint else None,
        "context_encoder_frozen": context_checkpoint is not None,
        "context_history_steps": (
            context_checkpoint.config.context_history_steps if context_checkpoint else 0
        ),
        "dynamics_latent_dim": latent_dim,
        "residual_scale": float(args.residual_scale),
        "residual_hidden_dims": list(args.residual_hidden_dims),
        "removed_step_interval_events": removed_disturbances,
        "cleared_missing_motion_exclusions": cleared_exclusions,
        "contract": (
            "base action and base feature extractor are frozen SPV5-2A; PPO samples the final "
            "base-plus-residual Gaussian; the latent baseline executes only the frozen "
            "history Context Encoder, never Forward Predictor or simulator theta"
        ),
    }
    checkpoint_config = _checkpoint_configuration(
        source_config,
        train_config,
        residual_metadata,
    )
    run_config = {
        **residual_metadata,
        "task_id": prepared.checkpoint_task_id,
        "num_envs_per_rank": args.num_envs,
        "world_size": distributed.world_size,
        "global_num_envs": args.num_envs * distributed.world_size,
        "num_steps_per_env": args.num_steps_per_env,
        "iterations": args.iterations,
        "initial_policy_distribution": (
            "exact frozen tracker mean/std because residual output layer starts at zero"
        ),
    }
    if distributed.is_main:
        _prepare_output(
            output_dir,
            resume=str(resume_path) if resume_path else None,
            run_config=run_config,
            checkpoint_config=checkpoint_config,
        )
    distributed.barrier()

    env: ManagerBasedRlEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    try:
        # Constructing the frozen Context Encoder consumes PyTorch RNG state.  Re-seed
        # immediately before environment construction so the latent/no-latent runs see
        # the same startup DR samples for the same user seed.
        _seed_everything(rank_seed)
        env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=str(device))
        if context_checkpoint is None:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        else:
            wrapped = ResidualLatentVecEnvWrapper(
                env,
                clip_actions=prepared.clip_actions,
                context_checkpoint=context_checkpoint,
                use_bfloat16=args.context_bfloat16,
            )
        runner = ResidualOnPolicyRunner(
            wrapped,
            copy.deepcopy(train_config),
            str(output_dir),
            str(device),
            checkpoint_cfg=checkpoint_config,
            residual_metadata=residual_metadata,
        )
        if resume_path is not None:
            runner.load(str(resume_path), map_location=str(device))
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=args.randomize_initial_episode_length,
        )
        return output_dir / "checkpoint_final.pt"
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def run(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/intact-matplotlib")
    os.environ["WANDB_MODE"] = args.wandb_mode
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
