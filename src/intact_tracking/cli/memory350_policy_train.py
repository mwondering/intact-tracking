"""Matched residual PPO with frozen hierarchical Memory350 or original inputs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import capture_original_rewards, assert_rewards_unchanged
from intact_tracking.cli.residual_policy_train import (
    _build_train_configuration, _prepare_output, _seed_everything,
)
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.distributed import DistributedContext
from intact_tracking.limb_context_distributed import (
    audit_rank_agreement, main_process_call, synchronize_critic_normalization, tensor_digest,
)
from intact_tracking.memory350_policy_env import Memory350PolicyWrapper as LimbContextWrapper
from intact_tracking.limb_context_policy import (
    FUSIONS, SCRATCH_INITIALIZATION, SCRATCH_ACTION_STD, configure_context_models,
)
from intact_tracking.limb_context_sampling import (
    SamplingCheckpoint, checkpoint_configuration, configure_motion_sampling,
    phase_budget, state_digest, validate_sampling_resume,
)
from intact_tracking.limb_context_terminations import (
    PROFILES, audit_runtime_terminations, configure_training_terminations,
    resolve_profile, validate_termination_resume,
)
from intact_tracking.limb_context_protocol import (
    FULL_DATASET, TRACKER, TRACKER_SHA256, PROJECT_ROOT,
)
from intact_tracking.memory350_policy_protocol import VERSION, EPISODE_STEPS
from intact_tracking.limb_context_dr import (
    DR_PROFILES, TRACKER_DR, configure_limb_dr, audit_limb_dr, resolve_dr_profile, validate_context_dr,
)
from intact_tracking.preview_protocol import dataset_identity
from intact_tracking.memory350_inference import load_memory350_checkpoint as load_frozen_context_checkpoint
from intact_tracking.memory350_checkpoint import BUNDLE_VERSION, make_context_bundle, make_tracker_bundle
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.wandb_logger import WandbLogger
from intact_tracking.memory350_policy_precision import (
    POLICY_PRECISIONS, configure_policy_precision, resolve_policy_precision,
)

PPO_CLASS_NAME = "intact_tracking.limb_context_distributed:DistributedResidualPPO"
ALLOWED_DR_PROFILES = (TRACKER_DR,)
CONTEXT_SOURCE_DR_PROFILE = None
TRAINING_START_PROFILE = None
# Alternate conditioning providers reuse the physics/PPO lifecycle without
# pretending to load a pretrained Memory350 encoder.
CONTEXT_CHECKPOINT_REQUIRED = True
METADATA_ADAPTER = None
EMBED_TRACKER_WITHOUT_CONTEXT = False
WANDB_TAGS = None
RESIDUAL_WANDB_METRICS = {
    "residual_action_rms": "Residual/mean_rms",
    "residual_action_abs_mean": "Residual/mean_abs",
    "residual_action_abs_max": "Residual/mean_abs_max",
    "residual_to_base_rms_ratio": "Residual/mean_to_tracker_rms_ratio",
    "base_action_rms": "Residual/tracker_action_rms",
    "residual_target_rms_rad": "Residual/target_rms_rad",
    "residual_target_abs_max_rad": "Residual/target_abs_max_rad",
    "residual_output_bounded": "Residual/output_bounded",
    "cuda_cache_released_gib": "Perf/cuda_cache_released_gib",
    "cuda_reserved_after_release_gib": "Perf/cuda_reserved_after_release_gib",
    "cuda_allocated_after_release_gib": "Perf/cuda_allocated_after_release_gib",
    "cuda_free_after_release_gib": "Perf/cuda_free_after_release_gib",
    "cuda_cache_release_seconds": "Perf/cuda_cache_release_seconds",
    "cuda_peak_allocated_gib": "Perf/torch_peak_allocated_gib",
    "cuda_peak_reserved_gib": "Perf/torch_peak_reserved_gib",
}
TRACKER_WANDB_GROUPS = (
    "Loss", "Policy", "Perf", "Train", "Episode", "Episode_Reward",
    "Episode_Termination", "Metrics", "Residual", "AuxDR",
)


def tracker_wandb_metrics(record, *, rollout_steps):
    """Use SP/RSL's scalar names with the already pooled multi-rank values."""
    payload = {key if key.split("/", 1)[0] in ("AuxDR", "Teacher", "WorldModel", "Adapter") else f"Loss/{key}": value
               for key, value in record["loss"].items()}
    duration = record["collect_seconds"] + record["learn_seconds"]
    payload.update({
        "Loss/learning_rate": record["learning_rate"],
        "Policy/mean_std": record["action_std"],
        "Perf/collection_time": record["collect_seconds"],
        "Perf/learning_time": record["learn_seconds"],
        "Perf/total_fps": int(record["global_num_envs"] * rollout_steps / duration) if duration > 0 else 0,
    })
    for name in ("mean_reward", "mean_episode_length"):
        if record[name] is not None:
            payload[f"Train/{name}"] = record[name]
    if "ppo_optimizer_steps" in record:
        payload["Policy/optimizer_steps"] = record["ppo_optimizer_steps"]
    for key, value in record.get("episode_metrics", {}).items():
        # RSL preserves pre-grouped environment keys, adding Episode/ only to
        # unqualified names. Flattening under training/ alone loses these panels.
        payload[key if "/" in key else f"Episode/{key}"] = value
    payload.update({path: record["loss"][key] for key, path in RESIDUAL_WANDB_METRICS.items()
                    if key in record["loss"]})
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", choices=("baseline", "film", "concat"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-checkpoint")
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--dr-profile", choices=DR_PROFILES,
                        help="Fresh runs default to legacy load_only; resumes inherit their saved DR profile")
    parser.add_argument("--dr-sampling", choices=("independent_uniform", "grid256_shared"),
                        default="independent_uniform", help="Static DR/load distribution used for PPO training")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-file")
    motion.add_argument("--motion-path")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Target total COMPLETED PPO updates (also on resume)")
    parser.add_argument("--until-user-stop", action="store_true", help="No total update cap")
    parser.add_argument("--training-ranks", type=int, choices=(2, 4), default=2)
    parser.add_argument("--num-envs", type=int, default=8192,
                        help="Environment count PER GPU/process")
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--device")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.0002)
    parser.add_argument("--initial-action-std", type=float,
                        help="Fresh policy Gaussian std; defaults to the model initialization contract")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--residual-output-mode", choices=("bounded", "unbounded"),
                        help="Unbounded uses the raw MLP output, without tanh, scaling or clipping")
    parser.add_argument("--resume")
    parser.add_argument("--reset-adaptive-sampling", action="store_true",
                        help="On resume, keep model/optimizer state but reset adaptive statistics to priors")
    parser.add_argument("--resume-uniform-sampling", action="store_true",
                        help="Resume with uniform motion sampling, disabling rewind and discarding adaptive statistics")
    parser.add_argument("--align-sampling-to-tracker", action="store_true",
                        help="Resume legacy native PPO with the original tracker rewind, preserving sampler counts")
    parser.add_argument("--policy-precision", choices=POLICY_PRECISIONS,
                        help="Inherit checkpoint precision; legacy checkpoints use TF32")
    parser.add_argument("--training-terminations", choices=PROFILES,
                        help="New experiments default to no_ee_body_pos; pinned experiments/resumes preserve their profile")
    parser.add_argument("--motion-sampling", choices=("uniform", "adaptive"), default="uniform")
    parser.add_argument("--adaptive-after-update", type=int, default=1000,
                        help="Uniform prefix, then restart from this exact checkpoint with adaptive sampling")
    parser.add_argument("--endpoint-eval-protocol",
                        help="Fixed cold/warm tracking evaluation at each 100 completed updates")
    parser.add_argument("--allow-evaluation-protocol-change", action="store_true",
                        help="On resume, explicitly migrate legacy endpoints to bounded warm evaluation")
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--wandb-project", default="intact-preview-v2")
    parser.add_argument("--wandb-entity", default="2486344338-zhejiang-university")
    parser.add_argument("--wandb-group", default="memory350-residual-stage2")
    parser.add_argument("--wandb-name")
    parser.set_defaults(dr_profile=TRACKER_DR, training_terminations="no_ee_body_pos",
                        motion_sampling="adaptive", save_interval=100)
    return parser


def audit_initial_models(actor, critic, obs, fusion):
    initial_std = actor.initial_action_std
    if initial_std is None:
        initial_std = SCRATCH_ACTION_STD
    with torch.no_grad():
        features, base_action = actor._base_features_and_action(obs)
        action = actor(obs)
        torch.testing.assert_close(action, base_action, atol=0, rtol=0)
        torch.testing.assert_close(action, actor.tracker(obs), atol=0, rtol=0)
        actor.distribution.update(action)
        torch.testing.assert_close(actor.output_std, torch.full_like(actor.output_std, initial_std),
                                   atol=0, rtol=0)
        normal = critic.obs_normalizer(critic._flat_obs(obs))
        original = critic.mlp if fusion == "baseline" else critic.mlp.base
        torch.testing.assert_close(critic(obs), original(normal), atol=1e-6, rtol=1e-6)
        if fusion != "baseline":
            alternate = obs.clone()
            alternate["dynamics_latent"] = torch.randn_like(alternate["dynamics_latent"])
            torch.testing.assert_close(actor(alternate), action, atol=0, rtol=0)
            torch.testing.assert_close(critic(alternate), critic(obs), atol=1e-6, rtol=1e-6)
    return {"actor_original_features": features.shape[-1], "critic_original_features": critic.obs_dim,
            "latent_dimensions": 0 if fusion == "baseline" else 64,
            "original_obs_groups": {"actor": list(actor.tracker.obs_groups),
                                    "critic": list(critic.obs_groups)},
            "identical_initial_action": True, "identical_initial_value": True,
            "action_std_initialization": initial_std,
            "residual_output_mode": actor.residual_output_mode,
            "actor_initialization_seed": actor.initialization_seed,
            "critic_initialization_seed": critic.initialization_seed,
            "actor_common_trunk_sha256": tensor_digest(
                (actor.residual_mlp if fusion == "baseline" else actor.residual_mlp.base).named_parameters()),
            "critic_common_trunk_sha256": tensor_digest(original.named_parameters()),
            "critic_initial_normalizer_sha256": tensor_digest(critic.obs_normalizer.state_dict().items()),
            "critic_initial_normalizer_count": float(critic.obs_normalizer.count),
            "actor_trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
            "critic_trainable_parameters": sum(p.numel() for p in critic.parameters() if p.requires_grad)}


def memory_checkpoint_configuration(source, train, metadata):
    cfg = checkpoint_configuration(source, train, metadata)
    seconds = metadata["episode_length_control_steps"] * float(source.task.decimation) * float(source.task.sim.timestep)
    OmegaConf.update(cfg, "task.episode_length_s", seconds, merge=False)
    return cfg


def attach_json_logger(runner, output, distributed, wandb_logger=None):
    original = runner.logger.log
    original_process = runner.logger.process_env_step

    def process(*args, **kwargs):
        # RSL guards buffer collection on writer presence. Non-main ranks need
        # those buffers too, but must never create a writer or checkpoint.
        writer = runner.logger.writer
        if writer is None:
            runner.logger.writer = object()
        try:
            return original_process(*args, **kwargs)
        finally:
            runner.logger.writer = writer

    runner.logger.process_env_step = process

    def log(**kw):
        losses = distributed.mean_scalars({k: float(v) for k, v in kw["loss_dict"].items()})
        peak_names = [name for name in ("cuda_peak_allocated_gib", "cuda_peak_reserved_gib")
                      if name in kw["loss_dict"]]
        if peak_names:
            peaks = torch.tensor([kw["loss_dict"][name] for name in peak_names], device=distributed.device)
            if distributed.enabled:
                torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
            losses.update(zip(peak_names, peaks.tolist(), strict=True))
        moments = torch.tensor([sum(runner.logger.rewbuffer), len(runner.logger.rewbuffer),
                                sum(runner.logger.lenbuffer), len(runner.logger.lenbuffer)],
                               device=distributed.device, dtype=torch.float64)
        distributed.all_reduce_sum(moments)
        durations = torch.tensor([kw["collect_time"], kw["learn_time"]], device=distributed.device)
        if distributed.enabled:
            torch.distributed.all_reduce(durations, op=torch.distributed.ReduceOp.MAX)
        record = {"completed_updates": kw["it"] + 1, "unix_time": time.time(),
                  "world_size": distributed.world_size, "num_envs_per_rank": runner.env.num_envs,
                  "global_num_envs": runner.env.num_envs * distributed.world_size,
                  "global_transitions": (kw["it"] + 1) * runner.env.num_envs * distributed.world_size * runner.cfg["num_steps_per_env"],
                  "collect_seconds": float(durations[0]), "learn_seconds": float(durations[1]),
                  "learning_rate": kw["learning_rate"],
                  "loss": losses,
                  "action_std": float(kw["action_std"].mean()),
                  "mean_reward": float(moments[0] / moments[1]) if moments[1] else None,
                  "mean_episode_length": float(moments[2] / moments[3]) if moments[3] else None,
                  "episode_window_count_across_ranks": int(moments[1])}
        record["policy_precision"] = runner.residual_metadata.get("policy_precision", "tf32")
        epochs, batches = getattr(runner.alg, "num_learning_epochs", None), getattr(runner.alg, "num_mini_batches", None)
        if epochs is not None and batches is not None:
            record["ppo_optimizer_steps"] = record["completed_updates"] * epochs * batches
        record["policy_fp32"] = record["policy_precision"] == "fp32"
        sampling = getattr(runner, "residual_metadata", {}).get("motion_sampling")
        terminations = getattr(runner, "residual_metadata", {}).get("training_terminations")
        if terminations:
            record["training_terminations"] = {
                "ee_body_pos_enabled": "ee_body_pos" not in terminations["disabled_terms"]}
        if sampling:
            command = runner.env.unwrapped.command_manager.get_term("motion")
            record["motion_sampling"] = {
                "adaptive_enabled": command.cfg.sampling_mode == "adaptive",
                "failure_rewind_enabled": command.cfg.rewind.enabled,
                "adaptive_after_update": sampling["adaptive_after_update"]}
        endpoint = getattr(runner, "pending_endpoint_evaluation", None)
        if endpoint is not None:
            record["endpoint_eval"] = endpoint
            record["evaluation_seconds"] = runner.pending_evaluation_seconds
        local_keys = sorted({key for row in runner.logger.ep_extras for key in row})
        keys = sorted({key for names in distributed.all_gather_object(local_keys) for key in names})
        if keys:
            stats = torch.zeros(len(keys), 2, dtype=torch.float64, device=distributed.device)
            for i, key in enumerate(keys):
                values = [torch.as_tensor(row[key], device=distributed.device).reshape(-1)
                          for row in runner.logger.ep_extras if key in row]
                if values:
                    flat = torch.cat(values).double()
                    stats[i, 0], stats[i, 1] = flat.sum(), flat.numel()
            distributed.all_reduce_sum(stats)
            record["episode_metrics"] = {key: float(stats[i, 0] / stats[i, 1])
                                          for i, key in enumerate(keys) if stats[i, 1]}
        if distributed.is_main:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            progress = output / "progress.json"
            temporary = progress.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"completed_updates": record["completed_updates"],
                                             "unix_time": record["unix_time"]}) + "\n")
            temporary.replace(progress)
            payload = {"completed_updates": record["completed_updates"]}
            def flatten(value, prefix):
                for key, item in value.items():
                    path = prefix + key
                    if isinstance(item, dict):
                        flatten(item, path + "/")
                    elif isinstance(item, (int, float, bool)):
                        payload[path] = item
            flatten(record, "training/")
            payload.update(tracker_wandb_metrics(record, rollout_steps=runner.cfg["num_steps_per_env"]))
            if wandb_logger is not None:
                wandb_logger.log(payload, step=record["completed_updates"])
        runner.pending_endpoint_evaluation = None
        runner.pending_evaluation_seconds = 0.
        original(**{**kw, "loss_dict": losses, "collect_time": float(durations[0]), "learn_time": float(durations[1])})
        if not distributed.is_main:
            runner.logger.ep_extras.clear()

    runner.logger.log = log


def validate_resume_models(old_train, train, args):
    """Strict by default; experiment entry points may authorize a narrow transition."""
    for key in ("actor", "critic", "algorithm", "obs_groups"):
        if old_train[key] != train[key]:
            raise ValueError(f"Resume changed {key}")
    return None


def _run(args, distributed):
    reset_sampling = getattr(args, "reset_adaptive_sampling", False)
    align_sampling = getattr(args, "align_sampling_to_tracker", False)
    uniform_resume = getattr(args, "resume_uniform_sampling", False)
    if uniform_resume and (not args.resume or args.motion_sampling != "uniform" or reset_sampling or align_sampling):
        raise ValueError("--resume-uniform-sampling requires --resume, uniform sampling, and no adaptive reset/alignment")
    if reset_sampling and (not args.resume or args.motion_sampling != "adaptive"):
        raise ValueError("--reset-adaptive-sampling requires --resume and adaptive motion sampling")
    if align_sampling and (not args.resume or args.motion_sampling != "adaptive"):
        raise ValueError("--align-sampling-to-tracker requires --resume and adaptive motion sampling")
    if args.until_user_stop:
        args.iterations = None
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("iterations must be positive")
    for key in ("num_envs", "rollout_steps", "epochs", "mini_batches", "save_interval"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if CONTEXT_CHECKPOINT_REQUIRED and (args.fusion in ("film", "concat")) != bool(args.context_checkpoint):
        raise ValueError("Film/concat require a context checkpoint; baseline/constant do not")
    if args.episode_steps != EPISODE_STEPS or distributed.world_size != args.training_ranks:
        raise ValueError("Training must use the requested rank count and 1000-step episodes")
    if args.dr_profile not in ALLOWED_DR_PROFILES:
        raise ValueError(f"Unsupported deployment DR profile: {args.dr_profile}")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Outputs must remain in the project directory")
    def validate_output():
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(output)
    main_process_call(distributed, validate_output)
    device = str(distributed.device)
    previous = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    args.policy_precision = resolve_policy_precision(getattr(args, "policy_precision", None), previous)
    precision_audit = configure_policy_precision(args.policy_precision)
    args.dr_profile = resolve_dr_profile(args.dr_profile, previous["residual_policy"] if previous else None)
    resume_update = previous["completed_updates"] if previous else 0
    args.training_terminations = resolve_profile(args.training_terminations, output, PROJECT_ROOT, previous)
    rank_seed = args.seed + 1000003 * distributed.rank
    motion_file = str(Path(args.motion_file).resolve()) if args.motion_file else None
    motion_path = None if motion_file else str(Path(args.motion_path or FULL_DATASET).resolve())
    files, dataset = dataset_identity(motion_file, motion_path)
    tracker = Path(args.tracker_checkpoint).resolve()
    tracker_hash = _sha256(tracker)
    if tracker_hash != TRACKER_SHA256:
        raise ValueError("Wrong frozen tracker checkpoint")
    _seed_everything(rank_seed)
    prepared = prepare_rollout(checkpoint_file=str(tracker), num_envs=args.num_envs,
                               motion_file=motion_file, motion_path=motion_path)
    prepared.env.episode_length_s = args.episode_steps * prepared.env.decimation * prepared.env.sim.mujoco.timestep
    source = _load_saved_config(tracker)
    rewards = capture_original_rewards(prepared.env)
    if args.dr_sampling == "grid256_shared":
        from intact_tracking.limb_context_grouped_dr import configure_grid256_dr, group_ids
        group_ids(args.num_envs)
        physics = configure_grid256_dr(prepared.env, args.seed)
    else:
        physics = configure_limb_dr(prepared.env, rank_seed, profile=args.dr_profile)
    terminations = configure_training_terminations(prepared.env, source, args.training_terminations)
    sampling = configure_motion_sampling(prepared.env, args.motion_sampling,
                                         args.adaptive_after_update, resume_update)
    if reset_sampling and sampling["active_mode"] != "adaptive":
        raise ValueError("--reset-adaptive-sampling requires an active adaptive sampling phase")
    if align_sampling and (sampling.get("sampling_contract") != "tracker_checkpoint_adaptive_v1"
                           or sampling.get("reference_tracker_sha256") != tracker_hash):
        raise ValueError("Sampling alignment requires the verified native tracker contract")
    if args.dr_sampling == "grid256_shared":
        sampling["scope"] = "motion/bin sampling only; 256 shared static DR profiles with {0,1,2,4}^4 limb loads"
    prepared.env.seed = rank_seed
    starts = configure_training_starts(prepared.env, TRAINING_START_PROFILE or
                                       ("original" if args.dr_profile == TRACKER_DR else "reference"))
    train = _build_train_configuration(
        source, tracker_checkpoint=tracker, tracker_actor_kwargs=prepared.actor_kwargs,
        tracker_obs_groups=prepared.obs_groups, baseline="no-latent", dynamics_latent_dim=64,
        residual_hidden_dims=(512, 256, 128), residual_scale=args.residual_scale,
        iterations=args.iterations or 0, num_steps_per_env=args.rollout_steps,
        save_interval=args.save_interval, seed=args.seed, logger="tensorboard",
        wandb_project="limb-context", actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr, check_for_nan=True,
    )
    train = configure_context_models(train, args.fusion, scratch_seed=args.seed)
    if args.initial_action_std is not None:
        if not 0 < args.initial_action_std < float("inf"):
            raise ValueError("initial_action_std must be finite and positive")
        train["actor"]["initial_action_std"] = args.initial_action_std
    if args.residual_output_mode is not None:
        if args.residual_output_mode == "unbounded" and args.residual_scale != 1.0:
            raise ValueError("Unbounded residuals use raw MLP outputs; set residual_scale=1.0")
        if args.residual_output_mode == "unbounded":
            action_cfg = prepared.env.actions["joint_pos"]
            if (prepared.clip_actions is not None or action_cfg.clip is not None
                    or action_cfg.raw_action_clip is not None):
                raise ValueError("Unbounded residual PPO requires an unclipped action chain")
        train["actor"]["residual_output_mode"] = args.residual_output_mode
    train["max_iterations"] = args.iterations
    train["algorithm"].update(schedule="fixed", entropy_coef=args.entropy_coef,
                              num_learning_epochs=args.epochs, num_mini_batches=args.mini_batches,
                              adaptive_critic_learning_rate=False,
                              class_name=PPO_CLASS_NAME)
    context = (load_frozen_context_checkpoint(args.context_checkpoint, device=device,
                                              expected_tracker_sha256=tracker_hash)
               if args.context_checkpoint else None)
    frozen_dependencies = ({"inference_bundle_version": BUNDLE_VERSION}
                           if EMBED_TRACKER_WITHOUT_CONTEXT else {})
    if context is not None:
        context_meta = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)
        validate_context_dr(context_meta, CONTEXT_SOURCE_DR_PROFILE or args.dr_profile)
        frozen_dependencies = {
            "inference_bundle_version": BUNDLE_VERSION,
            "frozen_context": make_context_bundle(context_meta, source_path=context.path,
                                                   source_sha256=context.sha256),
        }
        del context_meta
    if context is not None and context.config.dynamics_latent_dim != 64:
        raise ValueError("This experiment requires a 64-dimensional frozen context")
    metadata = {
        "version": VERSION, "fusion": args.fusion, "arguments": vars(args),
        "dr_profile": args.dr_profile, "dr_sampling": args.dr_sampling,
        "execution_protocol": f"memory350_{distributed.world_size}_gpu_scratch_v1",
        "maximum_updates": args.iterations,
        "episode_length_control_steps": args.episode_steps,
        "context_protocol": "short50_disjoint_chunk10_long30_cross_reset_v1",
        "frozen_inference": "cached chunk and long summaries; final short-context attention every control step",
        "initialization_protocol": SCRATCH_INITIALIZATION,
        "residual_diagnostics": {
            "scope": "post-update observation batch on all ranks, not the entire rollout",
            "action_units": "policy commands; deterministic residual mean excludes exploration noise",
            "target_units": "radians before SP delay, smoothing and joint offsets",
            "aggregation": "pooled RMS and mean absolute value; global maximum; equal rank batch sizes",
        },
        "wandb_logging": {
            "schema": "tracker_namespaces_v1",
            "metric_groups": list(TRACKER_WANDB_GROUPS),
            "step_metric": "completed_updates",
            "experiment_group": args.wandb_group,
            "legacy_training_metrics": True,
        },
        "distributed": {"world_size": distributed.world_size, "num_envs_per_rank": args.num_envs,
                        "global_num_envs": args.num_envs * distributed.world_size,
                        "rank_seed_formula": "seed + 1000003 * rank",
                        "gradient_synchronization": "mean across all ranks at each optimizer step",
                        "advantage_normalization": "global rollout mean and sample standard deviation",
                        "critic_normalization": "global pooled sums, squared sums and count at every environment step"},
        "tracker_checkpoint": str(tracker), "tracker_sha256": tracker_hash,
        "context_checkpoint": context.path if context else None,
        "context_sha256": context.sha256 if context else None,
        "dataset": dataset, "motion_count": len(files), "motion_file": motion_file,
        "motion_path": motion_path, "physics": physics, "training_starts": starts,
        "reward_contract": rewards, "reward_changes": {},
        "motion_sampling": sampling,
        "training_terminations": terminations,
        "actor_initialization": (
            "fresh residual MLP; random hidden layers, zero output layer, fresh scalar action std "
            f"{train['actor'].get('initial_action_std', SCRATCH_ACTION_STD)}"),
        "critic_initialization": "fresh random MLP; no source critic weights",
        "critic_normalization_initialization": "fresh moments from one pooled initial observation batch; then updated online",
        "initialization_seeds": {"actor": args.seed + 10007, "critic": args.seed + 20003},
        "frozen_tracker_role": "pretrained base action and original 1645-dimensional preprocessing",
        "encoder_frozen": True, "context_normalization_frozen": True,
        "policy_precision": args.policy_precision, "policy_precision_audit": precision_audit,
        "predictor_executed_in_ppo": False, "extra_current_state_physics_privilege": False,
        "stop_contract": "finish current update and save; total completed count is authoritative",
        "resume_contract": "restore model optimizer normalizers and update count; restart simulator episodes",
        "research_source_sha256": {str(p.relative_to(PROJECT_ROOT)): _sha256(p) for p in (
            Path(__file__).resolve(), *sorted((PROJECT_ROOT / "src/intact_tracking").glob("limb_context_*.py")),
            *sorted((PROJECT_ROOT / "src/intact_tracking").glob("memory350_*.py")),
            PROJECT_ROOT / "src/intact_tracking/residual_policy.py",
            PROJECT_ROOT / "src/intact_tracking/residual_dr_aux.py",
            PROJECT_ROOT / "src/intact_tracking/residual_runner.py")},
    }
    if METADATA_ADAPTER is not None:
        METADATA_ADAPTER(metadata, args, previous)
    endpoint_evaluator = None
    if args.endpoint_eval_protocol:
        from intact_tracking.memory350_policy_checkpoint_eval import PeriodicEndpointEvaluator, digest
        endpoint_evaluator = PeriodicEndpointEvaluator(output, args.endpoint_eval_protocol, distributed)
        metadata["periodic_evaluation"] = {"protocol": endpoint_evaluator.protocol,
            "protocol_file": str(Path(args.endpoint_eval_protocol).resolve()),
            "protocol_sha256": digest(args.endpoint_eval_protocol),
            "checkpoint_counting": "completed PPO updates; checkpoint_update_XXXXXX.pt",
            "execution": "all ranks pause PPO; separate endpoint simulators on the first two allocated GPUs"}
    if args.resume:
        old = previous["residual_policy"]
        if old.get("dr_sampling", "independent_uniform") != args.dr_sampling:
            raise ValueError("Resume changed the DR sampling contract; start a separate experiment")
        validate_termination_resume(previous, terminations)
        previous_sampling_mode = validate_sampling_resume(
            old, sampling, previous["completed_updates"], align_to_tracker=align_sampling,
            switch_to_uniform=uniform_resume)
        if align_sampling:
            previous_rewind = OmegaConf.to_container(previous["cfg"].task.command.command.rewind, resolve=True)
            previous_rewind["enabled"] = sampling["rewind"]["enabled"]
            if previous_rewind != sampling["rewind"]:
                raise ValueError("Sampling alignment changed tracker rewind probability or offsets")
        expected_resume_digest = state_digest(previous.get("rsl_rl", previous))
        for key in ("version", "fusion", "tracker_sha256", "context_sha256", "reward_contract", "initialization_protocol"):
            if old.get(key) != metadata[key]:
                raise ValueError(f"Resume changed {key}")
        if old["dataset"]["manifest_sha256"] != dataset["manifest_sha256"]:
            raise ValueError("Resume changed dataset")
        for key in ("seed", "num_envs", "rollout_steps", "episode_steps"):
            if old["arguments"][key] != vars(args)[key]:
                raise ValueError(f"Resume changed {key}")
        if old.get("distributed") != metadata["distributed"]:
            raise ValueError("Resume changed distributed training scale")
        old_train = OmegaConf.to_container(previous["cfg"].agent, resolve=True)
        auxiliary_transition = validate_resume_models(old_train, train, args)
        metadata["resume_history"] = [*copy.deepcopy(old.get("resume_history", [])), {
            "checkpoint": str(Path(args.resume).resolve()), "checkpoint_sha256": _sha256(Path(args.resume)),
            "completed_updates": previous["completed_updates"], "unix_time": time.time(),
            "motion_sampling_from": previous_sampling_mode, "motion_sampling_to": sampling["active_mode"],
            "tracker_sampling_alignment": ({"from": copy.deepcopy(old["motion_sampling"]),
                                             "to": copy.deepcopy(sampling)} if align_sampling else None),
            "adaptive_statistics": ("disabled_and_discarded" if sampling["active_mode"] == "uniform" else
                                    "reset_to_priors" if reset_sampling else
                                    "restore_from_checkpoint" if previous_sampling_mode == "adaptive" else
                                    "initialize_from_priors"),
            "policy_precision_from": resolve_policy_precision(None, previous),
            "policy_precision_to": args.policy_precision,
            "model_optimizer_and_normalization_restored": True, "simulator_episodes_restarted": True}]
        if auxiliary_transition is not None:
            metadata["resume_history"][-1]["dr_auxiliary_transition"] = auxiliary_transition
        if endpoint_evaluator is not None:
            original = old.get("periodic_evaluation", {})
            from intact_tracking.memory350_policy_checkpoint_eval import resumed_evaluation_metadata
            metadata["periodic_evaluation"] = resumed_evaluation_metadata(
                original, metadata["periodic_evaluation"], previous["completed_updates"],
                allow_change=args.allow_evaluation_protocol_change)
        del previous
    elif endpoint_evaluator is not None:
        metadata["periodic_evaluation"]["enabled_after_update"] = 0
    assert_rewards_unchanged(rewards, prepared.env)
    _seed_everything(rank_seed)
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=device)
    handlers = {}
    wandb_logger = None
    completed_successfully = False
    try:
        if env.max_episode_length != args.episode_steps:
            raise RuntimeError("Runtime episode limit differs from the matched contract")
        termination_audit = audit_runtime_terminations(env.termination_manager, terminations)
        metadata["training_termination_runtime_audits"] = distributed.all_gather_object(termination_audit)
        command = env.command_manager.get_term("motion")
        if (command.cfg.sampling_mode != sampling["active_mode"]
                or command.cfg.rewind.enabled != sampling["failure_rewind_enabled"]):
            raise ValueError("Runtime motion sampling differs from the recorded configuration")
        if sampling.get("sampling_contract") in ("tracker_checkpoint_adaptive_v1", "tracker_checkpoint_uniform_v1"):
            from intact_tracking.memory350_native_policy import audit_sampling
            metadata["sampling_runtime_audits"] = distributed.all_gather_object(audit_sampling(command, sampling))
        expected = files[distributed.rank::distributed.world_size] if len(files) > 1 else files
        if not distributed.all_true(tuple(map(str, expected)) == command.motion_files):
            raise ValueError("Loaded rank shard differs from the full requested motion catalog")
        rank_audit = {"rank": distributed.rank, "seed": rank_seed,
                      "episode_length_control_steps": env.max_episode_length,
                      "motion_count": command.motion.num_files,
                      "loaded_frames": int(command.motion.file_lengths.sum()),
                      "physics": audit_limb_dr(env, physics)}
        ranks = distributed.all_gather_object(rank_audit)
        if args.dr_sampling == "grid256_shared":
            from intact_tracking.limb_context_grouped_dr import audit_grouped_ranks
            physics["grouped_rank_agreement"] = audit_grouped_ranks(ranks)
        dataset.update(loaded_motion_count=sum(r["motion_count"] for r in ranks) if len(files) > 1 else 1,
                       loaded_frames=sum(r["loaded_frames"] for r in ranks) if len(files) > 1 else ranks[0]["loaded_frames"],
                       rank_partition="sorted files [rank::world_size]; single-file smoke replicated",
                       loaded_motion_counts_per_rank=[r["motion_count"] for r in ranks],
                       reference_storage_mode=command.cfg.reference_storage_mode)
        physics["runtime_audits_by_rank"] = ranks
        physics = distributed.broadcast_object(physics)
        metadata["physics"] = physics
        if dataset["loaded_motion_count"] != len(files):
            raise ValueError("Distributed motion shards do not cover the full dataset")
        wrapped = (RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions) if args.fusion == "baseline"
                   else LimbContextWrapper(env, prepared.clip_actions, context))
        if hasattr(wrapped, "physics_schema"):
            metadata["physics_input_schema"] = copy.deepcopy(wrapped.physics_schema)
            if args.resume and old.get("physics_input_schema") != metadata["physics_input_schema"]:
                raise ValueError("Resume changed physical input names/ranges")
        _seed_everything(args.seed)
        cfg = memory_checkpoint_configuration(source, train, metadata)
        main_process_call(distributed, lambda: output.mkdir(parents=True, exist_ok=True))
        runner = ResidualOnPolicyRunner(wrapped, train, str(output), device,
                                       checkpoint_cfg=cfg, residual_metadata=metadata,
                                       frozen_dependencies=frozen_dependencies)
        if hasattr(wrapped, "bind_policy"):
            wrapped.bind_policy(runner.alg.actor)
        if hasattr(runner.alg, "bind_environment"):
            runner.alg.bind_environment(wrapped)
        if frozen_dependencies:
            runner.frozen_dependencies["frozen_tracker"] = make_tracker_bundle(
                source, runner.alg.actor.tracker.state_dict(), source_path=tracker,
                source_sha256=tracker_hash)
        boundary_contract = getattr(runner.alg, "motion_boundary_contract", None)
        if boundary_contract is not None:
            metadata["motion_boundary_contract"] = copy.deepcopy(boundary_contract)
            if args.resume:
                prior_contract = old.get("motion_boundary_contract")
                if prior_contract is not None and prior_contract != boundary_contract:
                    raise ValueError("Resume changed an existing motion boundary contract")
                if prior_contract is None:
                    metadata["resume_history"][-1]["motion_boundary_fix"] = {
                        "from": "environment_dones_only",
                        "to": copy.deepcopy(boundary_contract),
                        "applied_after_completed_updates": resume_update,
                    }
        if args.dr_sampling == "grid256_shared":
            runner.alg.actor.configure_grouped_diagnostics(env.num_envs)
        if distributed.enabled:
            synchronize_critic_normalization(runner.alg.critic)
        # Extra FiLM parameter construction must not change the observation-noise
        # draw used to initialize the common critic's pooled moments.
        _seed_everything(rank_seed + 1700000)
        initial_obs = wrapped.get_observations()
        runner.alg.critic.train()
        # Bootstrap fresh statistics from the current DR worlds before the first
        # value estimate. Frozen tracker/context normalization is not changed.
        if not distributed.all_true(float(runner.alg.critic.obs_normalizer.count) == 0):
            raise ValueError("Scratch critic inherited old observation statistics")
        runner.alg.critic.update_normalization(initial_obs)
        metadata["input_audit"] = audit_initial_models(
            runner.alg.actor, runner.alg.critic, initial_obs, args.fusion)
        if args.resume:
            runner.load(args.resume, map_location=device)
            restored_digest = state_digest(runner.alg.save())
            if not distributed.all_true(restored_digest == expected_resume_digest):
                raise ValueError("Model/optimizer/normalization restoration differs from the selected checkpoint")
            metadata["resume_state_audit"] = {"passed": True, "checkpoint_update": resume_update,
                                              "model_optimizer_state_sha256": restored_digest}
            metadata["resume_history"][-1]["restoration_audit"] = copy.deepcopy(metadata["resume_state_audit"])
            # Preserve the true original initialization audit across simulator restarts.
            current_auxiliary_audit = metadata["input_audit"].get("dr_auxiliary")
            metadata["input_audit"] = copy.deepcopy(old["input_audit"])
            if auxiliary_transition is not None:
                metadata["input_audit"]["dr_auxiliary"] = copy.deepcopy(current_auxiliary_audit)
        sampler_checkpoint = SamplingCheckpoint(output, distributed, sampling)
        if args.resume:
            restored_sampling = sampler_checkpoint.restore(runner, previous_sampling_mode,
                                                           reset=reset_sampling, align_to_tracker=align_sampling)
            metadata["sampling_resume_audits"] = distributed.all_gather_object(restored_sampling)
            metadata["resume_history"][-1]["sampling_audits"] = copy.deepcopy(metadata["sampling_resume_audits"])
        runner.checkpoint_state_preparer = sampler_checkpoint.prepare
        cfg = memory_checkpoint_configuration(source, train, metadata)
        runner.checkpoint_cfg, runner.residual_metadata = cfg, copy.deepcopy(metadata)
        main_process_call(distributed, lambda: _prepare_output(output, resume=args.resume, run_config=metadata, checkpoint_config=cfg))
        def initialize_wandb():
            nonlocal wandb_logger
            wandb_logger = WandbLogger(enabled=True, is_main=True, project=args.wandb_project,
                entity=args.wandb_entity, group=args.wandb_group, name=args.wandb_name or output.name,
                output_dir=output, config=metadata, tags=WANDB_TAGS or ("memory350", "residual", args.fusion))
            wandb_logger.run.config.update({"policy_precision": args.policy_precision,
                "policy_precision_audit": precision_audit,
                "periodic_evaluation": metadata.get("periodic_evaluation"),
                "motion_boundary_contract": metadata.get("motion_boundary_contract"),
                "resume_history": metadata.get("resume_history", [])}, allow_val_change=True)
            wandb_logger.run.define_metric("completed_updates")
            wandb_logger.run.define_metric("training/*", step_metric="completed_updates")
            for group in TRACKER_WANDB_GROUPS:
                wandb_logger.run.define_metric(f"{group}/*", step_metric="completed_updates")
            (output / "wandb_run.json").write_text(json.dumps({"id": wandb_logger.id,
                "url": wandb_logger.url, "project": args.wandb_project, "group": args.wandb_group}, indent=2) + "\n")
        main_process_call(distributed, initialize_wandb)
        attach_json_logger(runner, output, distributed, wandb_logger)
        if not args.resume:
            main_process_call(distributed, lambda: runner.save(str(output / "checkpoint_initial.pt")))
        elif reset_sampling or align_sampling or uniform_resume:
            sampler_checkpoint.prepare(runner)
            main_process_call(distributed, lambda: runner.save(str(output / "checkpoint_resume.pt")))
        if args.resume:
            # Saved snapshots keep the last completed iteration; the loop
            # starts from the next one only after writing the resume snapshot.
            runner.current_learning_iteration = runner.completed_learning_updates

        def stop(signum, _frame):
            runner.request_stop()
            os.write(2, f"Signal {signum}: saving after the current complete PPO update.\n".encode())

        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, stop)
        if endpoint_evaluator is not None:
            runner.checkpoint_evaluator = endpoint_evaluator
            # The first evaluation after enabling this feature also covers the
            # current saved model. Older unsaved 100-update snapshots cannot be reconstructed.
            endpoint_evaluator(runner, force=True)
        _seed_everything(rank_seed + 1 + runner.completed_learning_updates)
        remaining = phase_budget(args.motion_sampling, args.adaptive_after_update,
                                 runner.completed_learning_updates, args.iterations)
        print(json.dumps({"phase": "ppo_ready", "fusion": args.fusion,
                          "input_audit": metadata["input_audit"], "remaining_updates": remaining,
                          "training_terminations": termination_audit,
                          "motion_sampling": sampling}), flush=True)
        if remaining is None or remaining > 0:
            runner.learn(remaining, init_at_random_ep_len=True)
        final_grouped_agreement = None
        if args.dr_sampling == "grid256_shared":
            final_physics = audit_limb_dr(env, physics)
            final_ranks = distributed.all_gather_object({"physics": final_physics})
            final_grouped_agreement = audit_grouped_ranks(final_ranks)
            if final_grouped_agreement["static_bank_sha256"] != physics["grouped_rank_agreement"]["static_bank_sha256"]:
                raise RuntimeError("Grid256 physics changed during PPO or episode resets")
        agreement = audit_rank_agreement(runner, distributed)
        if hasattr(runner.alg, "world_model"):
            digests = distributed.all_gather_object(state_digest({
                "encoder": runner.alg.actor.history_encoder.state_dict(),
                "world_model": runner.alg.world_model.state_dict(),
                "optimizer": runner.alg.world_optimizer.state_dict()}))
            if len(set(digests)) != 1:
                raise RuntimeError("Distributed world-model parameters/optimizer diverged")
            agreement["world_model_state_sha256"] = digests[0]
        stopped = not distributed.all_true(not runner.stop_requested)
        planned_transition = (not stopped and args.motion_sampling == "adaptive"
                              and sampling["active_mode"] == "uniform"
                              and runner.completed_learning_updates == args.adaptive_after_update
                              and (args.iterations is None or args.iterations > args.adaptive_after_update))
        if planned_transition:
            boundary = output / f"checkpoint_update_{runner.completed_learning_updates:06d}.pt"
            main_process_call(distributed, lambda: runner.save(str(boundary)) if not boundary.exists() else None)
        main_process_call(distributed, lambda: (output / "completion.json").write_text(json.dumps({
            "target_updates": args.iterations, "completed_updates": runner.completed_learning_updates,
            "complete": args.iterations is not None and runner.completed_learning_updates >= args.iterations,
            "stopped": stopped, "distributed_parameter_agreement": agreement,
            "planned_sampling_transition": planned_transition, "motion_sampling": sampling,
            "training_terminations": terminations,
            "grouped_dr_final_agreement": final_grouped_agreement,
            "initialization_protocol": SCRATCH_INITIALIZATION,
            "distributed": metadata["distributed"], "unix_time": time.time()}, indent=2) + "\n"))
        completed_successfully = True
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        env.close()
        if wandb_logger is not None:
            wandb_logger.finish(exit_code=0 if completed_successfully else 1)


def run(args):
    distributed = DistributedContext.initialize(requested_device=args.device,
                                                requested_backend=args.distributed_backend)
    if distributed.device.type == "cuda":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(distributed.local_rank)
    try:
        return _run(args, distributed)
    finally:
        distributed.close()


if __name__ == "__main__":
    run(build_parser().parse_args())
