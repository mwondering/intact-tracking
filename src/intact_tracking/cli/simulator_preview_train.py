"""User-controlled v2 training: five-step preview versus original-input residual.

No training is launched on import. --iterations is explicitly chosen by the user;
SIGINT/SIGTERM requests a checkpoint after the current complete PPO update.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import (
    assert_fixed_reward_checkpoint,
    assert_rewards_unchanged,
    capture_original_rewards,
)
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER, configure_physics
from intact_tracking.cli.residual_policy_train import (
    _audit_nominal_runtime,
    _build_train_configuration,
    _checkpoint_configuration,
    _prepare_output,
    _seed_everything,
)
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.preview_protocol import (
    DR_PROFILES,
    LEGACY_PROFILE,
    LIMB_PROFILE,
    audit_limb_payloads,
    dataset_identity,
)
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.simulator_preview_experiment import (
    EXPERIMENT_VERSION,
    HORIZON,
    PREVIEW_DIM,
    PreviewObservationWrapper,
    audit_input_contract,
    configure_preview_comparison,
)

DEFAULT_MOTION = str(Path(DATASET) / "lafan_qingtong/dance1_subject2.motion.npz")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("preview", "baseline"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, required=True,
                        help="Number of additional PPO updates for this invocation; choose yourself")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Real training worlds; preview additionally allocates this many shadows")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-file", help="Single motion; defaults to the historical dance clip")
    motion.add_argument("--motion-path", help="Load every NPZ recursively, with no exclusions/subset")
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--physics", choices=("dr", "nominal"), default="dr")
    parser.add_argument("--dr-profile", choices=DR_PROFILES, default=LEGACY_PROFILE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--critic-warmup-updates", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.0002)
    parser.add_argument("--initial-action-std", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256, 128],
                        help="Actor residual hidden widths; critic is locked to 1024 512 512 256")
    parser.add_argument("--resume", help="Only v2 checkpoints; restores actor/critic/optimizer/normalizers")
    return parser


def validate_arguments(args):
    for name in ("iterations", "num_envs", "rollout_steps", "save_interval", "epochs", "mini_batches"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.critic_warmup_updates < 0 or any(width <= 0 for width in args.hidden_dims):
        raise ValueError("Warmup must be nonnegative and actor widths positive")
    for name in ("actor_lr", "critic_lr", "initial_action_std", "residual_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("entropy_coef must be finite and nonnegative")
    if args.num_envs * args.rollout_steps % args.mini_batches:
        raise ValueError("Rollout size must be divisible by the minibatch count")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point is single GPU per arm; select a GPU with CUDA_VISIBLE_DEVICES")


def validate_resume(checkpoint, metadata, train):
    previous = checkpoint.get("residual_policy", {})
    if previous.get("version") != EXPERIMENT_VERSION:
        raise ValueError("Only v2 checkpoints can resume; legacy critic/input dimensions are incompatible")
    for name in ("variant", "tracker_sha256", "motion_sha256", "physics_mode"):
        if previous.get(name) != metadata[name]:
            raise ValueError(f"Resume changes locked experiment field {name}")
    if previous.get("dr_profile", LEGACY_PROFILE) != metadata.get("dr_profile", LEGACY_PROFILE):
        raise ValueError("Resume changes locked DR profile")
    if previous.get("physics") and metadata.get("physics"):
        if previous["physics"]["details"] != metadata["physics"]["details"]:
            raise ValueError("Resume changes locked physics configuration")
    if previous.get("dataset", {}).get("manifest_sha256") != metadata.get("dataset", {}).get("manifest_sha256"):
        # Old single-motion v2 checkpoints have a full content hash but no manifest.
        if previous.get("dataset") or metadata.get("motion_count", 1) != 1:
            raise ValueError("Resume changes locked dataset manifest")
    saved_agent = OmegaConf.to_container(checkpoint["cfg"].agent, resolve=True)
    for name in ("actor", "critic", "obs_groups", "algorithm"):
        if saved_agent[name] != train[name]:
            raise ValueError(f"Resume changes {name}; reuse the original hyperparameters")
    for name in ("num_envs", "seed", "rollout_steps", "initial_action_std"):
        if name not in previous["arguments"] or name not in metadata["arguments"]:
            raise ValueError(f"Resume metadata is missing {name}")
        if previous["arguments"][name] != metadata["arguments"][name]:
            raise ValueError(f"Resume changes {name}; reuse the original experiment configuration")


def run(args):
    validate_arguments(args)
    tracker_path = Path(args.tracker_checkpoint).resolve()
    motion_file = None if args.motion_path else str(Path(args.motion_file or DEFAULT_MOTION).resolve())
    files, dataset = dataset_identity(motion_file, args.motion_path)
    output = Path(args.output_dir).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    if args.resume and output.exists() and any(output.iterdir()):
        if Path(args.resume).resolve().parent != output:
            raise ValueError("Resume into its original directory or a new empty directory, not another run")
    _seed_everything(args.seed)
    prepared = prepare_rollout(
        checkpoint_file=str(tracker_path), num_envs=args.num_envs,
        motion_file=motion_file, motion_path=args.motion_path,
    )
    source = _load_saved_config(tracker_path)
    reward_contract = capture_original_rewards(prepared.env)
    physics = configure_physics(prepared.env, args.physics, args.dr_profile)
    prepared.env.commands["motion"].motion_manifest_file = ""
    prepared.env.seed = args.seed
    training_starts = configure_training_starts(prepared.env, "reference")
    prepared.env.commands["motion"].sampling_mode = "uniform"
    prepared.env.commands["motion"].rewind.enabled = False
    train = _build_train_configuration(
        source, tracker_checkpoint=tracker_path, tracker_actor_kwargs=prepared.actor_kwargs,
        tracker_obs_groups=prepared.obs_groups, baseline="no-latent", dynamics_latent_dim=0,
        residual_hidden_dims=tuple(args.hidden_dims), residual_scale=args.residual_scale,
        iterations=args.iterations, num_steps_per_env=args.rollout_steps,
        save_interval=args.save_interval, seed=args.seed, logger="tensorboard",
        wandb_project="intact-preview-v2", actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr, check_for_nan=True,
    )
    train = configure_preview_comparison(train, args.variant)
    train["algorithm"].update(
        schedule="fixed", entropy_coef=args.entropy_coef, num_learning_epochs=args.epochs,
        num_mini_batches=args.mini_batches, adaptive_critic_learning_rate=False,
    )
    if args.critic_warmup_updates:
        train["algorithm"].update(
            class_name="intact_tracking.adaptation_ppo:CriticWarmupPPO",
            critic_warmup_updates=args.critic_warmup_updates,
        )
    metadata = {
        "version": EXPERIMENT_VERSION, "variant": args.variant, "arguments": vars(args),
        "physics_mode": args.physics, "physics": physics, "training_starts": training_starts,
        "tracker_checkpoint": str(tracker_path), "tracker_sha256": _sha256(tracker_path),
        "motion_file": motion_file, "motion_path": dataset["root"] if args.motion_path else None,
        "motion_sha256": _sha256(Path(motion_file)) if motion_file else dataset["manifest_sha256"],
        "motion_count": len(files), "dataset": dataset, "dr_profile": args.dr_profile,
        "reward_contract": reward_contract, "reward_changes": {},
        "simulator_preview": "true" if args.variant == "preview" else None,
        "simulator_preview_horizon": HORIZON if args.variant == "preview" else None,
        "simulator_preview_critic": args.variant == "preview",
        "extra_input_dim": PREVIEW_DIM if args.variant == "preview" else 0,
        "extra_input_layout": "5x71 targets then 5x(71 simulated state + 71 signed state error)",
        "extra_current_state_physics_privilege": False, "distance_scalars": False,
        "critic_initialization": "from scratch on a fresh run; resume restores its own learned state",
        "critic_additive_branches": False, "actor_preview_encoding": "normalized raw concatenation",
        "stop_contract": "SIGINT/SIGTERM finishes one PPO update then saves; never use SIGKILL to save",
        "resume_contract": "restore model/optimizer/normalizers/update count; simulator episodes restart",
        "fixed_reward_lineage": {"original_tracker_pretraining_dr": True,
                                 "historical_shaped_checkpoint_used": False},
        "research_source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[3])): _sha256(path)
            for path in (Path(__file__).resolve(),
                         Path(__file__).resolve().parents[1] / "simulator_preview_experiment.py",
                         Path(__file__).resolve().parents[1] / "simulator_preview.py",
                         Path(__file__).resolve().parents[1] / "preview_protocol.py",
                         Path(__file__).resolve().parents[1] / "environment/mdp/shared_motion.py",
                         Path(__file__).resolve().parents[1] / "environment/mdp/multi_commands.py",
                         Path(__file__).resolve().parents[1] / "residual_runner.py")
        },
    }
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert_fixed_reward_checkpoint(checkpoint, reward_contract)
        validate_resume(checkpoint, metadata, train)
        metadata["resumed_from"] = str(Path(args.resume).resolve())
        metadata["resumed_checkpoint_sha256"] = _sha256(Path(args.resume))
        del checkpoint
    assert_rewards_unchanged(reward_contract, prepared.env)
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=args.device)
    wrapped = None
    handlers = {}
    try:
        command = env.command_manager.get_term("motion")
        if tuple(map(str, files)) != command.motion_files:
            raise ValueError("Loaded motion catalog/order does not match the complete requested dataset")
        metadata["dataset"]["loaded_motion_count"] = command.motion.num_files
        metadata["dataset"]["loaded_frames"] = int(command.motion.file_lengths.sum())
        metadata["dataset"]["reference_storage_mode"] = command.cfg.reference_storage_mode
        if args.physics == "dr" and args.dr_profile == LIMB_PROFILE:
            physics["runtime_payload_audit"] = audit_limb_payloads(env)
        if args.physics == "nominal":
            physics["runtime_audit"] = _audit_nominal_runtime(env)
        wrapped = (
            PreviewObservationWrapper(env, prepared.clip_actions) if args.variant == "preview"
            else RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        )
        # Model initialization is independent of startup DR/shadow RNG consumption.
        _seed_everything(args.seed)
        cfg = _checkpoint_configuration(source, train, metadata)
        output.mkdir(parents=True, exist_ok=True)
        runner = ResidualOnPolicyRunner(
            wrapped, train, str(output), args.device, checkpoint_cfg=cfg, residual_metadata=metadata,
        )
        if args.variant == "preview":
            wrapped.bind(runner.alg.actor)
        obs = wrapped.get_observations()
        metadata["input_audit"] = audit_input_contract(runner.alg.actor, runner.alg.critic, obs, args.variant)
        if args.resume:
            runner.load(args.resume, map_location=args.device)
            # Stored iter is zero-based LAST completed update, not the next one.
            runner.current_learning_iteration = runner.completed_learning_updates
        else:
            with torch.no_grad():
                actor = runner.alg.actor
                _, base = actor._base_features_and_action(obs)
                torch.testing.assert_close(actor(obs), base, atol=0, rtol=0)
                distribution = actor.distribution
                value = math.log(args.initial_action_std) if distribution.std_type == "log" else args.initial_action_std
                distribution.std_param.fill_(value)
        cfg = _checkpoint_configuration(source, train, metadata)
        runner.checkpoint_cfg, runner.residual_metadata = cfg, copy.deepcopy(metadata)
        _prepare_output(output, resume=args.resume, run_config=metadata, checkpoint_config=cfg)
        if not args.resume:
            runner.save(str(output / "checkpoint_initial.pt"), infos={"initial_action_max_difference": 0.0})

        def stop(signum, _frame):
            runner.request_stop()
            # Avoid re-entering TextIO's buffered print while handling a signal.
            os.write(2, f"Received {signal.Signals(signum).name}: finish this PPO update, save, then exit.\n".encode())

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, stop)
        _seed_everything(args.seed + 1 + runner.completed_learning_updates)
        print(json.dumps({"variant": args.variant, "real_envs": args.num_envs,
                          "shadow_envs": args.num_envs if args.variant == "preview" else 0,
                          "additional_updates": args.iterations,
                          "completed_updates": runner.completed_learning_updates,
                          "input_audit": metadata["input_audit"]}, indent=2), flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=True)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if isinstance(wrapped, PreviewObservationWrapper):
            wrapped.close_preview()
        env.close()


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
