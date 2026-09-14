"""Matched A/B/C LaFAN fine-tuning; torchrun uses num-envs per rank."""

from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.adaptation_curriculum import configure_training_starts
from intact_tracking.adaptation_reward_contract import (
    assert_rewards_unchanged,
    capture_original_rewards,
)
from intact_tracking.cli.adaptation_eval import TRACKER
from intact_tracking.cli.residual_policy_train import (
    _audit_nominal_runtime,
    _build_train_configuration,
    _checkpoint_configuration,
    _prepare_output,
    _seed_everything,
)
from intact_tracking.distributed import DistributedContext
from intact_tracking.environment.runtime import _load_saved_config, prepare_rollout
from intact_tracking.residual_runner import ResidualOnPolicyRunner
from intact_tracking.rollout.mjlab_adapter import _sha256
from intact_tracking.tracker_finetune import LAFAN, VERSION, configure_abc_physics, lafan_files


def run(args, distributed):
    output = Path(args.output_dir).resolve()
    if distributed.is_main:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True)
    distributed.barrier()
    files = lafan_files(args.motion_path)
    manifest = output / "motions.txt"
    if distributed.is_main:
        manifest.write_text("\n".join(map(str, files)) + "\n")
    distributed.barrier()
    rank_seed = args.seed + distributed.rank
    _seed_everything(rank_seed)
    tracker = Path(args.tracker_checkpoint).resolve()
    prepared = prepare_rollout(checkpoint_file=str(tracker), num_envs=args.num_envs,
                               motion_path=args.motion_path, motion_file=None)
    rewards = capture_original_rewards(prepared.env)
    physics = configure_abc_physics(prepared.env, args.condition, rank_seed)
    prepared.env.seed = rank_seed
    # Explicit manifest bypasses automatic per-rank motion sharding: every rank
    # can sample every selected LaFAN clip, regardless of its load condition.
    prepared.env.commands["motion"].motion_manifest_file = str(manifest)
    starts = configure_training_starts(prepared.env, "reference")
    prepared.env.commands["motion"].sampling_mode = "uniform"
    prepared.env.commands["motion"].rewind.enabled = False
    source = _load_saved_config(tracker)
    train = _build_train_configuration(
        source, tracker_checkpoint=tracker, tracker_actor_kwargs=prepared.actor_kwargs,
        tracker_obs_groups=prepared.obs_groups, baseline="no-latent", dynamics_latent_dim=0,
        residual_hidden_dims=(512, 256, 128), residual_scale=1, iterations=args.iterations,
        num_steps_per_env=24, save_interval=100, seed=args.seed, logger="tensorboard",
        wandb_project="intact-preview-v2", actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr, check_for_nan=True)
    train["actor"] = {
        **copy.deepcopy(prepared.actor_kwargs),
        "class_name": "intact_tracking.tracker_finetune:TrackerFinetuneActor",
        "tracker_checkpoint": str(tracker)}
    train["critic"]["class_name"] = "intact_tracking.tracker_finetune:SynchronizedWarmStartedCritic"
    train["algorithm"].update(
        class_name="intact_tracking.adaptation_ppo:CriticWarmupPPO",
        critic_warmup_updates=args.critic_warmup, schedule="fixed", entropy_coef=0.0002,
        num_learning_epochs=5, num_mini_batches=4, adaptive_critic_learning_rate=False)
    manifest_data = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    metadata = {
        "version": VERSION, "condition": args.condition, "variant": args.condition,
        "arguments": vars(args), "tracker_checkpoint": str(tracker), "tracker_sha256": _sha256(tracker),
        "motion_path": str(Path(args.motion_path).resolve()), "motion_file": None,
        "motion_count": len(files), "motion_files": list(map(str, files)),
        "motion_sha256": hashlib.sha256(json.dumps(manifest_data).encode()).hexdigest(),
        "physics_mode": physics["physics"], "physics": physics, "dr_profile": "abc-load-only",
        "reward_contract": rewards, "reward_changes": {}, "training_starts": starts,
        "world_size": distributed.world_size, "num_envs_per_rank": args.num_envs,
        "global_num_envs": args.num_envs * distributed.world_size,
        "real_transitions_per_update": args.num_envs * distributed.world_size * 24,
        "trainable_actor": "original control MLP and exploration std only; no residual",
        "frozen_actor": "height/contact estimator, reference encoder, all actor normalizers",
        "critic": "original 6330-D input, restored 1024-512-512-1 weights and DecayVecNorm; synchronized statistics",
        "extra_input_dim": 0, "preview": False, "latent": False,
        "fixed_reward_lineage": {"original_tracker_pretraining_dr": True,
                                  "historical_shaped_checkpoint_used": False},
        "research_source_sha256": {str(path.relative_to(Path.cwd())): _sha256(path) for path in (
            Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "tracker_finetune.py")},
    }
    assert_rewards_unchanged(rewards, prepared.env)
    env = ManagerBasedRlEnv(cfg=prepared.env, device=str(distributed.device))
    try:
        command = env.command_manager.get_term("motion")
        if command.motion_files != tuple(map(str, files)):
            raise RuntimeError("Every rank must see the complete selected LaFAN catalog")
        runtime = (_audit_nominal_runtime(env) if args.condition == "A"
                   else env.event_manager.get_term_cfg("abc_payload").func.audit())
        rank_audits = [None] * distributed.world_size
        if distributed.enabled:
            torch.distributed.all_gather_object(rank_audits, runtime)
        else:
            rank_audits = [runtime]
        metadata["runtime_audits_by_rank"] = rank_audits
        metadata["loaded_frames"] = int(command.motion.file_lengths.sum())
        wrapped = RslRlVecEnvWrapper(env, prepared.clip_actions)
        _seed_everything(args.seed)
        cfg = _checkpoint_configuration(source, train, metadata)
        runner = ResidualOnPolicyRunner(wrapped, train, str(output), str(distributed.device),
                                        checkpoint_cfg=cfg, residual_metadata=metadata)
        actor, critic = runner.alg.actor, runner.alg.critic
        if actor.obs_groups != prepared.obs_groups["actor"] or critic.obs_dim != 6330:
            raise RuntimeError("Original actor/critic input contract changed")
        names = [name for name, param in actor.named_parameters() if param.requires_grad]
        if not names or any(not name.startswith(("mlp.", "distribution.")) for name in names):
            raise RuntimeError(f"Unexpected trainable actor parameters: {names}")
        original = torch.load(tracker, map_location="cpu", weights_only=False)
        for network, key in ((actor, "actor_state_dict"), (critic, "critic_state_dict")):
            for name, value in network.state_dict().items():
                if not torch.equal(value.cpu(), original[key][name]):
                    raise RuntimeError(f"Initial checkpoint tensor not restored exactly: {key} {name}")
        del original
        metadata["initial_actor_and_critic_bitwise_restored"] = True
        frozen_before = {name: value.clone() for name, value in actor.state_dict().items()
                         if not name.startswith(("mlp.", "distribution."))}
        metadata["input_audit"] = {"actor_groups": actor.obs_groups, "critic_groups": critic.obs_groups,
                                   "actor_control_input_dim": actor.policy_input_dim,
                                   "critic_input_dim": critic.obs_dim, "trainable_actor_parameters": names}
        cfg = _checkpoint_configuration(source, train, metadata)
        runner.checkpoint_cfg, runner.residual_metadata = cfg, copy.deepcopy(metadata)
        if distributed.is_main:
            _prepare_output(output, resume=None, run_config=metadata, checkpoint_config=cfg)
            runner.save(str(output / "checkpoint_initial.pt"))
            print(json.dumps({"condition": args.condition, "world_size": distributed.world_size,
                              "global_envs": metadata["global_num_envs"], "motions": len(files),
                              "runtime_audits": rank_audits}, indent=2), flush=True)
        distributed.barrier()
        _seed_everything(rank_seed + 10000)
        runner.learn(args.iterations, init_at_random_ep_len=True)
        for name, value in frozen_before.items():
            if not torch.equal(value, actor.state_dict()[name]):
                raise RuntimeError(f"Frozen actor frontend changed: {name}")
        digest = hashlib.sha256()
        for network in (actor, critic):
            for name, value in network.state_dict().items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite learned tensor: {name}")
                digest.update(value.detach().cpu().numpy().tobytes())
        hashes = [None] * distributed.world_size
        if distributed.enabled:
            torch.distributed.all_gather_object(hashes, digest.hexdigest())
        else:
            hashes = [digest.hexdigest()]
        if len(set(hashes)) != 1:
            raise RuntimeError("Distributed actor/critic parameters or normalizers diverged")
        if distributed.is_main:
            (output / "completion_audit.json").write_text(json.dumps({
                "completed_updates": runner.completed_learning_updates, "rank_network_sha256": hashes,
                "frozen_frontend_tensors_unchanged": len(frozen_before), "all_weights_finite": True,
                "reward_sha256": rewards["sha256"], "global_envs": metadata["global_num_envs"],
                "motion_count": len(files)}, indent=2) + "\n")
        distributed.barrier()
    finally:
        env.close()


def main():
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("A", "B", "C"), required=True)
    parser.add_argument("--num-envs", type=int, default=2048, help="Environments per GPU/rank")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--motion-path", default=LAFAN)
    parser.add_argument("--tracker-checkpoint", default=TRACKER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--critic-warmup", type=int, default=0)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.actor_lr, args.critic_lr) <= 0 or args.critic_warmup < 0:
        parser.error("Invalid training settings")
    distributed = DistributedContext.initialize()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(distributed.local_rank)
    try:
        run(args, distributed)
    except Exception:
        # A failed rank must not enter NCCL destruction while its peer is
        # waiting in a different collective: that hides the original exception.
        traceback.print_exc()
        sys.stderr.flush()
        if distributed.enabled:
            os._exit(1)
        raise
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
