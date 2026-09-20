"""Measure PPO/DR gradients on a checkpoint copy without taking optimizer steps."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf
from rsl_rl.storage import RolloutStorage

from intact_tracking.cli.memory350_proprio_native_policy_train import configure_proprio_physics
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.memory350_inference import load_memory350_checkpoint
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.memory350_tracker_action_policy import (
    TrackerActionResidualActor, TrackerActionCritic, TrackerActionPPO,
)
from intact_tracking.residual_dr_aux import DR_TARGET_GROUP, DR_HISTORY_WEIGHT_GROUP
from intact_tracking.limb_context_distributed import tensor_digest
from intact_tracking import memory350_native_policy as native


def gradient_summary(ppo, auxiliary):
    p2 = sum(x.double().square().sum() for x in ppo)
    a2 = sum(x.double().square().sum() for x in auxiliary)
    dot = sum((x.double() * y.double()).sum() for x, y in zip(ppo, auxiliary, strict=True))
    pn, an = float(p2.sqrt()), float(a2.sqrt())
    return {"ppo_norm": pn, "weighted_aux_norm": an,
            "aux_to_ppo_ratio": an / pn if pn else None,
            "cosine": float(dot) / (pn * an) if pn and an else None}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure_policy_precision("fp32")
    torch.cuda.set_per_process_memory_fraction(.30)
    _seed_everything(args.seed)
    from intact_tracking.environment.config import EVENT_TERMS
    EVENT_TERMS["terrain_height_offset"] = native.FlatTerrainHeightOffset
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = state["residual_policy"]
    agent = OmegaConf.to_container(state["cfg"].agent, resolve=True)
    prepared = prepare_rollout(checkpoint_file=meta["tracker_checkpoint"], num_envs=args.num_envs,
                               motion_file=None, motion_path=meta["motion_path"])
    cfg = prepared.env
    configure_proprio_physics(cfg, args.seed, profile=native.PROFILE)
    cfg.commands["motion"].motion_manifest_file = str(Path(args.manifest).resolve())
    cfg.episode_length_s = 500 * cfg.decimation * cfg.sim.mujoco.timestep
    cfg.seed, cfg.auto_reset = args.seed, True
    context = load_memory350_checkpoint(meta["context_checkpoint"], device="cuda:0",
                                       expected_tracker_sha256=meta["tracker_sha256"])
    _seed_everything(args.seed)
    env = native.environment_factory(ManagerBasedRlEnv, cfg=cfg, device="cuda:0")
    try:
        wrapped = ProprioNativePolicyWrapper(env, prepared.clip_actions, context,
            latent_history_frames=5, dr_aux_schema=agent["actor"]["dr_aux_schema"])
        obs = wrapped.get_observations()
        kwargs = copy.deepcopy(agent["actor"]); kwargs.pop("class_name")
        actor = TrackerActionResidualActor(obs, agent["obs_groups"], "actor", 29, **kwargs).to(env.device)
        actor.load_state_dict(state["actor_state_dict"], strict=True)
        kwargs = copy.deepcopy(agent["critic"]); kwargs.pop("class_name")
        critic = TrackerActionCritic(obs, agent["obs_groups"], "critic", 1, **kwargs).to(env.device)
        critic.load_state_dict(state["critic_state_dict"], strict=True)
        if args.std_override is not None:
            # This modifies only the diagnostic copy, for a stated noise ablation.
            with torch.no_grad():
                actor.distribution.std_param.fill_(args.std_override)
        storage = RolloutStorage("rl", args.num_envs, 24, obs, [29], device=env.device)
        kwargs = copy.deepcopy(agent["algorithm"]); kwargs.pop("class_name")
        kwargs.update(num_learning_epochs=1, num_mini_batches=1)
        algorithm = TrackerActionPPO(actor, critic, storage, device=env.device, **kwargs)
        actor_before = tensor_digest(actor.named_parameters())
        actor.train(); critic.train()
        action_term = env.action_manager.get_term("joint_pos")
        with torch.inference_mode():
            for step in range(args.warmup_steps + args.settling_steps):
                if step < args.warmup_steps:
                    action = actor._base_features_and_action(obs)[1]
                    mean = action
                else:
                    action = actor(obs, stochastic_output=True)
                    mean = actor.output_mean
                action_term.record_policy_mean(mean)
                obs, _, _, _ = wrapped.step(action)
                if (step + 1) % 128 == 0:
                    print(json.dumps({"warmup_step": step + 1}), flush=True)
            reward_rows, failure_rows, counterfactual_rows = [], [], []
            terms = torch.zeros(len(env.reward_manager.active_terms), device=env.device)
            rate_names = ("action_rate_l2", "waist_action_rate_l2")
            rate_indices = [env.reward_manager.active_terms.index(name) for name in rate_names]
            rate_cfgs = [env.reward_manager.get_term_cfg(name) for name in rate_names]
            patterns = rate_cfgs[1].params["asset_cfg"].joint_names
            waist_mask = torch.tensor([any(re.search(pattern, name) for pattern in patterns)
                for name in action_term.target_names], dtype=torch.bool, device=env.device)
            reward_scale = env.step_dt if env.reward_manager._scale_by_dt else 1
            mean_rate_sums = torch.zeros(2, device=env.device)

            def rate_values(diff):
                return torch.stack((-diff.square().sum(-1) * rate_cfgs[0].weight,
                    diff[:, waist_mask].square().sum(-1) * rate_cfgs[1].weight), -1) * reward_scale

            for _ in range(24):
                action = algorithm.act(obs)
                action_term.record_policy_mean(actor.output_mean)
                assert action_term.cfg.prev_action_obs == "mean"
                history = action_term._action_mean_history
                mean_rates = rate_values(history[:, 0] - history[:, 1])
                sampled_rates = rate_values(action - action_term.get_recent_actions(1)[:, 0])
                obs, rewards, dones, extras = wrapped.step(action)
                actual_rates = env.reward_manager._step_reward[:, rate_indices] * reward_scale
                torch.testing.assert_close(sampled_rates, actual_rates, atol=2e-6, rtol=2e-5)
                reward_rows.append(rewards.clone())
                counterfactual_rows.append(rewards - actual_rates.sum(-1) + mean_rates.sum(-1))
                mean_rate_sums += mean_rates.mean(0)
                failure_rows.append(env.reset_terminated.clone())
                terms += env.reward_manager._step_reward.mean(0)
                algorithm.process_env_step(obs, rewards, dones, extras)
            algorithm.compute_returns(obs)
        batch = next(storage.mini_batch_generator(1, 1))
        if algorithm.normalize_advantage_per_mini_batch:
            batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        actor(batch.observations, stochastic_output=True)
        logprob = actor.get_output_log_prob(batch.actions)
        old_logprob = batch.old_actions_log_prob.squeeze(-1)
        ratio = (logprob - old_logprob).exp()
        advantages = batch.advantages.squeeze(-1)
        surrogate = torch.maximum(-advantages * ratio,
            -advantages * ratio.clamp(1 - algorithm.clip_param, 1 + algorithm.clip_param)).mean()
        policy_loss = surrogate - algorithm.entropy_coef * actor.output_entropy.mean()
        hidden = actor.dr_aux_features
        raw_aux, _ = actor.dr_aux_objective(actor.dr_aux_head(hidden),
            batch.observations[DR_TARGET_GROUP], batch.observations[DR_HISTORY_WEIGHT_GROUP])
        weighted_aux = algorithm.dr_aux_coef * raw_aux
        output_ids = {id(p) for p in actor.residual_mlp.base[-1].parameters()}
        shared = [(n, p) for n, p in actor.residual_mlp.named_parameters() if id(p) not in output_ids]
        targets = [p for _, p in shared] + [hidden]
        pg = torch.autograd.grad(policy_loss, targets, retain_graph=True)
        ag = torch.autograd.grad(weighted_aux, targets)
        assert all(bool(torch.isfinite(x).all()) for x in (*pg, *ag))
        assert tensor_digest(actor.named_parameters()) == actor_before
        rewards = torch.stack(reward_rows)
        counterfactual = torch.stack(counterfactual_rows)
        failures = torch.stack(failure_rows).bool()
        raw_advantage = (storage.returns - storage.values).squeeze(-1)
        def conditional_mean(value, mask):
            return float(value[mask].mean()) if bool(mask.any()) else None
        with Path(args.checkpoint).open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checkpoint_hash,
            "completed_updates": state["completed_updates"], "arguments": vars(args),
            "scope": "Independent local on-policy rollout on selected training motions and new native DR; no optimizer steps or training-process changes; gradients before clipping, no distributed reduction",
            "batch_size": len(batch.actions), "dr_aux_coef": algorithm.dr_aux_coef,
            "supervised_groups": actor.dr_aux_objective.supervised_groups,
            "raw_aux_loss": float(raw_aux.detach()), "weighted_aux_loss": float(weighted_aux.detach()),
            "shared_parameters": gradient_summary(pg[:-1], ag[:-1]),
            "shared_activations": gradient_summary(pg[-1:], ag[-1:]),
            "per_parameter": {n: gradient_summary([p], [a]) for (n, _), p, a in zip(shared, pg[:-1], ag[:-1], strict=True)},
            "max_preupdate_logprob_error": float((logprob.detach() - old_logprob).abs().max()),
            "rollout_reward_mean": float(rewards.mean()), "rollout_negative_reward_fraction": float((rewards < 0).float().mean()),
            "mean_action_rate_counterfactual": {
                "scope": "Same sampled actions, physics and transitions; replace only both action-rate reward terms by policy-mean differences; rewards passed to PPO remain unchanged",
                "reward_mean": float(counterfactual.mean()),
                "negative_reward_fraction": float((counterfactual < 0).float().mean()),
                "action_rate_terms_per_step": dict(zip(rate_names, (mean_rate_sums / 24).tolist(), strict=True)),
                "sampled_reward_reconstruction_verified": True},
            "failure_transition_fraction": float(failures.float().mean()),
            "terminal_advantage_mean": conditional_mean(raw_advantage, failures),
            "continuing_advantage_mean": conditional_mean(raw_advantage, ~failures),
            "reward_terms_per_control_step": dict(zip(env.reward_manager.active_terms,
                (terms / 24 * (env.step_dt if env.reward_manager._scale_by_dt else 1)).tolist(), strict=True)),
            "context": wrapped.latent_metrics,
            "actor_parameters_unchanged": True, "optimizer_steps": 0, "training_process_modified": False}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report[k] for k in ("completed_updates", "shared_parameters", "shared_activations", "rollout_reward_mean", "failure_transition_fraction")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=384)
    parser.add_argument("--settling-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=390919)
    parser.add_argument("--std-override", type=float)
    run(parser.parse_args())
