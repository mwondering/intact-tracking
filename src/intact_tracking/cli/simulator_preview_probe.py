"""Separate, non-acceptance run measuring actor/critic response to preview input."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from intact_tracking.adaptation_policy import PRIVILEGE
from intact_tracking.adaptation_reward_contract import (
    assert_fixed_reward_checkpoint,
    capture_original_rewards,
)
from intact_tracking.cli.adaptation_eval import TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.residual_policy import WarmStartedHeftCritic
from intact_tracking.simulator_preview import (
    OUTCOME_DIM,
    PreviewAwareHeftCritic,
    SimulatorPreviewWrapper,
)


def intervention(obs, mode):
    value = obs.clone(recurse=True)
    if mode == "zero":
        value[PRIVILEGE][:, -OUTCOME_DIM:] = 0
    elif mode == "shuffle":
        value[PRIVILEGE][:, -OUTCOME_DIM:] = obs[PRIVILEGE][:, -OUTCOME_DIM:].roll(1, 0)
    else:
        raise ValueError(mode)
    return value


def run(args):
    _seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prepared = prepare_rollout(checkpoint_file=TRACKER, motion_file=args.motion_file,
                               motion_path=None, num_envs=args.num_envs)
    assert_fixed_reward_checkpoint(checkpoint, capture_original_rewards(prepared.env))
    configure_physics(prepared.env, args.physics)
    prepared.env.seed = args.seed
    prepared.env.commands["motion"].sampling_mode = "uniform"
    prepared.env.commands["motion"].rewind.enabled = False
    env = ManagerBasedRlEnv(cfg=copy.deepcopy(prepared.env), device=args.device)
    wrapped = None
    try:
        wrapped = SimulatorPreviewWrapper(env, prepared.clip_actions, preview_mode="true")
        obs = wrapped.get_observations()
        actor = load_actor(args.checkpoint, prepared, obs, wrapped)
        wrapped.bind(actor)
        obs = wrapped.get_observations()
        agent = OmegaConf.to_container(checkpoint["cfg"].agent, resolve=True)
        kwargs = dict(agent["critic"])
        name = kwargs.pop("class_name")
        classes = {"WarmStartedHeftCritic": WarmStartedHeftCritic, "PreviewAwareHeftCritic": PreviewAwareHeftCritic}
        cls = classes[name.split(":")[-1]]
        critic = cls(obs, agent["obs_groups"], "critic", 1, **kwargs).to(env.device)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        critic.eval().requires_grad_(False)
        del checkpoint
        records = []
        for step in range(args.steps):
            with torch.no_grad():
                action, value = actor(obs), critic(obs)
                if step % 8 == 0:
                    base_action = actor._base_features_and_action(obs)[1]
                    record = {"step": step, "actor_action_rms": float(action.square().mean().sqrt()),
                              "actor_residual_rms": float((action - base_action).square().mean().sqrt()),
                              "critic_value_mean": float(value.mean()), "critic_value_std": float(value.std())}
                    for mode in ("zero", "shuffle"):
                        changed = intervention(obs, mode)
                        record[f"actor_{mode}_rms_difference"] = float((actor(changed) - action).square().mean().sqrt())
                        record[f"critic_{mode}_rms_difference"] = float((critic(changed) - value).square().mean().sqrt())
                    records.append(record)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, *_ = wrapped.step(action)
        # Verify a differentiable route to the actual outcome slots, not just
        # to common state/physics/reference. This is NOT an efficacy test.
        differentiated = obs.clone(recurse=True)
        differentiated[PRIVILEGE].requires_grad_(True)
        gradient_stats = {}
        for label, model in (("actor", actor), ("critic", critic)):
            output = model(differentiated)
            grad = torch.autograd.grad(output.sum(), differentiated[PRIVILEGE], allow_unused=True)[0] if output.requires_grad else None
            gradient_stats[label + "_outcome_gradient_rms"] = float(grad[:, -OUTCOME_DIM:].square().mean().sqrt()) if grad is not None else 0.0
        result = {"checkpoint": args.checkpoint, "physics": args.physics, "seed": args.seed,
                  "critic_class": name, "gradient": gradient_stats, "samples": records,
                  "warning": "Input intervention/gradient proves functional use only; mismatched or zero inputs can be out of distribution. Does NOT establish improved training or tracking.",
                  "actor_and_critic_parameters_shared": False}
        for key in records[0]:
            if key != "step":
                result[key] = sum(record[key] for record in records) / len(records)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close_preview()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physics", choices=("nominal", "dr"), default="dr")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13091)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
