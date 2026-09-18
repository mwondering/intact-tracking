"""Frozen tracker FiLM, with or without latent in both actor and critic."""

from functools import partial
import os
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.memory350_tracker_film import (
    VERSION, INITIALIZATION, CONDITIONING, configure_models, audit_initial_models,
)
from intact_tracking.memory350_tracker_film_training import TrackerFiLMRunner
from intact_tracking.residual_uniform_protocol import configure_physics, audit_physics, physics_contract


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    for action in list(parser._actions):
        if action.dest in ("fusion", "residual_scale", "endpoint_eval_protocol"):
            parser._remove_action(action)
            for option in action.option_strings:
                parser._option_string_actions.pop(option, None)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
        elif action.dest == "training_ranks":
            action.choices = (1, 2, 4, 8)
        elif action.dest == "motion_sampling":
            action.choices = ("uniform",)
        elif action.dest == "training_terminations":
            action.choices = ("original",)
        elif action.dest == "dr_sampling":
            action.choices = ("independent_uniform",)
        elif action.dest == "context_checkpoint":
            action.required = True
    parser.set_defaults(fusion="film", residual_scale=1.0, endpoint_eval_protocol=None,
        motion_sampling="uniform", adaptive_after_update=0, training_terminations="original",
        iterations=None, policy_precision="fp32", wandb_group="memory350-direct-tracker-film")
    parser.add_argument("--conditioning", dest="actor_conditioning", choices=CONDITIONING, required=True,
                        help="latent: actor/critic add real latent after their separate observation compressors; constant: both latent slots are zeros.")
    parser.add_argument("--tracker-warmup-steps", type=int, default=0,
                        help="Optional frozen-tracker history steps before PPO; default 0 starts with empty memory.")
    parser.add_argument("--wandb-mode", choices=("online", "offline"),
                        default=os.environ.get("WANDB_MODE", "online"))
    return parser


def main():
    args = build_parser().parse_args()
    if args.tracker_warmup_steps < 0:
        raise ValueError("Tracker warmup steps must be nonnegative")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
        if saved["version"] != VERSION or saved["physics"].get("residual_physics_contract") != physics_contract():
            raise ValueError("Resume requires the same tracker-FiLM and physics protocol")
        if saved["arguments"]["actor_conditioning"] != args.actor_conditioning:
            raise ValueError("Resume cannot change the actor's access to environment information")
    base.VERSION, base.SCRATCH_INITIALIZATION = VERSION, INITIALIZATION
    base.WandbLogger = partial(base.WandbLogger, mode=args.wandb_mode)
    base.ResidualOnPolicyRunner = TrackerFiLMRunner
    base.configure_context_models = partial(configure_models, actor_conditioning=args.actor_conditioning)
    base.audit_initial_models = audit_initial_models
    base.configure_limb_dr, base.audit_limb_dr = configure_physics, audit_physics
    original_configuration = base.memory_checkpoint_configuration
    original_sampling = base.configure_motion_sampling

    def sampling(*values, **kwargs):
        result = original_sampling(*values, **kwargs)
        result.update(
            scope="Uniform motion sampling; independent hand [0,2.5] / shin [0,4] kg plus original tracker DR",
            statistics="Uniform throughout; no adaptive curriculum or failure rewind")
        return result

    def configuration(source, train, metadata):
        metadata.update(initialization_protocol=INITIALIZATION,
            execution_protocol=f"direct_tracker_film_{args.training_ranks}_gpu_v1",
            actor_initialization="original tracker weights; zero FiLM output heads; fresh exploration std 0.25",
            frozen_tracker_role="original perception and control weights; hidden activations modulated at 512/256/128",
            actor_conditioning=args.actor_conditioning,
            baseline_contract="both actors' FiLM reads compressed observations; constant arm masks the 64-D latent in both actor and critic",
            critic_initialization="scratch compressed MLP; concat real latent in latent arm, zeros in constant arm; no FiLM",
            residual_network=False,
            action_formula="modulated_tracker_mean + Gaussian exploration; no external residual or action clamp",
            evaluation_module="intact_tracking.cli.memory350_tracker_film_eval")
        for path in (Path(__file__), Path(__file__).with_name("memory350_tracker_film_eval.py"),
                     Path(__file__).parents[1] / "residual_uniform_protocol.py"):
            metadata["research_source_sha256"][str(path.resolve().relative_to(base.PROJECT_ROOT))] = base._sha256(path)
        return original_configuration(source, train, metadata)

    base.configure_motion_sampling = sampling
    base.memory_checkpoint_configuration = configuration
    base.run(args)


if __name__ == "__main__":
    main()
