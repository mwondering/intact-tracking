"""Temporal observation/reference/latent/action Transformer versus a 1645-D MLP."""

from functools import partial
import os
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.limb_context_dr import LOAD_ONLY, TRACKER_DR
from intact_tracking.memory350_token_env import TokenPolicyWrapper
from intact_tracking.memory350_token_policy import (
    ARCHITECTURES, INITIALIZATION, VERSION, audit_initial_models, configure_models,
)
from intact_tracking.memory350_token_training import TemporalTokenRunner, audit_physics


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
        elif action.dest == "dr_profile":
            action.choices = (LOAD_ONLY, TRACKER_DR)
    parser.set_defaults(residual_scale=1.0, endpoint_eval_protocol=None,
                        dr_profile=LOAD_ONLY, training_ranks=4,
                        motion_sampling="uniform", adaptive_after_update=0,
                        training_terminations="original", iterations=None,
                        policy_precision="fp32", save_interval=250,
                        wandb_group="memory350-temporal29-vs-mlp")
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--critic-observation", choices=("privileged", "common"), default="privileged",
                        help="Transformer critic: add compressed original critic obs as token 30, or use exactly the common 29 tokens")
    parser.add_argument("--anchor-fraction", type=float, default=0.0,
                        help="Load-only physics: optional fraction on 256 balanced anchors; default all worlds continuous uniform")
    parser.add_argument("--wandb-mode", choices=("online", "offline"),
                        default=os.environ.get("WANDB_MODE", "online"))
    return parser


def main():
    args = build_parser().parse_args()
    args.fusion = "concat" if args.architecture == "transformer" else "baseline"
    if not 0 <= args.anchor_fraction <= 1:
        raise ValueError("anchor-fraction must be in [0,1]")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
        if saved["version"] != VERSION:
            raise ValueError("Resume requires a checkpoint from this temporal architecture experiment")
        for name in ("architecture", "critic_observation", "anchor_fraction"):
            if saved["arguments"][name] != getattr(args, name):
                raise ValueError(f"Resume changed {name}")
    base.VERSION, base.SCRATCH_INITIALIZATION = VERSION, INITIALIZATION
    base.ALLOWED_DR_PROFILES = (LOAD_ONLY, TRACKER_DR)
    base.CONTEXT_SOURCE_DR_PROFILE = TRACKER_DR
    base.TRAINING_START_PROFILE = "original"
    base.ResidualOnPolicyRunner = TemporalTokenRunner
    base.LimbContextWrapper = TokenPolicyWrapper
    base.configure_context_models = partial(configure_models, architecture=args.architecture,
        critic_privileged_token=args.critic_observation == "privileged")
    base.audit_initial_models = audit_initial_models
    base.audit_limb_dr = audit_physics
    if args.dr_profile == LOAD_ONLY:
        from intact_tracking.payload_prototype_physics import configure_physics
        base.configure_limb_dr = partial(configure_physics, anchor_fraction=args.anchor_fraction)
    else:
        from intact_tracking.residual_uniform_protocol import configure_physics
        base.configure_limb_dr = configure_physics
    base.WandbLogger = partial(base.WandbLogger, mode=args.wandb_mode)
    original_configuration = base.memory_checkpoint_configuration

    def configuration(source, train, metadata):
        metadata.update(
            initialization_protocol=INITIALIZATION, architecture=args.architecture,
            actor_initialization="zero residual output; frozen tracker unchanged; fresh Gaussian std 0.25",
            critic_initialization="scratch; independent Transformer or original-observation MLP",
            actor_critic_parameters_shared=False, tracker_warmup_steps=0,
            cuda_allocator_handoff=("release unused PyTorch cache after each complete PPO update before Warp simulation"
                                    if args.architecture == "transformer" else "unchanged"),
            baseline_contract="Actor reads only original 1645-D processed obs; critic reads original 6330-D obs; ordinary MLPs; no context encoder or latent input",
            comparison_scope="Complete architectures: temporal Transformer plus latent versus original-observation MLP; not an isolated latent ablation",
            token_contract={"history_frames": 5, "future_reference_frames": 4,
                "frame_types": ["proprio", "reference", "error", "latent", "tracker_action"],
                "frame_dimensions": [320, 269, 260, 64, 29], "future_frame_dimension": 77,
                "actor_tokens": 29, "critic_tokens": 30 if args.critic_observation == "privileged" else 29,
                "width": 128, "layers": 2, "heads": 4, "ffn_width": 256, "dropout": 0,
                "normalization": "frozen tracker feature normalization before reindexing; unit latent; trainable Pre-LayerNorm",
                "history": "raw frozen outputs captured before actions; immutable PPO snapshots; clear on reset/motion discontinuity",
                "reference": "current 269 plus four known future 77-D targets; preserve source coordinates",
                "future_visibility": "planned references only; no future actual state/action/latent",
                "critic_privileged_input": args.critic_observation == "privileged"},
            action_formula="frozen tracker raw deterministic mean + unbounded linear residual + Gaussian exploration",
            residual_output={"bounded": False, "scale": 1.0},
            evaluation_module="intact_tracking.cli.memory350_token_policy_eval")
        for path in (Path(__file__), Path(__file__).with_name("memory350_token_policy_eval.py")):
            metadata["research_source_sha256"][str(path.resolve().relative_to(base.PROJECT_ROOT))] = base._sha256(path)
        return original_configuration(source, train, metadata)

    base.memory_checkpoint_configuration = configuration
    base.run(args)


if __name__ == "__main__":
    main()
