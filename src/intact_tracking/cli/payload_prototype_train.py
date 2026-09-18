"""256 experts: observation/PPO routing versus fixed load-latent prototype routing."""

from functools import partial
import hashlib
import os
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.limb_context_dr import LOAD_ONLY, TRACKER_DR
from intact_tracking.payload_prototype_moe import (
    VERSION, INITIALIZATION, configure_models, audit_initial_models, PayloadMoEWrapper,
)
from intact_tracking.payload_prototype_training import PayloadMoERunner
from intact_tracking.payload_prototype_physics import configure_physics, audit_physics, load_grid


def build_parser():
    parser = base.build_parser()
    for action in list(parser._actions):
        if action.dest in ("fusion","residual_scale","endpoint_eval_protocol"):
            parser._remove_action(action)
            for option in action.option_strings: parser._option_string_actions.pop(option,None)
            for group in parser._action_groups:
                if action in group._group_actions: group._group_actions.remove(action)
        elif action.dest == "training_ranks": action.choices = (1,2,4,8)
        elif action.dest == "motion_sampling": action.choices = ("uniform",)
        elif action.dest == "training_terminations": action.choices = ("original",)
        elif action.dest == "dr_sampling": action.choices = ("independent_uniform",)
        elif action.dest == "dr_profile": action.choices = (LOAD_ONLY,)
    parser.set_defaults(residual_scale=1., endpoint_eval_protocol=None,dr_profile=LOAD_ONLY,
                        motion_sampling="uniform",adaptive_after_update=0,training_terminations="original",
                        iterations=None,policy_precision="fp32",save_interval=250,
                        wandb_group="memory350-payload256-top5-moe")
    parser.add_argument("--route-mode",choices=("learned","latent"),required=True)
    parser.add_argument("--prototype-file",required=True)
    parser.add_argument("--anchor-fraction",type=float,default=0.5)
    parser.add_argument("--wandb-mode",choices=("online","offline"),default=os.environ.get("WANDB_MODE","online"))
    return parser


def main():
    args = build_parser().parse_args()
    args.fusion = "concat" if args.route_mode == "latent" else "baseline"
    args.prototype_file = str(Path(args.prototype_file).resolve())
    prototype = torch.load(args.prototype_file,map_location="cpu",weights_only=False)
    torch.testing.assert_close(prototype["loads_kg"],load_grid(),atol=0,rtol=0)
    if args.context_checkpoint:
        if hashlib.sha256(Path(args.context_checkpoint).read_bytes()).hexdigest() != prototype["metadata"]["context_sha256"]:
            raise ValueError("Prototypes must come from this exact frozen encoder")
    if args.route_mode == "learned" and args.context_checkpoint:
        raise ValueError("The learned-router baseline must not load a context encoder")
    if args.resume:
        old = torch.load(args.resume,map_location="cpu",weights_only=False,mmap=True)["residual_policy"]
        if old["version"] != VERSION or old["arguments"]["anchor_fraction"] != args.anchor_fraction:
            raise ValueError("Resume cannot change the MoE or environment contract")
        if old["prototype_sha256"] != hashlib.sha256(Path(args.prototype_file).read_bytes()).hexdigest():
            raise ValueError("Resume changed prototype calibration")
    base.VERSION,base.SCRATCH_INITIALIZATION = VERSION,INITIALIZATION
    base.ALLOWED_DR_PROFILES = (LOAD_ONLY,)
    # Keep full-DR encoder provenance; deploy in an explicitly audited nominal-background subset.
    base.CONTEXT_SOURCE_DR_PROFILE = TRACKER_DR
    base.TRAINING_START_PROFILE = "original"
    base.PPO_CLASS_NAME = "intact_tracking.payload_prototype_training:PayloadMoEPPO"
    base.ResidualOnPolicyRunner = PayloadMoERunner
    base.configure_context_models = partial(configure_models,route_mode=args.route_mode,prototype_file=args.prototype_file)
    base.audit_initial_models = audit_initial_models
    base.configure_limb_dr = partial(configure_physics,anchor_fraction=args.anchor_fraction)
    base.audit_limb_dr = audit_physics
    base.LimbContextWrapper = PayloadMoEWrapper
    base.RslRlVecEnvWrapper = PayloadMoEWrapper
    base.WandbLogger = partial(base.WandbLogger,mode=args.wandb_mode)
    original_sampling = base.configure_motion_sampling
    def sampling(*values,**kwargs):
        result = original_sampling(*values,**kwargs)
        result.update(scope="Uniform motions; balanced 256 anchors plus continuous hand [0,2.5] / shin [0,4] loads; nominal background",
                      statistics="Uniform throughout; no adaptive curriculum or failure rewind")
        return result
    base.configure_motion_sampling = sampling
    original_configuration = base.memory_checkpoint_configuration
    def configuration(source,train,metadata):
        metadata.update(execution_protocol=f"payload256_top5_{args.training_ranks}gpu_v1",
                        context_source_dr_profile=TRACKER_DR if args.context_checkpoint else None,
                        initialization_protocol=INITIALIZATION,tracker_warmup_steps=0,
                        prototype_sha256=hashlib.sha256(Path(args.prototype_file).read_bytes()).hexdigest(),
                        prototype_calibration=prototype["metadata"],
                        baseline_contract="Actor and critic share observation-derived assignments; critic detaches them and uses zero latent; only PPO actor loss trains router",
                        actor_initialization="Independent private expert hidden layers; zero residual outputs; frozen tracker mean plus Gaussian std 0.25",
                        critic_initialization="Scratch independent obs encoder and expert branches; normalized latent only in latent arm",
                        action_formula="raw tracker mean + weighted sum of five unbounded residual means + one shared Gaussian exploration",
                        critic_routing="Same current assignments as actor, detached; no actor/critic parameter sharing",
                        evaluation_module="intact_tracking.cli.payload_prototype_eval")
        for path in [Path(__file__),*Path(__file__).parents[1].glob("payload_prototype_*.py")]:
            metadata["research_source_sha256"][str(path.resolve().relative_to(base.PROJECT_ROOT))] = base._sha256(path)
        return original_configuration(source,train,metadata)
    base.memory_checkpoint_configuration = configuration
    base.run(args)


if __name__ == "__main__":
    main()
