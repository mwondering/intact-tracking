"""Future residual PPO: uniform motions, 10 Nm wrists, hands <=2.5 kg, shins <=4 kg."""

from functools import partial
from pathlib import Path

import torch

from intact_tracking.cli import memory350_policy_train as base
from intact_tracking.cli.memory350_moe_policy_train import build_parser as parent_parser
from intact_tracking.fixed_dr_profiles import load_profile, file_sha256
from intact_tracking.memory350_moe_training import OnlineMoERunner
from intact_tracking.memory350_moe_critic_action import (
    TrackerActionBaselineWrapper, TrackerActionContextWrapper, audit_critic_action_models,
)
from intact_tracking.residual_uniform_protocol import (
    VERSION, physics_contract, configure_physics, audit_physics, configure_models, sample_masses,
)


def build_parser():
    parser = parent_parser()
    parser.description = __doc__
    for action in list(parser._actions):
        if action.dest == "motion_sampling":
            action.choices = ("uniform",)
        elif action.dest == "training_ranks":
            action.choices = (1, 2, 4)
        elif action.dest == "fusion":
            action.choices = ("baseline", "concat")
        elif action.dest == "dr_sampling":
            action.choices = ("independent_uniform",)
        elif action.dest == "residual_scale":
            parser._remove_action(action)
            for option in action.option_strings:
                parser._option_string_actions.pop(option, None)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
    parser.set_defaults(motion_sampling="uniform", adaptive_after_update=0,
                        training_terminations="original", residual_scale=1.0,
                        wandb_group="residual-uniform-wrist10-hand2p5")
    parser.add_argument("--dr-bank", help="Optional complete fixed DR bank for an independent specialist")
    parser.add_argument("--specialist-id", type=int, choices=range(8))
    parser.add_argument("--eval-motion-manifest")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=20001)
    return parser


class UniformResidualRunner(OnlineMoERunner):
    def learn(self, *args, **kwargs):
        def save_completed(runner):
            settings = runner.residual_metadata["arguments"]
            due = settings.get("eval_motion_manifest") and runner.completed_learning_updates % settings["eval_interval"] == 0
            if runner.completed_learning_updates % runner.cfg["save_interval"] and not due:
                return
            audit_physics(runner.env.unwrapped, runner.residual_metadata["physics"])
            preparer = getattr(runner, "checkpoint_state_preparer", None)
            if preparer is not None:
                preparer(runner)
            if runner.logger.writer is not None:
                checkpoint = Path(runner.logger.log_dir) / f"checkpoint_update_{runner.completed_learning_updates:06d}.pt"
                runner.save(str(checkpoint))
                if due:
                    from intact_tracking.residual_uniform_monitor import evaluate
                    evaluate(runner, checkpoint)
        self.checkpoint_evaluator = save_completed
        return super().learn(*args, **kwargs)


def main():
    args = build_parser().parse_args()
    if bool(args.dr_bank) != (args.specialist_id is not None):
        raise ValueError("Provide both --dr-bank and --specialist-id")
    if args.dr_bank and (args.fusion != "baseline" or args.training_ranks != 1):
        raise ValueError("A fixed specialist uses one GPU and a baseline MLP")
    if args.endpoint_eval_protocol:
        raise ValueError("Legacy 4-kg-hand endpoint protocols do not apply; use residual_uniform_eval")
    if args.eval_interval <= 0 or args.eval_steps <= 0:
        raise ValueError("Evaluation interval and horizon must be positive")
    if args.eval_motion_manifest and args.training_ranks != 1:
        raise ValueError("Periodic specialist evaluation uses one training GPU")
    if args.dr_bank:
        selected, _, _ = load_profile(args.dr_bank, args.specialist_id)
        sample_masses(1, args.seed, selected["masses_kg"])
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)["residual_policy"]
        if saved["version"] != VERSION or saved["physics"].get("residual_physics_contract") != physics_contract():
            raise ValueError("Resume requires this same physics protocol; old-physics checkpoints are a different experiment")
        if saved["motion_sampling"]["requested_mode"] != "uniform":
            raise ValueError("Resume must retain uniform motion sampling")
        old_fixed = saved["physics"].get("fixed_dr")
        if bool(old_fixed) != bool(args.dr_bank):
            raise ValueError("Resume changed fixed/random DR training")
        if old_fixed and (old_fixed["id"] != args.specialist_id or old_fixed["bank_sha256"] != file_sha256(args.dr_bank)):
            raise ValueError("Resume changed the fixed DR bank")
    base.VERSION = VERSION
    base.PPO_CLASS_NAME = "intact_tracking.memory350_moe_critic_action:TrackerActionMoEPPO"
    base.RslRlVecEnvWrapper = TrackerActionBaselineWrapper
    base.LimbContextWrapper = TrackerActionContextWrapper
    base.ResidualOnPolicyRunner = UniformResidualRunner
    base.configure_context_models = partial(configure_models, num_experts=args.num_experts,
        center_rate=args.router_center_rate, max_switch_fraction=args.router_max_switch_fraction)
    base.audit_initial_models = audit_critic_action_models
    base.configure_limb_dr = partial(configure_physics, bank_path=args.dr_bank, profile_id=args.specialist_id)
    base.audit_limb_dr = audit_physics
    original_configuration = base.memory_checkpoint_configuration
    original_sampling = base.configure_motion_sampling

    def sampling(*values, **kwargs):
        result = original_sampling(*values, **kwargs)
        result.update(scope="Uniform motion sampling; independent hand [0,2.5] / shin [0,4] kg or the selected complete fixed DR",
                      statistics="Uniform throughout; no adaptive curriculum or failure rewind")
        return result

    def configuration(source, train, metadata):
        for path in (Path(__file__), Path(__file__).with_name("residual_uniform_eval.py"),
                     Path(__file__).parents[1] / "residual_uniform_protocol.py",
                     Path(__file__).parents[1] / "residual_uniform_monitor.py"):
            metadata["research_source_sha256"][str(path.resolve().relative_to(base.PROJECT_ROOT))] = file_sha256(path)
        metadata["residual_output"] = {"bounded": False, "activation": "linear", "scale": 1.0,
                                       "formula": "action_mean = frozen_tracker_action + residual_mlp(inputs)",
                                       "actuator_force_limits": "retained; wrist pitch/yaw 10 Nm"}
        metadata["evaluation_module"] = "intact_tracking.cli.residual_uniform_eval"
        return original_configuration(source, train, metadata)
    base.configure_motion_sampling = sampling
    base.memory_checkpoint_configuration = configuration
    base.run(args)


if __name__ == "__main__":
    main()
