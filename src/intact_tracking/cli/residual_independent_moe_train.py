"""Uniform residual PPO with sixteen completely independent trainable experts."""

from pathlib import Path

from intact_tracking.cli import residual_uniform_train as uniform
from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.memory350_independent_moe_policy import (
    VERSION, configure_independent_models, audit_independent_models,
)


def main():
    original_parser = uniform.build_parser

    def parser():
        result = original_parser()
        result.description = __doc__
        for action in result._actions:
            if action.dest == "fusion":
                action.choices = ("concat",)
        result.set_defaults(fusion="concat")
        return result

    uniform.build_parser = parser
    uniform.VERSION = VERSION
    uniform.configure_models = configure_independent_models
    uniform.audit_critic_action_models = audit_independent_models
    original_configuration = uniform.base.memory_checkpoint_configuration

    def configuration(source, train, metadata):
        for path in (Path(__file__), Path(__file__).with_name("residual_independent_moe_eval.py"),
                     Path(__file__).parents[1] / "memory350_independent_moe_policy.py"):
            metadata["research_source_sha256"][str(path.resolve().relative_to(uniform.base.PROJECT_ROOT))] = file_sha256(path)
        metadata["evaluation_module"] = "intact_tracking.cli.residual_independent_moe_eval"
        metadata["expert_architecture"] = {
            "version": VERSION, "count": 16, "shared_trainable_parameters": False,
            "independent_components": ["actor observation encoder", "actor mean head", "actor Gaussian std",
                                       "critic observation encoder", "critic value head"],
            "common_nontrainable_components": ["frozen tracker", "frozen context encoder",
                                                "online K-means routing centers", "critic running normalization moments"],
        }
        return original_configuration(source, train, metadata)

    uniform.base.memory_checkpoint_configuration = configuration
    uniform.main()


if __name__ == "__main__":
    main()
