"""Resume u10000 to u12000 with weak weights 0.008 and negative margin 1.1."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path

from intact_tracking.cli import forward_memory_weak_pairs_train as weak
from intact_tracking.memory350_scale_nominal_comparison import reference_contract as physics_contract
from intact_tracking.memory350_weak_margin_tuning import (
    FIXED_LOSS, MILESTONES, TUNED_LOSS, TARGET_TOP1, validate_tuning_arguments, validate_tuning_losses,
)


trainer = weak.trainer


def build_parser():
    parser = weak.build_parser()
    parser.description = __doc__
    parser.set_defaults(**TUNED_LOSS, **FIXED_LOSS, stop_after_updates=MILESTONES[-1])
    return parser


def reference_contract(reference, actual):
    import torch
    source = json.loads((Path(reference) / "run_config.json").read_text())
    checkpoint = Path(actual["arguments"]["resume"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    changes = validate_tuning_arguments(source["arguments"], actual["arguments"], state["update"])
    validate_tuning_losses(state["loss_config"], actual["loss"])
    if state["loss_config"] != source["loss"] or state["model_config"] != actual["model"]:
        raise ValueError("Tuning checkpoint disagrees with its parent configuration or model")
    if state["optimizer_steps"] != state["update"] * 4:
        raise ValueError("Tuning requires four optimizer steps per update")
    actual["objective_weights"].update({key: actual["loss"][key] for key in weak.WEAK_KEYS})
    checked = deepcopy(actual)
    for key in TUNED_LOSS:
        checked["arguments"][key] = source["arguments"][key]
        checked["loss"][key] = source["loss"][key]
        checked["objective_weights"][key] = source["objective_weights"][key]
    for key in ("stop_after_updates", "comparison_reference_dir"):
        checked["arguments"][key] = source["arguments"][key]
    result = physics_contract(reference, checked)
    actual["method"] = "nominal50 Memory350 encoder2x u10000 weak-pair weights and margin tuning"
    actual["weak_pair_contract"] = deepcopy(source["weak_pair_contract"])
    actual["replay"]["weak_positive_pairs"] = source["replay"]["weak_positive_pairs"]
    actual["memory_contract"]["collector"] = source["memory_contract"]["collector"]
    actual["memory_contract"]["representation_eligibility"] = source["memory_contract"]["representation_eligibility"]
    actual["tuning_stage"] = {
        "parent_checkpoint": str(checkpoint.resolve()), "parent_sha256": trainer._sha256(checkpoint),
        "from_update": state["update"], "stop_after_updates": MILESTONES[-1],
        "additional_updates": MILESTONES[-1] - state["update"], "loss_changes": changes,
        "optimizer_state_restored": True, "learning_rate_schedule_unchanged": True,
        "primary_metric": "cross-motion, disjoint-history known-environment Top-1",
        "profiles": ["common", "memory_training"], "evaluation_updates": list(MILESTONES),
        "primary_profile": "memory_training", "target_top1": TARGET_TOP1,
        "simulator_replay_and_weak_archive_restarted": True,
        "conclusion_scope": "joint effect of three loss changes plus continued training; one seed",
    }
    for module in (Path(__file__), Path(weak.__file__),
                   Path(__file__).parents[1] / "memory350_weak_margin_tuning.py"):
        actual["research_source_sha256"][str(module.resolve().relative_to(Path(__file__).resolve().parents[3]))] = trainer._sha256(module)
    result.update(architecture_change="none", initialization="restore u10000 model and AdamW state",
                  authorized_supervision_changes=changes, paired_checkpoint_updates=list(MILESTONES),
                  primary_comparison_update=MILESTONES[-1], conclusion_scope=actual["tuning_stage"]["conclusion_scope"])
    return result


def main():
    args = build_parser().parse_args()
    if not args.resume or args.bounded_smoke:
        raise ValueError("Use this entry only for the explicit u10000 margin tuning branch")
    trainer.ForwardPredictorReplayBuffer = partial(
        weak.WeakPairReplayBuffer, weak_archive_slots=args.weak_archive_slots,
        weak_archive_interval=args.weak_archive_interval)
    trainer.ForwardPredictorObjective = weak.WeakPairObjective
    trainer.ForwardPredictorLossConfig = partial(
        weak.WeakPairLossConfig, **{key: getattr(args, key) for key in weak.WEAK_KEYS})
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | weak.WEAK_BATCH_FIELDS
    trainer._validate_resume_loss_config = validate_tuning_losses
    trainer.reference_contract = reference_contract
    print(trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
