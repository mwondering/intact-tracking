"""Train nominal50 encoder2x with weak DR pairs, or continue with unchanged settings."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path

from intact_tracking.cli import forward_memory_scale_nominal_train as trainer
from intact_tracking.memory350_scale_nominal_comparison import reference_contract as depth_contract
from intact_tracking.memory350_weak_pairs import (
    WEAK_BATCH_FIELDS, WeakPairLossConfig, WeakPairObjective, WeakPairReplayBuffer,
)


WEAK_KEYS = ("weak_positive_weight", "weak_negative_weight", "weak_negative_margin")


def validate_continuation(previous, actual, completed_update):
    """Only the resume path and independent stopping cap may change on continuation."""
    before, after = previous["arguments"], actual["arguments"]
    operational = {"resume", "stop_after_updates"}
    canonical = lambda value: json.loads(json.dumps(value))
    mismatched = {key: (before.get(key), after.get(key))
                  for key in set(before) | set(after)
                  if key not in operational and canonical(before.get(key)) != canonical(after.get(key))}
    for key in ("model", "loss", "objective_weights", "optimization", "distributed"):
        if canonical(previous[key]) != canonical(actual[key]):
            mismatched[key] = "configuration differs"
    if mismatched:
        raise ValueError(f"Continuation must preserve training settings: {mismatched}")
    target = after.get("stop_after_updates")
    if not after.get("resume") or target is None or target <= completed_update:
        raise ValueError("Continuation requires a checkpoint and a stopping cap beyond its update")
    if not after.get("until_user_stop"):
        raise ValueError("Continuation must retain the existing cosine schedule and learning-rate floor")
    return {"from_update": completed_update, "stop_after_updates": target,
            "unchanged_training_arguments": sorted(set(before) - operational),
            "cosine_horizon_unchanged": True, "losses_unchanged": True}


def reference_contract(reference, actual):
    """Permit only the explicitly requested supervision changes against encoder2x."""
    source = json.loads((Path(reference) / "run_config.json").read_text())
    if actual["model"] != source["model"]:
        raise ValueError("Weak-pair comparison must use exactly the same expanded model")
    # The base trainer creates only its original objective-weight metadata.
    # Populate the actual weak coefficients before comparing a resumed run.
    actual["objective_weights"].update({key: actual["loss"][key] for key in WEAK_KEYS})
    continuation = None
    if actual["arguments"].get("resume"):
        import torch
        checkpoint = Path(actual["arguments"]["resume"])
        previous = json.loads((checkpoint.parent / "run_config.json").read_text())
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        continuation = validate_continuation(previous, actual, state["update"])
        if state["loss_config"] != actual["loss"] or state["model_config"] != actual["model"]:
            raise ValueError("Continuation checkpoint model/loss differs from the unchanged run")
    elif actual["arguments"]["stop_after_updates"] != 5000:
        raise ValueError("Formal weak-pair comparison must stop at update 5000")
    changes = {"response_distance_scale": {"before": source["loss"]["response_distance_scale"],
                                             "after": actual["loss"]["response_distance_scale"]}}
    for key in WEAK_KEYS:
        changes[key] = {"before": source["loss"].get(key), "after": actual["loss"][key]}
    checked = deepcopy(actual)
    for field in ("loss", "objective_weights"):
        for key in WEAK_KEYS:
            checked[field].pop(key, None)
        checked[field]["response_distance_scale"] = source[field]["response_distance_scale"]
    for key in ("response_distance_scale", "comparison_reference_dir"):
        checked["arguments"][key] = source["arguments"][key]
    if continuation:
        checked["arguments"]["stop_after_updates"] = source["arguments"].get("stop_after_updates")
    result = depth_contract(reference, checked)
    actual["method"] = "nominal50 Memory350 encoder2x weak cross-motion positives and direct DR negatives v1"
    actual["objective_weights"].update({key: actual["loss"][key] for key in WEAK_KEYS})
    actual["weak_pair_contract"] = {
        "positive": "same fixed-physics world and session, different motion, disjoint full 350-step raw histories",
        "negative": "all distinct DR world pairs in a microbatch, full histories, no motion/phase matching",
        "positive_loss": "mean(1 - cosine(unit_anchor, unit_positive)) over valid DR positives",
        "negative_loss": "mean(relu(margin - unit_L2_distance)^2) over unordered distinct DR pairs",
        "weights_are_total_loss_coefficients": True,
        "nominal_worlds_are_not_negative_classes": True,
        "archive_slots_per_dr_world": actual["arguments"]["weak_archive_slots"],
        "archive_interval_control_steps": actual["arguments"]["weak_archive_interval"],
        "archive_dtype": "float32 raw interactions; re-encoded with current weights",
        "additional_sampling_rng_independent_of_base_replay": True,
    }
    actual["memory_contract"]["collector"] = "Original replay plus a separate raw cross-motion archive for weak positives"
    actual["memory_contract"]["representation_eligibility"] = (
        "Original local/response losses keep their eligibility; additional weak pairs require full short50 and long30")
    actual["replay"]["weak_positive_pairs"] = actual["weak_pair_contract"]["positive"]
    actual["research_source_sha256"][str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]))] = trainer._sha256(Path(__file__))
    result.update(architecture_change="none; expanded 2/4/4 encoder on both sides",
                  authorized_supervision_changes=changes, paired_checkpoint_updates=[5000],
                  primary_comparison_update=5000,
                  conclusion_scope="one seed; joint effect of weak positive/negative losses and response scale")
    if continuation:
        result["continuation"] = continuation
        result["continuation_comparison_updates"] = list(range(
            (continuation["from_update"] // 1000 + 1) * 1000,
            continuation["stop_after_updates"] + 1, 1000))
    return result


def build_parser():
    parser = trainer.build_parser()
    parser.description = __doc__
    parser.add_argument("--weak-positive-weight", type=float, default=.002)
    parser.add_argument("--weak-negative-weight", type=float, default=.002)
    parser.add_argument("--weak-negative-margin", type=float, default=1.)
    parser.add_argument("--weak-archive-slots", type=int, default=4)
    parser.add_argument("--weak-archive-interval", type=int, default=200)
    parser.set_defaults(response_distance_scale=.5, stop_after_updates=5000)
    return parser


def main():
    args = build_parser().parse_args()
    if args.bounded_smoke:
        args.stop_after_updates = args.updates
    trainer.ForwardPredictorReplayBuffer = partial(
        WeakPairReplayBuffer, weak_archive_slots=args.weak_archive_slots,
        weak_archive_interval=args.weak_archive_interval)
    trainer.ForwardPredictorObjective = WeakPairObjective
    trainer.ForwardPredictorLossConfig = partial(
        WeakPairLossConfig, **{key: getattr(args, key) for key in WEAK_KEYS})
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | WEAK_BATCH_FIELDS
    trainer.reference_contract = reference_contract
    print(trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
