"""Resume the matched response-window arms to u15000; predictor always has five steps."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path

from intact_tracking.cli import forward_memory_weak_pairs_train as weak
from intact_tracking.memory350_response_window import (
    RESPONSE_FIELDS, ResponseWindowCollector, ResponseWindowObjective, ResponseWindowReplay,
)

trainer = weak.trainer


def build_parser():
    parser = weak.build_parser()
    parser.description = __doc__
    parser.add_argument("--response-label-horizon", type=int, choices=(5, 10), required=True)
    parser.set_defaults(weak_positive_weight=.008, weak_negative_weight=.008,
                        weak_negative_margin=1.1, response_distance_scale=.3,
                        stop_after_updates=15000)
    return parser


def validate_arguments(previous, actual, completed_update):
    operational = {"output_dir", "resume", "stop_after_updates", "comparison_reference_dir",
                   "wandb_group", "wandb_name", "wandb_tag", "response_label_horizon"}
    canonical = lambda value: json.loads(json.dumps(value))
    changed = {key: (previous.get(key), actual.get(key))
               for key in previous.keys() | actual.keys()
               if key not in operational and canonical(previous.get(key)) != canonical(actual.get(key))}
    if changed:
        raise ValueError(f"Response-window comparison changed other training settings: {changed}")
    horizon = actual["response_label_horizon"]
    initial_update = 11000 if horizon == 10 else 12000
    if completed_update != initial_update or actual["stop_after_updates"] != 15000:
        raise ValueError(f"The {horizon}-step arm must resume u{initial_update} and stop at u15000")
    for key, value in {"weak_positive_weight": .008, "weak_negative_weight": .008,
                       "weak_negative_margin": 1.1, "response_distance_scale": .3}.items():
        if actual[key] != value:
            raise ValueError(f"The response-window experiment must preserve {key}={value}")
    if not actual.get("resume") or not actual.get("until_user_stop"):
        raise ValueError("Resume must preserve the original scheduler and learning-rate floor")
    return {"before": previous.get("response_label_horizon", 5), "after": horizon}


def reference_contract(reference, actual):
    import torch
    reference = Path(reference)
    source = json.loads((reference / "run_config.json").read_text())
    checkpoint = Path(actual["arguments"]["resume"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    change = validate_arguments(source["arguments"], actual["arguments"], state["update"])
    if checkpoint.resolve().parent != reference.resolve():
        raise ValueError("Validation and normalization must come from the resume checkpoint directory")
    if state["model_config"] != actual["model"] or state["loss_config"] != actual["loss"]:
        raise ValueError("Model architecture and every loss coefficient must stay unchanged")
    actual["objective_weights"].update({key: actual["loss"][key] for key in weak.WEAK_KEYS})
    for key in ("model", "loss", "objective_weights", "optimization", "distributed"):
        if source[key] != actual[key]:
            raise ValueError(f"Response-window comparison changed {key}")
    if state["optimizer_steps"] != state["update"] * 4:
        raise ValueError("Expected four restored optimizer steps per update")
    horizon = actual["arguments"]["response_label_horizon"]
    actual["method"] = f"nominal50 Memory350 encoder2x response{horizon} predictor5 comparison"
    for key in ("weak_pair_contract",):
        actual[key] = deepcopy(source[key])
    actual["replay"]["weak_positive_pairs"] = source["replay"]["weak_positive_pairs"]
    actual["response_window_experiment"] = {
        "predictor_horizon": 5, "response_label_horizon": horizon,
        "response_label_change": change, "local_positive_offset": 5,
        "a_steps_per_update_after_initial_lookahead": 5,
        "initial_a_lookahead_steps": 5 if horizon == 10 else 0,
        "response_reset_mask": "exclude any reset, motion switch, or physics change over all label steps",
        "invalid_label_behavior": "retain five-step prediction and local/weak losses; mask response relation only",
        "validation": "the same frozen five-step validation batches and cached raw diagnostic histories",
        "from_update": state["update"], "stop_after_updates": 15000,
        "evaluation_updates": list(range(state["update"] + 1000, 15001, 1000)),
        "primary_metric": "memory_training strict cross-motion disjoint-history known-environment Top1",
        "restart_asymmetry": "response10 restarts at u11000; response5 continues to u12000 before restart",
        "optimizer_and_normalization_restored": True, "loss_coefficients_unchanged": True,
    }
    actual["architecture"]["counterfactual_response"] = f"{horizon} physical PD targets from the same A/B initial state"
    actual["replay"]["response_label_horizon"] = horizon
    actual["replay"]["response_pairs"] = f"continuous RMS of {horizon}-step A-minus-B response differences"
    actual["training_control"]["stage2_requires_explicit_user_convergence_decision"] = False
    root = Path(__file__).resolve().parents[3]
    for module in (Path(__file__).resolve(), root / "src/intact_tracking/memory350_response_window.py"):
        actual["research_source_sha256"][str(module.relative_to(root))] = trainer._sha256(module)
    return deepcopy(actual["response_window_experiment"])


def configure(args):
    is_ten = args.response_label_horizon == 10
    trainer.ForwardPredictorReplayBuffer = partial(
        ResponseWindowReplay if is_ten else weak.WeakPairReplayBuffer,
        weak_archive_slots=args.weak_archive_slots, weak_archive_interval=args.weak_archive_interval)
    trainer.ForwardPredictorObjective = ResponseWindowObjective if is_ten else weak.WeakPairObjective
    trainer.ForwardPredictorLossConfig = partial(
        weak.WeakPairLossConfig, **{key: getattr(args, key) for key in weak.WEAK_KEYS})
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | weak.WEAK_BATCH_FIELDS | RESPONSE_FIELDS
    if is_ten:
        original_config = trainer.NominalPairRolloutConfig
        def nominal_config(**kwargs):
            return original_config(**{**kwargs, "horizon": 10})
        trainer.NominalPairRolloutConfig = nominal_config
        trainer._collect_counterfactual_block = ResponseWindowCollector()
    trainer.reference_contract = reference_contract


def main():
    args = build_parser().parse_args()
    if not args.resume or args.bounded_smoke:
        raise ValueError("Use this entry only for the authorized response-window continuations")
    configure(args)
    print(trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
