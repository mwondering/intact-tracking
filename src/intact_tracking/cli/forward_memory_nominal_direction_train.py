"""Memory350 response10: fixed nominal anchor, DR-to-anchor distance, no negatives."""

from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from intact_tracking.cli import forward_memory_scale_nominal_train as trainer
from intact_tracking.memory350_nominal_direction import (
    NominalDirectionLossConfig, NominalDirectionObjective,
)
from intact_tracking.memory350_response_window import (
    RESPONSE_FIELDS, ResponseWindowCollector, ResponseWindowReplay,
)
from intact_tracking.memory350_weak_pairs import WEAK_BATCH_FIELDS


EXTRA_METRICS = (
    "dr_nominal_relation_loss", "dr_nominal_weighted_loss", "nominal_anchor_loss",
    "nominal_anchor_weighted_loss", "nominal_anchor_samples", "nominal_anchor_cosine",
    "weak_positive_loss", "weak_positive_weighted_loss", "weak_positive_pairs",
    "weak_positive_fraction_of_full_dr", "response_label_valid_fraction",
)


def validate_resume_losses(previous, actual, *, new_stage):
    if previous.get("nominal_direction_objective_version") == 1:
        if previous != actual:
            raise ValueError("Continuation must preserve all nominal-direction loss settings")
        return
    if not new_stage:
        raise ValueError("Changing response10 supervision requires --resume-new-stage")
    if not {"weak_positive_weight", "weak_negative_weight"}.issubset(previous):
        raise ValueError("Initialize this variant from the original weak-pair response10 checkpoint")
    expected = dict(previous, representation_relation_weight=actual["representation_relation_weight"],
                    weak_negative_weight=0., nominal_anchor_weight=actual["nominal_anchor_weight"],
                    nominal_direction_objective_version=1)
    if expected != actual:
        raise ValueError("New-stage resume changed unrelated prediction or positive losses")


def validate_response_contract(previous, actual, *, new_stage=False,
                               allow_response_scale_retuning=False):
    if allow_response_scale_retuning and not new_stage:
        raise ValueError("Response-scale retuning requires a separate new resume stage")
    if previous is None:
        return
    ignored = {"response_scale"} if allow_response_scale_retuning else set()
    if ({k: v for k, v in previous.items() if k not in ignored}
            != {k: v for k, v in actual.items() if k not in ignored}):
        raise ValueError("Continuation changed the frozen-anchor supervision contract")


def configure_metadata(actual):
    args, loss = actual["arguments"], actual["loss"]
    actual["method"] = f"nominal{100*args['nominal_fraction']:g} Memory350 encoder2x fixed nominal direction response10 v1"
    actual["nominal_direction_contract"] = {
        "version": 1, "predictor_horizon": 5, "response_label_horizon": 10,
        "anchor": "normalize(global mean of unit nominal latents from full training histories before updates); frozen thereafter",
        "anchor_restore": "checkpoint stores the exact frozen vector; never recalibrated on continuation",
        "nominal_loss": "mean(1 - dot(normalize(z), anchor)) over nominal rows with usable history",
        "relation": "SmoothL1(unit latent distance to fixed anchor, 2*RMS(A-B)/(RMS(A-B)+scale)); DR rows only",
        "response_scale": loss["response_distance_scale"], "smooth_l1_beta": .25,
        "local_positive": "original same-world/episode/motion exact +/-5-step cosine",
        "cross_motion_positive": "same DR world/session, different motion, full disjoint 350-interaction histories; 1-cosine",
        "weak_negative": False, "cross_world_response_relation": False,
        "response_aggregation": "per sample, all ten steps and 70 normalized physical differences",
        "invalid_response": "exclude from relation only; retain nominal, local, weak-positive and prediction losses",
    }
    actual["objective_weights"] = {
        "teacher_forced_weight": 1., "recursive_weight": args["recursive_weight"],
        "local_positive_weight": loss["representation_weight"],
        "dr_nominal_response_weight": loss["representation_weight"] * loss["representation_relation_weight"],
        "weak_positive_weight": loss["weak_positive_weight"], "weak_negative_weight": 0.,
        "nominal_anchor_weight": loss["nominal_anchor_weight"],
    }
    actual["architecture"]["context"] = "Memory350 short50 + long30x10, attention depths 2/4/4, latent64; original LayerNorm and unit representation geometry"
    actual["architecture"]["physics"] = (
        f"A has {args['nominal_fraction']:.1%} compiled nominal worlds; remaining worlds retain tracker DR "
        f"and per-limb uniform maxima {args.get('limb_max_masses_kg') or [4,4,4,4]} kg, with each fixed "
        f"DR coordinate independently nominal with probability {args['dr_nominal_probability']}; "
        "B starts at each A anchor and replays ten identical physical PD targets")
    actual["validation"]["nominal_metrics"] = "separate nominal/DR held-out errors; exact per-rank counts recorded in dataset runtime audits"
    actual["replay"]["response_label_horizon"] = 10
    actual["replay"]["response_pairs"] = actual["nominal_direction_contract"]["relation"]
    actual["replay"]["weak_negative_pairs"] = False
    actual["validation"]["representation_probe"] = "new held-out world probes with genuine ten-step labels and the training-calibrated frozen anchor"
    actual["research_source_sha256"][str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]))] = trainer._sha256(Path(__file__))
    if args.get("resume"):
        source = json.loads((Path(args["resume"]).resolve().parent / "run_config.json").read_text())
        validate_response_contract(
            source.get("nominal_direction_contract"), actual["nominal_direction_contract"],
            new_stage=args.get("resume_new_stage", False),
            allow_response_scale_retuning=args.get("resume_retune_response_scale", False))


@torch.no_grad()
def prepare_objective(objective, replay, normalization, distributed, output_dir, resumed,
                      *, args, anchor_state):
    existing = resumed.get("nominal_direction_anchor") if resumed else None
    if existing is not None:
        objective.set_anchor(existing["direction"])
        metadata = deepcopy(existing)
    else:
        if resumed and resumed.get("supervision_horizons", {}).get("response_label") != 10:
            raise ValueError("Initial checkpoint must use ten-step A-B response labels")
        model, device = objective.model, objective.nominal_anchor.device
        was_training = model.training
        model.eval()
        total = torch.zeros_like(objective.nominal_anchor, dtype=torch.float64)
        count = torch.zeros((), dtype=torch.float64, device=device)
        try:
            for _ in range(args.anchor_calibration_batches):
                batch = replay.sample_batch(args.anchor_calibration_batch_size, normalization)
                valid = (batch["is_nominal"].bool() & batch["history_valid"].all(1)
                         & batch["memory_valid"].all(1))
                if not valid.any():
                    continue
                z = model.encode_context(
                    batch["history_state"][valid], batch["history_action"][valid],
                    batch["state"][valid, 0], batch["history_valid"][valid],
                    history_next_state=batch["history_next_state"][valid],
                    memory_interactions=batch["memory_interactions"][valid],
                    memory_valid=batch["memory_valid"][valid])
                unit = F.normalize(z.float(), dim=-1, eps=1e-8)
                total += unit.double().sum(0)
                count += len(unit)
        finally:
            model.train(was_training)
        counts = distributed.all_gather_object(int(count))
        distributed.all_reduce_sum(total)
        distributed.all_reduce_sum(count)
        if count < 1 or not torch.isfinite(total).all() or total.norm() < 1e-8:
            raise RuntimeError("No finite full-history training nominal direction available")
        anchor = F.normalize(total.float(), dim=0)
        objective.set_anchor(anchor)
        metadata = {
            "version": 1, "direction": anchor.cpu().tolist(),
            "source_update": resumed["update"] if resumed else 0,
            "source": "training replay only; full short50 and long30; source encoder before optimizer updates",
            "nominal_samples_by_rank": counts, "nominal_samples_total": int(count),
            "mean_resultant_length": float(total.norm() / count),
            "calibration_batches_per_rank": args.anchor_calibration_batches,
            "calibration_batch_size": args.anchor_calibration_batch_size,
            "validation_used": False,
        }
    digest = hashlib.sha256(objective.nominal_anchor.cpu().numpy().tobytes()).hexdigest()
    digests = distributed.all_gather_object(digest)
    if len(set(digests)) != 1:
        raise RuntimeError("Ranks disagree on the nominal anchor")
    metadata["sha256"] = digest
    metadata["sha256_by_rank"] = digests
    anchor_state.update(metadata)
    if distributed.is_main:
        trainer._write_json(output_dir / "nominal_anchor.json", metadata)
        print(json.dumps({"event": "nominal_anchor_ready", "sha256": digest,
                          "restored": existing is not None,
                          "nominal_samples_total": metadata["nominal_samples_total"]}), flush=True)
    return {"nominal_direction_anchor": metadata}


def configure_checkpoint(state, rollout, *, anchor_state):
    if not anchor_state:
        raise RuntimeError("Cannot save an uninitialized nominal anchor")
    state["nominal_direction_anchor"] = deepcopy(anchor_state)
    state["representation_supervision"] = "fixed_nominal_direction_response10_v1"
    state["supervision_horizons"].update(predictor=5, response_label=10)


def build_parser():
    parser = trainer.build_parser()
    parser.description = __doc__
    parser.add_argument("--nominal-anchor-weight", type=float, default=.01)
    parser.add_argument("--weak-positive-weight", type=float, default=.008)
    parser.add_argument("--weak-archive-slots", type=int, default=4)
    parser.add_argument("--weak-archive-interval", type=int, default=200)
    parser.add_argument("--anchor-calibration-batches", type=int, default=8)
    parser.add_argument("--anchor-calibration-batch-size", type=int, default=256)
    parser.add_argument("--limb-max-masses-kg", type=float, nargs=4,
                        metavar=("LEFT_HAND", "RIGHT_HAND", "LEFT_SHIN", "RIGHT_SHIN"))
    parser.set_defaults(representation_weight=.01, representation_relation_weight=4.,
                        response_distance_scale=.3, stop_after_updates=None,
                        batch_size=512, micro_batch_size=256, comparison_reference_dir=None)
    return parser


def configure_trainer(args):
    if args.comparison_reference_dir:
        raise ValueError("New supervision uses new ten-step held-out probes; do not reuse old five-step probes")
    if min(args.anchor_calibration_batches, args.anchor_calibration_batch_size) < 1:
        raise ValueError("Anchor calibration sample counts must be positive")
    NominalDirectionLossConfig(nominal_anchor_weight=args.nominal_anchor_weight,
                               weak_positive_weight=args.weak_positive_weight)
    anchor_state = {}
    trainer.ForwardPredictorReplayBuffer = partial(
        ResponseWindowReplay, weak_archive_slots=args.weak_archive_slots,
        weak_archive_interval=args.weak_archive_interval)
    trainer.ForwardPredictorObjective = NominalDirectionObjective
    trainer.ForwardPredictorLossConfig = partial(
        NominalDirectionLossConfig, nominal_anchor_weight=args.nominal_anchor_weight,
        weak_positive_weight=args.weak_positive_weight)
    nominal_config = trainer.NominalPairRolloutConfig
    trainer.NominalPairRolloutConfig = lambda **kwargs: nominal_config(**{**kwargs, "horizon": 10})
    trainer._collect_counterfactual_block = ResponseWindowCollector()
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | WEAK_BATCH_FIELDS | RESPONSE_FIELDS
    trainer._validate_arguments = partial(trainer._validate_arguments,
        require_comparison_reference=False, allow_multimotion_smoke=True, allow_nominal_fraction=True)
    trainer._normalize_resume_loss_config = dict
    trainer._validate_resume_loss_config = partial(validate_resume_losses,
        new_stage=args.resume_new_stage)
    trainer._configure_run_metadata = configure_metadata
    trainer._prepare_objective = partial(prepare_objective, args=args, anchor_state=anchor_state)
    trainer._configure_checkpoint = partial(configure_checkpoint, anchor_state=anchor_state)
    trainer._REPRESENTATION_PROBE_METRICS = (
        *trainer._REPRESENTATION_PROBE_METRICS, *EXTRA_METRICS)
    if args.bounded_smoke:
        args.stop_after_updates = args.updates


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
