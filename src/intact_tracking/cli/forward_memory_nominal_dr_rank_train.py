"""Memory350: preserve nominal response losses and add DR-center distance ranking."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path

from intact_tracking.cli import forward_memory_nominal_direction_train as nominal
from intact_tracking.memory350_dr_center_rollout import DRCenterTrackerRollout
from intact_tracking.memory350_nominal_dr_rank import (
    RANK_LOSS_FIELDS, RANK_METRICS, NominalDRRankLossConfig,
    NominalDRRankObjective, NominalDRRankReplay,
)


RETUNABLE_WEIGHTS = frozenset({
    "representation_relation_weight", "nominal_anchor_weight", "dr_center_rank_weight",
})


def validate_resume_losses(previous, actual, *, new_stage, allow_weight_retuning=False,
                           allow_response_scale_retuning=False):
    if (allow_weight_retuning or allow_response_scale_retuning) and not new_stage:
        raise ValueError("Weight retuning requires --resume-new-stage and a separate output directory")
    ignored = RETUNABLE_WEIGHTS if allow_weight_retuning else frozenset()
    if allow_response_scale_retuning:
        ignored = ignored | {"response_distance_scale"}

    def unchanged(before, after):
        return ({k: v for k, v in before.items() if k not in ignored}
                == {k: v for k, v in after.items() if k not in ignored})

    if previous.get("dr_center_rank_version") == 1:
        if not unchanged(previous, actual):
            raise ValueError("Continuation changed loss settings outside the explicitly retuned weights")
        return
    if not new_stage or previous.get("nominal_direction_objective_version") != 1:
        raise ValueError("Adding DR ranking requires a nominal-direction checkpoint and --resume-new-stage")
    if not unchanged(previous, {k: v for k, v in actual.items() if k not in RANK_LOSS_FIELDS}):
        raise ValueError("Adding DR ranking cannot change existing losses without explicit weight retuning")


def configure_metadata(actual):
    nominal.configure_metadata(actual)
    args, loss = actual["arguments"], actual["loss"]
    actual["method"] += " plus DR parameter center ranking v1"
    contract = {
        "version": 1, "schema": actual["batch_a_rollout"]["dr_metric_schema"],
        "center": "mean of current-encoder unit anchor/archive latents, pooled by world/session; no renormalization after averaging",
        "eligibility": "DR only; same world/session, different motion, full disjoint histories; reuse weak-positive raw archive",
        "pairing": "all unordered eligible center pairs within each rank/microbatch; sort by DR distance; compare rank offsets 1,2,4,...",
        "loss": "gap-weighted mean temperature*softplus((near_latent_distance-far_latent_distance)/temperature)",
        "absolute_target_distance": False, "fixed_latent_margin": False,
        "weight": loss["dr_center_rank_weight"], "temperature": loss["dr_rank_temperature"],
        "minimum_parameter_distance_gap": loss["dr_rank_min_gap"],
        "nominal_centers_in_ranking": False,
        "archive_slots": args["weak_archive_slots"], "archive_interval": args["weak_archive_interval"],
        "inference": "unchanged history-only encoder; simulator DR labels are training targets only",
    }
    if args.get("resume"):
        source = json.loads((Path(args["resume"]).resolve().parent / "run_config.json").read_text())
        previous = source.get("dr_center_rank_contract")
        retune = args.get("resume_retune_weights", False)
        if previous is None:
            if not args["resume_new_stage"]:
                raise ValueError("New DR supervision needs new-stage probes and a separate output directory")
        elif previous != contract:
            without_weight = lambda c: {k: v for k, v in c.items() if k != "weight"}
            if not (args["resume_new_stage"] and retune
                    and without_weight(previous) == without_weight(contract)):
                raise ValueError("Continuation changed DR ranking schema, pairing or coefficients")
        validate_resume_losses(source["loss"], loss, new_stage=args["resume_new_stage"],
                               allow_weight_retuning=retune,
                               allow_response_scale_retuning=args.get("resume_retune_response_scale", False))
        actual["new_stage_loss_changes"] = {
            key: {"previous": source["loss"].get(key), "current": value}
            for key, value in loss.items() if source["loss"].get(key) != value
        }
        actual["explicit_weight_retuning"] = retune
        actual["explicit_response_scale_retuning"] = args.get("resume_retune_response_scale", False)
    actual["dr_center_rank_contract"] = contract
    actual["objective_weights"]["dr_center_rank_weight"] = loss["dr_center_rank_weight"]
    actual["validation"]["dr_center_rank_probe"] = (
        "warmup requires at least three eligible DR worlds in train and held-out replay; "
        "fixed probe includes nominal and full cross-motion DR histories; always report pair/comparison counts")
    actual["research_source_sha256"][str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]))] = nominal.trainer._sha256(Path(__file__))


def configure_checkpoint(state, rollout, *, base):
    base(state, rollout)
    state["representation_supervision"] = "fixed_nominal_direction_response10_dr_center_rank_v1"
    state["dr_metric_schema"] = deepcopy(rollout.dr_metric_schema)
    state["dr_center_rank_supervision"] = {k: state["loss_config"][k] for k in RANK_LOSS_FIELDS}
    state["privileged_dynamics"]["inference_contract"] = (
        "history-only encoder; simulator DR parameters supervise center-distance ordering during training only")


def build_parser():
    parser = nominal.build_parser()
    parser.description = __doc__
    parser.add_argument("--dr-center-rank-weight", type=float, default=.002,
                        help="Additional total coefficient; zero disables its gradients")
    parser.add_argument("--dr-rank-temperature", type=float, default=.1)
    parser.add_argument("--dr-rank-min-gap", type=float, default=.01,
                        help="Skip comparisons whose normalized DR distances differ by at most this value")
    parser.add_argument("--resume-retune-weights", action="store_true",
                        help="In a new resume stage only, allow changing AB, nominal-anchor and DR-rank weights")
    parser.add_argument("--resume-retune-response-scale", action="store_true",
                        help="In a new resume stage only, allow changing the response-to-distance scale")
    parser.set_defaults(nominal_fraction=.1, dr_nominal_probability=.5,
                        limb_max_masses_kg=(2.5, 2.5, 4., 4.), warmup_steps=1000)
    return parser


def configure_trainer(args):
    if args.resume_retune_weights and not (args.resume and args.resume_new_stage):
        raise ValueError("--resume-retune-weights requires --resume and --resume-new-stage")
    if args.resume_retune_response_scale and not (args.resume and args.resume_new_stage):
        raise ValueError("--resume-retune-response-scale requires --resume and --resume-new-stage")
    config = NominalDRRankLossConfig(
        dr_center_rank_weight=args.dr_center_rank_weight,
        dr_rank_temperature=args.dr_rank_temperature, dr_rank_min_gap=args.dr_rank_min_gap)
    if args.weak_archive_slots < 2 or args.weak_archive_interval < 1:
        raise ValueError("Cross-motion centers need at least two archive slots and a positive interval")
    nominal.configure_trainer(args)
    trainer = nominal.trainer
    trainer.FixedDRTrackerRollout = DRCenterTrackerRollout
    trainer.ForwardPredictorReplayBuffer = partial(
        NominalDRRankReplay, weak_archive_slots=args.weak_archive_slots,
        weak_archive_interval=args.weak_archive_interval,
        require_rank_probe=config.dr_center_rank_weight > 0)
    trainer.ForwardPredictorObjective = NominalDRRankObjective
    trainer.ForwardPredictorLossConfig = partial(
        NominalDRRankLossConfig, nominal_anchor_weight=args.nominal_anchor_weight,
        weak_positive_weight=args.weak_positive_weight,
        dr_center_rank_weight=args.dr_center_rank_weight,
        dr_rank_temperature=args.dr_rank_temperature, dr_rank_min_gap=args.dr_rank_min_gap)
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | {"dr_metric"}
    trainer._validate_resume_loss_config = partial(
        validate_resume_losses, new_stage=args.resume_new_stage,
        allow_weight_retuning=args.resume_retune_weights,
        allow_response_scale_retuning=args.resume_retune_response_scale)
    trainer._configure_run_metadata = configure_metadata
    trainer._configure_checkpoint = partial(configure_checkpoint, base=trainer._configure_checkpoint)
    trainer._REPRESENTATION_PROBE_METRICS = (*trainer._REPRESENTATION_PROBE_METRICS, *RANK_METRICS)


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(nominal.trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
