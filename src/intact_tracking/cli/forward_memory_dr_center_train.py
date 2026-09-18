"""Nominal50 encoder2x: DR-center relation, optional cross-motion positives, five-step predictor."""

from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

from intact_tracking.cli import forward_memory_scale_nominal_train as trainer
from intact_tracking.memory350_dr_center_rollout import DRCenterTrackerRollout
from intact_tracking.limb_context_protocol import validate_limb_max_masses
from intact_tracking.memory350_dr_centers import (
    CENTER_BATCH_FIELDS, DRCenterLossConfig, DRCenterObjective, DRCenterReplayBuffer,
)


REPRESENTATION_METRICS = (
    "dr_center_relation_loss", "dr_center_weighted_loss", "dr_center_pairs",
    "dr_center_valid_worlds", "dr_center_eligible_fraction",
    "dr_center_distance_mean", "dr_center_target_distance_mean", "dr_parameter_distance_mean",
    "dr_center_parameter_correlation", "dr_center_norm_mean", "dr_within_center_distance_mean",
    "dr_cross_motion_distance_mean", "dr_center_dr_pairs", "dr_center_dr_distance_mean",
    "dr_center_dr_target_distance_mean", "dr_center_nominal_pairs", "dr_center_nominal_distance_mean",
    "dr_center_nominal_target_distance_mean",
    "dr_positive_loss", "dr_positive_weighted_loss", "dr_positive_pairs",
    "dr_positive_fraction_of_dr", "dr_positive_unit_distance",
    *(f"dr_center_{name}_batch_p{q}" for name in ("distance", "target_distance") for q in (10, 50, 90)),
)

DR_POSITIVE_PAIRING = "same nonnominal world and unchanged physics session; different motion; full, causally disjoint 350-interaction histories"
DR_POSITIVE_LOSS = "mean sum_k (normalize(z_anchor)_k - normalize(z_archive)_k)^2 over eligible DR pairs only; gradients through both current-encoder views"


def normalize_resume_loss_config(config):
    # Do not inherit the current CLI partial's positive coefficient when an
    # old checkpoint lacks this new field: its historical value was zero.
    return asdict(DRCenterLossConfig(**config))


def validate_resume_loss_config(previous, actual, *, allow_weight_change=False, allow_scale_change=False,
                                allow_positive_change=False):
    expected = dict(previous)
    expected.setdefault("dr_positive_weight", 0.0)
    if allow_weight_change:
        expected["representation_weight"] = actual["representation_weight"]
    if allow_scale_change:
        expected["dr_distance_scale"] = actual["dr_distance_scale"]
    if allow_positive_change:
        expected["dr_positive_weight"] = actual["dr_positive_weight"]
    if expected != actual:
        raise ValueError("Resume changed DR-center losses beyond the explicitly allowed tuning fields")


def configure_run_metadata(actual):
    args, loss = actual["arguments"], actual["loss"]
    positive_weight = loss["dr_positive_weight"]
    actual["method"] = "nominal50 Memory350 encoder2x DR-center relation only v1"
    if positive_weight:
        actual["method"] = "nominal50 Memory350 encoder2x DR-center relation plus DR cross-motion positives v1"
    actual["dr_center_contract"] = {
        "objective_version": 1,
        "schema": actual["batch_a_rollout"]["dr_metric_schema"],
        "center": "mean of current-encoder unit latents; anchor plus one archived different-motion view per eligible row; pool rows by world/session; do not renormalize mean",
        "views": "same fixed world/session, different motion, full 350 interactions each, causally disjoint raw histories",
        "pairing": "all unordered eligible center pairs per rank and microbatch; distinct world IDs with identical parameters receive target zero",
        "target_distance": "2*d_DR/(d_DR+dr_distance_scale)",
        "loss": "representation_weight * mean SmoothL1(||mu_i-mu_j||, target_distance; beta=dr_relation_beta)",
        "dr_distance_scale": loss["dr_distance_scale"],
        "dr_relation_beta": loss["dr_relation_beta"],
        "representation_weight": loss["representation_weight"],
        "center_archive_slots_per_world": args["center_archive_slots"],
        "center_archive_interval_steps": args["center_archive_interval"],
        "nominal_worlds_archived": True,
        "within_radius_loss": False, "local_positive_loss": False,
        "weak_positive_loss": positive_weight > 0, "weak_negative_loss": False, "response_relation_loss": False,
        "dr_positive_weight": positive_weight,
        "dr_positive_pairing": DR_POSITIVE_PAIRING, "dr_positive_loss": DR_POSITIVE_LOSS,
        "nominal_origin_anchor": False,
        "inference": "unchanged history-only encoder; no simulator DR input or decoder at inference",
    }
    actual["architecture"]["context"] = actual["dr_center_contract"]["center"] + "; short50 + chunk10*30, depths 2/4/4, latent64"
    actual["architecture"]["physics"] = (
        "50% compiled nominal with zero loads and no pulses; 50% original tracker DR plus "
        f"independent per-limb U(0,max) loads, maxima {list(args['limb_max_masses_kg'])} kg; "
        "nominal B is used only for diagnostics")
    if args.get("dr_nominal_probability", 0.0):
        actual["architecture"]["physics"] += (
            f"; within the DR population, each fixed scalar parameter independently takes nominal "
            f"with probability {args['dr_nominal_probability']}, retaining original random ranges otherwise"
        )
    actual["architecture"]["normalization"] = (
        "predictor statistics frozen after warmup; DR coordinates use fixed physical sampling ranges and equal factor weights")
    actual["memory_contract"].update(
        collector="Memory350 replay plus float32 raw cross-motion archives for every nominal and DR world",
        representation_eligibility=actual["dr_center_contract"]["views"])
    actual["replay"].update(
        positive_pairs=DR_POSITIVE_PAIRING if positive_weight else "unused; no local or weak positive loss",
        response_pairs="unused by the objective; nominal B retained only for diagnostics",
        center_pairs=actual["dr_center_contract"]["pairing"],
        center_archive_slots=args["center_archive_slots"],
        center_archive_interval_steps=args["center_archive_interval"])
    actual["objective_weights"] = {
        "teacher_forced_weight": 1.0, "recursive_weight": args["recursive_weight"],
        "dr_center_relation_total_weight": loss["representation_weight"],
        "effective_positive_weight": 0.0, "effective_relation_weight": 0.0,
        "weak_positive_weight": positive_weight, "dr_cross_motion_positive_total_weight": positive_weight,
        "weak_negative_weight": 0.0, "within_radius_weight": 0.0,
    }
    actual["validation"].update(
        representation_probe="own held-out fixed batch with full, disjoint cross-motion histories and DR labels; single-motion bounded smoke has no center supervision",
        convergence="diagnostic only: prediction NMSE plateau plus stable center/DR correlation and within-center distance; no automatic stop with the default update cap",
        comparable_to_old_frozen_probes=False)
    actual["representation_distance_diagnostics"] = (
        "Center geometry and within-center motion drift are separate metrics. "
        + ("DR cross-motion drift is also penalized by the separate positive loss. " if positive_weight
           else "Within-center motion drift is diagnostic only. ") +
        "Zero pairs yield zero values, not evidence of good clustering. Quantiles are rank-local.")
    actual["research_source_sha256"][str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]))] = trainer._sha256(Path(__file__))
    if args.get("resume"):
        source_dir = (Path(args["resume"]).resolve().parent if args.get("resume_new_stage")
                      else Path(args["output_dir"]))
        old = json.loads((source_dir / "run_config.json").read_text())
        expected = dict(old.get("dr_center_contract", {}))
        expected.setdefault("dr_positive_weight", 0.0)
        expected.setdefault("dr_positive_pairing", DR_POSITIVE_PAIRING)
        expected.setdefault("dr_positive_loss", DR_POSITIVE_LOSS)
        if args.get("retune_representation_weight"):
            old_weight = expected["representation_weight"]
            expected["representation_weight"] = loss["representation_weight"]
            actual["representation_weight_tuning"] = {
                "source_run_dir": str(source_dir), "previous_weight": old_weight,
                "weight": loss["representation_weight"],
                "factor": loss["representation_weight"] / old_weight if old_weight else None,
                "other_losses_and_dr_metric_preserved": True,
            }
        if args.get("retune_dr_distance_scale"):
            old_scale = expected["dr_distance_scale"]
            expected["dr_distance_scale"] = loss["dr_distance_scale"]
            actual["dr_distance_scale_tuning"] = {
                "source_run_dir": str(source_dir), "previous_scale": old_scale,
                "scale": loss["dr_distance_scale"], "physical_metric_schema_preserved": True,
            }
            if "representation_weight_tuning" in actual:
                actual["representation_weight_tuning"]["other_losses_and_dr_metric_preserved"] = False
        if args.get("retune_dr_positive_weight"):
            old_positive = expected["dr_positive_weight"]
            expected["dr_positive_weight"] = positive_weight
            expected["weak_positive_loss"] = positive_weight > 0
            actual["dr_positive_weight_tuning"] = {
                "source_run_dir": str(source_dir), "previous_weight": old_positive,
                "weight": positive_weight, "pairing": DR_POSITIVE_PAIRING, "loss": DR_POSITIVE_LOSS,
            }
            if "representation_weight_tuning" in actual:
                actual["representation_weight_tuning"]["other_losses_and_dr_metric_preserved"] = False
        if expected != actual["dr_center_contract"]:
            raise ValueError("Resume requires the same DR-center objective, metric schema and archive settings")


def configure_checkpoint(state, rollout):
    state["nominal_counterfactual_representation_supervision"] = False
    state["representation_supervision"] = "dr_parameter_center_distance_v1"
    state["supervision_horizons"].update(response_label=None, counterfactual_diagnostic=5)
    state["dr_metric_schema"] = rollout.dr_metric_schema
    state["dr_positive_supervision"] = {
        "weight": state["loss_config"].get("dr_positive_weight", 0.0),
        "pairing": DR_POSITIVE_PAIRING, "loss": DR_POSITIVE_LOSS,
    }
    state["privileged_dynamics"]["inference_contract"] = (
        "history-only inference; simulator parameters supervise center distances only during training")


def build_parser():
    parser = trainer.build_parser()
    parser.description = __doc__
    parser.add_argument("--dr-distance-scale", type=float, default=.3)
    parser.add_argument("--dr-relation-beta", type=float, default=.25)
    parser.add_argument("--dr-positive-weight", type=float, default=0.,
                        help="Independent coefficient of same-DR, different-motion squared unit-latent distance")
    parser.add_argument("--center-archive-slots", type=int, default=4)
    parser.add_argument("--center-archive-interval", type=int, default=200)
    parser.add_argument("--retune-representation-weight", action="store_true",
                        help="Explicitly allow only the representation coefficient to change in a new resume stage")
    parser.add_argument("--retune-dr-distance-scale", action="store_true",
                        help="Explicitly allow the DR target-distance scale to change in a new resume stage")
    parser.add_argument("--retune-dr-positive-weight", action="store_true",
                        help="Explicitly allow the DR cross-motion positive coefficient to change in a new resume stage")
    parser.add_argument("--limb-max-masses-kg", type=float, nargs=4,
                        metavar=("LEFT_HAND", "RIGHT_HAND", "LEFT_SHIN", "RIGHT_SHIN"),
                        default=(2.5, 2.5, 4.0, 4.0),
                        help="Independent uniform load maxima; also define DR-label normalization")
    parser.set_defaults(representation_weight=.02, representation_relation_weight=0.,
                        replay_sampling="uniform", stop_after_updates=None)
    help_text = {
        "representation_weight": "Total coefficient of the DR-center relation loss (default: 0.02)",
        "representation_relation_weight": "Old response-relation coefficient; must remain zero in this variant",
        "response_distance_scale": "Unused legacy field; use --dr-distance-scale for this variant",
        "positive_offset_steps": "Unused local-view compatibility field; remains 5",
        "comparison_reference_dir": "Unsupported: this variant needs its own DR-labeled cross-motion probes",
        "resume": "Restore this DR-center variant; weight tuning requires --resume-new-stage --retune-representation-weight",
        "stop_after_updates": "Optional exact update cap; by default training continues until stopped",
        "bounded_smoke": "At most 3 updates; a small multi-motion directory exercises center supervision; a single motion only checks prediction plumbing",
    }
    for action in parser._actions:
        if action.dest in help_text:
            action.help = help_text[action.dest]
    return parser


def configure_trainer(args):
    args.limb_max_masses_kg = validate_limb_max_masses(args.limb_max_masses_kg)
    if args.retune_representation_weight and not (args.resume and args.resume_new_stage):
        raise ValueError("Representation weight tuning requires a checkpoint and resume-new-stage")
    if args.retune_dr_distance_scale and not (args.resume and args.resume_new_stage):
        raise ValueError("DR distance scale tuning requires a checkpoint and resume-new-stage")
    if args.retune_dr_positive_weight and not (args.resume and args.resume_new_stage):
        raise ValueError("DR positive weight tuning requires a checkpoint and resume-new-stage")
    if args.comparison_reference_dir:
        raise ValueError("DR-center training builds its own labeled held-out histories; do not reuse old response-only probes")
    if args.center_archive_slots < 2 or args.center_archive_interval < 1:
        raise ValueError("Center archive needs at least two slots and a positive interval")
    DRCenterLossConfig(representation_weight=args.representation_weight,
                      representation_relation_weight=args.representation_relation_weight,
                      dr_distance_scale=args.dr_distance_scale, dr_relation_beta=args.dr_relation_beta,
                      dr_positive_weight=args.dr_positive_weight)
    if args.bounded_smoke:
        args.stop_after_updates = args.updates
    trainer.ForwardPredictorReplayBuffer = partial(
        DRCenterReplayBuffer, center_archive_slots=args.center_archive_slots,
        center_archive_interval=args.center_archive_interval,
        require_center_probe=not args.bounded_smoke or bool(args.motion_path))
    trainer.FixedDRTrackerRollout = DRCenterTrackerRollout
    trainer.ForwardPredictorObjective = DRCenterObjective
    trainer.ForwardPredictorLossConfig = partial(
        DRCenterLossConfig, dr_distance_scale=args.dr_distance_scale, dr_relation_beta=args.dr_relation_beta,
        dr_positive_weight=args.dr_positive_weight)
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | CENTER_BATCH_FIELDS
    trainer._validate_arguments = partial(trainer._validate_arguments, require_comparison_reference=False,
                                          allow_multimotion_smoke=True)
    trainer._configure_run_metadata = configure_run_metadata
    trainer._configure_checkpoint = configure_checkpoint
    trainer._normalize_resume_loss_config = normalize_resume_loss_config
    trainer._validate_resume_loss_config = partial(
        validate_resume_loss_config, allow_weight_change=args.retune_representation_weight,
        allow_scale_change=args.retune_dr_distance_scale,
        allow_positive_change=args.retune_dr_positive_weight)
    trainer._CORE_PROBE_METRICS = (
        "one_step_nmse", "nominal_five_step_nmse", "dr_five_step_nmse",
        "latent_shuffle_dr_error_ratio", "dr_counterfactual_rms", "nominal_counterfactual_rms")
    trainer._CONTEXT_PROBE_METRICS = ("latent_shuffle_dr_error_ratio",)
    trainer._REPRESENTATION_PROBE_METRICS = REPRESENTATION_METRICS
    trainer._PLATEAU_DIAGNOSTICS = ("dr_center_parameter_correlation", "dr_within_center_distance_mean")


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
