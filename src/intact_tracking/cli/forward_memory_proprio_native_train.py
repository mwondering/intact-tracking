"""Native DR Memory350 with noisy policy proprioception and control-step commands."""

from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from intact_tracking.cli import forward_memory_native_dr_train as native
from intact_tracking.memory350_model import Memory350Predictor
from intact_tracking.memory350_proprio_inputs import INPUT_CONTRACT, ProprioMemory350Config, validate_input_checkpoint
from intact_tracking.memory350_proprio_replay import ProprioMemory350Replay, ProprioNormalizationStats
from intact_tracking.memory350_proprio_rollout import ProprioNativeDRTrackerRollout


def build_parser():
    parser = native.build_parser()
    parser.description = __doc__
    return parser


def configure_metadata(actual):
    native.configure_metadata(actual)
    actual["method"] = "Memory350 noisy proprio122, control-step commands, native flat DR, soft supervision"
    actual["context_input_contract"] = deepcopy(INPUT_CONTRACT)
    architecture = actual["architecture"]
    architecture["predictor_input"] = architecture["input"]
    architecture["input"] = "50 short + 30x10 long completed interactions: cached noisy proprio122, control-step command29, next noisy proprio122"
    architecture["normalization"] = "separate frozen encoder observation/action moments and privileged predictor/label moments; training worlds only"
    actual["research_source_sha256"].update({
        str(path.relative_to(Path(__file__).resolve().parents[3])): native.soft.nominal.trainer._sha256(path)
        for path in (Path(__file__).resolve(), *Path(__file__).resolve().parents[1].glob("memory350_proprio_*.py"))
    })


def configure_checkpoint(state, rollout, *, base):
    base(state, rollout)
    state["context_input_contract"] = deepcopy(INPUT_CONTRACT)
    state["privileged_dynamics"]["inference_contract"] = (
        "encoder reads cached noisy proprio122 and control-step command29 only; "
        "physical states, feet/contacts and DR parameters are training predictor/label data")


def configure_trainer(args):
    if args.resume:
        validate_input_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=False))
    native.configure_trainer(args)
    trainer = native.soft.nominal.trainer
    trainer.FixedDRTrackerRollout = ProprioNativeDRTrackerRollout
    trainer.ForwardPredictorReplayBuffer = partial(
        ProprioMemory350Replay, weak_archive_slots=args.weak_archive_slots,
        weak_archive_interval=args.weak_archive_interval, require_rank_probe=args.dr_soft_weight > 0)
    trainer.ForwardPredictorConfig = ProprioMemory350Config
    trainer.ForwardDynamicsTransformer = Memory350Predictor
    trainer.ForwardPredictorNormalizationStats = ProprioNormalizationStats
    trainer._BATCH_FIELDS = trainer._BATCH_FIELDS | {"predictor_history_state", "predictor_history_action"}
    trainer._configure_run_metadata = configure_metadata
    trainer._configure_checkpoint = partial(configure_checkpoint, base=trainer._configure_checkpoint)


def main():
    args = build_parser().parse_args()
    configure_trainer(args)
    print(native.soft.nominal.trainer.run(args), flush=True)


if __name__ == "__main__":
    main()
