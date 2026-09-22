import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("resume,mode", [(None, "adaptive"), ("checkpoint.pt", "uniform")])
def test_sampling_reset_requires_an_explicit_adaptive_resume(resume, mode):
    from intact_tracking.cli.memory350_policy_train import _run

    args = SimpleNamespace(reset_adaptive_sampling=True, resume=resume, motion_sampling=mode)
    with pytest.raises(ValueError, match="requires --resume and adaptive"):
        _run(args, None)


@pytest.mark.parametrize("resume,mode", [(None, "adaptive"), ("checkpoint.pt", "uniform")])
def test_tracker_alignment_requires_an_explicit_adaptive_resume(resume, mode):
    from intact_tracking.cli.memory350_policy_train import _run
    args = SimpleNamespace(reset_adaptive_sampling=False, align_sampling_to_tracker=True,
                           resume=resume, motion_sampling=mode)
    with pytest.raises(ValueError, match="requires --resume and adaptive"):
        _run(args, None)


def test_launcher_forwards_reset_without_changing_ppo_configuration(tmp_path, monkeypatch):
    from intact_tracking.cli.memory350_proprio_native_policy_train import build_parser

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("residual_resume_launcher_test", scripts / "run_144000_residual.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    (tmp_path / "source_identity.json").write_text(json.dumps({
        "tracker_checkpoint": "/tracker.pt", "root": "/motions"}))
    monkeypatch.setattr(launcher, "RUN", tmp_path)
    kwargs = dict(resume=tmp_path / "checkpoint.pt", wandb_name="resumed-sampling-reset")
    ordinary = launcher.command_for("train", tmp_path / "output", **kwargs)
    reset = launcher.command_for("train", tmp_path / "output", reset_adaptive_sampling=True, **kwargs)
    assert reset == ordinary + ["--reset-adaptive-sampling"]
    aligned = launcher.command_for("train", tmp_path / "output", align_sampling_to_tracker=True, **kwargs)
    assert aligned == ordinary + ["--align-sampling-to-tracker"]
    offset = reset.index("intact_tracking.cli.memory350_proprio_native_policy_train") + 1
    args = build_parser().parse_args(reset[offset:])
    assert args.reset_adaptive_sampling and args.resume == str(kwargs["resume"])
    assert args.wandb_name == "resumed-sampling-reset"
    assert args.entropy_coef == .005 and args.initial_action_std == 1.
    assert args.dr_aux_coef == .05 and args.dr_aux_motor_weight == 0.
    assert args.until_user_stop and args.residual_output_mode == "unbounded"
    with pytest.raises(ValueError, match="requires --resume"):
        launcher.command_for("train", tmp_path / "output", reset_adaptive_sampling=True)
    with pytest.raises(ValueError, match="requires --resume"):
        launcher.command_for("train", tmp_path / "output", align_sampling_to_tracker=True)
