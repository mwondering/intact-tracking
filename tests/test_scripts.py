from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def _fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-python"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "${CUDA_VISIBLE_DEVICES:-unset}" > "${CAPTURE_ENV}"
printf '%s\\n' "$@" > "${CAPTURE_ARGS}"
output_dir=""
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--output-dir" ]]; then
    output_dir="${argument}"
    break
  fi
  previous="${argument}"
done
test -n "${output_dir}"
printf 'fake checkpoint\n' > "${output_dir}/last.pt"
"""
    )
    executable.chmod(0o755)
    return executable


def _fake_residual_python(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-residual-python"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" > "${CAPTURE_ENV}"
printf '%s\n' "$@" > "${CAPTURE_ARGS}"
output_dir=""
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--output-dir" ]]; then
    output_dir="${argument}"
    break
  fi
  previous="${argument}"
done
test -n "${output_dir}"
printf 'fake residual checkpoint\n' > "${output_dir}/checkpoint_final.pt"
"""
    )
    executable.chmod(0o755)
    return executable


def _launcher_inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "tracker.pt"
    motion = tmp_path / "motion.npz"
    checkpoint.touch()
    motion.touch()
    return checkpoint, motion


def _last_option_value(arguments: list[str], option: str) -> str:
    index = len(arguments) - 1 - arguments[::-1].index(option)
    return arguments[index + 1]


def test_latent_residual_launcher_uses_context_checkpoint_and_multigpu(tmp_path: Path) -> None:
    tracker, motion = _launcher_inputs(tmp_path)
    forward = tmp_path / "forward.pt"
    forward.touch()
    output = tmp_path / "latent-output"
    captured_args = tmp_path / "latent-args"
    captured_env = tmp_path / "latent-environment"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_residual_python(tmp_path)),
        "GPUS": "2,5",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(captured_env),
    }

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_residual_policy_latent.sh"),
            str(tracker),
            str(forward),
            str(motion),
            str(output),
            "--iterations",
            "1",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert captured_env.read_text().strip() == "2,5"
    assert arguments[:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "intact_tracking.cli.residual_policy_train",
    ]
    assert _last_option_value(arguments, "--baseline") == "latent"
    assert _last_option_value(arguments, "--forward-checkpoint") == str(forward.resolve())
    assert _last_option_value(arguments, "--num-envs") == "2048"
    assert _last_option_value(arguments, "--iterations") == "1"
    assert "--no-include-disturbances" in arguments


def test_no_latent_residual_launcher_has_no_context_checkpoint(tmp_path: Path) -> None:
    tracker, motion = _launcher_inputs(tmp_path)
    output = tmp_path / "no-latent-output"
    captured_args = tmp_path / "no-latent-args"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_residual_python(tmp_path)),
        "DEVICE": "cuda:3",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(tmp_path / "no-latent-environment"),
    }
    environment.pop("GPUS", None)

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_residual_policy_no_latent.sh"),
            str(tracker),
            str(motion),
            str(output),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert arguments[:2] == ["-m", "intact_tracking.cli.residual_policy_train"]
    assert _last_option_value(arguments, "--baseline") == "no-latent"
    assert "--forward-checkpoint" not in arguments
    assert _last_option_value(arguments, "--device") == "cuda:3"


def test_model_gradient_residual_launcher_passes_both_checkpoints_and_multigpu(
    tmp_path: Path,
) -> None:
    tracker, motion = _launcher_inputs(tmp_path)
    forward = tmp_path / "forward.pt"
    forward.touch()
    output = tmp_path / "model-gradient-output"
    captured_args = tmp_path / "model-gradient-args"
    captured_env = tmp_path / "model-gradient-environment"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "GPUS": "4,7",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(captured_env),
    }

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_model_gradient_residual_training.sh"),
            str(tracker),
            str(forward),
            str(motion),
            str(output),
            "--updates",
            "1",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert captured_env.read_text().strip() == "4,7"
    assert arguments[:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "intact_tracking.cli.model_gradient_residual_train",
    ]
    assert _last_option_value(arguments, "--tracker-checkpoint") == str(tracker.resolve())
    assert _last_option_value(arguments, "--forward-checkpoint") == str(forward.resolve())
    assert _last_option_value(arguments, "--num-envs") == "2048"
    assert _last_option_value(arguments, "--nominal-fraction") == "0"


def test_training_launcher_builds_multigpu_torchrun_command(tmp_path: Path) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    output = tmp_path / "output"
    captured_args = tmp_path / "args"
    captured_env = tmp_path / "environment"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "GPUS": "2,5",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(captured_env),
    }

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_training.sh"),
            str(checkpoint),
            str(motion),
            str(output),
            "--updates",
            "1",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert captured_env.read_text().strip() == "2,5"
    assert arguments[:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "intact_tracking.cli.online_train",
    ]
    assert "--device" not in arguments


def test_training_launcher_preserves_single_device_mode(tmp_path: Path) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    output = tmp_path / "output"
    captured_args = tmp_path / "args"
    captured_env = tmp_path / "environment"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "DEVICE": "cuda:3",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(captured_env),
    }
    environment.pop("GPUS", None)

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_training.sh"),
            str(checkpoint),
            str(motion),
            str(output),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert captured_env.read_text().strip() == "unset"
    assert arguments[:2] == ["-m", "intact_tracking.cli.online_train"]
    device_index = arguments.index("--device")
    assert arguments[device_index + 1] == "cuda:3"
    assert "torch.distributed.run" not in arguments


def test_forward_nominal_launcher_locks_training_contract_and_preserves_wandb_flags(
    tmp_path: Path,
) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    output = tmp_path / "residual-output"
    captured_args = tmp_path / "residual-args"
    captured_env = tmp_path / "residual-environment"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "DEVICE": "cuda:4",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(captured_env),
    }
    environment.pop("GPUS", None)

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_forward_nominal_training.sh"),
            str(checkpoint),
            str(motion),
            str(output),
            "--wandb-project",
            "residual-test",
            "--wandb-name",
            "run-one",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert arguments[:2] == ["-m", "intact_tracking.cli.residual_train"]
    assert arguments[arguments.index("--device") + 1] == "cuda:4"
    assert _last_option_value(arguments, "--wandb-project") == "residual-test"
    assert _last_option_value(arguments, "--wandb-name") == "run-one"
    assert arguments[arguments.index("--num-envs") + 1] == "4096"
    assert arguments[arguments.index("--batch-size") + 1] == "768"
    assert arguments[arguments.index("--nominal-rollout-fraction") + 1] == "1.0"
    assert arguments[arguments.index("--nominal-pair-batch-size") + 1] == "0"
    assert arguments[arguments.index("--nominal-pair-weight") + 1] == "0.0"
    assert arguments[arguments.index("--context-steps") + 1] == "160"
    assert arguments[arguments.index("--transformer-dim") + 1] == "400"
    assert arguments[arguments.index("--transformer-depth") + 1] == "6"
    assert arguments[arguments.index("--transformer-heads") + 1] == "8"


def test_forward_nominal_launcher_rejects_managed_contract_options(tmp_path: Path) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "CAPTURE_ARGS": str(tmp_path / "args"),
        "CAPTURE_ENV": str(tmp_path / "environment"),
    }

    result = subprocess.run(
        [
            str(REPOSITORY / "scripts/run_forward_nominal_training.sh"),
            str(checkpoint),
            str(motion),
            str(tmp_path / "output"),
            "--nominal-rollout-fraction",
            "0.5",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fixed by the nominal Forward launcher" in result.stderr


def test_deprecated_residual_launcher_forwards_to_nominal_forward(tmp_path: Path) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    output = tmp_path / "forwarded-output"
    captured_args = tmp_path / "forwarded-args"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(tmp_path / "environment"),
    }

    result = subprocess.run(
        [
            str(REPOSITORY / "scripts/run_residual_training.sh"),
            str(checkpoint),
            str(motion),
            str(output),
            "--updates",
            "1",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert "run_residual_training.sh is deprecated" in result.stderr
    assert arguments[arguments.index("--nominal-rollout-fraction") + 1] == "1.0"
    assert arguments[arguments.index("--nominal-pair-batch-size") + 1] == "0"


def test_forward_predictor_launcher_builds_locked_causal_transformer_command(
    tmp_path: Path,
) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    output = tmp_path / "predictor-output"
    captured_args = tmp_path / "predictor-args"
    captured_env = tmp_path / "predictor-environment"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "GPUS": "1,3",
        "CAPTURE_ARGS": str(captured_args),
        "CAPTURE_ENV": str(captured_env),
    }

    subprocess.run(
        [
            str(REPOSITORY / "scripts/run_forward_predictor_training.sh"),
            str(checkpoint),
            str(motion),
            str(output),
            "--updates",
            "2",
            "--wandb-name",
            "predictor-test",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = captured_args.read_text().splitlines()
    assert captured_env.read_text().strip() == "1,3"
    assert arguments[:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "intact_tracking.cli.forward_predictor_train",
    ]
    assert _last_option_value(arguments, "--num-envs") == "2048"
    assert _last_option_value(arguments, "--nominal-fraction") == "0.5"
    assert "--payload" in arguments
    assert _last_option_value(arguments, "--payload-body-name") == "right_wrist_yaw_link"
    assert arguments[arguments.index("--payload-mass-range-kg") + 1 :][:2] == ["1", "3"]
    assert arguments[arguments.index("--payload-position-body-m") + 1 :][:3] == [
        "0.12",
        "0",
        "0",
    ]
    assert _last_option_value(arguments, "--batch-size") == "4096"
    assert _last_option_value(arguments, "--micro-batch-size") == "512"
    assert _last_option_value(arguments, "--amp-dtype") == "bfloat16"
    assert _last_option_value(arguments, "--replay-capacity") == "262144"
    assert _last_option_value(arguments, "--replay-sampling") == "motion_balanced"
    assert _last_option_value(arguments, "--gradient-steps-per-update") == "4"
    assert _last_option_value(arguments, "--rollout-steps-per-update") == "5"
    assert _last_option_value(arguments, "--history-steps") == "10"
    assert _last_option_value(arguments, "--context-history-steps") == "100"
    assert _last_option_value(arguments, "--positive-offset-steps") == "5"
    assert _last_option_value(arguments, "--transformer-dim") == "512"
    assert _last_option_value(arguments, "--transformer-depth") == "6"
    assert _last_option_value(arguments, "--transformer-heads") == "8"
    assert _last_option_value(arguments, "--context-dim") == "128"
    assert _last_option_value(arguments, "--context-depth") == "2"
    assert _last_option_value(arguments, "--context-heads") == "4"
    assert _last_option_value(arguments, "--dynamics-latent-dim") == "64"
    assert _last_option_value(arguments, "--dropout") == "0"


def test_forward_predictor_launcher_rejects_architecture_override(tmp_path: Path) -> None:
    checkpoint, motion = _launcher_inputs(tmp_path)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "CAPTURE_ARGS": str(tmp_path / "args"),
        "CAPTURE_ENV": str(tmp_path / "environment"),
    }

    result = subprocess.run(
        [
            str(REPOSITORY / "scripts/run_forward_predictor_training.sh"),
            str(checkpoint),
            str(motion),
            str(tmp_path / "output"),
            "--transformer-dim",
            "64",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fixed by the Forward Predictor launcher" in result.stderr

    second_output = tmp_path / "second-output"
    second = subprocess.run(
        [
            str(REPOSITORY / "scripts/run_forward_predictor_training.sh"),
            str(checkpoint),
            str(motion),
            str(second_output),
            "--nominal-fraction",
            "0.0",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 2
    assert "fixed by the Forward Predictor launcher" in second.stderr
