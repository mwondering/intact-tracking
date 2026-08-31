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


def _launcher_inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "tracker.pt"
    motion = tmp_path / "motion.npz"
    checkpoint.touch()
    motion.touch()
    return checkpoint, motion


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


def test_residual_launcher_uses_residual_entrypoint_and_preserves_wandb_flags(
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
            str(REPOSITORY / "scripts/run_residual_training.sh"),
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
    assert arguments[arguments.index("--wandb-project") + 1] == "residual-test"
    assert arguments[arguments.index("--wandb-name") + 1] == "run-one"
