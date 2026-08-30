#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CHECKPOINT MOTION_DIRECTORY" \
    "" \
    "Collect a minimal real MJLab rollout and execute one INTACT train/val batch." \
    "CHECKPOINT and MOTION_DIRECTORY may instead be supplied through" \
    "CHECKPOINT_FILE and MOTION_PATH." \
    "" \
    "Environment:" \
    "  PYTHON_BIN  Python executable (default: this repository's .venv)" \
    "  DEVICE      Rollout/training device (default: cuda:0)" \
    "  OUTPUT_ROOT Smoke artifact parent (default: ./runs)"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 2 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INTACT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${INTACT_ROOT}/.venv/bin/python}"
CHECKPOINT_FILE="${1:-${CHECKPOINT_FILE:-}}"
MOTION_PATH="${2:-${MOTION_PATH:-}}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INTACT_ROOT}/runs}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-3}"
SMOKE_STEPS_PER_ENV="${SMOKE_STEPS_PER_ENV:-120}"

if [[ -z "${CHECKPOINT_FILE}" || -z "${MOTION_PATH}" ]]; then
  usage >&2
  exit 2
fi

for value_name in SMOKE_NUM_ENVS SMOKE_STEPS_PER_ENV; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done

if (( SMOKE_NUM_ENVS < 3 )); then
  echo "SMOKE_NUM_ENVS must be at least 3 for world-disjoint train/val/test splits" >&2
  exit 2
fi

# 16 context blocks plus one H=5 query require at least (16 + 5) * B=5 steps.
MIN_STEPS_PER_ENV=$(( (16 + 5) * 5 ))
if (( SMOKE_STEPS_PER_ENV < MIN_STEPS_PER_ENV )); then
  echo "SMOKE_STEPS_PER_ENV must be at least ${MIN_STEPS_PER_ENV} for full context" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Create the repository environment first: uv sync --extra dev" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_FILE}" >&2
  exit 2
fi
if [[ ! -d "${MOTION_PATH}" ]]; then
  echo "Motion directory not found: ${MOTION_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
SMOKE_ROOT="$(mktemp -d "${OUTPUT_ROOT}/smoke.XXXXXX")"
ROLLOUT_DIR="${SMOKE_ROOT}/rollout"
TRAIN_DIR="${SMOKE_ROOT}/train"
TRANSITIONS=$(( SMOKE_NUM_ENVS * SMOKE_STEPS_PER_ENV ))

on_exit() {
  status=$?
  if (( status != 0 )); then
    echo "Smoke test failed; partial artifacts kept at: ${SMOKE_ROOT}" >&2
  fi
}
trap on_exit EXIT

echo "[1/3] Collecting ${TRANSITIONS} transitions from ${SMOKE_NUM_ENVS} static worlds"
(
  cd "${INTACT_ROOT}"
  env -u PYTHONPATH "${PYTHON_BIN}" "${INTACT_ROOT}/scripts/collect_rollouts.py" \
      --checkpoint-file "${CHECKPOINT_FILE}" \
      --motion-path "${MOTION_PATH}" \
      --output-dir "${ROLLOUT_DIR}" \
      --num-envs "${SMOKE_NUM_ENVS}" \
      --transitions "${TRANSITIONS}" \
      --shard-size "${TRANSITIONS}" \
      --world-session-steps 3000 \
      --device "${DEVICE}"
)

echo "[2/3] Running one full INTACT optimizer step with batch-size=1"
(
  cd "${INTACT_ROOT}"
  env -u PYTHONPATH "${PYTHON_BIN}" -m intact_tracking.cli.train \
      --manifest "${ROLLOUT_DIR}/manifest.json" \
      --output-dir "${TRAIN_DIR}" \
      --epochs 1 \
      --batch-size 1 \
      --workers 0 \
      --max-train-batches 1 \
      --max-validation-batches 1 \
      --block-size 5 \
      --horizon 5 \
      --device "${DEVICE}"
)

echo "[3/3] Verifying rollout, losses, architecture contract, and checkpoint"
env -u PYTHONPATH "${PYTHON_BIN}" "${INTACT_ROOT}/scripts/verify_smoke_run.py" \
    --manifest "${ROLLOUT_DIR}/manifest.json" \
    --run-dir "${TRAIN_DIR}" \
    --expected-transitions "${TRANSITIONS}"

echo "Smoke test passed: ${SMOKE_ROOT}"
