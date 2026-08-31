#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CHECKPOINT MOTION_SOURCE" \
    "" \
    "Run one real pure-online rollout/update with batch-size=1." \
    "MOTION_SOURCE may be a motion directory or one motion .npz file." \
    "Arguments may instead be supplied through CHECKPOINT_FILE and MOTION_PATH." \
    "" \
    "Environment:" \
    "  PYTHON_BIN     Python executable (default: this repository's .venv)" \
    "  DEVICE         Single-process device (default: cuda:0)" \
    "  GPUS           Comma-separated GPU IDs; 2+ IDs run a DDP smoke test" \
    "  OUTPUT_ROOT    Smoke artifact parent (default: ./runs)" \
    "  SMOKE_NUM_ENVS Vector environments per GPU/rank (default: 3)"
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
MOTION_SOURCE="${2:-${MOTION_PATH:-}}"
DEVICE="${DEVICE:-cuda:0}"
GPUS="${GPUS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INTACT_ROOT}/runs}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-3}"

if [[ -z "${CHECKPOINT_FILE}" || -z "${MOTION_SOURCE}" ]]; then
  usage >&2
  exit 2
fi
if ! [[ "${SMOKE_NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_NUM_ENVS must be a positive integer, got: ${SMOKE_NUM_ENVS}" >&2
  exit 2
fi
NPROC=1
if [[ -n "${GPUS}" ]]; then
  if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPUS must be comma-separated non-negative GPU IDs, got: ${GPUS}" >&2
    exit 2
  fi
  IFS=',' read -r -a gpu_ids <<< "${GPUS}"
  declare -A seen_gpu_ids=()
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
      echo "GPUS contains a duplicate GPU ID: ${gpu_id}" >&2
      exit 2
    fi
    seen_gpu_ids["${gpu_id}"]=1
  done
  NPROC="${#gpu_ids[@]}"
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
CHECKPOINT_FILE="$(realpath --canonicalize-existing -- "${CHECKPOINT_FILE}")"

motion_argument=()
if [[ -d "${MOTION_SOURCE}" ]]; then
  MOTION_SOURCE="$(realpath --canonicalize-existing -- "${MOTION_SOURCE}")"
  motion_argument=(--motion-path "${MOTION_SOURCE}")
elif [[ -f "${MOTION_SOURCE}" ]]; then
  MOTION_SOURCE="$(realpath --canonicalize-existing -- "${MOTION_SOURCE}")"
  motion_argument=(--motion-file "${MOTION_SOURCE}")
else
  echo "Motion source is neither a directory nor a file: ${MOTION_SOURCE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
SMOKE_ROOT="$(mktemp -d "${OUTPUT_ROOT}/smoke.XXXXXX")"
TRAIN_DIR="${SMOKE_ROOT}/train"
mkdir -p "${TRAIN_DIR}"

on_exit() {
  status=$?
  if (( status != 0 )); then
    echo "Smoke test failed; partial artifacts kept at: ${SMOKE_ROOT}" >&2
  fi
}
trap on_exit EXIT

echo "[1/2] Running ${NPROC}-rank frozen-tracker rollout and one immediate INTACT update"

command=(env -u PYTHONPATH)
if [[ -n "${GPUS}" ]]; then
  command+=(CUDA_VISIBLE_DEVICES="${GPUS}")
fi
command+=("${PYTHON_BIN}")
if (( NPROC > 1 )); then
  command+=(
    -m torch.distributed.run
    --standalone
    --nproc-per-node "${NPROC}"
    -m intact_tracking.cli.online_train
  )
else
  command+=(-m intact_tracking.cli.online_train)
fi
command+=(
    --checkpoint-file "${CHECKPOINT_FILE}"
    "${motion_argument[@]}"
    --output-dir "${TRAIN_DIR}"
    --num-envs "${SMOKE_NUM_ENVS}"
    --warmup-steps 105
    --max-warmup-steps 2000
    --updates 1
    --rollout-steps-per-update 1
    --gradient-steps-per-update 1
    --batch-size 1
    --replay-capacity 32
    --log-interval 1
    --checkpoint-interval 1
    --effect-steps 5
    --query-transitions 5
    --context-chunk-steps 5
    --sample-stride 1
)
if (( NPROC == 1 )); then
  command+=(--device "${DEVICE}")
fi
(
  cd "${INTACT_ROOT}"
  "${command[@]}" 2>&1 | tee "${TRAIN_DIR}/train.log"
)

echo "[2/2] Verifying online schedule, fixed DR contract, losses, and checkpoint"
env -u PYTHONPATH "${PYTHON_BIN}" "${INTACT_ROOT}/scripts/verify_online_smoke.py" \
    --run-dir "${TRAIN_DIR}" \
    --expected-num-envs "${SMOKE_NUM_ENVS}" \
    --expected-world-size "${NPROC}"

echo "Smoke test passed: ${SMOKE_ROOT}"
