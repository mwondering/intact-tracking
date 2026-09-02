#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CHECKPOINT MOTION_SOURCE OUTPUT_DIR [RESIDUAL_OPTIONS...]" \
    "" \
    "Train the context-conditioned Forward model from frozen-tracker rollouts." \
    "W&B logging is enabled by default; pass --no-wandb for local-only runs." \
    "MOTION_SOURCE may be one .npz file or a motion directory." \
    "" \
    "Environment:" \
    "  PYTHON_BIN  Python executable (default: repository .venv)" \
    "  DEVICE      Single-process device (default: cuda:0)" \
    "  GPUS        Comma-separated GPU IDs for torchrun"
}

if (( $# == 0 )) || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 3 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INTACT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${INTACT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
GPUS="${GPUS:-}"

CHECKPOINT_FILE="$1"
MOTION_SOURCE="$2"
OUTPUT_DIR="$3"
shift 3

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
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

if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
  echo "Output path exists and is not a directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to use a non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath --canonicalize-existing -- "${OUTPUT_DIR}")"

NPROC=1
if [[ -n "${GPUS}" ]]; then
  if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPUS must be comma-separated non-negative GPU IDs, got: ${GPUS}" >&2
    exit 2
  fi
  IFS=',' read -r -a gpu_ids <<< "${GPUS}"
  NPROC="${#gpu_ids[@]}"
fi

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
    -m intact_tracking.cli.residual_train
  )
else
  command+=(-m intact_tracking.cli.residual_train)
fi
command+=(
  --checkpoint-file "${CHECKPOINT_FILE}"
  "${motion_argument[@]}"
  --output-dir "${OUTPUT_DIR}"
)
if (( NPROC == 1 )); then
  command+=(--device "${DEVICE}")
fi
command+=("$@")

printf 'Launching Forward-only training (%d rank(s)):' "${NPROC}"
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

if [[ ! -s "${OUTPUT_DIR}/last.pt" ]]; then
  echo "Forward training returned without a non-empty last.pt" >&2
  exit 1
fi
echo "Forward training completed: ${OUTPUT_DIR}/last.pt"
