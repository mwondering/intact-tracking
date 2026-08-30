#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CHECKPOINT MOTION_SOURCE OUTPUT_DIR [ONLINE_OPTIONS...]" \
    "" \
    "Run pure online INTACT training: frozen tracker rollout and immediate updates." \
    "MOTION_SOURCE may be a motion directory or one motion .npz file." \
    "OUTPUT_DIR must be new or empty. No rollout manifest is required." \
    "" \
    "Environment:" \
    "  PYTHON_BIN  Python executable (default: this repository's .venv)" \
    "  DEVICE      Simulator/training device (default: cuda:0)" \
    "" \
    "Example:" \
    "  $0 checkpoint.pt motions/ runs/intact_online --updates 10000 --batch-size 64"
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

CHECKPOINT_FILE="$1"
MOTION_SOURCE="$2"
OUTPUT_DIR="$3"
shift 3

for argument in "$@"; do
  case "${argument}" in
    --checkpoint-file|--checkpoint-file=*|--motion-path|--motion-path=*|--motion-file|--motion-file=*|--output-dir|--output-dir=*|--device|--device=*)
      echo "Pass checkpoint/motion/output positionally and device through DEVICE: ${argument}" >&2
      exit 2
      ;;
  esac
done

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

on_exit() {
  status=$?
  if (( status != 0 )); then
    echo "Online training failed; partial artifacts kept at: ${OUTPUT_DIR}" >&2
  fi
}
trap on_exit EXIT

command=(
  env -u PYTHONPATH "${PYTHON_BIN}" -m intact_tracking.cli.online_train
  --checkpoint-file "${CHECKPOINT_FILE}"
  "${motion_argument[@]}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  "$@"
)

printf 'Launching pure online training:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

if [[ ! -s "${OUTPUT_DIR}/last.pt" ]]; then
  echo "Training returned without a non-empty last.pt" >&2
  exit 1
fi
echo "Training completed: ${OUTPUT_DIR}/last.pt"
