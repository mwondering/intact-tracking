#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 MANIFEST OUTPUT_DIR [TRAIN_OPTIONS...]" \
    "" \
    "Run a complete INTACT training job over every batch in every epoch." \
    "MANIFEST is a rollout manifest.json; OUTPUT_DIR must be new or empty." \
    "Additional options are forwarded to intact_tracking.cli.train." \
    "" \
    "Environment:" \
    "  PYTHON_BIN  Python executable (default: this repository's .venv)" \
    "  DEVICE      Training device (default: cuda:0)" \
    "" \
    "Example:" \
    "  $0 /data/rollout/manifest.json runs/intact_e5 --epochs 5 --batch-size 256"
}

if (( $# == 0 )) || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INTACT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${INTACT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

MANIFEST="$1"
OUTPUT_DIR="$2"
shift 2

for argument in "$@"; do
  case "${argument}" in
    --max-train-batches|--max-train-batches=*|--max-validation-batches|--max-validation-batches=*)
      echo "Formal training forbids smoke batch limits: ${argument}" >&2
      exit 2
      ;;
    --allow-padded-context)
      echo "Formal training requires all 16 context tokens; padded context is disabled" >&2
      exit 2
      ;;
    --manifest|--manifest=*|--output-dir|--output-dir=*|--device|--device=*)
      echo "Pass manifest/output positionally and device through DEVICE: ${argument}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Create the repository environment first: uv sync --extra dev" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Rollout manifest not found: ${MANIFEST}" >&2
  exit 2
fi
MANIFEST="$(realpath --canonicalize-existing -- "${MANIFEST}")"

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
    echo "Training failed; partial artifacts kept at: ${OUTPUT_DIR}" >&2
  fi
}
trap on_exit EXIT

command=(
  env -u PYTHONPATH "${PYTHON_BIN}" -m intact_tracking.cli.train
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  "$@"
)

printf 'Launching full training:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

if [[ ! -s "${OUTPUT_DIR}/last.pt" ]]; then
  echo "Training returned without a non-empty last.pt" >&2
  exit 1
fi
echo "Training completed: ${OUTPUT_DIR}/last.pt"
