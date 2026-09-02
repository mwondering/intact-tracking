#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CHECKPOINT MOTION_SOURCE OUTPUT_DIR [FORWARD_OPTIONS...]" \
    "" \
    "Train only the nominal Forward world model from frozen-tracker rollouts." \
    "" \
    "Fixed training contract:" \
    "  rollout physics       100% nominal; domain randomization disabled" \
    "  rollout controller    frozen tracker" \
    "  trainable modules     one unified history-to-future Transformer" \
    "  disabled modules      Residual Policy, Backward Predictor, Tracking loss" \
    "  sequence              160-step history + current condition + 5 query actions" \
    "  Transformer           327 tokens, width 400, depth 6, heads 8 (~11.80M total)" \
    "  counterfactual pairs  disabled" \
    "" \
    "Production defaults (override with FORWARD_OPTIONS):" \
    "  --num-envs 4096 --batch-size 768 --replay-capacity 8192" \
    "  --updates 100000 --wandb-project intact-forward-world-model" \
    "" \
    "MOTION_SOURCE may be one .npz file or a motion directory." \
    "W&B logging is enabled by default; pass --no-wandb for local-only runs." \
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

managed_options=(
  --nominal-rollout-fraction
  --nominal-pair-batch-size
  --nominal-pair-weight
  --nominal-effect-weight
  --nominal-consistency-weight
  --context-steps
  --transformer-dim
  --transformer-depth
  --transformer-heads
)
for argument in "$@"; do
  for managed in "${managed_options[@]}"; do
    if [[ "${argument}" == "${managed}" || "${argument}" == "${managed}="* ]]; then
      echo "${managed} is fixed by the nominal Forward launcher and cannot be overridden." >&2
      exit 2
    fi
  done
done

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
  --num-envs 4096
  --batch-size 768
  --replay-capacity 8192
  --updates 100000
  --wandb-project intact-forward-world-model
)
if (( NPROC == 1 )); then
  command+=(--device "${DEVICE}")
fi
command+=("$@")
command+=(
  --nominal-rollout-fraction 1.0
  --nominal-pair-batch-size 0
  --nominal-pair-weight 0.0
  --nominal-effect-weight 0.0
  --nominal-consistency-weight 0.0
  --context-steps 160
  --transformer-dim 400
  --transformer-depth 6
  --transformer-heads 8
)

printf '%s\n' \
  "Training contract: nominal-only Forward world model" \
  "  trainable: one unified history-to-future causal Transformer" \
  "  disabled: DR, nominal pairs, Residual Policy, Backward Predictor, Tracking loss" \
  "  sequence: 160-step history + CURRENT + 5 query actions = 327 tokens" \
  "  model: width 400, depth 6, heads 8, ~11.80M parameters" \
  "  distributed ranks: ${NPROC}"
printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

if [[ ! -s "${OUTPUT_DIR}/last.pt" ]]; then
  echo "Forward training returned without a non-empty last.pt" >&2
  exit 1
fi
echo "Nominal Forward training completed: ${OUTPUT_DIR}/last.pt"
