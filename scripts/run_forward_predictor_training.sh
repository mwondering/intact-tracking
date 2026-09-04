#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CHECKPOINT MOTION_SOURCE OUTPUT_DIR [PREDICTOR_OPTIONS...]" \
    "" \
    "Train the local-contrastive dynamics-context Forward Predictor." \
    "" \
    "Fixed contract:" \
    "  physics          128 fixed startup-DR prototypes; never resampled" \
    "  disturbance      checkpoint step/interval random pushes are not reproduced" \
    "  controller       frozen tracker" \
    "  model            causal Transformer + lightweight dynamics Context Encoder" \
    "  context input    100 completed robot-state/applied-target transitions (2 s); no foot/contact" \
    "  predictor input  10 historical full-state/PD-target tokens + 1 current token" \
    "  output           70-D robot delta + 8-D foot state + 6-D contact force + 2-D logits" \
    "  rollout          shared one-step model recursively applied 5 times" \
    "  supervision      dynamics prediction + local contrastive representation learning" \
    "  positive pairs   same world/episode/motion, nearby context windows" \
    "  negative pairs   every other world at the exact same motion and phase" \
    "  ignored pairs    different motion/phase; no theta or response-distance threshold" \
    "  disabled         Residual Policy, Backward, gradient clipping" \
    "" \
    "Production defaults (override with PREDICTOR_OPTIONS):" \
    "  --num-envs 2048 --batch-size 4096 --micro-batch-size 512 --amp-dtype bfloat16" \
    "  --replay-capacity 262144" \
    "  --gradient-steps-per-update 4 --updates 100000" \
    "  --contrastive-weight 0.01 --contrastive-temperature 0.1" \
    "  --dynamics-classes 128 --context-history-steps 100" \
    "  --contrastive-positive-max-offset-steps 20" \
    "" \
    "Use --fixed-batch-overfit for the mandatory model-capacity diagnostic." \
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
  --history-steps
  --context-history-steps
  --dynamics-classes
  --transformer-dim
  --transformer-depth
  --transformer-heads
  --context-dim
  --context-depth
  --context-heads
  --dynamics-latent-dim
  --dropout
  --rollout-steps-per-update
)
for argument in "$@"; do
  for managed in "${managed_options[@]}"; do
    if [[ "${argument}" == "${managed}" || "${argument}" == "${managed}="* ]]; then
      echo "${managed} is fixed by the Forward Predictor launcher." >&2
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
    -m intact_tracking.cli.forward_predictor_train
  )
else
  command+=(-m intact_tracking.cli.forward_predictor_train)
fi
command+=(
  --checkpoint-file "${CHECKPOINT_FILE}"
  "${motion_argument[@]}"
  --output-dir "${OUTPUT_DIR}"
  --num-envs 2048
  --batch-size 4096
  --micro-batch-size 512
  --amp-dtype bfloat16
  --replay-capacity 262144
  --replay-sampling motion_balanced
  --gradient-steps-per-update 4
  --updates 100000
  --wandb-project intact-forward-predictor
)
if (( NPROC == 1 )); then
  command+=(--device "${DEVICE}")
fi
command+=("$@")
command+=(
  --rollout-steps-per-update 5
  --history-steps 10
  --context-history-steps 100
  --dynamics-classes 128
  --transformer-dim 512
  --transformer-depth 6
  --transformer-heads 8
  --context-dim 128
  --context-depth 2
  --context-heads 4
  --dynamics-latent-dim 64
  --dropout 0
)

printf '%s\n' \
  "Training contract: local-contrastive Context Forward Predictor v11" \
  "  model: Context Encoder sees robot state/action only; privileged features stay in predictor" \
  "  rollout: predicted robot/foot/contact state recurs 5 steps; no articulated FK in model" \
  "  disturbance: checkpoint step/interval random pushes are removed" \
  "  loss: dynamics + local exact-cohort contrastive (0.01); no theta decoder" \
  "  context: 100 proprioceptive frames; reset-padded contexts predict but do not contrast" \
  "  positives: same-world windows shifted 5/10/15/20 steps within one episode and motion" \
  "  negatives: all other worlds in the same 128-class motion/phase cohort" \
  "  replay: motion-balanced, 262144 samples per rank by default" \
  "  optimizer: effective batch 4096, micro-batch 512, BF16 autocast, fused AdamW" \
  "  diagnostics: full metric/probe evaluation every --log-interval updates" \
  "  normalization: robot/target/foot/contact/delta/DR-provenance stats frozen after warmup" \
  "  inference: history-only; theta is never a model input or prediction target" \
  "  pair filter: cross-motion/phase pairs ignored; no dynamics-distance threshold" \
  "  policy/backward: disabled" \
  "  distributed ranks: ${NPROC}"
printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

if [[ ! -s "${OUTPUT_DIR}/last.pt" ]]; then
  echo "Forward Predictor training returned without a non-empty last.pt" >&2
  exit 1
fi
echo "Forward Predictor training completed: ${OUTPUT_DIR}/last.pt"
