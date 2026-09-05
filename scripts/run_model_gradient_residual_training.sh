#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 TRACKER_CHECKPOINT FORWARD_CHECKPOINT MOTION_SOURCE OUTPUT_DIR [OPTIONS...]" \
    "" \
    "Train a residual policy by differentiating a five-step tracking surrogate" \
    "through the frozen v12 Forward Predictor and its frozen Context Encoder." \
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
if (( $# < 4 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INTACT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${INTACT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
GPUS="${GPUS:-}"

TRACKER_CHECKPOINT="$1"
FORWARD_CHECKPOINT="$2"
MOTION_SOURCE="$3"
OUTPUT_DIR="$4"
shift 4

for argument in "$@"; do
  case "${argument}" in
    --num-envs|--num-envs=*|--task-id|--task-id=*)
      echo "${argument%%=*} is fixed by the model-gradient launcher." >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
for checkpoint in "${TRACKER_CHECKPOINT}" "${FORWARD_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
done
TRACKER_CHECKPOINT="$(realpath --canonicalize-existing -- "${TRACKER_CHECKPOINT}")"
FORWARD_CHECKPOINT="$(realpath --canonicalize-existing -- "${FORWARD_CHECKPOINT}")"

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

resume_requested=false
for argument in "$@"; do
  if [[ "${argument}" == "--resume" || "${argument}" == --resume=* ]]; then
    resume_requested=true
  fi
done
if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
  echo "Output path exists and is not a directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ "${resume_requested}" == false && -d "${OUTPUT_DIR}" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to use a non-empty output directory without --resume: ${OUTPUT_DIR}" >&2
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
GLOBAL_ENVS=4096
if (( GLOBAL_ENVS % NPROC != 0 )); then
  echo "4096 global environments must divide evenly across ${NPROC} ranks." >&2
  exit 2
fi
LOCAL_ENVS=$((GLOBAL_ENVS / NPROC))

command=(env -u PYTHONPATH MPLCONFIGDIR=/tmp/intact-matplotlib)
if [[ -n "${GPUS}" ]]; then
  command+=(CUDA_VISIBLE_DEVICES="${GPUS}")
fi
command+=("${PYTHON_BIN}")
if (( NPROC > 1 )); then
  command+=(
    -m torch.distributed.run
    --standalone
    --nproc-per-node "${NPROC}"
    -m intact_tracking.cli.model_gradient_residual_train
  )
else
  command+=(-m intact_tracking.cli.model_gradient_residual_train)
fi
command+=(
  --tracker-checkpoint "${TRACKER_CHECKPOINT}"
  --forward-checkpoint "${FORWARD_CHECKPOINT}"
  "${motion_argument[@]}"
  --output-dir "${OUTPUT_DIR}"
  --task-id SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward
  --num-envs "${LOCAL_ENVS}"
  --updates 100000
  --gradient-steps-per-update 1
  --batch-size 1024
  --micro-batch-size 128
  --probe-batch-size 256
  --learning-rate 0.0001
  --residual-hidden-dims 512 256 128
  --residual-scale 0.25
  --nominal-fraction 0
  --wandb
  --wandb-project intact-model-gradient-residual
)
if (( NPROC == 1 )); then
  command+=(--device "${DEVICE}")
fi
command+=("$@")

printf '%s\n' \
  "Training contract: frozen-model-gradient residual policy v1" \
  "  tracker: frozen SPV5-2A; residual input keeps its exact 1645-D features" \
  "  latent: frozen history-only 100-frame Context Encoder" \
  "  gradient: five recursive Forward Predictor steps -> residual MLP only" \
  "  action: raw tracker+residual -> exact differentiable physical PD target" \
  "  physics: ${GLOBAL_ENVS} fixed startup-DR worlds; random pushes removed" \
  "  trust region: bounded residual plus magnitude and temporal-smoothness penalties" \
  "  devices: ${NPROC} rank(s), ${LOCAL_ENVS} environments per rank"
printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
if [[ "${resume_requested}" == true ]]; then
  "${command[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
else
  "${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

if [[ ! -s "${OUTPUT_DIR}/last.pt" ]]; then
  echo "Model-gradient residual training returned without a non-empty last.pt" >&2
  exit 1
fi
echo "Model-gradient residual training completed: ${OUTPUT_DIR}/last.pt"
