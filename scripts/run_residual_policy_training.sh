#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 TRACKER_CHECKPOINT MOTION_SOURCE OUTPUT_DIR {latent|no-latent} [OPTIONS...]" \
    "" \
    "Train a residual PPO on the exact SPV5-2A task/reward/critic contract." \
    "" \
    "latent requires: --forward-checkpoint PATH" \
    "no-latent rejects: --forward-checkpoint" \
    "" \
    "The frozen tracker supplies the base action. The trainable actor outputs only" \
    "a bounded residual; the latent variant runs only the frozen Context Encoder." \
    "Checkpoint step/interval disturbances (including random pushes) remain disabled." \
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
MOTION_SOURCE="$2"
OUTPUT_DIR="$3"
BASELINE="$4"
shift 4

if [[ "${BASELINE}" != "latent" && "${BASELINE}" != "no-latent" ]]; then
  echo "Baseline must be latent or no-latent, got: ${BASELINE}" >&2
  exit 2
fi
for argument in "$@"; do
  case "${argument}" in
    --num-envs|--num-envs=*|--baseline|--baseline=*|--task-id|--task-id=*)
      echo "${argument%%=*} is fixed by the residual-policy launcher." >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${TRACKER_CHECKPOINT}" ]]; then
  echo "Tracker checkpoint not found: ${TRACKER_CHECKPOINT}" >&2
  exit 2
fi
TRACKER_CHECKPOINT="$(realpath --canonicalize-existing -- "${TRACKER_CHECKPOINT}")"

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
    -m intact_tracking.cli.residual_policy_train
  )
else
  command+=(-m intact_tracking.cli.residual_policy_train)
fi
command+=(
  --tracker-checkpoint "${TRACKER_CHECKPOINT}"
  "${motion_argument[@]}"
  --output-dir "${OUTPUT_DIR}"
  --baseline "${BASELINE}"
  --task-id SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward
  --num-envs "${LOCAL_ENVS}"
  --iterations 100000
  --num-steps-per-env 24
  --save-interval 1000
  --residual-hidden-dims 512 256 128
  --residual-scale 0.25
  --no-include-disturbances
  --logger wandb
  --wandb-project intact-residual-policy
)
if (( NPROC == 1 )); then
  command+=(--device "${DEVICE}")
fi
command+=("$@")

printf '%s\n' \
  "Training contract: frozen SPV5-2A tracker + residual PPO (${BASELINE})" \
  "  task/reward/critic: restored from the tracker checkpoint" \
  "  base policy: frozen; final PPO Gaussian is base mean + bounded residual" \
  "  initialization: zero residual and tracker checkpoint action std" \
  "  actor observations: exact frozen-tracker processed observation" \
  "  latent: $([[ "${BASELINE}" == "latent" ]] && echo 'frozen 100-frame Context Encoder' || echo 'disabled')" \
  "  Forward Predictor/theta: never executed or exposed to the residual actor" \
  "  disturbances: checkpoint random pushes removed" \
  "  environments: ${GLOBAL_ENVS} global (${LOCAL_ENVS} per rank, ${NPROC} ranks)"
printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${INTACT_ROOT}"
if [[ "${resume_requested}" == true ]]; then
  "${command[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
else
  "${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

if [[ ! -s "${OUTPUT_DIR}/checkpoint_final.pt" ]]; then
  echo "Residual PPO returned without a non-empty checkpoint_final.pt" >&2
  exit 1
fi
echo "Residual PPO training completed: ${OUTPUT_DIR}/checkpoint_final.pt"

