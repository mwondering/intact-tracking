#!/usr/bin/env bash

set -Eeuo pipefail

if (( $# == 0 )) || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: $0 TRACKER_CHECKPOINT FORWARD_CHECKPOINT MOTION_SOURCE OUTPUT_DIR [OPTIONS...]" \
    "" \
    "Train the latent-conditioned frozen-tracker residual PPO baseline."
  exit 0
fi
if (( $# < 4 )); then
  echo "Usage: $0 TRACKER_CHECKPOINT FORWARD_CHECKPOINT MOTION_SOURCE OUTPUT_DIR [OPTIONS...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_CHECKPOINT="$1"
FORWARD_CHECKPOINT="$2"
MOTION_SOURCE="$3"
OUTPUT_DIR="$4"
shift 4

exec "${SCRIPT_DIR}/run_residual_policy_training.sh" \
  "${TRACKER_CHECKPOINT}" \
  "${MOTION_SOURCE}" \
  "${OUTPUT_DIR}" \
  latent \
  --forward-checkpoint "${FORWARD_CHECKPOINT}" \
  "$@"

