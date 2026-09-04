#!/usr/bin/env bash

set -Eeuo pipefail

if (( $# == 0 )) || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: $0 TRACKER_CHECKPOINT MOTION_SOURCE OUTPUT_DIR [OPTIONS...]" \
    "" \
    "Train the otherwise identical residual PPO baseline without dynamics latent."
  exit 0
fi
if (( $# < 3 )); then
  echo "Usage: $0 TRACKER_CHECKPOINT MOTION_SOURCE OUTPUT_DIR [OPTIONS...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_CHECKPOINT="$1"
MOTION_SOURCE="$2"
OUTPUT_DIR="$3"
shift 3

exec "${SCRIPT_DIR}/run_residual_policy_training.sh" \
  "${TRACKER_CHECKPOINT}" \
  "${MOTION_SOURCE}" \
  "${OUTPUT_DIR}" \
  no-latent \
  "$@"

