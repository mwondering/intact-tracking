#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

printf '%s\n' \
  "run_residual_training.sh is deprecated." \
  "Forward training is now explicitly nominal-only; forwarding to:" \
  "  scripts/run_forward_nominal_training.sh" >&2

exec "${SCRIPT_DIR}/run_forward_nominal_training.sh" "$@"
