#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_v4d_no_local_matcher.yaml}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-}"

if [[ -z "${REAL_EXPERIMENT_NAME}" ]]; then
  RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
  export RUN_TAG
  REAL_EXPERIMENT_NAME="pccr_oasis_real_v4d_no_local_matcher_${RUN_TAG}"
fi

export CONFIG_PATH REAL_EXPERIMENT_NAME
"${REPO_ROOT}/run_pccr_oasis_v4d.sh"
