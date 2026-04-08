#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"

echo "[run_pccr_oasis_v4_all.sh] RUN_TAG=${RUN_TAG}"
echo "[run_pccr_oasis_v4_all.sh] INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH}"

(
  cd "${REPO_ROOT}"
  RUN_TAG="${RUN_TAG}" INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH}" REAL_EXPERIMENT_NAME="pccr_oasis_real_v4a_${RUN_TAG}" ./run_pccr_oasis_v4a.sh
) &
pid_a=$!

(
  cd "${REPO_ROOT}"
  RUN_TAG="${RUN_TAG}" INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH}" REAL_EXPERIMENT_NAME="pccr_oasis_real_v4b_${RUN_TAG}" ./run_pccr_oasis_v4b.sh
) &
pid_b=$!

(
  cd "${REPO_ROOT}"
  RUN_TAG="${RUN_TAG}" INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH}" REAL_EXPERIMENT_NAME="pccr_oasis_real_v4c_${RUN_TAG}" ./run_pccr_oasis_v4c.sh
) &
pid_c=$!

wait "${pid_a}" "${pid_b}" "${pid_c}"
