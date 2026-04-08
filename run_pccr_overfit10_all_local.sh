#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-2}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pccr_overfit10_local/${RUN_TAG}}"

mkdir -p "${LOG_DIR}"

(
  cd "${REPO_ROOT}"
  VARIANT=v4a GPU_ID="${GPU_A}" CONFIG_PATH=src/pccr/configs/pairwise_oasis_v4a_overfit10.yaml INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH}" EXPERIMENT_NAME="pccr_v4a_overfit10_${RUN_TAG}" ./run_pccr_overfit10_local.sh
) > "${LOG_DIR}/v4a.log" 2>&1 &
pid_a=$!

(
  cd "${REPO_ROOT}"
  VARIANT=v4b GPU_ID="${GPU_B}" CONFIG_PATH=src/pccr/configs/pairwise_oasis_v4b_overfit10.yaml INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH}" EXPERIMENT_NAME="pccr_v4b_overfit10_${RUN_TAG}" ./run_pccr_overfit10_local.sh
) > "${LOG_DIR}/v4b.log" 2>&1 &
pid_b=$!

(
  cd "${REPO_ROOT}"
  VARIANT=v4c GPU_ID="${GPU_C}" CONFIG_PATH=src/pccr/configs/pairwise_oasis_v4c_overfit10.yaml INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH}" EXPERIMENT_NAME="pccr_v4c_overfit10_${RUN_TAG}" ./run_pccr_overfit10_local.sh
) > "${LOG_DIR}/v4c.log" 2>&1 &
pid_c=$!

wait "${pid_a}" "${pid_b}" "${pid_c}"

echo "[run_pccr_overfit10_all_local.sh] Logs: ${LOG_DIR}"
