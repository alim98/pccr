#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-2}"

echo "[run_pccr_proxy20_all_local.sh] RUN_TAG=${RUN_TAG}"

(
  cd "${REPO_ROOT}"
  VARIANT=v4a GPU_ID="${GPU_A}" EXPERIMENT_NAME="pccr_v4a_proxy20_${RUN_TAG}" ./run_pccr_proxy20_local.sh
) &
pid_a=$!

(
  cd "${REPO_ROOT}"
  VARIANT=v4b GPU_ID="${GPU_B}" EXPERIMENT_NAME="pccr_v4b_proxy20_${RUN_TAG}" ./run_pccr_proxy20_local.sh
) &
pid_b=$!

(
  cd "${REPO_ROOT}"
  VARIANT=v4c GPU_ID="${GPU_C}" EXPERIMENT_NAME="pccr_v4c_proxy20_${RUN_TAG}" ./run_pccr_proxy20_local.sh
) &
pid_c=$!

wait "${pid_a}" "${pid_b}" "${pid_c}"
