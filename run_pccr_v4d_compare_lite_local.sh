#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
TEST_KIND="${TEST_KIND:-overfit}"
VARIANTS="${VARIANTS:-v4b v4d_no_local_matcher v4d}"

case "${TEST_KIND}" in
  overfit)
    RUNNER="${REPO_ROOT}/run_pccr_overfit10_lite_local.sh"
    SUFFIX="overfit10_lite"
    ;;
  freeze)
    RUNNER="${REPO_ROOT}/run_pccr_freeze_lite_local.sh"
    SUFFIX="freeze_lite"
    ;;
  proxy)
    RUNNER="${REPO_ROOT}/run_pccr_proxy20_lite_local.sh"
    SUFFIX="proxy20_lite"
    ;;
  *)
    echo "Unsupported TEST_KIND=${TEST_KIND}. Use overfit, freeze, or proxy." >&2
    exit 1
    ;;
esac

for variant in ${VARIANTS}; do
  echo "[run_pccr_v4d_compare_lite_local.sh] Running ${TEST_KIND} for ${variant}"
  VARIANT="${variant}" \
  GPU_ID="${GPU_ID}" \
  EXPERIMENT_NAME="pccr_${variant}_${SUFFIX}_${RUN_TAG}" \
  "${RUNNER}"
done
