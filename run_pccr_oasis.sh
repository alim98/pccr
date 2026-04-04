#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PIPELINE_LAUNCHER="${REPO_ROOT}/launch_pccr_full.sh"

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
AIM_REPO_DEFAULT="${AIM_REPO_DEFAULT:-${REPO_ROOT}/aim}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
SYNTH_EXPERIMENT_NAME="${SYNTH_EXPERIMENT_NAME:-pccr_oasis_synth_${RUN_TAG}}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-pccr_oasis_real_${RUN_TAG}}"

# Default OASIS pipeline settings. Override via env vars if needed.
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-1}"
NUM_WORKERS_DEFAULT="${NUM_WORKERS_DEFAULT:-8}"
PRECISION_DEFAULT="${PRECISION_DEFAULT:-bf16-mixed}"

LR_SYNTH_DEFAULT="${LR_SYNTH_DEFAULT:-1e-4}"
MAX_EPOCHS_SYNTH_DEFAULT="${MAX_EPOCHS_SYNTH_DEFAULT:-50}"

LR_REAL_DEFAULT="${LR_REAL_DEFAULT:-5e-5}"
MAX_EPOCHS_REAL_DEFAULT="${MAX_EPOCHS_REAL_DEFAULT:-200}"
TRAIN_NUM_STEPS_REAL_DEFAULT="${TRAIN_NUM_STEPS_REAL_DEFAULT:-200}"
MAX_VAL_PAIRS_REAL_DEFAULT="${MAX_VAL_PAIRS_REAL_DEFAULT:-20}"

echo "=========================================="
echo "PCCR OASIS Full Runner"
echo "Repository: ${REPO_ROOT}"
echo "Environment: ${ENV_PATH}"
echo "Data root: ${DATA_ROOT_DEFAULT}"
echo "Aim repo: ${AIM_REPO_DEFAULT}"
echo "Synthetic experiment: ${SYNTH_EXPERIMENT_NAME}"
echo "Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "Timestamp: $(date)"
echo "=========================================="

export ENV_PATH
export DATA_ROOT_DEFAULT
export AIM_REPO_DEFAULT
export NUM_GPUS_PER_NODE
export BATCH_SIZE_DEFAULT
export NUM_WORKERS_DEFAULT
export PRECISION_DEFAULT
export LR_SYNTH_DEFAULT
export MAX_EPOCHS_SYNTH_DEFAULT
export LR_REAL_DEFAULT
export MAX_EPOCHS_REAL_DEFAULT
export TRAIN_NUM_STEPS_REAL_DEFAULT
export MAX_VAL_PAIRS_REAL_DEFAULT

CMD=(
  "${PIPELINE_LAUNCHER}"
  --synthetic_experiment_name "${SYNTH_EXPERIMENT_NAME}"
  --real_experiment_name "${REAL_EXPERIMENT_NAME}"
)

printf '[run_pccr_oasis.sh] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
