#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PIPELINE_LAUNCHER="${REPO_ROOT}/launch_pccr_full.sh"

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
AIM_REPO_DEFAULT="${AIM_REPO_DEFAULT:-${REPO_ROOT}/aim}"
CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_v5.yaml}"
DRY_RUN="${DRY_RUN:-0}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
SYNTH_EXPERIMENT_NAME="${SYNTH_EXPERIMENT_NAME:-pccr_v5_synth_${RUN_TAG}}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-pccr_v5_real_${RUN_TAG}}"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-1}"
NUM_WORKERS_DEFAULT="${NUM_WORKERS_DEFAULT:-8}"
PRECISION_DEFAULT="${PRECISION_DEFAULT:-bf16-mixed}"

LR_SYNTH_DEFAULT="${LR_SYNTH_DEFAULT:-1e-4}"
MAX_EPOCHS_SYNTH_DEFAULT="${MAX_EPOCHS_SYNTH_DEFAULT:-50}"

LR_REAL_DEFAULT="${LR_REAL_DEFAULT:-2e-5}"
MAX_EPOCHS_REAL_DEFAULT="${MAX_EPOCHS_REAL_DEFAULT:-200}"
TRAIN_NUM_STEPS_REAL_DEFAULT="${TRAIN_NUM_STEPS_REAL_DEFAULT:-600}"
MAX_VAL_PAIRS_REAL_DEFAULT="${MAX_VAL_PAIRS_REAL_DEFAULT:-20}"
ITER_EVAL_EVERY_N_EPOCHS_DEFAULT="${ITER_EVAL_EVERY_N_EPOCHS_DEFAULT:-25}"
ITER_EVAL_NUM_PAIRS_DEFAULT="${ITER_EVAL_NUM_PAIRS_DEFAULT:-100}"
ITER_VIZ_EVERY_N_EPOCHS_DEFAULT="${ITER_VIZ_EVERY_N_EPOCHS_DEFAULT:-50}"
ITER_VIZ_PAIR_INDEX_DEFAULT="${ITER_VIZ_PAIR_INDEX_DEFAULT:-0}"

echo "=========================================="
echo "PCCR OASIS V5 Full Runner"
echo "Repository: ${REPO_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Synthetic experiment: ${SYNTH_EXPERIMENT_NAME}"
echo "Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "Timestamp: $(date)"
echo "=========================================="

if [[ ! -x "${PIPELINE_LAUNCHER}" ]]; then
  echo "[run_pccr_oasis_v5.sh] Launcher not found or not executable: ${PIPELINE_LAUNCHER}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG_PATH}" ]]; then
  echo "[run_pccr_oasis_v5.sh] Config not found: ${REPO_ROOT}/${CONFIG_PATH}" >&2
  exit 1
fi

export ENV_PATH
export DATA_ROOT_DEFAULT
export AIM_REPO_DEFAULT
export NUM_GPUS_PER_NODE
export BATCH_SIZE_DEFAULT
export NUM_WORKERS_DEFAULT
export PRECISION_DEFAULT

CMD=(
  "${PIPELINE_LAUNCHER}"
  --synthetic_experiment_name "${SYNTH_EXPERIMENT_NAME}"
  --real_experiment_name "${REAL_EXPERIMENT_NAME}"
  --config "${CONFIG_PATH}"
  --max_epochs "${MAX_EPOCHS_SYNTH_DEFAULT}"
  --lr "${LR_SYNTH_DEFAULT}"
  --
  --config "${CONFIG_PATH}"
  --max_epochs "${MAX_EPOCHS_REAL_DEFAULT}"
  --train_num_steps "${TRAIN_NUM_STEPS_REAL_DEFAULT}"
  --lr "${LR_REAL_DEFAULT}"
  --max_val_pairs "${MAX_VAL_PAIRS_REAL_DEFAULT}"
  --iter_eval_every_n_epochs "${ITER_EVAL_EVERY_N_EPOCHS_DEFAULT}"
  --iter_eval_num_pairs "${ITER_EVAL_NUM_PAIRS_DEFAULT}"
  --iter_eval_skip_hd95
  --iter_viz_every_n_epochs "${ITER_VIZ_EVERY_N_EPOCHS_DEFAULT}"
  --iter_viz_pair_index "${ITER_VIZ_PAIR_INDEX_DEFAULT}"
)

printf '[run_pccr_oasis_v5.sh] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[run_pccr_oasis_v5.sh] DRY_RUN=1, skipping execution."
  exit 0
fi

"${CMD[@]}"
