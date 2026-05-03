#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SYNTH_EXPERIMENT_NAME="${SYNTH_EXPERIMENT_NAME:-pccr_transmorph_pkl_half_v5_synthetic}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-pccr_transmorph_pkl_half_v5_real_now_${RUN_ID}}"
JOB_NAME="${JOB_NAME:-pccr_tm_pkl_half_v5_now}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_half_v5.yaml}"
FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-off}"
FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-0}"
FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-16}"
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-250}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-10}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"
REAL_MAX_EPOCHS="${REAL_MAX_EPOCHS:-200}"
REAL_TRAIN_NUM_STEPS="${REAL_TRAIN_NUM_STEPS:-600}"
REAL_LR="${REAL_LR:-2e-5}"

latest_checkpoint() {
  local root="$1"
  local last_ckpt=""
  local any_ckpt=""

  last_ckpt="$(find "${root}" -maxdepth 2 -type f -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"
  any_ckpt="$(find "${root}" -maxdepth 2 -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"

  if [[ -n "${last_ckpt}" ]]; then
    printf '%s\n' "${last_ckpt}"
  elif [[ -n "${any_ckpt}" ]]; then
    printf '%s\n' "${any_ckpt}"
  else
    printf '\n'
  fi
}

SYNTH_CHECKPOINT_ROOT="/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/${SYNTH_EXPERIMENT_NAME}"
CHECKPOINT_PATH="$(latest_checkpoint "${SYNTH_CHECKPOINT_ROOT}")"

if [[ -z "${CHECKPOINT_PATH}" || ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "[half_v5_real_now] No usable checkpoint found yet under ${SYNTH_CHECKPOINT_ROOT}" >&2
  exit 1
fi

echo "=========================================="
echo "PCCR TransMorph PKL Half V5-Style Immediate Real Launcher"
echo "Synthetic experiment: ${SYNTH_EXPERIMENT_NAME}"
echo "Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "Job name: ${JOB_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Final cost volume mode: ${FINAL_COST_VOLUME_MODE}"
echo "Starting from checkpoint: ${CHECKPOINT_PATH}"
echo "Timestamp: $(date)"
echo "=========================================="

CONFIG_PATH="${CONFIG_PATH}" \
FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE}" \
FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS}" \
FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE}" \
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS}" \
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH}" \
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS}" \
MAX_VAL_PAIRS="${MAX_VAL_PAIRS}" \
MAX_EPOCHS="${REAL_MAX_EPOCHS}" \
TRAIN_NUM_STEPS="${REAL_TRAIN_NUM_STEPS}" \
LR="${REAL_LR}" \
JOB_NAME="${JOB_NAME}" \
"${LAUNCHER}" \
  --phase real \
  --job_name "${JOB_NAME}" \
  --experiment_name "${REAL_EXPERIMENT_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}"
