#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SYNTH_EXPERIMENT_NAME="${SYNTH_EXPERIMENT_NAME:-pccr_transmorph_pkl_v3_vec_full_20260410_125634_synthetic}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-pccr_transmorph_pkl_v3_vec_full_real_optimized_${RUN_ID}}"
JOB_NAME="${JOB_NAME:-pccr_tm_pkl_real_full_optimized}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl.yaml}"
FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
# Low-risk optimization variant: same full-size setup, but reduce final CV radius.
FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-6}"
FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-16}"
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-25}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"
# Optimized variant for this hardware: keep the radius-6 speedup, but force
# inner chunk checkpointing so the full-size real run stays memory-safe.
PCCR_DISABLE_INNER_CHKPT_MAX_VOXELS="${PCCR_DISABLE_INNER_CHKPT_MAX_VOXELS:-0}"

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
  echo "[full_real_optimized] No usable checkpoint found yet under ${SYNTH_CHECKPOINT_ROOT}" >&2
  exit 1
fi

export CONFIG_PATH
export FINAL_COST_VOLUME_MODE
export FINAL_COST_VOLUME_RADIUS
export FINAL_COST_VOLUME_CHUNK_SIZE
export CHECKPOINT_EVERY_N_TRAIN_STEPS
export CHECK_VAL_EVERY_N_EPOCH
export LOG_EVERY_N_STEPS
export MAX_VAL_PAIRS
export JOB_NAME
export PCCR_DISABLE_INNER_CHKPT_MAX_VOXELS

echo "=========================================="
echo "PCCR TransMorph PKL Full-Size Optimized Real Launcher"
echo "Synthetic experiment: ${SYNTH_EXPERIMENT_NAME}"
echo "Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "Job name: ${JOB_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Final cost volume mode: ${FINAL_COST_VOLUME_MODE}"
echo "Final cost volume radius: ${FINAL_COST_VOLUME_RADIUS}"
echo "Final cost volume chunk size: ${FINAL_COST_VOLUME_CHUNK_SIZE}"
echo "Inner checkpoint voxel threshold: ${PCCR_DISABLE_INNER_CHKPT_MAX_VOXELS}"
echo "Checkpoint every n train steps: ${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
echo "Starting from checkpoint: ${CHECKPOINT_PATH}"
echo "Timestamp: $(date)"
echo "=========================================="

"${LAUNCHER}" \
  --phase real \
  --job_name "${JOB_NAME}" \
  --experiment_name "${REAL_EXPERIMENT_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}"
