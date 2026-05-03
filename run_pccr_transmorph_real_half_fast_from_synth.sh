#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_half_fast.yaml}"
FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-legacy}"
FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-16}"
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-500}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SYNTH_EXPERIMENT_NAME="${SYNTH_EXPERIMENT_NAME:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-pccr_transmorph_pkl_half_fast_real_${RUN_ID}}"

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

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if [[ -z "${SYNTH_EXPERIMENT_NAME}" ]]; then
    echo "[half_fast_real] Set either CHECKPOINT_PATH or SYNTH_EXPERIMENT_NAME." >&2
    exit 1
  fi

  SYNTH_CHECKPOINT_ROOT="/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/${SYNTH_EXPERIMENT_NAME}"
  echo "[half_fast_real] Waiting for checkpoint under ${SYNTH_CHECKPOINT_ROOT}"

  while true; do
    CHECKPOINT_PATH="$(latest_checkpoint "${SYNTH_CHECKPOINT_ROOT}")"
    if [[ -n "${CHECKPOINT_PATH}" && -s "${CHECKPOINT_PATH}" ]]; then
      break
    fi
    sleep "${WAIT_POLL_SECONDS}"
  done
fi

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "[half_fast_real] Checkpoint does not exist or is empty: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

export CONFIG_PATH
export FINAL_COST_VOLUME_MODE
export FINAL_COST_VOLUME_CHUNK_SIZE
export CHECKPOINT_EVERY_N_TRAIN_STEPS
export LOG_EVERY_N_STEPS

echo "[half_fast_real] Config: ${CONFIG_PATH}"
echo "[half_fast_real] Final cost volume mode: ${FINAL_COST_VOLUME_MODE}"
echo "[half_fast_real] Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "[half_fast_real] Starting from checkpoint: ${CHECKPOINT_PATH}"

"${LAUNCHER}" \
  --phase real \
  --experiment_name "${REAL_EXPERIMENT_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}"
