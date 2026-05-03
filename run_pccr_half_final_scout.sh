#!/usr/bin/env bash
# Scout run: validates all loss fixes at half-resolution in ~4-5 days.
#
# Purpose:
#   Before committing to a 50-day full-res run, this gives you a fast signal.
#   Same warm-start checkpoint, same config fixes, just 80x96x112 instead of full-res.
#   Half-res runs at ~12 epochs/day → 50 epochs = 4 days.
#
# Decision rule:
#   - val_dice > 0.79  →  fixes work, full-res run is correct, let it continue
#   - val_dice ≤ 0.76  →  no improvement, something is still broken, investigate before
#                         spending 50 days on the full-res run
#
# Launch both this AND run_pccr_full_final.sh at the same time.
# If the scout shows improvement around day 4, you have confirmation.
# The full-res run will already be ~25 epochs ahead by then.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_half_final_scout_${RUN_ID}}"

SYNTHETIC_CKPT="/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/pccr_transmorph_pkl_half_fast_full_v2_synthetic/last.ckpt"

if [[ ! -s "${SYNTHETIC_CKPT}" ]]; then
  echo "[run_pccr_half_final_scout] ERROR: synthetic checkpoint not found: ${SYNTHETIC_CKPT}" >&2
  exit 1
fi

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_half_final_scout.yaml}"
export PHASE="real"
export MAX_EPOCHS="${MAX_EPOCHS:-50}"
export TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-1000}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"

# Half-res: no cost volume memory tricks needed
export FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-legacy}"
export FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-8}"
export FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-16}"

export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXPERIMENT_NAME

echo "=========================================="
echo "PCCR Half-Res Scout Run (config validation)"
echo "Experiment:     ${EXPERIMENT_NAME}"
echo "Config:         ${CONFIG_PATH}"
echo "Warm-start:     ${SYNTHETIC_CKPT}"
echo "Max epochs:     ${MAX_EPOCHS}  (~4-5 days)"
echo "Target signal:  val_dice > 0.79 confirms fixes are working"
echo "Timestamp:      $(date)"
echo "=========================================="

"${LAUNCHER}" \
  --phase real \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${SYNTHETIC_CKPT}"
