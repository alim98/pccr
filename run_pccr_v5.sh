#!/usr/bin/env bash
# PCCR v5 — full-resolution real training with all full_final fixes applied.
#
# What changed vs full_final (pccr_full_final_20260418_210108):
#   * segmentation_supervision_weight 0.05 → 0.20   primary bottleneck: 4× direct dice gradient
#   * smoothness_weight 0.01 → 0.015                stabilise rising SDlogJ (0.59→0.67 trend)
#   * cost-volume proj/feature channels 8 → 16       more expressive final refinement head
#   * val_dice logged with sync_dist=True             fixes noisy 1-pair-per-GPU dice in 16-GPU DDP
#
# Warm-started from the strongest monitored v5 real checkpoint currently available:
# pccr_v5_20260420_130721/best-dice-epoch033-dice0.8067.ckpt.
# No synthetic re-training needed: encoder/pointmap/matcher weights carry over cleanly.
# The cost-volume encoder layers are re-initialised (shape changed); everything else loads.
#
# Usage:
#   bash run_pccr_v5.sh
#   JOB_SCRIPT="${PWD}/jobs/active/train_pccr_transmorph_pkl_4node.sh" bash run_pccr_v5.sh
#   MAX_EPOCHS=300 bash run_pccr_v5.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

if [[ -n "${JOB_SCRIPT:-}" ]]; then
  export JOB_SCRIPT
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_v5_${RUN_ID}}"

# Warm-start from the strongest monitored v5 real checkpoint we have on disk.
# Use the explicit best-dice checkpoint rather than a stale early warm-start.
# --checkpoint_path triggers strict=False loading so the resized cost-volume
# encoder is re-initialised rather than causing a shape mismatch error.
WARMSTART_CKPT="${WARMSTART_CKPT:-/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/pccr_v5_20260420_130721/best-dice-epoch033-dice0.8067.ckpt}"

if [[ ! -s "${WARMSTART_CKPT}" ]]; then
  echo "[run_pccr_v5] ERROR: warm-start checkpoint not found: ${WARMSTART_CKPT}" >&2
  exit 1
fi

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_v5.yaml}"
export PHASE="real"
export MAX_EPOCHS="${MAX_EPOCHS:-300}"
export TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"

export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-500}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"

# Cost volume at full radius — same as full_final.
export FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
export FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-8}"
export FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-4}"

# Real phase: no unused parameters in DDP.
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

# Keep LR identical to full_final. The 5-epoch warmup protects the warm-started
# weights from a sudden jump (LR ramps 2e-5 → 1e-4 over epochs 0-4).
export LR="${LR:-1e-4}"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXPERIMENT_NAME

echo "=========================================="
echo "PCCR v5 Experiment"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Config:           ${CONFIG_PATH}"
echo "Warm-start ckpt:  ${WARMSTART_CKPT}"
echo "Max epochs:       ${MAX_EPOCHS}"
echo "Val every N ep:   ${CHECK_VAL_EVERY_N_EPOCH}"
echo "Cost vol radius:  ${FINAL_COST_VOLUME_RADIUS}"
echo "Timestamp:        $(date)"
echo "=========================================="

"${LAUNCHER}" \
  --phase real \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${WARMSTART_CKPT}"
