#!/usr/bin/env bash
# PCCR v6 — full-resolution real training.
#
# Improvements over v5:
#   * Full-resolution multiscale GDL supervision (was half-res soft-Dice)
#       - scales [1x, 0.5x, 0.25x] with weights [1.0, 0.5, 0.25]
#       - inverse-volume label weighting (GDL) so small structures contribute properly
#   * Full-resolution smoothness + Jacobian regularisation (was half-res)
#   * Stage-2 local correlation restored at full resolution; dense global
#     pointmap matching stays at stage 3 because stage 2 is too large for cdist
#   * Cost-volume proj/feature channels 8→16 (bug fix from v5 launch notes)
#   * Real-phase per-sample intensity + small affine augmentation
#
# Warm-started from the strongest available v5 real checkpoint
# pccr_v5_20260420_130721/best-dice-epoch033-dice0.8067.ckpt.
# Strict=False load with partial tensor warm-start: resized refinement/cost-volume
# tensors keep overlapping checkpoint channels and initialise only new channels.
#
# Usage:
#   bash run_pccr_v6_real.sh
#   _SCRIPT="${PWD}/jobs/active/train_pccr_v6_4node.sh" bash run_pccr_v6_real.sh
#   MAX_EPOCHS=150 bash run_pccr_v6_real.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

JOB_SCRIPT="${_SCRIPT:-${REPO_ROOT}/jobs/active/train_pccr_v6_4node.sh}"
if [[ -n "${JOB_SCRIPT:-}" ]]; then
  export JOB_SCRIPT
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_v6_${RUN_ID}}"

WARMSTART_CKPT="${WARMSTART_CKPT:-/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/pccr_v5_20260420_130721/best-dice-epoch033-dice0.8067.ckpt}"

if [[ ! -s "${WARMSTART_CKPT}" ]]; then
  echo "[run_pccr_v6_real] ERROR: warm-start checkpoint not found: ${WARMSTART_CKPT}" >&2
  exit 1
fi

export QOS="${QOS:-g0016}"
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_v6.yaml}"
export PHASE="real"
export MAX_EPOCHS="${MAX_EPOCHS:-250}"
export TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"

export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-0}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"
export ITER_EVAL_EVERY_N_EPOCHS="${ITER_EVAL_EVERY_N_EPOCHS:-5}"
export ITER_EVAL_NUM_PAIRS="${ITER_EVAL_NUM_PAIRS:-${MAX_VAL_PAIRS}}"
export ITER_EVAL_SKIP_HD95="${ITER_EVAL_SKIP_HD95:-1}"

export FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
export FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-8}"
export FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-4}"

export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

export LR="${LR:-1e-4}"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Force the initial submit to use WARMSTART_CKPT even if EXPERIMENT_NAME
# already has stale checkpoints. Launcher restarts still resume in-experiment.
export START_CHECKPOINT_POLICY="${START_CHECKPOINT_POLICY:-requested}"
export EXPERIMENT_NAME

echo "=========================================="
echo "PCCR v6 Experiment"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Config:           ${CONFIG_PATH}"
echo "Job script:       ${JOB_SCRIPT}"
echo "Warm-start ckpt:  ${WARMSTART_CKPT}"
echo "Max epochs:       ${MAX_EPOCHS}"
echo "Val every N ep:   ${CHECK_VAL_EVERY_N_EPOCH}"
echo "Iter eval:        every ${ITER_EVAL_EVERY_N_EPOCHS} epochs, pairs=${ITER_EVAL_NUM_PAIRS}, skip_hd95=${ITER_EVAL_SKIP_HD95}"
echo "Cost vol radius:  ${FINAL_COST_VOLUME_RADIUS}"
echo "Timestamp:        $(date)"
echo "=========================================="

"${LAUNCHER}" \
  --phase real \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${WARMSTART_CKPT}"
