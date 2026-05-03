#!/usr/bin/env bash
# PCCR v7 — last-shot, full-resolution real training.
#
# What changed vs v5 (architecturally identical, byte-for-byte compatible weights):
#   1. Multi-window LNCC                  windows=[5, 9, 13]
#   2. Per-stage Dice supervision         {stage1: 0.10, stage2: 0.05}
#                                         (final-φ Dice unchanged at 0.20)
#   3. Hyperelastic regularizer           weight=0.005, power=2.0
#                                         (smoothness reduced 0.015 → 0.012)
#   4. Symmetric inverse-consistent       enabled at val/test
#   5. LR warmup                          5 → 8 epochs (protects warm-start)
#
# All five are activated via config flags; the model architecture is
# unchanged so warm-start from any v5 checkpoint loads strict.
#
# Usage:
#   bash run_pccr_v7.sh
#   WARMSTART_CKPT=<path> bash run_pccr_v7.sh
#   MAX_EPOCHS=300 JOB_SCRIPT=<path/to/4node.sh> bash run_pccr_v7.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

# 4-node SLURM launcher by default (16 x A100). Override with JOB_SCRIPT.
JOB_SCRIPT="${JOB_SCRIPT:-${REPO_ROOT}/jobs/active/train_pccr_transmorph_pkl_4node.sh}"
export JOB_SCRIPT

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_v7_${RUN_ID}}"

# ──────────────────────────────────────────────────────────────────────────────
# Warm-start checkpoint resolution.
#
# Priority:
#   1. WARMSTART_CKPT env var (explicit override)
#   2. The newest best-dice-* under the most-recent v5 run on disk
#   3. Hardcoded fallback to pccr_v5_20260425_115302/best-dice-epoch014-dice0.8115.ckpt
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_V5_CKPT_DIR="${DEFAULT_V5_CKPT_DIR:-/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/pccr_v5_20260425_115302}"
HARDCODED_FALLBACK="${DEFAULT_V5_CKPT_DIR}/best-dice-epoch014-dice0.8115.ckpt"

if [[ -z "${WARMSTART_CKPT:-}" ]]; then
  if compgen -G "${DEFAULT_V5_CKPT_DIR}/best-dice-*.ckpt" > /dev/null 2>&1; then
    # Pick the best by alphabetical order of the dice-suffix (sort -V handles 0.8115 > 0.8114).
    WARMSTART_CKPT="$(ls -1 "${DEFAULT_V5_CKPT_DIR}"/best-dice-*.ckpt 2>/dev/null | sort -V | tail -n 1)"
  fi
  WARMSTART_CKPT="${WARMSTART_CKPT:-${HARDCODED_FALLBACK}}"
fi

if [[ ! -s "${WARMSTART_CKPT}" ]]; then
  echo "[run_pccr_v7] ERROR: warm-start checkpoint not found: ${WARMSTART_CKPT}" >&2
  echo "[run_pccr_v7]   set WARMSTART_CKPT=<path> to point to a valid v5 .ckpt" >&2
  exit 1
fi

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_v7.yaml}"
export PHASE="real"
export MAX_EPOCHS="${MAX_EPOCHS:-300}"
export TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"

export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-500}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"

# Cost volume settings — identical to v5 (radius=8, channels=8, streamed).
export FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
export FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-8}"
export FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-4}"

# Real phase: no unused parameters in DDP.
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

# LR identical to v5. The 8-epoch warmup (set in v7.yaml) protects warm-started
# weights from the new loss-landscape gradients.
export LR="${LR:-1e-4}"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXPERIMENT_NAME

echo "=========================================="
echo "PCCR v7 — Final / Last-Shot Run"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Config:           ${CONFIG_PATH}"
echo "Job script:       ${JOB_SCRIPT}"
echo "Warm-start ckpt:  ${WARMSTART_CKPT}"
echo "Max epochs:       ${MAX_EPOCHS}"
echo "Cost-vol radius:  ${FINAL_COST_VOLUME_RADIUS}"
echo "v7 deltas:        multi-window LNCC, per-stage Dice (s1/s2),"
echo "                  hyperelastic, symmetric inference, 8-epoch warmup"
echo "Timestamp:        $(date)"
echo "=========================================="

"${LAUNCHER}" \
  --phase real \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${WARMSTART_CKPT}"
