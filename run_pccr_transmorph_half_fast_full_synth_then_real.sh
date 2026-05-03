#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
FULL_LAUNCHER="${FULL_LAUNCHER:-${REPO_ROOT}/run_pccr_transmorph_full_synth_then_real.sh}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-pccr_transmorph_pkl_half_fast_full_${RUN_ID}}"

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_half_fast.yaml}"
export FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
export FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-16}"
export FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-6}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-250}"
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-10}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXPERIMENT_PREFIX

echo "=========================================="
echo "PCCR TransMorph PKL Half-Fast Full Launcher"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "Config: ${CONFIG_PATH}"
echo "Final cost volume mode: ${FINAL_COST_VOLUME_MODE}"
echo "Final cost volume radius: ${FINAL_COST_VOLUME_RADIUS}"
echo "Checkpoint every n train steps: ${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
echo "Timestamp: $(date)"
echo "=========================================="

"${FULL_LAUNCHER}"
