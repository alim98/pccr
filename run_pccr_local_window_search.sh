#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-0}"
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_v4b.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_local_window_search}"
SEARCH_RADIUS="${SEARCH_RADIUS:-1}"
PATCH_WINDOW="${PATCH_WINDOW:-5}"
VAL_PAIRS="${VAL_PAIRS:-50}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/local_window_search_eval.py \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --accelerator gpu \
  --num_gpus 1 \
  --precision bf16-mixed \
  --train_num_steps 200 \
  --max_val_pairs "${VAL_PAIRS}" \
  --num_workers 4 \
  --search_radius "${SEARCH_RADIUS}" \
  --patch_window "${PATCH_WINDOW}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --output_dir "logs/pccr/${EXPERIMENT_NAME}" \
  "$@"
