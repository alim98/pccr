#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

VARIANT="${VARIANT:-custom}"
GPU_ID="${GPU_ID:-0}"
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_v4a_overfit10.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_${VARIANT}_overfit10}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase real \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${INIT_CHECKPOINT_PATH}" \
  --accelerator gpu \
  --num_gpus 1 \
  --dataset_format oasis_fs \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --logger_backend csv \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs 300 \
  --lr 5e-4 \
  --precision bf16-mixed \
  --overfit_num_pairs 10 \
  --overfit_split val \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0 \
  --iter_eval_every_n_epochs 25 \
  --iter_eval_num_pairs 10 \
  "$@"
