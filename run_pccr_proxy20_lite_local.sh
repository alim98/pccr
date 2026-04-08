#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

VARIANT="${VARIANT:-v4d}"
GPU_ID="${GPU_ID:-0}"
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_${VARIANT}.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
VAL_PAIRS="${VAL_PAIRS:-200}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
LR="${LR:-2e-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_${VARIANT}_proxy20_lite}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_ARGS=()
if [[ -n "${INIT_CHECKPOINT_PATH}" && -f "${INIT_CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint_path "${INIT_CHECKPOINT_PATH}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config "${CONFIG_PATH}" \
  "${CHECKPOINT_ARGS[@]}" \
  --accelerator gpu \
  --num_gpus 1 \
  --dataset_format oasis_fs \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --logger_backend csv \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs "${MAX_EPOCHS}" \
  --lr "${LR}" \
  --precision bf16-mixed \
  --train_num_steps "${TRAIN_STEPS}" \
  --max_val_pairs "${VAL_PAIRS}" \
  --iter_eval_every_n_epochs 5 \
  --iter_eval_num_pairs 50 \
  "$@"
