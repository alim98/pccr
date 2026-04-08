#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-0}"
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr_v6/configs/pairwise_oasis_v6a.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_v6a_overfit10_diag}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LR="${LR:-5e-4}"
OVERFIT_PAIRS="${OVERFIT_PAIRS:-10}"
ITER_EVAL_EVERY_N_EPOCHS="${ITER_EVAL_EVERY_N_EPOCHS:-10}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr_v6/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${INIT_CHECKPOINT_PATH}" \
  --config_override multiscale_similarity_factors='[1]' \
  --config_override multiscale_similarity_weights='[1.0]' \
  --config_override segmentation_supervision_weight=0.0 \
  --config_override smoothness_weight=0.0 \
  --config_override jacobian_weight=0.0 \
  --config_override inverse_consistency_weight=0.0 \
  --config_override correspondence_weight=0.2 \
  --config_override decoder_fitting_weight=0.1 \
  --config_override residual_velocity_weight=0.0 \
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
  --overfit_num_pairs "${OVERFIT_PAIRS}" \
  --overfit_split val \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0 \
  --iter_eval_every_n_epochs "${ITER_EVAL_EVERY_N_EPOCHS}" \
  --iter_eval_num_pairs "${OVERFIT_PAIRS}" \
  "$@"
