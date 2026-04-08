#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-0}"
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_v4e.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_v4e_oracle_handoff_10pairs}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
LR="${LR:-2e-4}"
OVERFIT_PAIRS="${OVERFIT_PAIRS:-10}"
ITER_EVAL_EVERY_N_EPOCHS="${ITER_EVAL_EVERY_N_EPOCHS:-10}"
ORACLE_SIGMA="${ORACLE_SIGMA:-1.0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${INIT_CHECKPOINT_PATH}" \
  --freeze_mode decoder_and_refinement \
  --oracle_handoff \
  --config_override "oracle_handoff_gaussian_sigma=${ORACLE_SIGMA}" \
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
