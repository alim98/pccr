#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

VARIANT="${VARIANT:-v4b}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"

CONFIG_PATH="${CONFIG_PATH:-src/pccr/configs/pairwise_oasis_${VARIANT}.yaml}"
SYNTH_EPOCHS="${SYNTH_EPOCHS:-50}"
REAL_EPOCHS="${REAL_EPOCHS:-50}"
SYNTH_STEPS="${SYNTH_STEPS:-400}"
REAL_STEPS="${REAL_STEPS:-400}"
VAL_PAIRS="${VAL_PAIRS:-100}"
LR_SYNTH="${LR_SYNTH:-1e-4}"
LR_REAL="${LR_REAL:-2e-5}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SYNTH_ONLY_EXP="pccr_${VARIANT}_synthetic_only_${RUN_TAG}"
REAL_ONLY_EXP="pccr_${VARIANT}_real_only_${RUN_TAG}"
MIXED_SYNTH_EXP="pccr_${VARIANT}_mixed_synth_${RUN_TAG}"
MIXED_REAL_EXP="pccr_${VARIANT}_mixed_real_${RUN_TAG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase synthetic \
  --config "${CONFIG_PATH}" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --experiment_name "${SYNTH_ONLY_EXP}" \
  --logger_backend csv \
  --accelerator gpu \
  --num_gpus 1 \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs "${SYNTH_EPOCHS}" \
  --lr "${LR_SYNTH}" \
  --precision bf16-mixed \
  --train_num_steps "${SYNTH_STEPS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${REPO_ROOT}/checkpoints/pccr/${SYNTH_ONLY_EXP}/last.ckpt" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --accelerator gpu \
  --num_gpus 1 \
  --precision bf16-mixed \
  --train_num_steps "${REAL_STEPS}" \
  --max_val_pairs "${VAL_PAIRS}" \
  --num_workers 4 \
  --experiment_name "${SYNTH_ONLY_EXP}_eval" \
  --output_dir "logs/pccr/${SYNTH_ONLY_EXP}_eval"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config "${CONFIG_PATH}" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --experiment_name "${REAL_ONLY_EXP}" \
  --logger_backend csv \
  --accelerator gpu \
  --num_gpus 1 \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs "${REAL_EPOCHS}" \
  --lr "${LR_REAL}" \
  --precision bf16-mixed \
  --train_num_steps "${REAL_STEPS}" \
  --max_val_pairs "${VAL_PAIRS}" \
  --iter_eval_every_n_epochs 10 \
  --iter_eval_num_pairs 50

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${REPO_ROOT}/checkpoints/pccr/${REAL_ONLY_EXP}/last.ckpt" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --accelerator gpu \
  --num_gpus 1 \
  --precision bf16-mixed \
  --train_num_steps "${REAL_STEPS}" \
  --max_val_pairs "${VAL_PAIRS}" \
  --num_workers 4 \
  --experiment_name "${REAL_ONLY_EXP}_eval" \
  --output_dir "logs/pccr/${REAL_ONLY_EXP}_eval"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase synthetic \
  --config "${CONFIG_PATH}" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --experiment_name "${MIXED_SYNTH_EXP}" \
  --logger_backend csv \
  --accelerator gpu \
  --num_gpus 1 \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs "${SYNTH_EPOCHS}" \
  --lr "${LR_SYNTH}" \
  --precision bf16-mixed \
  --train_num_steps "${SYNTH_STEPS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
  --mode train \
  --phase real \
  --data_source real \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${REPO_ROOT}/checkpoints/pccr/${MIXED_SYNTH_EXP}/last.ckpt" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --experiment_name "${MIXED_REAL_EXP}" \
  --logger_backend csv \
  --accelerator gpu \
  --num_gpus 1 \
  --batch_size 1 \
  --num_workers 4 \
  --max_epochs "${REAL_EPOCHS}" \
  --lr "${LR_REAL}" \
  --precision bf16-mixed \
  --train_num_steps "${REAL_STEPS}" \
  --max_val_pairs "${VAL_PAIRS}" \
  --iter_eval_every_n_epochs 10 \
  --iter_eval_num_pairs 50

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --checkpoint_path "${REPO_ROOT}/checkpoints/pccr/${MIXED_REAL_EXP}/last.ckpt" \
  --train_data_path "${DATA_ROOT_DEFAULT}" \
  --val_data_path "${DATA_ROOT_DEFAULT}" \
  --dataset_format oasis_fs \
  --accelerator gpu \
  --num_gpus 1 \
  --precision bf16-mixed \
  --train_num_steps "${REAL_STEPS}" \
  --max_val_pairs "${VAL_PAIRS}" \
  --num_workers 4 \
  --experiment_name "${MIXED_REAL_EXP}_eval" \
  --output_dir "logs/pccr/${MIXED_REAL_EXP}_eval"
