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
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
VAL_PAIRS="${VAL_PAIRS:-200}"
EVAL_VAL_PAIRS="${EVAL_VAL_PAIRS:-200}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-2e-5}"
PROXY_DATA_SIZE="${PROXY_DATA_SIZE:-[64,64,64]}"
JAC_WEIGHTS="${JAC_WEIGHTS:-0.0 0.001 0.005 0.01 0.02}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_ARGS=()
if [[ -n "${INIT_CHECKPOINT_PATH}" && -f "${INIT_CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint_path "${INIT_CHECKPOINT_PATH}")
fi

IFS=' ' read -r -a JAC_ARRAY <<< "${JAC_WEIGHTS}"
SUMMARY_PATHS=()
LABELS=()

for JAC in "${JAC_ARRAY[@]}"; do
  SAFE_JAC="${JAC//./p}"
  EXPERIMENT_NAME="pccr_${VARIANT}_jac_${SAFE_JAC}_${RUN_TAG}"
  EVAL_OUTPUT="logs/pccr/${EXPERIMENT_NAME}_eval"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/train.py \
    --mode train \
    --phase real \
    --data_source real \
    --config "${CONFIG_PATH}" \
    --config_override "data_size=${PROXY_DATA_SIZE}" \
    --config_override "jacobian_weight=${JAC}" \
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
    --iter_eval_num_pairs 50

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/evaluate.py \
    --config "${CONFIG_PATH}" \
    --config_override "data_size=${PROXY_DATA_SIZE}" \
    --checkpoint_path "${REPO_ROOT}/checkpoints/pccr/${EXPERIMENT_NAME}/last.ckpt" \
    --train_data_path "${DATA_ROOT_DEFAULT}" \
    --val_data_path "${DATA_ROOT_DEFAULT}" \
    --dataset_format oasis_fs \
    --accelerator gpu \
    --num_gpus 1 \
    --precision bf16-mixed \
    --train_num_steps "${TRAIN_STEPS}" \
    --max_val_pairs "${EVAL_VAL_PAIRS}" \
    --num_workers 4 \
    --experiment_name "${EXPERIMENT_NAME}_eval" \
    --output_dir "${EVAL_OUTPUT}"

  SUMMARY_PATHS+=("${REPO_ROOT}/${EVAL_OUTPUT}/summary.json")
  LABELS+=("jac=${JAC}")
done

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" src/pccr/scripts/plot_jacobian_tradeoff.py \
  "${SUMMARY_PATHS[@]}" \
  --labels "${LABELS[@]}" \
  --output_csv "${REPO_ROOT}/logs/pccr/pccr_${VARIANT}_jacobian_tradeoff_${RUN_TAG}.csv" \
  --output_png "${REPO_ROOT}/logs/pccr/pccr_${VARIANT}_jacobian_tradeoff_${RUN_TAG}.png"
