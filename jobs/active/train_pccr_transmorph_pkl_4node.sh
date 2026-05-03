#!/bin/bash -l
#SBATCH -J pccr_tm_pkl
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_pccr_tm_pkl/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_pccr_tm_pkl/%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --qos=g0016
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=NONE
#SBATCH --mail-user=ali.Mikaeili@brain.mpg.de
#SBATCH --time=23:00:00

# 4-node, 16xA100 variant.
# Expected speedup: ~3-3.5x over single node → 300 epochs in ~15-17 days.
#
# Important: gradient_accumulation_steps is forced to 1 here.
#   - With 16 GPUs and accum=4, each optimizer step = 64 samples (wasteful and slow per epoch).
#   - With 16 GPUs and accum=1, effective batch = 16 (same as 4-GPU/accum=4), epoch is 4x faster.
#   - The config override below enforces this regardless of what the YAML says.
#
# Usage:
#   JOB_SCRIPT="${REPO_ROOT}/jobs/active/train_pccr_transmorph_pkl_4node.sh" \
#   bash run_pccr_full_final.sh
#
# If the cluster QOS does not allow 4-node jobs, try 2-node:
#   JOB_SCRIPT="${REPO_ROOT}/jobs/active/train_pccr_transmorph_pkl_2node.sh" \
#   bash run_pccr_full_final.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

NUM_GPUS_PER_NODE=4
TOTAL_GPUS=$(( SLURM_NNODES * NUM_GPUS_PER_NODE ))

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl.yaml}"
PHASE="${PHASE:-real}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/All}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/Test}"
LOGGER_BACKEND="${LOGGER_BACKEND:-aim,csv}"
AIM_REPO="${AIM_REPO:-${REPO_ROOT}/aim}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-auto}"
PRECISION="${PRECISION:-bf16-mixed}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"
LR="${LR:-1e-4}"
CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-500}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"
FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-8}"
FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-8}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_transmorph_pkl_4node_${RUN_TAG}}"
PCCR_CHECKPOINT_DIR="${PCCR_CHECKPOINT_DIR:-/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/${EXPERIMENT_NAME}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
AIM_RUN_NAME="${AIM_RUN_NAME:-${EXPERIMENT_NAME}}"
AIM_RUN_HASH_FILE="${AIM_RUN_HASH_FILE:-${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}/aim_run_hash.txt}"
AIM_RUN_HASH="${AIM_RUN_HASH:-}"
if [[ -z "${AIM_RUN_HASH}" && -s "${AIM_RUN_HASH_FILE}" ]]; then
  AIM_RUN_HASH="$(head -n 1 "${AIM_RUN_HASH_FILE}" | tr -d '[:space:]')"
fi

EXTRA_ARGS=("$@")

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr_tm_pkl" \
  "${REPO_ROOT}/slurm/error_pccr_tm_pkl" \
  "${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}" \
  "${PCCR_CHECKPOINT_DIR}" \
  "${AIM_REPO}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_PATH}"
else
  export PATH="${ENV_PATH}/bin:${PATH}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

# With 16 GPUs total, force accumulation=1 so effective batch stays at 16.
# This makes each epoch ~4x faster than accumulation=4 would be.
COMMON_OVERRIDES=(
  "lr_scheduler=cosine"
  "lr_warmup_epochs=5"
  "gradient_accumulation_steps=1"
)

case "${FINAL_COST_VOLUME_MODE}" in
  streamed)
    COMMON_OVERRIDES+=(
      "final_refinement_use_local_cost_volume=true"
      "final_refinement_cost_volume_radius=${FINAL_COST_VOLUME_RADIUS}"
      "final_refinement_memory_efficient_cost_volume=true"
      "final_refinement_cost_volume_offset_chunk_size=${FINAL_COST_VOLUME_CHUNK_SIZE}"
    )
    ;;
  legacy)
    COMMON_OVERRIDES+=(
      "final_refinement_use_local_cost_volume=true"
      "final_refinement_cost_volume_radius=${FINAL_COST_VOLUME_RADIUS}"
      "final_refinement_memory_efficient_cost_volume=false"
      "final_refinement_cost_volume_offset_chunk_size=${FINAL_COST_VOLUME_CHUNK_SIZE}"
    )
    ;;
  off)
    COMMON_OVERRIDES+=(
      "final_refinement_use_local_cost_volume=false"
      "final_refinement_cost_volume_radius=0"
    )
    ;;
  *)
    echo "[train_pccr_transmorph_pkl_4node.sh] Unknown FINAL_COST_VOLUME_MODE=${FINAL_COST_VOLUME_MODE}" >&2
    exit 1
    ;;
esac

CMD=(
  "${PYTHON_BIN}" src/pccr/scripts/train.py
  --config "${CONFIG_PATH}"
  --mode train
  --phase "${PHASE}"
  --dataset_format pkl
  --train_data_path "${TRAIN_DATA_PATH}"
  --val_data_path "${VAL_DATA_PATH}"
  --accelerator gpu
  --num_gpus "${NUM_GPUS_PER_NODE}"
  --num_nodes "${SLURM_NNODES}"
  --experiment_name "${EXPERIMENT_NAME}"
  --checkpoint_dir "${PCCR_CHECKPOINT_DIR}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}"
  --logger_backend "${LOGGER_BACKEND}"
  --aim_repo "${AIM_REPO}"
  --run_name "${AIM_RUN_NAME}"
  --aim_run_hash_file "${AIM_RUN_HASH_FILE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --train_num_steps "${TRAIN_NUM_STEPS}"
  --max_epochs "${MAX_EPOCHS}"
  --lr "${LR}"
  --precision "${PRECISION}"
  --checkpoint_every_n_epochs "${CHECKPOINT_EVERY_N_EPOCHS}"
  --checkpoint_every_n_train_steps "${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
  --check_val_every_n_epoch "${CHECK_VAL_EVERY_N_EPOCH}"
  --log_every_n_steps "${LOG_EVERY_N_STEPS}"
  --max_val_pairs "${MAX_VAL_PAIRS}"
)

if [[ -n "${AIM_RUN_HASH}" ]]; then
  CMD+=(--aim_run_hash "${AIM_RUN_HASH}")
fi

for override in "${COMMON_OVERRIDES[@]}"; do
  CMD+=(--config_override "${override}")
done

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[train_pccr_transmorph_pkl_4node.sh] Nodes: ${SLURM_NNODES}, total GPUs: ${TOTAL_GPUS}"
echo "[train_pccr_transmorph_pkl_4node.sh] Config: ${CONFIG_PATH}"
echo "[train_pccr_transmorph_pkl_4node.sh] Checkpoints: ${PCCR_CHECKPOINT_DIR}"
echo "[train_pccr_transmorph_pkl_4node.sh] Aim run: ${AIM_RUN_NAME} (${AIM_RUN_HASH:-new})"
echo "[train_pccr_transmorph_pkl_4node.sh] Phase: ${PHASE}"
printf '[train_pccr_transmorph_pkl_4node.sh] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

srun \
  --ntasks="${TOTAL_GPUS}" \
  --ntasks-per-node="${NUM_GPUS_PER_NODE}" \
  --kill-on-bad-exit=1 \
  "${CMD[@]}"
