#!/bin/bash -l
#SBATCH -J hvit_tm_pkl
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_hvit_tm_pkl/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_hvit_tm_pkl/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=128000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=NONE
#SBATCH --time=24:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "${SLURM_NTASKS_PER_NODE}" != "${NUM_GPUS_PER_NODE}" ]]; then
  echo "[train_hvit_transmorph_pkl.sh] SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE} does not match NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}." >&2
  exit 1
fi

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/hvit_transmorph_pkl.yaml}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/All}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasistransmorph/OASIS_L2R_2021_task03/Test}"
LOGGER_BACKEND="${LOGGER_BACKEND:-aim,csv}"
AIM_REPO="${AIM_REPO:-${REPO_ROOT}/aim}"
PRECISION="${PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"
LR="${LR:-1e-4}"
SAVE_MODEL_EVERY_N_EPOCHS="${SAVE_MODEL_EVERY_N_EPOCHS:-25}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hvit_transmorph_pkl_${RUN_TAG}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

EXTRA_ARGS=("$@")

mkdir -p \
  "${REPO_ROOT}/slurm/output_hvit_tm_pkl" \
  "${REPO_ROOT}/slurm/error_hvit_tm_pkl" \
  "${REPO_ROOT}/logs/${EXPERIMENT_NAME}" \
  "/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/${EXPERIMENT_NAME}" \
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
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

if [[ ",${LOGGER_BACKEND}," == *",aim,"* ]] && command -v aim >/dev/null 2>&1; then
  if [[ ! -d "${AIM_REPO}/.aim" ]]; then
    aim init --repo "${AIM_REPO}" >/dev/null 2>&1 || true
  fi
fi

CMD=(
  "${PYTHON_BIN}" src/scripts/main.py
  --config "${CONFIG_PATH}"
  --mode train
  --dataset_format pkl
  --train_data_path "${TRAIN_DATA_PATH}"
  --val_data_path "${VAL_DATA_PATH}"
  --accelerator gpu
  --num_gpus "${NUM_GPUS_PER_NODE}"
  --experiment_name "${EXPERIMENT_NAME}"
  --logger_backend "${LOGGER_BACKEND}"
  --aim_repo "${AIM_REPO}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --train_num_steps "${TRAIN_NUM_STEPS}"
  --max_epochs "${MAX_EPOCHS}"
  --lr "${LR}"
  --precision "${PRECISION}"
  --save_model_every_n_epochs "${SAVE_MODEL_EVERY_N_EPOCHS}"
  --max_val_pairs 19
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[train_hvit_transmorph_pkl.sh] Repository: ${REPO_ROOT}"
echo "[train_hvit_transmorph_pkl.sh] Environment: ${ENV_PATH}"
echo "[train_hvit_transmorph_pkl.sh] Config: ${CONFIG_PATH}"
echo "[train_hvit_transmorph_pkl.sh] Train data: ${TRAIN_DATA_PATH}"
echo "[train_hvit_transmorph_pkl.sh] Val data: ${VAL_DATA_PATH}"
echo "[train_hvit_transmorph_pkl.sh] Experiment: ${EXPERIMENT_NAME}"
printf '[train_hvit_transmorph_pkl.sh] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if command -v srun >/dev/null 2>&1; then
  srun \
    --ntasks="${NUM_GPUS_PER_NODE}" \
    --ntasks-per-node="${NUM_GPUS_PER_NODE}" \
    --kill-on-bad-exit=1 \
    "${CMD[@]}"
else
  "${CMD[@]}"
fi
