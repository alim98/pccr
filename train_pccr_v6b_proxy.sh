#!/bin/bash -l
#SBATCH -J pccr_v6b_proxy
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_pccr_v6/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_pccr_v6/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=NONE
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${TRAIN_DATA_PATH}}"
CONFIG_PATH="${CONFIG_PATH:-src/pccr_v6/configs/pairwise_oasis_v6b.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/pccr/pccr_oasis_real_20260329-100814/epoch1365-val1.5977.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_v6b_real_proxy_${SLURM_JOB_ID}}"
LOGGER_BACKEND="${LOGGER_BACKEND:-csv}"
AIM_REPO="${AIM_REPO:-${REPO_ROOT}/aim}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-0.1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.2}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-20}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-200}"

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr_v6" \
  "${REPO_ROOT}/slurm/error_pccr_v6" \
  "${REPO_ROOT}/logs/pccr_v6/${EXPERIMENT_NAME}" \
  "${REPO_ROOT}/checkpoints/pccr_v6/${EXPERIMENT_NAME}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_PATH}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CMD=(
  "${PYTHON_BIN}" src/pccr_v6/scripts/train.py
  --mode train
  --phase real
  --data_source real
  --config "${CONFIG_PATH}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --accelerator gpu
  --num_gpus 1
  --dataset_format oasis_fs
  --train_data_path "${TRAIN_DATA_PATH}"
  --val_data_path "${VAL_DATA_PATH}"
  --experiment_name "${EXPERIMENT_NAME}"
  --logger_backend "${LOGGER_BACKEND}"
  --aim_repo "${AIM_REPO}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_epochs "${MAX_EPOCHS}"
  --lr "${LR}"
  --precision "${PRECISION}"
  --train_num_steps "${TRAIN_NUM_STEPS}"
  --max_val_pairs "${MAX_VAL_PAIRS}"
  --limit_train_batches "${LIMIT_TRAIN_BATCHES}"
  --limit_val_batches "${LIMIT_VAL_BATCHES}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[train_pccr_v6b_proxy.sh] Repository: ${REPO_ROOT}"
echo "[train_pccr_v6b_proxy.sh] Experiment: ${EXPERIMENT_NAME}"
echo "[train_pccr_v6b_proxy.sh] Checkpoint: ${CHECKPOINT_PATH}"
printf '[train_pccr_v6b_proxy.sh] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if command -v srun >/dev/null 2>&1; then
  srun --ntasks=1 --kill-on-bad-exit=1 "${CMD[@]}"
else
  "${CMD[@]}"
fi
