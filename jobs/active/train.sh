#!/bin/bash -l
#SBATCH -J hvit_oasis
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_hvit/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_hvit/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=128000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=NONE
#SBATCH --time=24:00:00

set -euo pipefail

# Under `sbatch`, Slurm executes a staged copy of the script from `/var/spool/...`.
# Prefer the original submit directory so logs/checkpoints stay in the repository.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

# Keep Slurm's task layout aligned with Lightning's DDP device count.
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "${SLURM_NTASKS_PER_NODE}" != "${NUM_GPUS_PER_NODE}" ]]; then
  echo "[train.sh] SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE} does not match NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}." >&2
  echo "[train.sh] Keep Slurm ntasks-per-node and Lightning devices aligned for DDP." >&2
  exit 1
fi
ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
AIM_REPO_DEFAULT="${AIM_REPO_DEFAULT:-${REPO_ROOT}/aim}"

mkdir -p \
  "${REPO_ROOT}/slurm/output_hvit" \
  "${REPO_ROOT}/slurm/error_hvit" \
  "${REPO_ROOT}/logs" \
  "/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints" \
  "${AIM_REPO_DEFAULT}"

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

if command -v aim >/dev/null 2>&1; then
  mkdir -p "${AIM_REPO_DEFAULT}"
  if [[ ! -d "${AIM_REPO_DEFAULT}/.aim" ]]; then
    aim init --repo "${AIM_REPO_DEFAULT}" >/dev/null 2>&1 || true
  fi
fi

RESUME_FROM_CHECKPOINT=""
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hvit_slurm_${SLURM_JOB_ID}}"
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume_from_checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

export HVIT_LOG_ROOT="${REPO_ROOT}/logs/${EXPERIMENT_NAME}"
export HVIT_CHECKPOINT_ROOT="/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/${EXPERIMENT_NAME}"
mkdir -p "${HVIT_LOG_ROOT}" "${HVIT_CHECKPOINT_ROOT}"

TRAIN_CMD=(
  python src/scripts/main.py
  --mode train
  --accelerator gpu
  --num_gpus "${NUM_GPUS_PER_NODE}"
  --dataset_format oasis_fs
  --train_data_path "${DATA_ROOT_DEFAULT}"
  --val_data_path "${DATA_ROOT_DEFAULT}"
  --experiment_name "${EXPERIMENT_NAME}"
  --logger_backend aim
  --aim_repo "${AIM_REPO_DEFAULT}"
  --batch_size 1
  --num_workers 8
  --max_epochs 1000
  --train_num_steps 200
  --precision bf16
  --save_model_every_n_epochs 1
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

TRAIN_CMD+=("${OTHER_ARGS[@]}")

echo "[train.sh] Repository: ${REPO_ROOT}"
echo "[train.sh] Environment: ${ENV_PATH}"
echo "[train.sh] Experiment: ${EXPERIMENT_NAME}"
echo "[train.sh] HVIT_LOG_ROOT: ${HVIT_LOG_ROOT}"
echo "[train.sh] HVIT_CHECKPOINT_ROOT: ${HVIT_CHECKPOINT_ROOT}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[train.sh] Resuming from: ${RESUME_FROM_CHECKPOINT}"
fi
printf '[train.sh] Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if command -v srun >/dev/null 2>&1; then
  # Be explicit about the task layout so Lightning's DDP world size matches Slurm.
  SRUN_CMD=(
    srun
    --ntasks="${NUM_GPUS_PER_NODE}"
    --ntasks-per-node="${NUM_GPUS_PER_NODE}"
    --kill-on-bad-exit=1
  )
  "${SRUN_CMD[@]}" "${TRAIN_CMD[@]}"
else
  "${TRAIN_CMD[@]}"
fi
