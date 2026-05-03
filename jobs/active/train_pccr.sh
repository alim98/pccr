#!/bin/bash -l
#SBATCH -J pccr_oasis
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_pccr/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_pccr/%j.err
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
  echo "[train_pccr.sh] SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE} does not match NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}." >&2
  exit 1
fi

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
AIM_REPO_DEFAULT="${AIM_REPO_DEFAULT:-${REPO_ROOT}/aim}"
CONFIG_PATH_DEFAULT="${CONFIG_PATH_DEFAULT:-src/pccr/configs/pairwise_oasis.yaml}"

PHASE="${PHASE:-real}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_${PHASE}_${SLURM_JOB_ID}}"
CHECKPOINT_PATH=""
RESUME_FROM_CHECKPOINT=""
CONFIG_PATH="${CONFIG_PATH_DEFAULT}"
TRAIN_DATA_PATH="${DATA_ROOT_DEFAULT}"
VAL_DATA_PATH="${DATA_ROOT_DEFAULT}"
AIM_REPO="${AIM_REPO_DEFAULT}"
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --resume_from_checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --train_data_path)
      TRAIN_DATA_PATH="$2"
      shift 2
      ;;
    --val_data_path)
      VAL_DATA_PATH="$2"
      shift 2
      ;;
    --aim_repo)
      AIM_REPO="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr" \
  "${REPO_ROOT}/slurm/error_pccr" \
  "${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}" \
  "/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/${EXPERIMENT_NAME}" \
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

if command -v aim >/dev/null 2>&1; then
  if [[ ! -d "${AIM_REPO}/.aim" ]]; then
    aim init --repo "${AIM_REPO}" >/dev/null 2>&1 || true
  fi
fi

BASE_CMD=(
  python src/pccr/scripts/train.py
  --mode train
  --phase "${PHASE}"
  --config "${CONFIG_PATH}"
  --accelerator gpu
  --num_gpus "${NUM_GPUS_PER_NODE}"
  --dataset_format oasis_fs
  --train_data_path "${TRAIN_DATA_PATH}"
  --experiment_name "${EXPERIMENT_NAME}"
  --logger_backend aim
  --aim_repo "${AIM_REPO}"
  --batch_size "${BATCH_SIZE_DEFAULT:-1}"
  --num_workers "${NUM_WORKERS_DEFAULT:-8}"
  --precision "${PRECISION_DEFAULT:-bf16-mixed}"
)

if [[ "${PHASE}" == "synthetic" ]]; then
  BASE_CMD+=(
    --lr "${LR_SYNTH_DEFAULT:-1e-4}"
    --max_epochs "${MAX_EPOCHS_SYNTH_DEFAULT:-50}"
  )
else
  BASE_CMD+=(
    --val_data_path "${VAL_DATA_PATH}"
    --lr "${LR_REAL_DEFAULT:-5e-5}"
    --max_epochs "${MAX_EPOCHS_REAL_DEFAULT:-200}"
    --train_num_steps "${TRAIN_NUM_STEPS_REAL_DEFAULT:-200}"
    --max_val_pairs "${MAX_VAL_PAIRS_REAL_DEFAULT:-20}"
  )
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  BASE_CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  BASE_CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

BASE_CMD+=("${OTHER_ARGS[@]}")

echo "[train_pccr.sh] Repository: ${REPO_ROOT}"
echo "[train_pccr.sh] Environment: ${ENV_PATH}"
echo "[train_pccr.sh] Phase: ${PHASE}"
echo "[train_pccr.sh] Experiment: ${EXPERIMENT_NAME}"
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  echo "[train_pccr.sh] Checkpoint: ${CHECKPOINT_PATH}"
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[train_pccr.sh] Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi
printf '[train_pccr.sh] Command:'
printf ' %q' "${BASE_CMD[@]}"
printf '\n'

if command -v srun >/dev/null 2>&1; then
  # Be explicit about the task layout so Lightning's DDP world size matches Slurm.
  SRUN_CMD=(
    srun
    --ntasks="${NUM_GPUS_PER_NODE}"
    --ntasks-per-node="${NUM_GPUS_PER_NODE}"
    --kill-on-bad-exit=1
  )
  "${SRUN_CMD[@]}" "${BASE_CMD[@]}"
else
  "${BASE_CMD[@]}"
fi
