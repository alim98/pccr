#!/bin/bash -l
#SBATCH -J pccr_eval
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_pccr_eval/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_pccr_eval/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=128000
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=NONE
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "${SLURM_NTASKS_PER_NODE}" != "${NUM_GPUS_PER_NODE}" ]]; then
  echo "[eval_pccr_4gpu.sh] SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE} does not match NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}." >&2
  exit 1
fi

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
CONFIG_PATH_DEFAULT="${CONFIG_PATH_DEFAULT:-src/pccr/configs/pairwise_oasis.yaml}"

CHECKPOINT_PATH=""
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_eval_${SLURM_JOB_ID}}"
OUTPUT_DIR=""
CONFIG_PATH="${CONFIG_PATH_DEFAULT}"
TRAIN_DATA_PATH="${DATA_ROOT_DEFAULT}"
VAL_DATA_PATH="${DATA_ROOT_DEFAULT}"
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
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
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "[eval_pccr_4gpu.sh] --checkpoint_path is required." >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}"
fi

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr_eval" \
  "${REPO_ROOT}/slurm/error_pccr_eval" \
  "${OUTPUT_DIR}"

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

COMMON_ARGS=(
  src/pccr/scripts/evaluate.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --config "${CONFIG_PATH}"
  --train_data_path "${TRAIN_DATA_PATH}"
  --val_data_path "${VAL_DATA_PATH}"
  --dataset_format oasis_fs
  --accelerator gpu
  --num_gpus 1
  --precision "${PRECISION_DEFAULT:-bf16-mixed}"
  --val_fraction 0.2
  --split_seed 42
  --train_num_steps "${TRAIN_NUM_STEPS_REAL_DEFAULT:-600}"
  --max_val_pairs 0
  --num_workers "${NUM_WORKERS_DEFAULT:-4}"
  --progress_every 25
)
COMMON_ARGS+=("${OTHER_ARGS[@]}")

echo "[eval_pccr_4gpu.sh] Repository: ${REPO_ROOT}"
echo "[eval_pccr_4gpu.sh] Environment: ${ENV_PATH}"
echo "[eval_pccr_4gpu.sh] Experiment: ${EXPERIMENT_NAME}"
echo "[eval_pccr_4gpu.sh] Checkpoint: ${CHECKPOINT_PATH}"
echo "[eval_pccr_4gpu.sh] Output dir: ${OUTPUT_DIR}"
printf '[eval_pccr_4gpu.sh] Common evaluate args:'
printf ' %q' "${COMMON_ARGS[@]}"
printf '\n'

srun --ntasks="${NUM_GPUS_PER_NODE}" --ntasks-per-node="${NUM_GPUS_PER_NODE}" --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    shard_index="${SLURM_PROCID}"
    num_shards="${SLURM_NTASKS}"
    local_gpu="${SLURM_LOCALID}"
    shard_dir="'"${OUTPUT_DIR}"'/shard_${shard_index}"
    mkdir -p "${shard_dir}"
    export CUDA_VISIBLE_DEVICES="${local_gpu}"
    echo "[eval_pccr_4gpu.sh] Task ${shard_index}/${num_shards} pinned to CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    python '"${COMMON_ARGS[*]}"' \
      --shard_index "${shard_index}" \
      --num_shards "${num_shards}" \
      --experiment_name "'"${EXPERIMENT_NAME}"'_shard_${shard_index}" \
      --output_dir "${shard_dir}" \
      > "${shard_dir}/stdout.log" 2> "${shard_dir}/stderr.log"
  '

python src/pccr/scripts/merge_eval_reports.py \
  "${OUTPUT_DIR}"/shard_* \
  --output_dir "${OUTPUT_DIR}"

echo "[eval_pccr_4gpu.sh] Done. Merged summary: ${OUTPUT_DIR}/summary.json"
