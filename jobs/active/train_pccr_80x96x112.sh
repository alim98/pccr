#!/bin/bash -l
#SBATCH -J pccr_80x96x112
#SBATCH -D /u/almik/others/hvit
#SBATCH -o /u/almik/others/hvit/slurm/output_pccr_80x96x112/%j.out
#SBATCH -e /u/almik/others/hvit/slurm/error_pccr_80x96x112/%j.err
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
  echo "[train_pccr_80x96x112.sh] SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE} does not match NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}." >&2
  exit 1
fi

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_fullres.yaml}"
DATA_ROOT="${DATA_ROOT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis_l2r}"
LOGGER_BACKEND="${LOGGER_BACKEND:-csv}"
AIM_REPO="${AIM_REPO:-${REPO_ROOT}/aim}"
PRECISION="${PRECISION:-bf16-mixed}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-200}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-50}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-200}"
PHASE1_LR="${PHASE1_LR:-1e-4}"
PHASE2_LR="${PHASE2_LR:-2e-5}"
CHECKPOINT_EVERY_SYNTH="${CHECKPOINT_EVERY_SYNTH:-10}"
CHECKPOINT_EVERY_REAL="${CHECKPOINT_EVERY_REAL:-25}"
CHECK_VAL_EVERY_REAL="${CHECK_VAL_EVERY_REAL:-10}"
MAX_VAL_PAIRS_REAL="${MAX_VAL_PAIRS_REAL:-5}"
PHASE_MODE="${PHASE_MODE:-both}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-pccr_l2r_80x96x112_${RUN_TAG}}"
PHASE1_EXPERIMENT="${PHASE1_EXPERIMENT:-${EXPERIMENT_ROOT}_phase1}"
PHASE2_EXPERIMENT="${PHASE2_EXPERIMENT:-${EXPERIMENT_ROOT}_phase2}"
PHASE1_CHECKPOINT_OVERRIDE="${PHASE1_CHECKPOINT_OVERRIDE:-}"
RESUME_PHASE1="${RESUME_PHASE1:-}"
RESUME_PHASE2="${RESUME_PHASE2:-}"

FINAL_COST_VOLUME="${FINAL_COST_VOLUME:-true}"
FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-4}"
STAGE2_LOCAL_REFINEMENT="${STAGE2_LOCAL_REFINEMENT:-true}"
STAGE2_LOCAL_REFINEMENT_RADIUS="${STAGE2_LOCAL_REFINEMENT_RADIUS:-3}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase-mode)
      PHASE_MODE="$2"
      shift 2
      ;;
    --experiment-root)
      EXPERIMENT_ROOT="$2"
      PHASE1_EXPERIMENT="${EXPERIMENT_ROOT}_phase1"
      PHASE2_EXPERIMENT="${EXPERIMENT_ROOT}_phase2"
      shift 2
      ;;
    --phase1-checkpoint)
      PHASE1_CHECKPOINT_OVERRIDE="$2"
      shift 2
      ;;
    --resume-phase1)
      RESUME_PHASE1="$2"
      shift 2
      ;;
    --resume-phase2)
      RESUME_PHASE2="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${PHASE_MODE}" in
  both|synthetic|real)
    ;;
  *)
    echo "[train_pccr_80x96x112.sh] Unsupported PHASE_MODE=${PHASE_MODE}. Use one of: both, synthetic, real." >&2
    exit 1
    ;;
esac

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr_80x96x112" \
  "${REPO_ROOT}/slurm/error_pccr_80x96x112" \
  "${REPO_ROOT}/logs/pccr/${PHASE1_EXPERIMENT}" \
  "${REPO_ROOT}/logs/pccr/${PHASE2_EXPERIMENT}" \
  "${REPO_ROOT}/checkpoints/pccr/${PHASE1_EXPERIMENT}" \
  "${REPO_ROOT}/checkpoints/pccr/${PHASE2_EXPERIMENT}" \
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${LOGGER_BACKEND}" == "aim" ]] && command -v aim >/dev/null 2>&1; then
  if [[ ! -d "${AIM_REPO}/.aim" ]]; then
    aim init --repo "${AIM_REPO}" >/dev/null 2>&1 || true
  fi
fi

COMMON_OVERRIDES=(
  "data_size=[80, 96, 112]"
  "align_data_size_to_native_shape=false"
)

if [[ "${STAGE2_LOCAL_REFINEMENT}" == "true" ]]; then
  COMMON_OVERRIDES+=(
    "use_stage2_local_refinement=true"
    "stage2_local_refinement_radius=${STAGE2_LOCAL_REFINEMENT_RADIUS}"
  )
else
  COMMON_OVERRIDES+=(
    "use_stage2_local_refinement=false"
    "stage2_local_refinement_radius=0"
  )
fi

if [[ "${FINAL_COST_VOLUME}" == "true" ]]; then
  COMMON_OVERRIDES+=(
    "final_refinement_use_local_cost_volume=true"
    "final_refinement_cost_volume_radius=${FINAL_COST_VOLUME_RADIUS}"
  )
else
  COMMON_OVERRIDES+=(
    "final_refinement_use_local_cost_volume=false"
    "final_refinement_cost_volume_radius=0"
  )
fi

append_common_args() {
  local -n cmd_ref=$1
  cmd_ref+=(
    --config "${CONFIG_PATH}"
    --mode train
    --dataset_format oasis_l2r
    --train_data_path "${DATA_ROOT}"
    --val_data_path "${DATA_ROOT}"
    --accelerator gpu
    --num_gpus "${NUM_GPUS_PER_NODE}"
    --logger_backend "${LOGGER_BACKEND}"
    --aim_repo "${AIM_REPO}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --train_num_steps "${TRAIN_NUM_STEPS}"
    --precision "${PRECISION}"
  )
  local override
  for override in "${COMMON_OVERRIDES[@]}"; do
    cmd_ref+=(--config_override "${override}")
  done
}

run_with_srun() {
  local -a cmd=("$@")
  printf '[train_pccr_80x96x112.sh] Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if command -v srun >/dev/null 2>&1; then
    srun \
      --ntasks="${NUM_GPUS_PER_NODE}" \
      --ntasks-per-node="${NUM_GPUS_PER_NODE}" \
      --kill-on-bad-exit=1 \
      "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_phase1() {
  local -a cmd=(
    "${PYTHON_BIN}" src/pccr/scripts/train.py
    --phase synthetic
    --experiment_name "${PHASE1_EXPERIMENT}"
    --max_epochs "${PHASE1_EPOCHS}"
    --lr "${PHASE1_LR}"
    --checkpoint_every_n_epochs "${CHECKPOINT_EVERY_SYNTH}"
  )
  append_common_args cmd
  cmd+=(
    --config_override "lr_scheduler=cosine"
    --config_override "lr_warmup_epochs=0"
  )
  if [[ -n "${RESUME_PHASE1}" ]]; then
    cmd+=(--resume_from_checkpoint "${RESUME_PHASE1}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  run_with_srun "${cmd[@]}"
}

run_phase2() {
  local phase1_checkpoint="$1"
  local -a cmd=(
    "${PYTHON_BIN}" src/pccr/scripts/train.py
    --phase real
    --experiment_name "${PHASE2_EXPERIMENT}"
    --max_epochs "${PHASE2_EPOCHS}"
    --lr "${PHASE2_LR}"
    --checkpoint_every_n_epochs "${CHECKPOINT_EVERY_REAL}"
    --check_val_every_n_epoch "${CHECK_VAL_EVERY_REAL}"
    --max_val_pairs "${MAX_VAL_PAIRS_REAL}"
  )
  append_common_args cmd
  cmd+=(
    --config_override "lr_scheduler=cosine"
    --config_override "lr_warmup_epochs=5"
  )
  if [[ -n "${phase1_checkpoint}" ]]; then
    cmd+=(--checkpoint_path "${phase1_checkpoint}")
  fi
  if [[ -n "${RESUME_PHASE2}" ]]; then
    cmd+=(--resume_from_checkpoint "${RESUME_PHASE2}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  run_with_srun "${cmd[@]}"
}

PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT_OVERRIDE:-${REPO_ROOT}/checkpoints/pccr/${PHASE1_EXPERIMENT}/last.ckpt}"

echo "[train_pccr_80x96x112.sh] Repository: ${REPO_ROOT}"
echo "[train_pccr_80x96x112.sh] Environment: ${ENV_PATH}"
echo "[train_pccr_80x96x112.sh] Config: ${CONFIG_PATH}"
echo "[train_pccr_80x96x112.sh] Data root: ${DATA_ROOT}"
echo "[train_pccr_80x96x112.sh] Phase mode: ${PHASE_MODE}"
echo "[train_pccr_80x96x112.sh] Phase-1 experiment: ${PHASE1_EXPERIMENT}"
echo "[train_pccr_80x96x112.sh] Phase-2 experiment: ${PHASE2_EXPERIMENT}"
echo "[train_pccr_80x96x112.sh] Stage-2 local refinement: ${STAGE2_LOCAL_REFINEMENT} (radius=${STAGE2_LOCAL_REFINEMENT_RADIUS})"
echo "[train_pccr_80x96x112.sh] Final cost volume: ${FINAL_COST_VOLUME} (radius=${FINAL_COST_VOLUME_RADIUS})"

if [[ "${PHASE_MODE}" == "synthetic" || "${PHASE_MODE}" == "both" ]]; then
  run_phase1
fi

if [[ "${PHASE_MODE}" == "real" || "${PHASE_MODE}" == "both" ]]; then
  if [[ -z "${PHASE1_CHECKPOINT_OVERRIDE}" && -n "${RESUME_PHASE2}" ]]; then
    PHASE1_CHECKPOINT=""
  fi
  if [[ -n "${PHASE1_CHECKPOINT}" && ! -f "${PHASE1_CHECKPOINT}" && -z "${RESUME_PHASE2}" ]]; then
    echo "[train_pccr_80x96x112.sh] Expected phase-1 checkpoint not found: ${PHASE1_CHECKPOINT}" >&2
    exit 1
  fi
  run_phase2 "${PHASE1_CHECKPOINT}"
fi
