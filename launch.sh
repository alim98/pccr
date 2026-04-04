#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
JOB_SCRIPT="${JOB_SCRIPT:-${REPO_ROOT}/train.sh}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
AIM_REPO_DEFAULT="${AIM_REPO_DEFAULT:-${REPO_ROOT}/aim}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RESTART_ON_STATES="${RESTART_ON_STATES:-TIMEOUT PREEMPTED NODE_FAIL FAILED CANCELLED OUT_OF_MEMORY}"
MAX_RESTARTS_WITHOUT_CHECKPOINT="${MAX_RESTARTS_WITHOUT_CHECKPOINT:-2}"

mkdir -p \
  "${REPO_ROOT}/slurm/output_hvit" \
  "${REPO_ROOT}/slurm/error_hvit" \
  "${REPO_ROOT}/logs" \
  "${REPO_ROOT}/checkpoints" \
  "${AIM_REPO_DEFAULT}"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hvit_slurm_${RUN_TAG}}"
RESUME_FROM_CHECKPOINT=""
EXTRA_ARGS=()
consecutive_restarts_without_checkpoint=0

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
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

LAUNCH_LOG_DIR="${REPO_ROOT}/logs/${EXPERIMENT_NAME}"
mkdir -p "${LAUNCH_LOG_DIR}"
LAUNCH_LOG_FILE="${LAUNCH_LOG_DIR}/launch.log"
exec > >(tee -a "${LAUNCH_LOG_FILE}") 2>&1

EXPERIMENT_LOG_ROOT="${REPO_ROOT}/logs/${EXPERIMENT_NAME}"
EXPERIMENT_CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_LOG_ROOT}" "${EXPERIMENT_CHECKPOINT_ROOT}"

echo "=========================================="
echo "H-ViT Slurm Launcher"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Timestamp: $(date)"
echo "=========================================="

latest_checkpoint() {
  local root="$1"
  local model_ckpt=""
  local best_ckpt=""

  model_ckpt=$(find "$root" -type f -name 'model_epoch_*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)
  best_ckpt=$(find "$root" -type f -name 'best_model.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)

  if [[ -n "${model_ckpt}" ]]; then
    printf '%s\n' "${model_ckpt}"
  elif [[ -n "${best_ckpt}" ]]; then
    printf '%s\n' "${best_ckpt}"
  else
    printf '\n'
  fi
}

job_state() {
  local jid="$1"
  local state=""

  state=$(squeue -j "$jid" -h -o "%T" 2>/dev/null || true)
  if [[ -n "${state}" ]]; then
    printf '%s\n' "${state}"
    return
  fi

  state=$(sacct -j "$jid" -n -o State 2>/dev/null | head -n1 | awk '{print $1}' || true)
  state="${state%%+*}"
  printf '%s\n' "${state}"
}

resolve_final_state() {
  local jid="$1"
  local state=""
  local retry=0

  while [[ ${retry} -lt 10 ]]; do
    state="$(job_state "${jid}")"
    if [[ -n "${state}" ]]; then
      printf '%s\n' "${state}"
      return
    fi
    retry=$((retry + 1))
    sleep 5
  done

  printf '\n'
}

submit_job() {
  local resume_ckpt="${1:-}"
  local sbatch_out=""
  local cmd=(
    sbatch --parsable
    "${JOB_SCRIPT}"
    --experiment_name "${EXPERIMENT_NAME}"
    --train_data_path "${DATA_ROOT_DEFAULT}"
    --val_data_path "${DATA_ROOT_DEFAULT}"
    --aim_repo "${AIM_REPO_DEFAULT}"
  )

  if [[ -n "${resume_ckpt}" ]]; then
    cmd+=(--resume_from_checkpoint "${resume_ckpt}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  printf '[launch.sh] Submitting:' >&2
  printf ' %q' "${cmd[@]}" >&2
  printf '\n' >&2

  if ! sbatch_out="$("${cmd[@]}" 2>&1)"; then
    printf '[launch.sh] Submission failed: %s\n' "${sbatch_out}" >&2
    return 1
  fi
  printf '%s\n' "${sbatch_out}"
}

if [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_FROM_CHECKPOINT="$(latest_checkpoint "${EXPERIMENT_CHECKPOINT_ROOT}")"
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[launch.sh] Initial resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
  echo "[launch.sh] No checkpoint found. Starting fresh."
fi

if ! job_id="$(submit_job "${RESUME_FROM_CHECKPOINT}")"; then
  echo "[launch.sh] Initial submission failed. Stopping launcher."
  exit 1
fi
if [[ -z "${job_id}" ]]; then
  echo "[launch.sh] Initial submission returned an empty job id. Stopping launcher."
  exit 1
fi
echo "[launch.sh] Submitted job ${job_id}"

trap 'echo "[launch.sh] Interrupted. Last submitted job: ${job_id}"' INT TERM

while true; do
  sleep "${POLL_SECONDS}"
  state="$(job_state "${job_id}")"

  while [[ "${state}" == "PENDING" || "${state}" == "CONFIGURING" || "${state}" == "RUNNING" || "${state}" == "COMPLETING" ]]; do
    sleep "${POLL_SECONDS}"
    state="$(job_state "${job_id}")"
  done

  state="$(resolve_final_state "${job_id}")"
  echo "[launch.sh] Job ${job_id} finished with state: ${state}"

  case " ${RESTART_ON_STATES} " in
    *" ${state} "*)
      RESUME_FROM_CHECKPOINT="$(latest_checkpoint "${EXPERIMENT_CHECKPOINT_ROOT}")"
      if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
        consecutive_restarts_without_checkpoint=0
        echo "[launch.sh] Restarting from checkpoint: ${RESUME_FROM_CHECKPOINT}"
      else
        consecutive_restarts_without_checkpoint=$((consecutive_restarts_without_checkpoint + 1))
        if (( consecutive_restarts_without_checkpoint > MAX_RESTARTS_WITHOUT_CHECKPOINT )); then
          echo "[launch.sh] Job failed ${consecutive_restarts_without_checkpoint} times without producing a checkpoint. Stopping launcher."
          exit 1
        fi
        echo "[launch.sh] No checkpoint found after ${state}. Restarting fresh (${consecutive_restarts_without_checkpoint}/${MAX_RESTARTS_WITHOUT_CHECKPOINT})."
      fi
      if ! job_id="$(submit_job "${RESUME_FROM_CHECKPOINT}")"; then
        echo "[launch.sh] Replacement submission failed. Stopping launcher."
        exit 1
      fi
      if [[ -z "${job_id}" ]]; then
        echo "[launch.sh] Replacement submission returned an empty job id. Stopping launcher."
        exit 1
      fi
      echo "[launch.sh] Submitted replacement job ${job_id}"
      ;;
    *)
      echo "[launch.sh] State ${state} is terminal and not configured for restart. Stopping launcher."
      break
      ;;
  esac
done
