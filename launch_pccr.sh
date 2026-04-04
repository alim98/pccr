#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
JOB_SCRIPT="${JOB_SCRIPT:-${REPO_ROOT}/train_pccr.sh}"
DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
AIM_REPO_DEFAULT="${AIM_REPO_DEFAULT:-${REPO_ROOT}/aim}"
POLL_SECONDS="${POLL_SECONDS:-60}"
HEARTBEAT_POLLS="${HEARTBEAT_POLLS:-5}"
RESTART_ON_STATES="${RESTART_ON_STATES:-TIMEOUT PREEMPTED NODE_FAIL FAILED CANCELLED OUT_OF_MEMORY}"
MAX_RESTARTS_WITHOUT_CHECKPOINT="${MAX_RESTARTS_WITHOUT_CHECKPOINT:-2}"

PHASE="${PHASE:-real}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_${PHASE}_${RUN_TAG}}"
CHECKPOINT_PATH=""
EXTRA_ARGS=()
consecutive_restarts_without_checkpoint=0

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
    --checkpoint_path|--resume_from_checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr" \
  "${REPO_ROOT}/slurm/error_pccr" \
  "${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}" \
  "${REPO_ROOT}/checkpoints/pccr/${EXPERIMENT_NAME}" \
  "${AIM_REPO_DEFAULT}"

LAUNCH_LOG_FILE="${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}/launch.log"
exec > >(tee -a "${LAUNCH_LOG_FILE}") 2>&1

EXPERIMENT_CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pccr/${EXPERIMENT_NAME}"

echo "=========================================="
echo "PCCR Slurm Launcher"
echo "Phase: ${PHASE}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Timestamp: $(date)"
echo "=========================================="

latest_checkpoint() {
  local root="$1"
  local last_ckpt=""
  local any_ckpt=""

  last_ckpt="$(find "$root" -type f -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"
  any_ckpt="$(find "$root" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"

  if [[ -n "${last_ckpt}" ]]; then
    printf '%s\n' "${last_ckpt}"
  elif [[ -n "${any_ckpt}" ]]; then
    printf '%s\n' "${any_ckpt}"
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
    --phase "${PHASE}"
    --experiment_name "${EXPERIMENT_NAME}"
    --train_data_path "${DATA_ROOT_DEFAULT}"
    --aim_repo "${AIM_REPO_DEFAULT}"
  )

  if [[ "${PHASE}" == "real" ]]; then
    cmd+=(--val_data_path "${DATA_ROOT_DEFAULT}")
  fi

  if [[ -n "${resume_ckpt}" ]]; then
    if [[ "${resume_ckpt}" == "${EXPERIMENT_CHECKPOINT_ROOT}"/* ]]; then
      cmd+=(--resume_from_checkpoint "${resume_ckpt}")
    else
      cmd+=(--checkpoint_path "${resume_ckpt}")
    fi
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  printf '[launch_pccr.sh] Submitting:' >&2
  printf ' %q' "${cmd[@]}" >&2
  printf '\n' >&2

  if ! sbatch_out="$("${cmd[@]}" 2>&1)"; then
    printf '[launch_pccr.sh] Submission failed: %s\n' "${sbatch_out}" >&2
    return 1
  fi
  printf '%s\n' "${sbatch_out}"
}

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(latest_checkpoint "${EXPERIMENT_CHECKPOINT_ROOT}")"
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  echo "[launch_pccr.sh] Initial checkpoint: ${CHECKPOINT_PATH}"
else
  echo "[launch_pccr.sh] No checkpoint found. Starting fresh."
fi

if ! job_id="$(submit_job "${CHECKPOINT_PATH}")"; then
  echo "[launch_pccr.sh] Initial submission failed. Stopping launcher."
  exit 1
fi
if [[ -z "${job_id}" ]]; then
  echo "[launch_pccr.sh] Initial submission returned an empty job id. Stopping launcher."
  exit 1
fi
echo "[launch_pccr.sh] Submitted job ${job_id}"

trap 'echo "[launch_pccr.sh] Interrupted. Last submitted job: ${job_id}"' INT TERM

while true; do
  sleep "${POLL_SECONDS}"
  state="$(job_state "${job_id}")"
  heartbeat_count=0

  while [[ "${state}" == "PENDING" || "${state}" == "CONFIGURING" || "${state}" == "RUNNING" || "${state}" == "COMPLETING" ]]; do
    heartbeat_count=$((heartbeat_count + 1))
    if (( heartbeat_count == 1 || heartbeat_count % HEARTBEAT_POLLS == 0 )); then
      echo "[launch_pccr.sh] Job ${job_id} current state: ${state}"
    fi
    sleep "${POLL_SECONDS}"
    state="$(job_state "${job_id}")"
  done

  state="$(resolve_final_state "${job_id}")"
  echo "[launch_pccr.sh] Job ${job_id} finished with state: ${state}"

  case " ${RESTART_ON_STATES} " in
    *" ${state} "*)
      CHECKPOINT_PATH="$(latest_checkpoint "${EXPERIMENT_CHECKPOINT_ROOT}")"
      if [[ -n "${CHECKPOINT_PATH}" ]]; then
        consecutive_restarts_without_checkpoint=0
        echo "[launch_pccr.sh] Restarting from checkpoint: ${CHECKPOINT_PATH}"
      else
        consecutive_restarts_without_checkpoint=$((consecutive_restarts_without_checkpoint + 1))
        if (( consecutive_restarts_without_checkpoint > MAX_RESTARTS_WITHOUT_CHECKPOINT )); then
          echo "[launch_pccr.sh] Job failed ${consecutive_restarts_without_checkpoint} times without a checkpoint. Stopping."
          exit 1
        fi
        echo "[launch_pccr.sh] No checkpoint found after ${state}. Restarting fresh (${consecutive_restarts_without_checkpoint}/${MAX_RESTARTS_WITHOUT_CHECKPOINT})."
      fi
      if ! job_id="$(submit_job "${CHECKPOINT_PATH}")"; then
        echo "[launch_pccr.sh] Replacement submission failed. Stopping launcher."
        exit 1
      fi
      if [[ -z "${job_id}" ]]; then
        echo "[launch_pccr.sh] Replacement submission returned an empty job id. Stopping launcher."
        exit 1
      fi
      echo "[launch_pccr.sh] Submitted replacement job ${job_id}"
      ;;
    *)
      echo "[launch_pccr.sh] State ${state} is terminal and not configured for restart. Stopping launcher."
      break
      ;;
  esac
done
