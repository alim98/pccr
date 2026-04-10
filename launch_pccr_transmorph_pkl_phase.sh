#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
JOB_SCRIPT="${JOB_SCRIPT:-${REPO_ROOT}/train_pccr_transmorph_pkl.sh}"

ACCOUNT="${ACCOUNT:-mhf_gpu}"
QOS="${QOS:-g0008}"
PARTITION="${PARTITION:-gpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-18}"

POLL_SECONDS="${POLL_SECONDS:-60}"
HEARTBEAT_POLLS="${HEARTBEAT_POLLS:-5}"
RESTART_ON_STATES="${RESTART_ON_STATES:-TIMEOUT PREEMPTED NODE_FAIL}"
MAX_RESTARTS_WITHOUT_CHECKPOINT="${MAX_RESTARTS_WITHOUT_CHECKPOINT:-2}"
MAX_TOTAL_RESTARTS="${MAX_TOTAL_RESTARTS:-100}"

PHASE="${PHASE:-real}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_transmorph_pkl_${PHASE}_${RUN_TAG}}"
CHECKPOINT_PATH=""
EXTRA_ARGS=()
consecutive_restarts_without_checkpoint=0
total_restarts=0

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

EXPERIMENT_CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pccr/${EXPERIMENT_NAME}"
LAUNCH_LOG_DIR="${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}"
LAUNCH_LOG_FILE="${LAUNCH_LOG_DIR}/launch.log"

mkdir -p \
  "${REPO_ROOT}/slurm/output_pccr_tm_pkl" \
  "${REPO_ROOT}/slurm/error_pccr_tm_pkl" \
  "${LAUNCH_LOG_DIR}" \
  "${EXPERIMENT_CHECKPOINT_ROOT}" \
  "${REPO_ROOT}/aim"

exec > >(tee -a "${LAUNCH_LOG_FILE}") 2>&1

echo "=========================================="
echo "PCCR TransMorph PKL Phase Launcher"
echo "Phase: ${PHASE}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Job script: ${JOB_SCRIPT}"
echo "Timestamp: $(date)"
echo "=========================================="

latest_checkpoint() {
  local root="$1"
  local last_ckpt=""
  local any_ckpt=""

  last_ckpt="$(find "${root}" -type f -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"
  any_ckpt="$(find "${root}" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"

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

  state="$(squeue -j "${jid}" -h -o "%T" 2>/dev/null || true)"
  if [[ -n "${state}" ]]; then
    printf '%s\n' "${state}"
    return
  fi

  state="$(sacct -j "${jid}" -n -o State 2>/dev/null | head -n1 | awk '{print $1}' || true)"
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

join_exports() {
  local joined="ALL"
  local item
  for item in "$@"; do
    if [[ -n "${item}" ]]; then
      joined="${joined},${item}"
    fi
  done
  printf '%s' "${joined}"
}

submit_job() {
  local resume_ckpt="${1:-}"
  local sbatch_out=""
  local resume_export=""
  local cmd=(
    sbatch --parsable
    -A "${ACCOUNT}"
    --qos="${QOS}"
    -p "${PARTITION}"
    --cpus-per-task="${CPUS_PER_TASK}"
  )

  local job_args=("${EXTRA_ARGS[@]}")
  if [[ -n "${resume_ckpt}" ]]; then
    if [[ "${resume_ckpt}" == "${EXPERIMENT_CHECKPOINT_ROOT}"/* ]]; then
      resume_export="RESUME_FROM_CHECKPOINT=${resume_ckpt}"
    else
      job_args+=(--checkpoint_path "${resume_ckpt}")
    fi
  fi

  local export_arg
  export_arg="$(join_exports \
    "EXPERIMENT_NAME=${EXPERIMENT_NAME}" \
    "PHASE=${PHASE}" \
    "DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-auto}" \
    "FINAL_COST_VOLUME_MODE=${FINAL_COST_VOLUME_MODE:-streamed}" \
    "FINAL_COST_VOLUME_CHUNK_SIZE=${FINAL_COST_VOLUME_CHUNK_SIZE:-16}" \
    "CHECKPOINT_EVERY_N_TRAIN_STEPS=${CHECKPOINT_EVERY_N_TRAIN_STEPS:-500}" \
    "CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH:-10}" \
    "LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS:-10}" \
    "MAX_VAL_PAIRS=${MAX_VAL_PAIRS:-19}" \
    "PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}" \
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "MAX_EPOCHS=${MAX_EPOCHS:-500}" \
    "TRAIN_NUM_STEPS=${TRAIN_NUM_STEPS:-394}" \
    "${resume_export}")"

  cmd+=(--export="${export_arg}" "${JOB_SCRIPT}" "${job_args[@]}")

  printf '[phase_launcher] Submitting:' >&2
  printf ' %q' "${cmd[@]}" >&2
  printf '\n' >&2

  if ! sbatch_out="$("${cmd[@]}" 2>&1)"; then
    printf '[phase_launcher] Submission failed: %s\n' "${sbatch_out}" >&2
    return 1
  fi
  printf '%s\n' "${sbatch_out}"
}

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(latest_checkpoint "${EXPERIMENT_CHECKPOINT_ROOT}")"
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  echo "[phase_launcher] Initial checkpoint: ${CHECKPOINT_PATH}"
else
  echo "[phase_launcher] No checkpoint found. Starting fresh."
fi

if ! job_id="$(submit_job "${CHECKPOINT_PATH}")"; then
  echo "[phase_launcher] Initial submission failed. Stopping launcher."
  exit 1
fi
if [[ -z "${job_id}" ]]; then
  echo "[phase_launcher] Initial submission returned an empty job id. Stopping launcher."
  exit 1
fi
echo "[phase_launcher] Submitted job ${job_id}"

trap 'echo "[phase_launcher] Interrupted. Last submitted job: ${job_id}"; exit 130' INT TERM

while true; do
  sleep "${POLL_SECONDS}"
  state="$(job_state "${job_id}")"
  heartbeat_count=0

  while [[ "${state}" == "PENDING" || "${state}" == "CONFIGURING" || "${state}" == "RUNNING" || "${state}" == "COMPLETING" ]]; do
    heartbeat_count=$((heartbeat_count + 1))
    if (( heartbeat_count == 1 || heartbeat_count % HEARTBEAT_POLLS == 0 )); then
      echo "[phase_launcher] Job ${job_id} current state: ${state}"
    fi
    sleep "${POLL_SECONDS}"
    state="$(job_state "${job_id}")"
  done

  state="$(resolve_final_state "${job_id}")"
  echo "[phase_launcher] Job ${job_id} finished with state: ${state}"

  if [[ "${state}" == "COMPLETED" ]]; then
    echo "[phase_launcher] Phase ${PHASE} completed."
    exit 0
  fi

  case " ${RESTART_ON_STATES} " in
    *" ${state} "*)
      total_restarts=$((total_restarts + 1))
      if (( total_restarts > MAX_TOTAL_RESTARTS )); then
        echo "[phase_launcher] Reached MAX_TOTAL_RESTARTS=${MAX_TOTAL_RESTARTS}. Stopping."
        exit 1
      fi

      CHECKPOINT_PATH="$(latest_checkpoint "${EXPERIMENT_CHECKPOINT_ROOT}")"
      if [[ -n "${CHECKPOINT_PATH}" ]]; then
        consecutive_restarts_without_checkpoint=0
        echo "[phase_launcher] Restarting from checkpoint: ${CHECKPOINT_PATH}"
      else
        consecutive_restarts_without_checkpoint=$((consecutive_restarts_without_checkpoint + 1))
        if (( consecutive_restarts_without_checkpoint > MAX_RESTARTS_WITHOUT_CHECKPOINT )); then
          echo "[phase_launcher] Job failed ${consecutive_restarts_without_checkpoint} times without a checkpoint. Stopping."
          exit 1
        fi
        echo "[phase_launcher] No checkpoint found after ${state}. Restarting fresh (${consecutive_restarts_without_checkpoint}/${MAX_RESTARTS_WITHOUT_CHECKPOINT})."
      fi

      if ! job_id="$(submit_job "${CHECKPOINT_PATH}")"; then
        echo "[phase_launcher] Replacement submission failed. Stopping launcher."
        exit 1
      fi
      if [[ -z "${job_id}" ]]; then
        echo "[phase_launcher] Replacement submission returned an empty job id. Stopping launcher."
        exit 1
      fi
      echo "[phase_launcher] Submitted replacement job ${job_id}"
      ;;
    *)
      echo "[phase_launcher] State ${state} is terminal and not configured for restart. Stopping launcher."
      exit 1
      ;;
  esac
done
