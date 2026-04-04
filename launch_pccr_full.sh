#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
LAUNCHER="${REPO_ROOT}/launch_pccr.sh"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
SYNTH_EXPERIMENT_NAME="${SYNTH_EXPERIMENT_NAME:-pccr_synth_${RUN_TAG}}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-pccr_real_${RUN_TAG}}"
SYNTH_CHECKPOINT=""
SYNTH_EXTRA_ARGS=()
REAL_EXTRA_ARGS=()
CURRENT_STAGE="synthetic"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --synthetic_experiment_name)
      SYNTH_EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --real_experiment_name)
      REAL_EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --synthetic_checkpoint_path)
      SYNTH_CHECKPOINT="$2"
      shift 2
      ;;
    --)
      shift
      CURRENT_STAGE="real"
      ;;
    *)
      if [[ "${CURRENT_STAGE}" == "synthetic" ]]; then
        SYNTH_EXTRA_ARGS+=("$1")
      else
        REAL_EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

echo "=========================================="
echo "PCCR Full Pipeline Launcher"
echo "Synthetic experiment: ${SYNTH_EXPERIMENT_NAME}"
echo "Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "Timestamp: $(date)"
echo "=========================================="

SYNTH_CMD=(
  "${LAUNCHER}"
  --phase synthetic
  --experiment_name "${SYNTH_EXPERIMENT_NAME}"
)
if [[ -n "${SYNTH_CHECKPOINT}" ]]; then
  SYNTH_CMD+=(--checkpoint_path "${SYNTH_CHECKPOINT}")
fi
SYNTH_CMD+=("${SYNTH_EXTRA_ARGS[@]}")

printf '[launch_pccr_full.sh] Synthetic launcher:'
printf ' %q' "${SYNTH_CMD[@]}"
printf '\n'
"${SYNTH_CMD[@]}"

FINAL_SYNTH_CKPT="${REPO_ROOT}/checkpoints/pccr/${SYNTH_EXPERIMENT_NAME}/last.ckpt"
if [[ ! -f "${FINAL_SYNTH_CKPT}" ]]; then
  echo "[launch_pccr_full.sh] Synthetic stage completed but ${FINAL_SYNTH_CKPT} was not found." >&2
  exit 1
fi

REAL_CMD=(
  "${LAUNCHER}"
  --phase real
  --experiment_name "${REAL_EXPERIMENT_NAME}"
  --checkpoint_path "${FINAL_SYNTH_CKPT}"
)
REAL_CMD+=("${REAL_EXTRA_ARGS[@]}")

printf '[launch_pccr_full.sh] Real launcher:'
printf ' %q' "${REAL_CMD[@]}"
printf '\n'
"${REAL_CMD[@]}"
