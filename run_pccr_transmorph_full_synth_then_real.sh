#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-pccr_transmorph_pkl_v3_vec_full_${RUN_ID}}"
SYN_EXPERIMENT_NAME="${SYN_EXPERIMENT_NAME:-${EXPERIMENT_PREFIX}_synthetic}"
REAL_EXPERIMENT_NAME="${REAL_EXPERIMENT_NAME:-${EXPERIMENT_PREFIX}_real}"
SYN_CHECKPOINT_PATH="${SYN_CHECKPOINT_PATH:-}"

FULL_LOG_DIR="${REPO_ROOT}/logs/pccr/${EXPERIMENT_PREFIX}_launcher"
FULL_LOG_FILE="${FULL_LOG_DIR}/launch_full.log"
mkdir -p "${FULL_LOG_DIR}"
exec > >(tee -a "${FULL_LOG_FILE}") 2>&1

echo "=========================================="
echo "PCCR TransMorph PKL Full Pipeline Launcher"
echo "Synthetic experiment: ${SYN_EXPERIMENT_NAME}"
echo "Real experiment: ${REAL_EXPERIMENT_NAME}"
echo "Launcher: ${LAUNCHER}"
echo "Timestamp: $(date)"
echo "=========================================="

SYN_CMD=(
  "${LAUNCHER}"
  --phase synthetic
  --experiment_name "${SYN_EXPERIMENT_NAME}"
)
if [[ -n "${SYN_CHECKPOINT_PATH}" ]]; then
  SYN_CMD+=(--checkpoint_path "${SYN_CHECKPOINT_PATH}")
fi

printf '[full_launcher] Synthetic launcher:'
printf ' %q' "${SYN_CMD[@]}"
printf '\n'
"${SYN_CMD[@]}"

FINAL_SYNTH_CKPT="/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/${SYN_EXPERIMENT_NAME}/last.ckpt"
if [[ ! -s "${FINAL_SYNTH_CKPT}" ]]; then
  echo "[full_launcher] Synthetic completed, but checkpoint was not found: ${FINAL_SYNTH_CKPT}" >&2
  exit 1
fi

echo "[full_launcher] Synthetic completed."
echo "[full_launcher] Real will initialize from: ${FINAL_SYNTH_CKPT}"

REAL_CMD=(
  "${LAUNCHER}"
  --phase real
  --experiment_name "${REAL_EXPERIMENT_NAME}"
  --checkpoint_path "${FINAL_SYNTH_CKPT}"
)

printf '[full_launcher] Real launcher:'
printf ' %q' "${REAL_CMD[@]}"
printf '\n'
"${REAL_CMD[@]}"

echo "[full_launcher] Full synthetic -> real pipeline completed."
