#!/bin/bash -l

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_fullres.yaml}"
DATA_ROOT="${DATA_ROOT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis_l2r}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_fullres_l2r}"
LOGGER_BACKEND="${LOGGER_BACKEND:-tensorboard}"
AIM_REPO="${AIM_REPO:-${REPO_ROOT}/aim}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-200}"
PRECISION="${PRECISION:-bf16-mixed}"
MEMORY_WARNING_RATIO="${MEMORY_WARNING_RATIO:-0.9}"

EXTRA_ARGS=("$@")

mkdir -p "${REPO_ROOT}/logs/pccr" "${REPO_ROOT}/checkpoints/pccr" "${AIM_REPO}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_PATH}"
else
  export PATH="${ENV_PATH}/bin:${PATH}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RESOLUTION_CANDIDATES=(
  "[160, 192, 224]"
  "[128, 160, 192]"
  "[80, 96, 112]"
)

resolution_tag() {
  echo "$1" | tr -d '[] ' | tr ',' 'x'
}

common_overrides() {
  local resolution="$1"
  printf '%s\n' \
    "data_size=${resolution}" \
    "align_data_size_to_native_shape=false" \
    "lncc_window_size=9" \
    "use_amp=true" \
    "use_gradient_checkpointing=true" \
    "batch_size=1"
}

run_probe() {
  local phase="$1"
  local resolution="$2"
  local lr="$3"
  local max_epochs="$4"
  local check_val_every_n_epoch="$5"
  local max_val_pairs="$6"
  local experiment="${EXPERIMENT_NAME}_probe_${phase}_$(resolution_tag "${resolution}")"

  local cmd=(
    "${PYTHON_BIN}" src/pccr/scripts/train.py
    --config "${CONFIG_PATH}"
    --mode train
    --phase "${phase}"
    --train_data_path "${DATA_ROOT}"
    --val_data_path "${DATA_ROOT}"
    --dataset_format oasis_l2r
    --accelerator gpu
    --num_gpus "${NUM_GPUS}"
    --experiment_name "${experiment}"
    --logger_backend none
    --num_workers "${NUM_WORKERS}"
    --train_num_steps "${TRAIN_NUM_STEPS}"
    --max_epochs "${max_epochs}"
    --lr "${lr}"
    --precision "${PRECISION}"
    --limit_train_batches 1
    --limit_val_batches 0
    --check_val_every_n_epoch "${check_val_every_n_epoch}"
    --checkpoint_every_n_epochs 100000
    --memory_probe_only
    --memory_warning_ratio "${MEMORY_WARNING_RATIO}"
    --max_val_pairs "${max_val_pairs}"
  )

  while IFS= read -r override; do
    cmd+=(--config_override "${override}")
  done < <(common_overrides "${resolution}")

  if [[ "${phase}" == "synthetic" ]]; then
    cmd+=(--config_override "lr_scheduler=cosine" --config_override "lr_warmup_epochs=0")
  else
    cmd+=(--config_override "lr_scheduler=cosine" --config_override "lr_warmup_epochs=5")
  fi

  echo "[train_fullres.sh] Probe ${phase} at ${resolution}"
  "${cmd[@]}"
}

run_phase1() {
  local resolution="$1"
  local phase1_experiment="$2"
  local cmd=(
    "${PYTHON_BIN}" src/pccr/scripts/train.py
    --config "${CONFIG_PATH}"
    --mode train
    --phase synthetic
    --train_data_path "${DATA_ROOT}"
    --val_data_path "${DATA_ROOT}"
    --dataset_format oasis_l2r
    --accelerator gpu
    --num_gpus "${NUM_GPUS}"
    --experiment_name "${phase1_experiment}"
    --logger_backend "${LOGGER_BACKEND}"
    --aim_repo "${AIM_REPO}"
    --num_workers "${NUM_WORKERS}"
    --train_num_steps "${TRAIN_NUM_STEPS}"
    --max_epochs 50
    --lr 1e-4
    --precision "${PRECISION}"
    --checkpoint_every_n_epochs 10
  )

  while IFS= read -r override; do
    cmd+=(--config_override "${override}")
  done < <(common_overrides "${resolution}")
  cmd+=(--config_override "lr_scheduler=cosine" --config_override "lr_warmup_epochs=0")
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[train_fullres.sh] Phase 1 experiment: ${phase1_experiment}"
  printf '[train_fullres.sh] Phase 1 command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"

}

run_phase2() {
  local resolution="$1"
  local phase1_checkpoint="$2"
  local phase2_experiment="$3"
  local cmd=(
    "${PYTHON_BIN}" src/pccr/scripts/train.py
    --config "${CONFIG_PATH}"
    --mode train
    --phase real
    --train_data_path "${DATA_ROOT}"
    --val_data_path "${DATA_ROOT}"
    --dataset_format oasis_l2r
    --accelerator gpu
    --num_gpus "${NUM_GPUS}"
    --experiment_name "${phase2_experiment}"
    --logger_backend "${LOGGER_BACKEND}"
    --aim_repo "${AIM_REPO}"
    --num_workers "${NUM_WORKERS}"
    --train_num_steps "${TRAIN_NUM_STEPS}"
    --max_epochs 200
    --lr 2e-5
    --precision "${PRECISION}"
    --checkpoint_path "${phase1_checkpoint}"
    --check_val_every_n_epoch 10
    --max_val_pairs 5
    --checkpoint_every_n_epochs 25
  )

  while IFS= read -r override; do
    cmd+=(--config_override "${override}")
  done < <(common_overrides "${resolution}")
  cmd+=(
    --config_override "lr_scheduler=cosine"
    --config_override "lr_warmup_epochs=5"
    --config_override "image_loss=lncc"
  )
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[train_fullres.sh] Phase 2 experiment: ${phase2_experiment}"
  echo "[train_fullres.sh] Phase 2 checkpoint init: ${phase1_checkpoint}"
  printf '[train_fullres.sh] Phase 2 command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

SELECTED_RESOLUTION=""
for resolution in "${RESOLUTION_CANDIDATES[@]}"; do
  if run_probe synthetic "${resolution}" 1e-4 50 100000 0 && \
     run_probe real "${resolution}" 2e-5 200 10 5; then
    SELECTED_RESOLUTION="${resolution}"
    break
  fi
  echo "[train_fullres.sh] Falling back from ${resolution}"
done

if [[ -z "${SELECTED_RESOLUTION}" ]]; then
  echo "[train_fullres.sh] No candidate resolution passed the startup memory probe." >&2
  exit 1
fi

RESOLUTION_TAG="$(resolution_tag "${SELECTED_RESOLUTION}")"
RESOLUTION_LOG="${REPO_ROOT}/logs/pccr/${EXPERIMENT_NAME}_selected_resolution.txt"
printf '%s\n' "${SELECTED_RESOLUTION}" | tee "${RESOLUTION_LOG}"
echo "[train_fullres.sh] Selected resolution: ${SELECTED_RESOLUTION}"
echo "[train_fullres.sh] Resolution log: ${RESOLUTION_LOG}"

PHASE1_EXPERIMENT="${EXPERIMENT_NAME}_phase1_synth_${RESOLUTION_TAG}"
PHASE2_EXPERIMENT="${EXPERIMENT_NAME}_phase2_real_${RESOLUTION_TAG}"
run_phase1 "${SELECTED_RESOLUTION}" "${PHASE1_EXPERIMENT}"
PHASE1_CHECKPOINT="${REPO_ROOT}/checkpoints/pccr/${PHASE1_EXPERIMENT}/last.ckpt"
if [[ ! -f "${PHASE1_CHECKPOINT}" ]]; then
  echo "[train_fullres.sh] Expected Phase 1 checkpoint not found: ${PHASE1_CHECKPOINT}" >&2
  exit 1
fi

run_phase2 "${SELECTED_RESOLUTION}" "${PHASE1_CHECKPOINT}" "${PHASE2_EXPERIMENT}"
