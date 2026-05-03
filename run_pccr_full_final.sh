#!/usr/bin/env bash
# Last experiment before publication.
#
# What this does:
#   - Skips synthetic pretraining — warm-starts from the existing
#     half-res synthetic checkpoint (encoder + pointmap + matcher weights
#     transfer perfectly; decoder / refinement head are random-init both
#     in that checkpoint and in the new full-res model, so strict=False is safe)
#   - Trains real phase at full resolution 160x192x224
#   - Applies all fixes versus previous runs:
#       * decoder_fitting_weight=0.1  (was 0.0 — decoder now follows matcher)
#       * segmentation_supervision_weight=0.05  (was 0.0 — direct Dice gradient)
#       * inverse_consistency_weight=0.05  (was 0.1 — less competing pressure)
#       * correspondence_weight=0.1  (was 0.2 — less competing pressure)
#       * smoothness_weight=0.01, jacobian_weight=0.005
#       * multiscale LNCC with image_loss_weight=1.25
#               * final refinement head: 32ch, 2 blocks (48ch/3-block OOMs at full 160x192x224)
#       * lr_min_ratio=0.01 (cosine floor so model does not collapse at epoch 250+)
#
# Time estimate (based on current full-res runs: ~6 epochs/day on 4xA100):
#   300 epochs  →  ~50 days
#   Model typically plateaus around epoch 150-180 with proper supervision.
#   Check val_dice curve after epoch 100. If no improvement for 25 eval cycles
#   (every 5 epochs = 125 epochs), it has converged — kill and report best.
#
# Usage:
#   bash run_pccr_full_final.sh
#   RUN_ID=myrun bash run_pccr_full_final.sh
#   MAX_EPOCHS=200 bash run_pccr_full_final.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/launch_pccr_transmorph_pkl_phase.sh}"

# To use 2-node (8xA100, ~halves wall-clock time), set this env var before launching:
#   JOB_SCRIPT="${REPO_ROOT}/jobs/active/train_pccr_transmorph_pkl_2node.sh" \
#   bash run_pccr_full_final.sh
if [[ -n "${JOB_SCRIPT:-}" ]]; then
  export JOB_SCRIPT
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pccr_full_final_${RUN_ID}}"

# Warm-start checkpoint: the best synthetic checkpoint we have.
# This is the half-res (80x96x112) synthetic run at step 49500.
# Encoder, pointmap head, and matcher weights load correctly (Conv3d, size-agnostic).
# Spatial transformer buffers and decoder weights are re-initialized for 160x192x224.
# train.py loads this with strict=False automatically when checkpoint is not in the
# current experiment's checkpoint directory.
SYNTHETIC_CKPT="/u/almik/others/hvit/symlinks/experiments_pccr/checkpoints/pccr/pccr_transmorph_pkl_half_fast_full_v2_synthetic/last.ckpt"

if [[ ! -s "${SYNTHETIC_CKPT}" ]]; then
  echo "[run_pccr_full_final] ERROR: synthetic checkpoint not found: ${SYNTHETIC_CKPT}" >&2
  exit 1
fi

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/pairwise_oasis_transmorph_pkl_full_final.yaml}"
export PHASE="real"
export MAX_EPOCHS="${MAX_EPOCHS:-300}"
export TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-394}"

# Validate every epoch so best-dice and restart state are updated continuously.
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"

# Save a full epoch checkpoint every epoch and keep them all.
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"

# Also keep a rolling step checkpoint in case a run dies mid-epoch.
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-500}"

export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-19}"

# Final cost volume at full radius.
# train_pccr_transmorph_pkl.sh will inject these as --config_override at runtime.
export FINAL_COST_VOLUME_MODE="${FINAL_COST_VOLUME_MODE:-streamed}"
export FINAL_COST_VOLUME_RADIUS="${FINAL_COST_VOLUME_RADIUS:-8}"
export FINAL_COST_VOLUME_CHUNK_SIZE="${FINAL_COST_VOLUME_CHUNK_SIZE:-4}"

# Real phase: all params are in the loss graph — no unused-param DDP penalty.
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXPERIMENT_NAME

echo "=========================================="
echo "PCCR Full Final Experiment"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Config:           ${CONFIG_PATH}"
echo "Synthetic ckpt:   ${SYNTHETIC_CKPT}"
echo "Max epochs:       ${MAX_EPOCHS}"
echo "Val every N ep:   ${CHECK_VAL_EVERY_N_EPOCH}"
echo "Cost vol radius:  ${FINAL_COST_VOLUME_RADIUS}"
echo "Timestamp:        $(date)"
echo "=========================================="

"${LAUNCHER}" \
  --phase real \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${SYNTHETIC_CKPT}"
