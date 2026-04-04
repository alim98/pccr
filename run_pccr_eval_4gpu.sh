#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_path> <output_dir> [extra evaluate args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

CHECKPOINT_PATH="$1"
OUTPUT_DIR="$2"
shift 2

ENV_PATH="${ENV_PATH:-/nexus/posix0/MBR-neuralsystems/alim/envs/hvit}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_WORKERS_PER_SHARD="${NUM_WORKERS_PER_SHARD:-4}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH}/bin/python}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  src/pccr/scripts/evaluate.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_data_path "${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
  --val_data_path "${DATA_ROOT_DEFAULT:-/nexus/posix0/MBR-neuralsystems/alim/regdata/oasis}"
  --dataset_format oasis_fs
  --accelerator gpu
  --num_gpus 1
  --precision bf16-mixed
  --val_fraction 0.2
  --split_seed 42
  --train_num_steps 600
  --max_val_pairs 0
  --num_workers "${NUM_WORKERS_PER_SHARD}"
)
COMMON_ARGS+=("$@")

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

pids=()
shard_dirs=()

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  shard_dir="${OUTPUT_DIR}/shard_${shard}"
  shard_dirs+=("${shard_dir}")
  mkdir -p "${shard_dir}"

  echo "[run_pccr_eval_4gpu.sh] Launching shard ${shard}/${NUM_SHARDS} on GPU ${shard}"
  CUDA_VISIBLE_DEVICES="${shard}" \
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
    --shard_index "${shard}" \
    --num_shards "${NUM_SHARDS}" \
    --experiment_name "pccr_eval_shard_${shard}" \
    --output_dir "${shard_dir}" \
    > "${shard_dir}/stdout.log" 2> "${shard_dir}/stderr.log" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[run_pccr_eval_4gpu.sh] At least one shard failed. Check shard_*/stderr.log" >&2
  exit 1
fi

"${PYTHON_BIN}" src/pccr/scripts/merge_eval_reports.py \
  "${shard_dirs[@]}" \
  --output_dir "${OUTPUT_DIR}"

echo "[run_pccr_eval_4gpu.sh] Done. Merged summary: ${OUTPUT_DIR}/summary.json"
