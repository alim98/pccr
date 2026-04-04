#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
JOB_SCRIPT="${REPO_ROOT}/eval_pccr_4gpu.sh"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --checkpoint_path <ckpt> [other eval_pccr_4gpu.sh args...]" >&2
  exit 1
fi

CMD=(
  sbatch --parsable
  "${JOB_SCRIPT}"
  "$@"
)

printf '[launch_pccr_eval_4gpu.sh] Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'

job_id="$("${CMD[@]}")"
echo "[launch_pccr_eval_4gpu.sh] Submitted job ${job_id}"
