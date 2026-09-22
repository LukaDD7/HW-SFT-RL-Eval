#!/usr/bin/env bash
#
# Run the four rule-based B6-v2 benchmarks on two eval GPUs in parallel.
# Shard A: GQA + DynaMath. Shard B: ViewSpatial + MMMU-Pro.
# The shard outputs are merged into the canonical EVAL_RUN_NAME directory.

set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

EVAL_ENV="${HW_EVAL_ENV:-${SFT_RL_EVAL_ENV:-}}"
[ -n "${EVAL_ENV}" ] || { echo "FATAL: set HW_EVAL_ENV or SFT_RL_EVAL_ENV" >&2; exit 1; }
PY="${EVAL_ENV}/bin/python"

BASE_RUN_NAME="${EVAL_RUN_NAME:-b6_mixed_avg4}"
PRIMARY_GPU="${EVAL_GPU:-0}"
SECOND_GPU="${EVAL_V2_SECOND_GPU:-${EVAL_JUDGE_GPU:-1}}"
PRIMARY_PORT="${EVAL_PORT:-8000}"
SECOND_PORT="${EVAL_V2_SECOND_PORT:-${EVAL_JUDGE_PORT:-8001}}"
SMOKE_SUFFIX=""
if [[ "${EVAL_SMOKE:-${SFT_RL_SMOKE:-0}}" == "1" ]]; then
  SMOKE_SUFFIX="_smoke"
fi

SHARD_A_BASE="${BASE_RUN_NAME}_v2a"
SHARD_B_BASE="${BASE_RUN_NAME}_v2b"
SHARD_A_RUN="${SHARD_A_BASE}${SMOKE_SUFFIX}"
SHARD_B_RUN="${SHARD_B_BASE}${SMOKE_SUFFIX}"
MERGED_RUN="${BASE_RUN_NAME}${SMOKE_SUFFIX}"
OUT_ROOT="${DTOPD_EVAL_ROOT:-${DTOPD_ROOT:-${REPO_ROOT}}/eval_runs/vision_opd_project_v2}"

echo "== v2 parallel: shard_a=${SHARD_A_RUN} GPU${PRIMARY_GPU}:${PRIMARY_PORT} =="
echo "== v2 parallel: shard_b=${SHARD_B_RUN} GPU${SECOND_GPU}:${SECOND_PORT} =="
echo "== v2 parallel: merged=${MERGED_RUN} out_root=${OUT_ROOT} =="

pids=()
failed=0

(
  export EVAL_RUN_NAME="${SHARD_A_BASE}"
  export V2_BENCHMARKS="gqa,dynamath"
  export EVAL_GPU="${PRIMARY_GPU}"
  export EVAL_PORT="${PRIMARY_PORT}"
  bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks_v2.sh"
) &
pids+=($!)

(
  export EVAL_RUN_NAME="${SHARD_B_BASE}"
  export V2_BENCHMARKS="viewspatial,mmmu_pro"
  export EVAL_GPU="${SECOND_GPU}"
  export EVAL_PORT="${SECOND_PORT}"
  bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks_v2.sh"
) &
pids+=($!)

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "FATAL: one or more v2 parallel shards failed; shard outputs preserved." >&2
  exit 1
fi

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${PY}" -m dual_track_opd.eval.merge_v2_shards \
  --output-run "${OUT_ROOT}/${MERGED_RUN}" \
  --shard-run "${OUT_ROOT}/${SHARD_A_RUN}" \
  --shard-run "${OUT_ROOT}/${SHARD_B_RUN}" \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd_v2.yaml"

echo "== v2 parallel summary =="
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${PY}" -m dual_track_opd.eval.project_summary "${OUT_ROOT}/${MERGED_RUN}" \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd_v2.yaml"

echo "== DONE. Raw outputs: ${OUT_ROOT}/${MERGED_RUN} =="
