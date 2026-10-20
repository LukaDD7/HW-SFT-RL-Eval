#!/usr/bin/env bash
#
# Run the nine Project15-complement benchmarks on four GPUs in parallel,
# using the offline local-parquet protocol for networkless GPU instances.
#
# Layout:
#   GPU 0: model server, VQAv2 + ScienceQA shard
#   GPU 1: model server, MindCube + MMSI + BLINK shard
#   GPU 2: model server, MV-MATH + judge-dependent shard
#   GPU 3: judge server for the judge-dependent shard
#
# The four output directories are merged into one canonical run directory.

set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

EVAL_ENV="${HW_EVAL_ENV:-${SFT_RL_EVAL_ENV:-}}"
[ -n "${EVAL_ENV}" ] || { echo "FATAL: set HW_EVAL_ENV or SFT_RL_EVAL_ENV" >&2; exit 1; }
PY="${EVAL_ENV}/bin/python"

BASE_RUN_NAME="${EVAL_RUN_NAME:-project15_complement_4gpu}"
MODEL_GPU_A="${EVAL_GPU:-0}"
MODEL_GPU_B="${EVAL_SECOND_GPU:-1}"
MODEL_GPU_C="${EVAL_THIRD_GPU:-2}"
JUDGE_GPU="${EVAL_JUDGE_GPU:-3}"
MODEL_PORT_A="${EVAL_PORT:-8000}"
MODEL_PORT_B="${EVAL_SECOND_PORT:-8001}"
MODEL_PORT_C="${EVAL_THIRD_PORT:-8002}"
JUDGE_PORT="${EVAL_JUDGE_PORT:-8003}"

SMOKE_SUFFIX=""
if [[ "${EVAL_SMOKE:-${SFT_RL_SMOKE:-0}}" == "1" ]]; then
  SMOKE_SUFFIX="_smoke"
fi

SHARD_A_RUN="${BASE_RUN_NAME}_vqav2${SMOKE_SUFFIX}"
SHARD_B_RUN="${BASE_RUN_NAME}_rule${SMOKE_SUFFIX}"
SHARD_C_RUN="${BASE_RUN_NAME}_judge${SMOKE_SUFFIX}"
MERGED_RUN="${BASE_RUN_NAME}${SMOKE_SUFFIX}"
RUN_SUFFIX="_nojudge"
JUDGED_RUN_SUFFIX="_judged"
if [[ -n "${SMOKE_SUFFIX}" ]]; then
  RUN_SUFFIX="${SMOKE_SUFFIX}_nojudge"
  JUDGED_RUN_SUFFIX="${SMOKE_SUFFIX}_judged"
fi
RUN_SUFFIX="_nojudge"
JUDGED_RUN_SUFFIX="_judged"
if [[ -n "${SMOKE_SUFFIX}" ]]; then
  RUN_SUFFIX="${SMOKE_SUFFIX}_nojudge"
  JUDGED_RUN_SUFFIX="${SMOKE_SUFFIX}_judged"
fi
OUT_ROOT="${DTOPD_EVAL_ROOT:-${DTOPD_ROOT:-${REPO_ROOT}}/eval_runs/vision_opd_project_baseline}"

echo "== Project15 complement 4-GPU parallel =="
echo "== shard_a=${SHARD_A_RUN} GPU${MODEL_GPU_A}:${MODEL_PORT_A} =="
echo "== shard_b=${SHARD_B_RUN} GPU${MODEL_GPU_B}:${MODEL_PORT_B} =="
echo "== shard_c=${SHARD_C_RUN} GPU${MODEL_GPU_C}:${MODEL_PORT_C}, judge GPU${JUDGE_GPU}:${JUDGE_PORT} =="
echo "== merged=${MERGED_RUN} out_root=${OUT_ROOT} =="

pids=()
shard_names=()
failed=0

(
  export EVAL_RUN_NAME="${SHARD_A_RUN}"
  export SFT_RL_EVAL_CONFIG="${REPO_ROOT}/configs/eval/project_vision_opd_v1_offline_complement.yaml"
  export SFT_RL_BENCHMARKS="vqav2,scienceqa"
  export SFT_RL_JUDGE_BENCHMARKS=""
  export EVAL_GPU="${MODEL_GPU_A}"
  export EVAL_JUDGE_GPU="${JUDGE_GPU}"
  export EVAL_PORT="${MODEL_PORT_A}"
  export EVAL_JUDGE_PORT="${JUDGE_PORT}"
  bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks.sh"
) &
pids+=($!)
shard_names+=("vqav2")

(
  export EVAL_RUN_NAME="${SHARD_B_RUN}"
  export SFT_RL_EVAL_CONFIG="${REPO_ROOT}/configs/eval/project_vision_opd_v1_offline_complement.yaml"
  export SFT_RL_BENCHMARKS="mindcube,mmsi_bench,blink"
  export SFT_RL_JUDGE_BENCHMARKS=""
  export EVAL_GPU="${MODEL_GPU_B}"
  export EVAL_JUDGE_GPU="${JUDGE_GPU}"
  export EVAL_PORT="${MODEL_PORT_B}"
  export EVAL_JUDGE_PORT="${JUDGE_PORT}"
  bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks.sh"
) &
pids+=($!)
shard_names+=("rule")

(
  export EVAL_RUN_NAME="${SHARD_C_RUN}"
  export SFT_RL_EVAL_CONFIG="${REPO_ROOT}/configs/eval/project_vision_opd_v1_offline_complement.yaml"
  export SFT_RL_BENCHMARKS="mv_math"
  export SFT_RL_JUDGE_BENCHMARKS="mathverse,mathvista,mmvet"
  export EVAL_GPU="${MODEL_GPU_C}"
  export EVAL_JUDGE_GPU="${JUDGE_GPU}"
  export EVAL_PORT="${MODEL_PORT_C}"
  export EVAL_JUDGE_PORT="${JUDGE_PORT}"
  bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks.sh"
) &
pids+=($!)
shard_names+=("judge")

for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  shard="${shard_names[$index]}"
  echo "[4gpu] waiting shard=${shard} pid=${pid}"
  if wait "${pid}"; then
    echo "[4gpu] shard=${shard} pid=${pid} rc=0"
  else
    rc=$?
    failed=1
    echo "[4gpu] shard=${shard} pid=${pid} rc=${rc}" >&2
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "FATAL: one or more Project15-complement 4-GPU shards failed; shard outputs preserved." >&2
  exit 1
fi

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${PY}" -m dual_track_opd.eval.merge_project15_complement_shards \
  --output-run "${OUT_ROOT}/${MERGED_RUN}" \
  --shard-run "${OUT_ROOT}/${SHARD_A_RUN}${RUN_SUFFIX}" \
  --shard-run "${OUT_ROOT}/${SHARD_B_RUN}${RUN_SUFFIX}" \
  --shard-run "${OUT_ROOT}/${SHARD_C_RUN}${RUN_SUFFIX}" \
  --shard-run "${OUT_ROOT}/${SHARD_C_RUN}${JUDGED_RUN_SUFFIX}" \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd_v1_offline_complement.yaml"

echo "== Project15 complement 4-GPU summary =="
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${PY}" -m dual_track_opd.eval.project_summary "${OUT_ROOT}/${MERGED_RUN}" \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd_v1_offline_complement.yaml"

echo "== DONE. Raw outputs: ${OUT_ROOT}/${MERGED_RUN} =="
