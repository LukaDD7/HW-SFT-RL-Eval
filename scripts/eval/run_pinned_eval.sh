#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EVAL_ENV="${HW_EVAL_ENV:-}"
[[ -n "${EVAL_ENV}" ]] || { echo "FATAL: set HW_EVAL_ENV" >&2; exit 1; }
PY="${EVAL_ENV}/bin/python"

usage() {
  cat <<'EOF'
Usage:
  EVAL_CKPT=/path/to/model EVAL_RUN_NAME=my_run HW_EVAL_ENV=/path/to/env \
    bash scripts/eval/run_pinned_eval.sh --protocol b6-mixed

  EVAL_CKPT=/path/to/model EVAL_RUN_NAME=my_run HW_EVAL_ENV=/path/to/env \
    bash scripts/eval/run_pinned_eval.sh --protocol project15

Protocols:
  b6-mixed   v2 GQA/DynaMath/ViewSpatial/MMMU-Pro + v1 ReMI/MMBench
  project15  15-task Project15 v1/offline suite

B6 prompt adapter:
  --think-mode auto|think|no-think
  Default: EVAL_THINK_MODE, then SFT_RL_THINK_MODE, then auto (native prompts).
  Think uses the Open-MOPD prompt and 8192 output tokens on all six tasks.
  Use a different EVAL_RUN_NAME for each mode and for new avg@4 runs.
EOF
}

protocol=""
think_mode="${EVAL_THINK_MODE:-${SFT_RL_THINK_MODE:-auto}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --protocol)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      protocol="$2"
      shift 2
      ;;
    --think-mode)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      think_mode="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$protocol" ]] || { usage; exit 2; }
case "$think_mode" in
  auto|think|no-think) ;;
  *) echo "Invalid --think-mode: $think_mode" >&2; exit 2 ;;
esac
if [[ "$think_mode" != "auto" && "$protocol" != "b6-mixed" ]]; then
  echo "The Think adapter currently supports --protocol b6-mixed only" >&2
  exit 2
fi
export EVAL_THINK_MODE="$think_mode" SFT_RL_THINK_MODE="$think_mode"

# Keep both stages of B6-mixed under one output root and one run name.  The
# underlying v1/v2 runners have different historical defaults, so exposing only
# this canonical entrypoint makes the protocol boundary explicit.
export DTOPD_EVAL_ROOT="${DTOPD_EVAL_ROOT:-${DTOPD_ROOT:-${REPO_ROOT}}/eval_runs/vision_opd_project_baseline}"
case "$protocol" in
  b6-mixed)
    export EVAL_RUN_NAME="${EVAL_RUN_NAME:-b6_mixed_avg4}"
    ;;
  project15)
    export EVAL_RUN_NAME="${EVAL_RUN_NAME:-project15_avg4}"
    ;;
esac

# Historical runs accidentally used literal closing braces in DTOPD_EVAL_ROOT.
# Refuse that pattern so future runs cannot silently create another
# `vision_opd_project_baseline}}...` directory.
if [[ "${DTOPD_EVAL_ROOT:-}" == *'}'* ]]; then
  echo "FATAL: DTOPD_EVAL_ROOT must not contain literal '}' characters" >&2
  echo "       Use a clean path or symlink, e.g.:" >&2
  echo "       ln -sfn <existing-output-root> /path/to/clean-output-root" >&2
  exit 2
fi

score_remi_avg4() {
  local suffix=""
  if [[ "${EVAL_SMOKE:-${SFT_RL_SMOKE:-0}}" == "1" ]]; then
    suffix="_smoke"
  fi
  local run_dir="${DTOPD_EVAL_ROOT}/${EVAL_RUN_NAME}${suffix}_nojudge"
  local -a files=( "${run_dir}"/replay/remi.repeat_*.jsonl )
  local -a existing=()
  local file
  for file in "${files[@]}"; do
    [[ -f "${file}" ]] && existing+=( "${file}" )
  done
  if [[ ${#existing[@]} -eq 0 ]]; then
    return 0
  fi
  if [[ ${#existing[@]} -ne 4 ]]; then
    echo "FATAL: ReMI avg@4 requires 4 replay files, found ${#existing[@]} in ${run_dir}/replay" >&2
    return 1
  fi

  "${PY}" scripts/sft_rl/remi_reeval.py \
    --mode exact \
    --jsonl "${existing[0]}" \
    --jsonl "${existing[1]}" \
    --jsonl "${existing[2]}" \
    --jsonl "${existing[3]}" \
    --label-jsonl assets/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl \
    --aggregate-mean \
    --out "${run_dir}/remi_avg4.json"

  "${PY}" -m dual_track_opd.eval.project_summary "${run_dir}" \
    --config configs/eval/project_vision_opd.yaml
}

case "$protocol" in
  b6-mixed)
    bash scripts/eval/run_target_benchmarks_v2.sh
    SFT_RL_BENCHMARKS=remi \
    SFT_RL_JUDGE_BENCHMARKS=mmbench \
    SFT_RL_EVAL_CONFIG=configs/eval/project_vision_opd.yaml \
      bash scripts/eval/run_target_benchmarks.sh
    score_remi_avg4
    ;;
  project15)
    SFT_RL_BENCHMARKS=viewspatial,mindcube,gqa,vqav2,scienceqa,mv_math,dynamath,mmsi_bench,blink,mmmu_pro,remi \
    SFT_RL_JUDGE_BENCHMARKS=mathverse,mathvista,mmbench,mmvet \
    SFT_RL_EVAL_CONFIG=configs/eval/project_vision_opd.yaml \
      bash scripts/eval/run_target_benchmarks.sh
    score_remi_avg4
    ;;
  *)
    echo "Unknown protocol: $protocol" >&2
    usage
    exit 2
    ;;
esac
