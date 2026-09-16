#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

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
EOF
}

protocol=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --protocol)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      protocol="$2"
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

case "$protocol" in
  b6-mixed)
    bash scripts/eval/run_target_benchmarks_v2.sh
    SFT_RL_BENCHMARKS=remi \
    SFT_RL_JUDGE_BENCHMARKS=mmbench \
    SFT_RL_EVAL_CONFIG=configs/eval/project_vision_opd.yaml \
      bash scripts/eval/run_target_benchmarks.sh
    ;;
  project15)
    SFT_RL_BENCHMARKS=viewspatial,mindcube,gqa,vqav2,scienceqa,dynamath,mmsi_bench,blink,mmmu_pro,remi \
    SFT_RL_JUDGE_BENCHMARKS=mathverse,mathvista,mmbench,mmvet \
    SFT_RL_EVAL_CONFIG=configs/eval/project_vision_opd.yaml \
      bash scripts/eval/run_target_benchmarks.sh
    ;;
  *)
    echo "Unknown protocol: $protocol" >&2
    usage
    exit 2
    ;;
esac
