#!/usr/bin/env bash
#
# Uniform entry point for the six target benchmarks:
# GQA / DynaMath / ViewSpatial-Bench / MMMU-Pro / ReMI / MMBench.
#
# Default model is the recovered Vision-OPD gs65 HF checkpoint.  To evaluate any
# later HF-format checkpoint, set EVAL_CKPT (and optionally EVAL_RUN_NAME).
#
# Usage:
#   bash scripts/eval/run_target_benchmarks.sh
#   EVAL_SMOKE=1 bash scripts/eval/run_target_benchmarks.sh
#   EVAL_CKPT=/path/to/model EVAL_RUN_NAME=my_ckpt \
#     bash scripts/eval/run_target_benchmarks.sh
#   EVAL_THINK_MODE=think ...  # Open-MOPD-compatible thinking protocol

set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

EVAL_CKPT="${EVAL_CKPT:-${SFT_RL_MODEL_HF:-${HW_MODEL_CKPT:-}}}"
[ -n "${EVAL_CKPT}" ] || { echo "FATAL: set EVAL_CKPT or SFT_RL_MODEL_HF" >&2; exit 1; }
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${SFT_RL_RUN_NAME:-vision_opd_gs65_target}}"
EVAL_SERVED_MODEL="${EVAL_SERVED_MODEL:-${SFT_RL_SERVED_MODEL:-Vision-OPD-4B}}"
EVAL_GPU="${EVAL_GPU:-${SFT_RL_EVAL_GPU:-0}}"
EVAL_JUDGE_GPU="${EVAL_JUDGE_GPU:-${SFT_RL_JUDGE_GPU:-1}}"
EVAL_PORT="${EVAL_PORT:-${SFT_RL_EVAL_PORT:-8000}}"
EVAL_JUDGE_PORT="${EVAL_JUDGE_PORT:-${SFT_RL_JUDGE_PORT:-8001}}"

export SFT_RL_MODEL_HF="${EVAL_CKPT}"
export SFT_RL_RUN_NAME="${EVAL_RUN_NAME}"
export SFT_RL_SERVED_MODEL="${EVAL_SERVED_MODEL}"
export SFT_RL_EVAL_GPU="${EVAL_GPU}"
export SFT_RL_JUDGE_GPU="${EVAL_JUDGE_GPU}"
export SFT_RL_EVAL_PORT="${EVAL_PORT}"
export SFT_RL_JUDGE_PORT="${EVAL_JUDGE_PORT}"
export SFT_RL_THINK_MODE="${EVAL_THINK_MODE:-${SFT_RL_THINK_MODE:-auto}}"

if [[ "${EVAL_SMOKE:-${SFT_RL_SMOKE:-0}}" == "1" ]]; then
  export SFT_RL_SMOKE=1
fi

exec bash "${REPO_ROOT}/scripts/sft_rl/run_sftrl_benchmarks.sh"
