#!/usr/bin/env bash
# Launch the four-task v2 diagnostic for base, TailSFT, and PTD-PO.
#
# Usage:
#   bash scripts/eval/launch_v2_eval_pack.sh [gpu] [arms]
#
# Default:
#   GPU 0, arms=base,tailsft,ptdpo
#
# Each arm runs sequentially on one model server. The v2 protocol has no judge
# tasks, so only one GPU is needed. Output is written under:
#   eval_runs/vision_opd_project_v2

set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DTOPD_ROOT="${DTOPD_ROOT:-${REPO_ROOT}}"
GPU_ID="${1:-0}"
ARMS="${2:-base,tailsft,ptdpo}"
PORT="${V2_PACK_PORT:-8100}"

PTDPO_HF="${HW_PTDPO_CKPT:-}"

IFS=',' read -r -a ARM_LIST <<< "${ARMS}"

for arm in "${ARM_LIST[@]}"; do
  case "${arm}" in
    base)
      ckpt="${HW_BASE_CKPT:-}"
      run_name="v2_base_qwen3vl8b"
      served_name="Qwen3-VL-8B-Base"
      ;;
    tailsft)
      ckpt="${HW_TAILSFT_CKPT:-}"
      run_name="v2_tailsft_mmf122k_1ep"
      served_name="Qwen3-VL-8B-TailSFT"
      ;;
    ptdpo)
      ckpt="${PTDPO_HF}"
      run_name="v2_ptdpo_r4_step390"
      served_name="Qwen3-VL-8B-PTDPO-R4"
      ;;
    *)
      echo "FATAL: unknown arm ${arm}" >&2
      exit 1
      ;;
  esac

  [ -n "${ckpt}" ] || {
    echo "FATAL: ${arm} checkpoint is not set; use HW_BASE_CKPT, HW_TAILSFT_CKPT, or HW_PTDPO_CKPT" >&2
    exit 1
  }

  echo "== launching v2 arm ${arm} on GPU ${GPU_ID} =="
  EVAL_CKPT="${ckpt}" \
  EVAL_RUN_NAME="${run_name}" \
  EVAL_SERVED_MODEL="${served_name}" \
  EVAL_GPU="${GPU_ID}" \
  EVAL_PORT="${PORT}" \
    bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks_v2.sh"

  PORT="$((PORT + 1))"
done

echo "== v2 eval pack complete =="
