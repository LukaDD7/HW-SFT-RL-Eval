#!/usr/bin/env bash
# MMF 14-bench VLMEvalKit eval launcher (single arm, one GPU, background).
#
# Why a script instead of the raw one-liner:
#   1. 2026-09-07 07:30 base-arm first launch died: run.py's in-process vLLM
#      engine warmup JIT-compiles flashinfer top-k/top-p sampling kernels and
#      needs nvcc — same crash as the judge's first launch (07:05). The raw
#      command was missing the cuda132-toolchain env vars. This script always
#      exports them.
#   2. 2026-09-07 07:28 tailsft-arm paste split mid-command ("env: '--work-dir':
#      No such file or directory"). One short command line, nothing to split.
#
# Usage (GPU node, judge on :8801 must already be online — see manuscript 7-A-1):
#   bash scripts/eval/launch_mmf_eval.sh base          # GPU 0 default
#   bash scripts/eval/launch_mmf_eval.sh tailsft       # GPU 1 default
#   bash scripts/eval/launch_mmf_eval.sh tailsft 2     # explicit GPU
#   bash scripts/eval/launch_mmf_eval.sh base 0 --reuse  # interrupt-resume
#   bash scripts/eval/launch/launch_mmf_eval.sh grpo    # future RL arm; config
#      qwen3vl_8b_grpo_tailsft_mfr.json must exist first (manuscript 第 6 步
#      optional block), or pass a full config path as arm argument.
set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
VLMEVAL_DIR="${DTOPD_ROOT}/third_party_runtime/VLMEvalKit"
CFG_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_cfg"
LMU_DATA="${DTOPD_ROOT}/fc-opd-storage/lmu_data"
WORK_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs"
LOG_DIR="${DTOPD_ROOT}/fc-opd-storage/logs"
JUDGE_URL="http://127.0.0.1:8801/v1"
JUDGE_KEY="dummy"
JUDGE_MODEL="Qwen3-VL-32B-Instruct"

ARM="${1:?usage: launch_mmf_eval.sh base|tailsft|<config-path> [gpu_id] [--reuse]}"
GPU_ID="${2:-}"
REUSE="${3:-}"

# Arm -> (config, default GPU, log name)
CFG=""; LOG=""
case "${ARM}" in
    base)    CFG="${CFG_DIR}/qwen3vl_8b_base_mfr.json";    LOG="vlmeval_base_mfr";    GPU_ID="${GPU_ID:-0}" ;;
    tailsft) CFG="${CFG_DIR}/qwen3vl_8b_tailsft_mfr.json"; LOG="vlmeval_tailsft_mfr"; GPU_ID="${GPU_ID:-1}" ;;
    grpo)    CFG="${CFG_DIR}/qwen3vl_8b_grpo_tailsft_mfr.json"; LOG="vlmeval_grpo_tailsft_mfr"; GPU_ID="${GPU_ID:-0}" ;;
    *)       CFG="${ARM}"; LOG="vlmeval_$(basename "${ARM}" .json)"; GPU_ID="${GPU_ID:-0}" ;;
esac

[ -f "${CFG}" ] || { echo "FATAL: config not found: ${CFG}" >&2; exit 1; }
[ -f "${VLMEVAL_DIR}/run.py" ] || { echo "FATAL: missing ${VLMEVAL_DIR}/run.py" >&2; exit 1; }

REUSE_ARG=""
if [ "${REUSE}" = "--reuse" ]; then REUSE_ARG="--reuse"; fi

LOG_FILE="${LOG_DIR}/${LOG}.log"

# Toolchain 4 vars (flashinfer sampling-kernel JIT needs nvcc; see CLAUDE.md /
# manuscript 7-A-1 postmortem). Judge must be checked online first.
# 2026-09-07 08:0x fix: first version curled ${JUDGE_URL%/v1}/chat/completions
# = /chat/completions (missing /v1) → vLLM 404 → false "judge not ready" while
# the judge was perfectly fine. JUDGE_URL already ends in /v1: just append.
echo "== launching ${LOG} on GPU ${GPU_ID} (judge ${JUDGE_URL}) =="
# no_proxy BEFORE the probe (README 坑 4: http_proxy hijacks 127.00.0.1 requests)
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
PROBE_RESP="$(curl -s "${JUDGE_URL}/chat/completions" \
    -H 'Content-Type: application/json' -H "Authorization: Bearer ${JUDGE_KEY}" \
    -d "{\"model\":\"${JUDGE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":8}")" || true
if ! printf '%s' "${PROBE_RESP}" | grep -q choices; then
    echo "FATAL: judge not ready on ${JUDGE_URL} — probe got: ${PROBE_RESP:0:200}" >&2
    echo "        start it first (manuscript 7-A-1)" >&2
    exit 1
fi

setsid nohup env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}" \
    LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost \
    PYTHONPATH="${VLMEVAL_DIR}" \
    LMUData="${LMU_DATA}" \
    "${ENV_PREFIX}/bin/python" "${VLMEVAL_DIR}/run.py" \
    --config "${CFG}" \
    --work-dir "${WORK_DIR}" \
    --mode all \
    --judge "${JUDGE_MODEL}" \
    --judge-base-url "${JUDGE_URL}" \
    --judge-key "${JUDGE_KEY}" \
    ${REUSE_ARG} \
    > "${LOG_FILE}" 2>&1 &

echo "started (bg pid $!), log: ${LOG_FILE}"
echo "tail -f ${LOG_FILE}"
echo "end marker = Run Summary Report; interrupt-resume = rerun with --reuse as 3rd arg"
