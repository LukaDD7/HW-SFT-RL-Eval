#!/usr/bin/env bash
# Judge launcher for the MMF 14-bench eval (Qwen3-VL-32B vLLM, GPU 3 :8801).
#
# Why a script instead of the raw one-liner (manuscript 6a / 7-A-1):
#   2026-09-07 14:57 incident: judge restart used the STALE 6a one-liner
#   (no CUDA toolchain env) — vLLM 0.27.1 warmup JIT-compiles flashinfer
#   top-k/top-p sampling kernels and dies without nvcc ("Could not find nvcc
#   and default cuda_home='/usr/local/cuda'"). Both restart attempts also
#   `>`-overwrote the same log file, destroying the crash evidence (log was
#   left at 22 bytes = "nohup: ignoring input"). This script:
#     1. always exports the cuda132-toolchain 4 vars (same fix as
#        launch_mmf_eval.sh / manuscript 7-A-1 postmortem);
#     2. rotates any existing judge log to a timestamped copy before restart;
#     3. pkills port-8801 residue first (no-op when clean);
#     4. probes until ready and FAILS FAST (dumps log tail) if the server dies.
#
# Usage (GPU node):  bash scripts/eval/launch_judge.sh [gpu_id]   # default 3
# Success = prints "judge ready on http://127.0.0.1:8801/v1".
# Keep it online for the whole eval: MathVista/MathVerse/MathVision/CharXiv/
# DynaMath scoring goes through it and errors out (skips) if it is down.
set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DTOPD_ROOT="${DTOPD_ROOT:-${REPO_ROOT}}"
ENV_PREFIX="${HW_EVAL_ENV:-${SFT_RL_EVAL_ENV:-}}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
JUDGE_PATH="${HW_JUDGE_MODEL:-${SFT_RL_JUDGE_HF:-}}"
SERVED_NAME="Qwen3-VL-32B-Instruct"
JUDGE_API_KEY="dummy"
PORT=8801
GPU_ID="${1:-3}"
LOG_DIR="${HW_EVAL_LOG_DIR:-${DTOPD_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/judge_qwen3vl32b_8801.log"

[ -x "${ENV_PREFIX}/bin/vllm" ] || { echo "FATAL: missing ${ENV_PREFIX}/bin/vllm" >&2; exit 1; }
[ -f "${JUDGE_PATH}/config.json" ] || { echo "FATAL: missing judge weights ${JUDGE_PATH}" >&2; exit 1; }
[ -d "${CUDA_HOME}/bin" ] || { echo "FATAL: missing CUDA toolchain ${CUDA_HOME}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"

# 1) rotate old log so crash evidence survives restarts (14:57 incident)
if [ -s "${LOG_FILE}" ]; then
    ROT="${LOG_FILE%.log}_$(date +%Y%m%d_%H%M%S).log"
    mv -f "${LOG_FILE}" "${ROT}"
    echo "== rotated old judge log -> ${ROT} =="
fi

# 2) clear residue (no-op when clean)
pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
sleep 3

# 3) launch with toolchain 4 vars + no_proxy (proxy hijacks 127.0.0.1 requests otherwise)
echo "== launching judge on GPU ${GPU_ID} :${PORT} (first boot incl. JIT ~3-4 min) =="
setsid nohup env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}" \
    LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost \
    "${ENV_PREFIX}/bin/vllm" serve "${JUDGE_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host 127.0.0.1 --port "${PORT}" \
    --api-key "${JUDGE_API_KEY}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > "${LOG_FILE}" 2>&1 &
echo "started (bg pid $!), log: ${LOG_FILE}"

# 4) readiness probe: default 15 min (JUDGE_WAIT_MINUTES). 2026-09-09: the
# 62-GiB BF16 checkpoint can take >5 min on a contended GPFS node before the
# OpenAI endpoint answers, even while the process is healthy and loading.
JUDGE_WAIT_MINUTES="${JUDGE_WAIT_MINUTES:-15}"
JUDGE_WAIT_TICKS="$((JUDGE_WAIT_MINUTES * 4))"
for i in $(seq 1 "${JUDGE_WAIT_TICKS}"); do
    sleep 15
    if curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -H "Authorization: Bearer ${JUDGE_API_KEY}" \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":8}" \
        | grep -q choices; then
        echo "judge ready on http://127.0.0.1:${PORT}/v1"
        echo "keep it online until both 6b evals print their Run Summary Report"
        exit 0
    fi
    if ! pgrep -f "vllm serve.*--port ${PORT}" > /dev/null; then
        echo "FATAL: judge process died during startup; last 30 log lines:" >&2
        tail -30 "${LOG_FILE}" >&2
        exit 1
    fi
    echo "waiting for judge (${i}/${JUDGE_WAIT_TICKS})..."
done
echo "FATAL: judge not ready after ${JUDGE_WAIT_MINUTES} min; tail ${LOG_FILE}" >&2
exit 1
