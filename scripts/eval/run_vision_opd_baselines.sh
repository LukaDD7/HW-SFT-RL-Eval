#!/usr/bin/env bash
# =============================================================================
# Vision-OPD Baseline Evaluation — All-in-One Script
#
# GPU allocation (8xH200, 141GB each):
#   GPU 0       — eval model vLLM serve (4B, ~8GB, port 8000)
#   GPU 7       — judge model, loaded in-process via judge_qwenlm.py (32B)
#   GPUs 1-6    — free for other experiments
#
# Judge strategy: pass JUDGE_MODEL_PATH so judge_qwenlm.py loads the 32B model
# directly with vLLM's LLM class (no server needed).  This mirrors how the
# project's GKD teacher_service loads 32B models — direct in-process, not vLLM
# serve — but uses vLLM's offline inference API for the judge.
#
# Usage:
#   conda activate vision-opd-cu128
#   bash scripts/eval/run_vision_opd_baselines.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment (verified working config from baselines/vision_opd/scripts/)
# ---------------------------------------------------------------------------
TOOLCHAIN_BIN="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain/bin"
if [ -d "${TOOLCHAIN_BIN}" ]; then
    export PATH="${TOOLCHAIN_BIN}:${PATH}"
fi

export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/usr/local/cuda}/lib:${CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/usr/local/cuda}/lib:${CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
unset VLLM_ATTENTION_BACKEND
# vLLM v1 + Triton + FlashInfer all JIT-compile — must have real gcc/nvcc
# gcc_linux-64 installed into vision-opd-cu128 via:
#   conda install -n vision-opd-cu128 -c conda-forge gcc_linux-64 gxx_linux-64 -y
CONDA_GCC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GPP="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
if [ -x "${CONDA_GCC}" ]; then
    export CC="${CONDA_GCC}"
    export CXX="${CONDA_GPP}"
elif [ -x "/usr/bin/gcc" ]; then
    export CC="/usr/bin/gcc"
    export CXX="/usr/bin/g++"
else
    echo "FATAL: No gcc found." >&2
    echo "Run: conda install -n vision-opd-cu128 -c conda-forge gcc_linux-64 gxx_linux-64 -y" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VOPD_ROOT="${PROJECT_ROOT}/third_party/Vision-OPD"
MODEL_ROOT="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models"

VISION_OPD_CKPT="${VOPD_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65"
BASE_MODEL="${MODEL_ROOT}/Qwen3.5-4B"
JUDGE_MODEL_PATH="${MODEL_ROOT}/Qwen3-VL-32B-Instruct"

EVAL_PORT=8000
VLLM_COMMON_ARGS="--trust-remote-code --enforce-eager --disable-custom-all-reduce --gdn-prefill-backend triton"

BENCH_ALL="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn,mme-realworld-lite,mmstar,pope,pope_adv,pope_pop,pope_random,cv-bench,mmvp,visualprobe"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN${NC} $*"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] FATAL${NC} $*"; exit 1; }

wait_for_health() {
    local url="$1" label="$2" max_wait="${3:-300}"
    log "Waiting for ${label} (max ${max_wait}s)..."
    local start=$(date +%s)
    while true; do
        if curl -s "${url}/health" >/dev/null 2>&1; then
            log "${label} ready (took $(( $(date +%s) - start ))s)"
            return 0
        fi
        if [ $(( $(date +%s) - start )) -ge "${max_wait}" ]; then
            die "${label} failed to start within ${max_wait}s"
        fi
        sleep 3
    done
}

stop_server() {
    local pid="$1" label="$2"
    if kill -0 "${pid}" 2>/dev/null; then
        log "Stopping ${label} (pid=${pid})..."
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
log "=== Pre-flight checks ==="
for d in "${VISION_OPD_CKPT}" "${BASE_MODEL}" "${JUDGE_MODEL_PATH}"; do
    [ -d "${d}" ] || die "Model not found: ${d}"
done
if ! command -v vllm &>/dev/null; then
    die "vLLM not found. Activate the correct conda env first."
fi
if ! command -v nvcc &>/dev/null; then
    die "nvcc not found. Check cuda128-toolchain PATH."
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${GPU_COUNT}" -ge 2 ] || die "Need ≥2 GPUs, found ${GPU_COUNT}"
JUDGE_GPU=$((GPU_COUNT - 1))
log "${GPU_COUNT} GPUs — GPU 0 (eval vLLM serve) + GPU ${JUDGE_GPU} (judge in-process)"

# ---------------------------------------------------------------------------
# Function: evaluate one model
# ---------------------------------------------------------------------------
evaluate_model() {
    local model_path="$1" model_id="$2" model_name="$3" enable_thinking="$4"

    log "============================================================"
    log "EVALUATING: ${model_name}"
    log "============================================================"

    # Start eval vLLM server on GPU 0
    log "Starting vLLM serve: ${model_id} on GPU 0 (port ${EVAL_PORT})..."
    CUDA_VISIBLE_DEVICES=0 vllm serve "${model_path}" \
        --gpu-memory-utilization 0.85 \
        --tensor-parallel-size 1 \
        --served-model-name "${model_id}" \
        ${VLLM_COMMON_ARGS} \
        --port "${EVAL_PORT}" &
    EVAL_PID=$!
    wait_for_health "http://localhost:${EVAL_PORT}" "${model_id}" 900

    cd "${VOPD_ROOT}/eval"
    log "Running 15 benchmarks (judge loads 32B in-process on GPU 7)..."

    # Judge runs on GPU 7, loaded directly by judge_qwenlm.py (no server)
    set +e
    env API_BASE="http://localhost:${EVAL_PORT}/v1/" \
        OPENAI_MODEL_ID="${model_id}" \
        MODEL_NAME="${model_name}" \
        CUDA_VISIBLE_DEVICES=${JUDGE_GPU} \
        JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}" \
        BENCHMARK="${BENCH_ALL}" \
        BENCHMARK="${BENCH_ALL}" \
        ${enable_thinking:+ENABLE_THINKING="${enable_thinking}"} \
        bash run_eval.sh
    EVAL_RC=$?
    set -e

    stop_server "${EVAL_PID}" "${model_id}"

    if [ ${EVAL_RC} -ne 0 ]; then
        warn "${model_name} exited with code ${EVAL_RC}"
    else
        log "${model_name} COMPLETE"
    fi
}

# ---------------------------------------------------------------------------
# Run both models
# ---------------------------------------------------------------------------
evaluate_model "${VISION_OPD_CKPT}" "Vision-OPD-4B" "Vision-OPD-4B" ""
evaluate_model "${BASE_MODEL}" "Qwen3.5-4B" "Qwen3.5-4B" "False"

log "============================================================"
log "ALL BASELINES COMPLETE"
log "Results: ${VOPD_ROOT}/eval/model_answer/"
log "Judges:  ${VOPD_ROOT}/eval/judge/"
log "Run:     bash scripts/eval/summarize_baselines.sh"
log "============================================================"
