#!/usr/bin/env bash
#
# SFT-RL 目标 benchmark 评测（共享文档口径）
#   Benchmark: MMMU-Pro / MMBench / REMI / DynaMath / ViewSpatial-Bench / GQA
#             (MathVerse can be appended explicitly for broader reporting.)
#   模型: SFT_RL_MODEL_HF 默认 = GRPO ckpt 184 导出的 HF 权重
#         （fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_grpo_mmf_184）
#   流程: GPU0 = vLLM serve 评测模型 :8000（OpenAI 兼容）
#         GPU1 = vLLM serve Qwen3-VL-32B-Instruct-FP8 judge :8001
#         先跑规则判分任务（gqa/dynamath/viewspatial/mmmu_pro/remi，judge-policy=defer），
#         再跑 judge 任务（mmbench，judge-policy=score）。
#         数据已预下载到共享缓存 $HF_HOME（download_bench_datasets.sh 在有网节点执行），
#         本脚本 HF_HUB_OFFLINE=1，GPU 节点无需网络。
#   用法（GPU 节点）:
#     bash scripts/sft_rl/run_sftrl_benchmarks.sh            # 全量
#     SFT_RL_SMOKE=1 bash scripts/sft_rl/run_sftrl_benchmarks.sh  # 每项 8 条冒烟
#   可选 env:
#     SFT_RL_MODEL_HF / SFT_RL_JUDGE_HF / SFT_RL_EVAL_GPU / SFT_RL_JUDGE_GPU
#     SFT_RL_RUN_NAME / SFT_RL_MAX_LEN / SFT_RL_HF_CACHE
#     SFT_RL_BENCHMARKS（规则判分列表，默认 gqa,dynamath,viewspatial,mmmu_pro,remi）
#     SFT_RL_JUDGE_BENCHMARKS（judge 列表，默认 mmbench；mathverse 可显式追加，置空跳过）
#     SFT_RL_LIMIT（可选：--limit 传给 benchmark_suite，例如 VQAv2 全量 214K 不可行时
#     用 SFT_RL_LIMIT=5000 对齐 2026-07 基线 replay 口径）
set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DTOPD_ROOT="${DTOPD_ROOT:-${REPO_ROOT}}"
EVAL_ENV="${SFT_RL_EVAL_ENV:-${HW_EVAL_ENV:-}}"
[ -n "${EVAL_ENV}" ] || { echo "FATAL: set HW_EVAL_ENV or SFT_RL_EVAL_ENV" >&2; exit 1; }
PY="${EVAL_ENV}/bin/python"
VLLM_BIN="${EVAL_ENV}/bin/vllm"
LOG_DIR="${HW_EVAL_LOG_DIR:-${DTOPD_ROOT}/logs}}"
CUDA_TOOLCHAIN="${SFT_RL_CUDA_TOOLCHAIN:-${CUDA_TOOLCHAIN:-/usr/local/cuda}}"
[ -x "${CUDA_TOOLCHAIN}/bin/nvcc" ] || { echo "FATAL: nvcc not found: ${CUDA_TOOLCHAIN}/bin/nvcc"; exit 1; }
mkdir -p "${LOG_DIR}"

# Vision-OPD's Qwen3.5 GDN attention triggers FlashInfer SM90 JIT on the first
# multimodal request.  Export the shared CUDA 12.8 toolchain so nvcc and CUDA
# libraries are available without relying on a login-node environment.
export CUDA_HOME="${CUDA_TOOLCHAIN}"
export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${DTOPD_ROOT}/.cache/flashinfer}"

HF_HOME="${SFT_RL_HF_CACHE:-${DTOPD_ROOT}/.cache/huggingface}"
export HF_HOME HF_HUB_OFFLINE=1
# lmms-eval 默认会把 HF datasets 缓存从远程 FS（GPFS）重定向到节点本地 scratch；
# 离线时必须强制用共享缓存目录（LMMS_EVAL_DATASETS_CACHE 优先且不做远程检查）。
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export LMMS_EVAL_DATASETS_CACHE="${HF_HOME}/datasets"

SFT_RL_MODEL_HF="${SFT_RL_MODEL_HF:-${HW_MODEL_CKPT:-}}"
[ -n "${SFT_RL_MODEL_HF}" ] || { echo "FATAL: set SFT_RL_MODEL_HF or HW_MODEL_CKPT" >&2; exit 1; }
SFT_RL_EVAL_CONFIG="${SFT_RL_EVAL_CONFIG:-${REPO_ROOT}/configs/eval/project_vision_opd.yaml}"
SFT_RL_JUDGE_HF="${SFT_RL_JUDGE_HF:-${HW_JUDGE_MODEL:-}}"
SERVED_NAME="${SFT_RL_SERVED_MODEL:-Qwen3-VL-8B-SFTRL}"
JUDGE_NAME="Qwen3-VL-32B-Instruct"
EVAL_GPU="${SFT_RL_EVAL_GPU:-0}"
JUDGE_GPU="${SFT_RL_JUDGE_GPU:-1}"
EVAL_PORT="${SFT_RL_EVAL_PORT:-8000}"
JUDGE_PORT="${SFT_RL_JUDGE_PORT:-8001}"
MAX_LEN="${SFT_RL_MAX_LEN:-65536}"
RUN_NAME="${SFT_RL_RUN_NAME:-sftrl_grpo184}"
OUT_ROOT="${DTOPD_EVAL_ROOT:-${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline}"
JUDGE_API_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
JUDGE_BENCHMARKS_RAW="${SFT_RL_JUDGE_BENCHMARKS-mmbench}"
JUDGE_ENABLED=0
[[ -n "${JUDGE_BENCHMARKS_RAW//,/}" ]] && JUDGE_ENABLED=1

[ -f "${SFT_RL_MODEL_HF}/model.safetensors" ] || compgen -G "${SFT_RL_MODEL_HF}/model-*.safetensors" >/dev/null || { echo "FATAL: no model.safetensors/model-*.safetensors in ${SFT_RL_MODEL_HF}"; exit 1; }
[ -x "${PY}" ] && [ -x "${VLLM_BIN}" ] || { echo "FATAL: eval env missing: ${EVAL_ENV}"; exit 1; }
if [[ "${JUDGE_ENABLED}" == "1" && -z "${SFT_RL_JUDGE_HF}" ]]; then
  echo "FATAL: set SFT_RL_JUDGE_HF or HW_JUDGE_MODEL for judge benchmarks" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ ! -d "${HF_HOME}/datasets" ]]; then
  echo "FATAL: benchmark dataset cache not found at ${HF_HOME}/datasets"
  echo "       Run scripts/sft_rl/download_bench_datasets.sh on a networked node first."
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "${SFT_RL_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${SFT_RL_LIMIT}")
fi
if [[ "${SFT_RL_SMOKE:-0}" == "1" ]]; then
  LIMIT_ARGS=(--limit 8)
  RUN_NAME="${RUN_NAME}_smoke"
fi

EVAL_LOG="${LOG_DIR}/sftrl_bench_eval_server_${RUN_NAME}.log"
JUDGE_LOG="${LOG_DIR}/sftrl_bench_judge_server_${RUN_NAME}.log"

cleanup() {
  echo "[sftrl-bench] stopping servers..."
  [[ -n "${EVAL_PID:-}" ]] && kill "${EVAL_PID}" 2>/dev/null || true
  [[ -n "${JUDGE_PID:-}" ]] && kill "${JUDGE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "== SFT-RL benchmarks: model=${SFT_RL_MODEL_HF} =="
if [[ "${JUDGE_ENABLED}" == "1" ]]; then
  echo "== eval: GPU${EVAL_GPU}:${EVAL_PORT}  judge: GPU${JUDGE_GPU}:${JUDGE_PORT} (${JUDGE_NAME}) =="
else
  echo "== eval: GPU${EVAL_GPU}:${EVAL_PORT}  judge: disabled (no judged benchmarks) =="
fi
echo "== run_name=${RUN_NAME}  out_root=${OUT_ROOT} =="

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  "${VLLM_BIN}" serve "${SFT_RL_MODEL_HF}" \
  --host 127.0.0.1 --port "${EVAL_PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len "${MAX_LEN}" \
  --trust-remote-code > "${EVAL_LOG}" 2>&1 &
EVAL_PID=$!

if [[ "${JUDGE_ENABLED}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" \
    "${VLLM_BIN}" serve "${SFT_RL_JUDGE_HF}" \
    --host 127.0.0.1 --port "${JUDGE_PORT}" \
    --served-model-name "${JUDGE_NAME}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --trust-remote-code > "${JUDGE_LOG}" 2>&1 &
  JUDGE_PID=$!
fi

echo "[sftrl-bench] waiting for endpoints..."
for i in $(seq 1 180); do
  ev_ok=$(curl -s --max-time 5 "http://127.0.0.1:${EVAL_PORT}/v1/models" >/dev/null 2>&1 && echo 1 || echo 0)
  if [[ "${JUDGE_ENABLED}" == "1" ]]; then
    jd_ok=$(curl -s --max-time 5 "http://127.0.0.1:${JUDGE_PORT}/v1/models" >/dev/null 2>&1 && echo 1 || echo 0)
  else
    jd_ok=1
  fi
  if [[ "${ev_ok}" == "1" && "${jd_ok}" == "1" ]]; then
    echo "[sftrl-bench] both servers ready after ~$((i * 5))s"
    break
  fi
  if ! kill -0 "${EVAL_PID}" 2>/dev/null || { [[ "${JUDGE_ENABLED}" == "1" ]] && ! kill -0 "${JUDGE_PID}" 2>/dev/null; }; then
    echo "FATAL: a vLLM server died. See ${EVAL_LOG} / ${JUDGE_LOG}"; tail -n 20 "${EVAL_LOG}" "${JUDGE_LOG}"; exit 1
  fi
  sleep 5
done
curl -s --max-time 5 "http://127.0.0.1:${EVAL_PORT}/v1/models" >/dev/null 2>&1 || { echo "FATAL: eval server not ready"; exit 1; }
if [[ "${JUDGE_ENABLED}" == "1" ]]; then
  curl -s --max-time 5 "http://127.0.0.1:${JUDGE_PORT}/v1/models" >/dev/null 2>&1 || { echo "FATAL: judge server not ready"; exit 1; }
fi

export VISION_OPD_CHECKPOINT="${SFT_RL_MODEL_HF}"
export VISION_OPD_SERVED_MODEL="${SERVED_NAME}"
export VISION_OPD_API_BASE="http://127.0.0.1:${EVAL_PORT}/v1"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

BENCH1="${SFT_RL_BENCHMARKS-gqa,dynamath,viewspatial,mmmu_pro,remi}"
BENCH2="${SFT_RL_JUDGE_BENCHMARKS-mmbench}"
KEEP_GOING_ARGS=()
if [[ "${SFT_RL_KEEP_GOING:-0}" == "1" ]]; then
  KEEP_GOING_ARGS=(--keep-going)
fi

# 自动续跑：同 run name 已有输出目录时用 --resume-from 跳过已完成 benchmark
# （进行中的 benchmark 由 lmms-eval response cache 逐条回放，不会白跑）
RESUME1=()
RESUME2=()
if [[ -d "${OUT_ROOT}/${RUN_NAME}_nojudge" ]]; then
  RESUME1=(--resume-from "${OUT_ROOT}/${RUN_NAME}_nojudge")
  echo "[sftrl-bench] resuming no-judge from ${OUT_ROOT}/${RUN_NAME}_nojudge"
fi
if [[ -d "${OUT_ROOT}/${RUN_NAME}_judged" ]]; then
  RESUME2=(--resume-from "${OUT_ROOT}/${RUN_NAME}_judged")
  echo "[sftrl-bench] resuming judged from ${OUT_ROOT}/${RUN_NAME}_judged"
fi

echo "== [1/2] rule-based benchmarks (no judge) =="
if [[ -n "${BENCH1}" ]]; then
  "${PY}" -m dual_track_opd.eval.benchmark_suite \
    --config "${SFT_RL_EVAL_CONFIG}" \
    --benchmarks "${BENCH1}" \
    --judge-policy defer \
    --run-name "${RUN_NAME}_nojudge" \
    "${LIMIT_ARGS[@]}" \
    "${RESUME1[@]}" \
    --execute \
    "${KEEP_GOING_ARGS[@]}"
fi

echo "== [2/2] judge benchmarks (${BENCH2}) =="
if [[ -n "${BENCH2}" ]]; then
  export JUDGE_API_KEY="EMPTY"
  export JUDGE_API_URL="${JUDGE_API_URL}"
  export JUDGE_MODEL="${JUDGE_NAME}"
  "${PY}" -m dual_track_opd.eval.benchmark_suite \
    --config "${SFT_RL_EVAL_CONFIG}" \
    --benchmarks "${BENCH2}" \
    --judge-policy score \
    --run-name "${RUN_NAME}_judged" \
    "${LIMIT_ARGS[@]}" \
    "${RESUME2[@]}" \
    --execute \
    "${KEEP_GOING_ARGS[@]}"
fi

echo "== summaries =="
[[ -d "${OUT_ROOT}/${RUN_NAME}_nojudge" ]] && "${PY}" -m dual_track_opd.eval.project_summary "${OUT_ROOT}/${RUN_NAME}_nojudge" || true
[[ -d "${OUT_ROOT}/${RUN_NAME}_judged" ]] && "${PY}" -m dual_track_opd.eval.project_summary "${OUT_ROOT}/${RUN_NAME}_judged" || true
echo "== DONE. Raw outputs: ${OUT_ROOT}/${RUN_NAME}_nojudge, ${OUT_ROOT}/${RUN_NAME}_judged =="
