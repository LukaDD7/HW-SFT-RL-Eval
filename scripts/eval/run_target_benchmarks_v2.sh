#!/usr/bin/env bash
#
# MMF SFT 复现审计后的重评测入口（v2 协议）。
#
# 与 scripts/eval/run_target_benchmarks.sh（v1）的区别：
#   - config: configs/eval/project_vision_opd_v2.yaml（任务名 *_v2）
#   - 每个 lmms-eval 命令自动带 --include_path eval_tasks/opd_v2
#   - 四项全 4096 解码预算；判分链 <answer> 标签优先 + 确定性 fallback
#   - 只跑 rule-based 四项（gqa/dynamath/viewspatial/mmmu_pro），
#     不含 remi（replay 口径无变化）与 mmbench（judge 口径不变）
#   - 输出根目录默认 eval_runs/vision_opd_project_v2，与 v1 结果并行保留
#
# 用法（GPU 节点，先确认 benchmark 数据已在共享 HF 缓存）:
#   bash scripts/eval/run_target_benchmarks_v2.sh
#   EVAL_SMOKE=1 bash scripts/eval/run_target_benchmarks_v2.sh   # 每项 8 条
#   EVAL_CKPT=/path/to/model EVAL_RUN_NAME=my_ckpt \
#     bash scripts/eval/run_target_benchmarks_v2.sh
#
# v2 协议无 judge 任务，因此不启动 judge 服务器（省一张 GPU）。
# 复用 v1 runner 的服务器管理（含 FlashInfer JIT 所需 CUDA toolchain 导出），但
# 直接调用 benchmark_suite 指定 v2 config —— 不复用 v1 wrapper 是因为其
# --config 硬编码为 project_vision_opd.yaml（v1 协议）。

set -euo pipefail

REPO_ROOT="${HW_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DTOPD_ROOT="${DTOPD_ROOT:-${REPO_ROOT}}"
EVAL_ENV="${SFT_RL_EVAL_ENV:-${HW_EVAL_ENV:-}}"
[ -n "${EVAL_ENV}" ] || { echo "FATAL: set HW_EVAL_ENV or SFT_RL_EVAL_ENV" >&2; exit 1; }
PY="${EVAL_ENV}/bin/python"
VLLM_BIN="${EVAL_ENV}/bin/vllm"
LOG_DIR="${HW_EVAL_LOG_DIR:-${DTOPD_ROOT}/logs}"
CUDA_TOOLCHAIN="${EVAL_CUDA_TOOLCHAIN:-${CUDA_TOOLCHAIN:-/usr/local/cuda}}"
[ -x "${CUDA_TOOLCHAIN}/bin/nvcc" ] || { echo "FATAL: nvcc not found: ${CUDA_TOOLCHAIN}/bin/nvcc"; exit 1; }
mkdir -p "${LOG_DIR}"

# Vision-OPD 的 Qwen3.5 GDN 注意力会在首个多模态请求触发 FlashInfer SM90 JIT；
# 导出共享 CUDA 12.8 toolchain（同 v1 runner）。
export CUDA_HOME="${CUDA_TOOLCHAIN}"
export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${DTOPD_ROOT}/.cache/flashinfer}"

HF_HOME="${EVAL_HF_CACHE:-${SFT_RL_HF_CACHE:-${DTOPD_ROOT}/.cache/huggingface}}"
export HF_HOME HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export LMMS_EVAL_DATASETS_CACHE="${HF_HOME}/datasets"

EVAL_CKPT="${EVAL_CKPT:-${SFT_RL_MODEL_HF:-${HW_MODEL_CKPT:-}}}"
[ -n "${EVAL_CKPT}" ] || { echo "FATAL: set EVAL_CKPT or SFT_RL_MODEL_HF" >&2; exit 1; }
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${SFT_RL_RUN_NAME:-vision_opd_gs65_target_v2}}"
EVAL_SERVED_MODEL="${EVAL_SERVED_MODEL:-${SFT_RL_SERVED_MODEL:-Vision-OPD-4B}}"
EVAL_GPU="${EVAL_GPU:-${SFT_RL_EVAL_GPU:-0}}"
EVAL_PORT="${EVAL_PORT:-${SFT_RL_EVAL_PORT:-8000}}"
MAX_LEN="${EVAL_MAX_LEN:-${SFT_RL_MAX_LEN:-65536}}"

# v2 protocol knobs
export DTOPD_EVAL_ROOT="${DTOPD_EVAL_ROOT:-${DTOPD_ROOT}/eval_runs/vision_opd_project_v2}}"
export DTOPD_EVAL_TASK_PATH="${REPO_ROOT}/eval_tasks/opd_v2"
V2_BENCHMARKS="${V2_BENCHMARKS:-gqa,dynamath,viewspatial,mmmu_pro}"

[ -f "${EVAL_CKPT}/model.safetensors" ] || compgen -G "${EVAL_CKPT}/model-*.safetensors" >/dev/null || { echo "FATAL: no model.safetensors/model-*.safetensors in ${EVAL_CKPT}"; exit 1; }
[ -x "${PY}" ] && [ -x "${VLLM_BIN}" ] || { echo "FATAL: eval env missing: ${EVAL_ENV}"; exit 1; }
cd "${REPO_ROOT}"

if [[ ! -d "${HF_HOME}/datasets" ]]; then
  echo "FATAL: benchmark dataset cache not found at ${HF_HOME}/datasets" >&2
  echo "       Run scripts/sft_rl/download_bench_datasets.sh on a networked node first." >&2
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT:-${SFT_RL_LIMIT:-}}" ]]; then
  LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
fi
if [[ "${EVAL_SMOKE:-${SFT_RL_SMOKE:-0}}" == "1" ]]; then
  LIMIT_ARGS=(--limit 8)
  EVAL_RUN_NAME="${EVAL_RUN_NAME}_smoke"
fi

OUT_ROOT="${DTOPD_EVAL_ROOT}"
EVAL_LOG="${LOG_DIR}/v2_bench_eval_server_${EVAL_RUN_NAME}.log"

cleanup() {
  echo "[v2-bench] stopping server..."
  [[ -n "${EVAL_PID:-}" ]] && kill "${EVAL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "== v2 benchmarks: model=${EVAL_CKPT} =="
echo "== eval: GPU${EVAL_GPU}:${EVAL_PORT} (no judge server) =="
echo "== run_name=${EVAL_RUN_NAME}  out_root=${OUT_ROOT} =="

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  "${VLLM_BIN}" serve "${EVAL_CKPT}" \
  --host 127.0.0.1 --port "${EVAL_PORT}" \
  --served-model-name "${EVAL_SERVED_MODEL}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len "${MAX_LEN}" \
  --trust-remote-code > "${EVAL_LOG}" 2>&1 &
EVAL_PID=$!

echo "[v2-bench] waiting for eval server..."
for i in $(seq 1 180); do
  ev_ok=$(curl -s --max-time 5 "http://127.0.0.1:${EVAL_PORT}/v1/models" >/dev/null 2>&1 && echo 1 || echo 0)
  if [[ "${ev_ok}" == "1" ]]; then
    echo "[v2-bench] eval server ready after ~$((i * 5))s"
    break
  fi
  if ! kill -0 "${EVAL_PID}" 2>/dev/null; then
    echo "FATAL: eval server died. See ${EVAL_LOG}"; tail -n 20 "${EVAL_LOG}"; exit 1
  fi
  sleep 5
done
curl -s --max-time 5 "http://127.0.0.1:${EVAL_PORT}/v1/models" >/dev/null 2>&1 || { echo "FATAL: eval server not ready"; exit 1; }

export VISION_OPD_CHECKPOINT="${EVAL_CKPT}"
export VISION_OPD_SERVED_MODEL="${EVAL_SERVED_MODEL}"
export VISION_OPD_API_BASE="http://127.0.0.1:${EVAL_PORT}/v1"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# 自动续跑：同 run name 已有输出目录时用 --resume-from 跳过已完成 benchmark
# （进行中的 benchmark 由 lmms-eval response cache 逐条回放，不会白跑）
RESUME_ARGS=()
if [[ -d "${OUT_ROOT}/${EVAL_RUN_NAME}" ]]; then
  RESUME_ARGS=(--resume-from "${OUT_ROOT}/${EVAL_RUN_NAME}")
  echo "[v2-bench] resuming from ${OUT_ROOT}/${EVAL_RUN_NAME}"
fi

echo "== v2 rule-based benchmarks (${V2_BENCHMARKS}) =="
"${PY}" -m dual_track_opd.eval.benchmark_suite \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd_v2.yaml" \
  --benchmarks "${V2_BENCHMARKS}" \
  --judge-policy defer \
  --run-name "${EVAL_RUN_NAME}" \
  "${LIMIT_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --execute

echo "== summary =="
"${PY}" -m dual_track_opd.eval.project_summary "${OUT_ROOT}/${EVAL_RUN_NAME}" \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd_v2.yaml" || true
echo "== DONE. Raw outputs: ${OUT_ROOT}/${EVAL_RUN_NAME} =="
