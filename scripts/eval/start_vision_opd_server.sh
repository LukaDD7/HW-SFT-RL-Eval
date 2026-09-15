#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="${VISION_OPD_CHECKPOINT:-${HW_MODEL_CKPT:-}}"
[ -n "${CHECKPOINT}" ] || { echo "ERROR: set VISION_OPD_CHECKPOINT or HW_MODEL_CKPT" >&2; exit 1; }
SERVED_MODEL="${VISION_OPD_SERVED_MODEL:-Vision-OPD-4B}"
HOST="${VISION_OPD_HOST:-127.0.0.1}"
PORT="${VISION_OPD_PORT:-8000}"
TP="${VISION_OPD_TP:-1}"
MAX_MODEL_LEN="${VISION_OPD_MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${VISION_OPD_GPU_MEMORY_UTILIZATION:-0.85}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: Vision-OPD checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm CLI is not available in the active environment." >&2
  exit 1
fi

echo "[vision-opd-server] repo=${REPO_ROOT}"
echo "[vision-opd-server] checkpoint=${CHECKPOINT}"
echo "[vision-opd-server] endpoint=http://${HOST}:${PORT}/v1"
echo "[vision-opd-server] served_model=${SERVED_MODEL} tp=${TP} max_model_len=${MAX_MODEL_LEN}"

exec vllm serve "${CHECKPOINT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser qwen3 \
  --trust-remote-code
