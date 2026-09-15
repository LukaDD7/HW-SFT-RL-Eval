#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export DTOPD_ROOT="${DTOPD_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

exec python -m dual_track_opd.eval.benchmark_suite \
  --config "${REPO_ROOT}/configs/eval/project_vision_opd.yaml" \
  "$@"
