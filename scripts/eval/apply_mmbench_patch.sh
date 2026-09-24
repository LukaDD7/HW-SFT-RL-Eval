#!/usr/bin/env bash
set -euo pipefail

# Apply the repository-owned MMBench fix to the pinned external lmms-eval
# checkout. HW-SFT-RL-Eval deliberately does not vendor lmms-eval, so this
# small adapter is kept as a patch and applied at the canonical entry point.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_mmbench_tag_first.patch"
EVAL_ENV="${HW_EVAL_ENV:-${SFT_RL_EVAL_ENV:-}}"
PY="${EVAL_ENV:+${EVAL_ENV}/bin/python}"

[[ -r "${PATCH_FILE}" ]] || { echo "FATAL: missing MMBench patch: ${PATCH_FILE}" >&2; exit 1; }

LMMS_ROOT="${LMMS_EVAL_ROOT:-}"
if [[ -z "${LMMS_ROOT}" ]]; then
  [[ -x "${PY}" ]] || { echo "FATAL: set LMMS_EVAL_ROOT or HW_EVAL_ENV" >&2; exit 1; }
  LMMS_ROOT="$(${PY} -c 'import lmms_eval; from pathlib import Path; print(Path(lmms_eval.__file__).resolve().parent.parent)')"
fi

MMBENCH_FILE="${LMMS_ROOT}/lmms_eval/tasks/mmbench/mmbench_evals.py"
REASONING_FILE="${LMMS_ROOT}/lmms_eval/api/reasoning.py"
[[ -r "${MMBENCH_FILE}" ]] || { echo "FATAL: missing lmms-eval MMBench file: ${MMBENCH_FILE}" >&2; exit 1; }
[[ -r "${REASONING_FILE}" ]] || { echo "FATAL: missing lmms-eval reasoning file: ${REASONING_FILE}" >&2; exit 1; }

if grep -q 'def extract_final_answer' "${MMBENCH_FILE}" \
    && grep -q 'terminal answer' "${REASONING_FILE}"; then
  echo "MMBench patch already present: ${LMMS_ROOT}"
  exit 0
fi

echo "Applying repository MMBench patch to ${LMMS_ROOT}"
(cd "${LMMS_ROOT}" && patch --forward --batch -p1 < "${PATCH_FILE}")

grep -q 'def extract_final_answer' "${MMBENCH_FILE}" \
  || { echo "FATAL: MMBench extractor patch did not apply" >&2; exit 1; }
grep -q 'terminal answer' "${REASONING_FILE}" \
  || { echo "FATAL: reasoning cleaner patch did not apply" >&2; exit 1; }
echo "MMBench patch applied: ${LMMS_ROOT}"
