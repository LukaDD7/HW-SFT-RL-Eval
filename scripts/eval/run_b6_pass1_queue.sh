#!/usr/bin/env bash
# Run an ordered four-checkpoint B6 single-sample pass@1 queue.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EVAL_SEED=42 EVAL_REPEAT_COUNT=1 EVAL_SAMPLING_TEMPERATURE=1.0
# The plan owns generation endpoints. These scoped settings fix GPU/TP layout
# and must not be inherited from an interactive shell.
export B6_V2A_GPUS=0,1 B6_V2B_GPUS=2,3 B6_REMI_GPUS=0,1 B6_MMBENCH_GPUS=2,3
export B6_JUDGE_GPUS=4,5 B6_MODEL_TP=2 B6_JUDGE_TP=2
export B6_V2A_PORT=8000 B6_V2B_PORT=8001
export B6_REMI_PORT=8000 B6_MMBENCH_PORT=8002 B6_JUDGE_PORT=8004
export B6_MAX_MODEL_LEN=65536 B6_JUDGE_MAX_MODEL_LEN=32768
PLAN="${B6_PASS1_PLAN:-}"
MODE=full
DRY_RUN=0
# Optional comma-separated expert filter (e.g. "C3,C4") for split instances.
B6_EXPERTS="${B6_EXPERTS:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN="${2:?missing plan}"; shift 2 ;;
    --smoke) MODE=smoke; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: bash scripts/eval/run_b6_pass1_queue.sh --plan plan.json [--smoke|--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PLAN" ]] || { echo "FATAL: --plan is required" >&2; exit 2; }
[[ -f "$PLAN" && ! -L "$PLAN" ]] || { echo "FATAL: plan must be a regular file" >&2; exit 1; }
PLAN="$(realpath "$PLAN")"
PY="${HW_EVAL_ENV:+${HW_EVAL_ENV}/bin/python}"
PY="${PY:-python}"
export B6_PYTHON="$PY"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

if [[ "$DRY_RUN" == 0 ]]; then
  # Install/check the two patches once before either v1 lane starts, then
  # require the pinned lmms-eval revision. Concurrent patch writes are unsafe.
  "$PY" - "$REPO_ROOT" <<'PY'
from pathlib import Path
import lmms_eval
import subprocess
import sys
expected = "88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"
root = Path(lmms_eval.__file__).resolve().parent.parent
actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                        check=True, text=True, capture_output=True).stdout.strip()
if actual != expected:
    raise SystemExit(f"lmms-eval revision mismatch: {actual} != {expected}")
patch = Path(sys.argv[1]) / "scripts/eval/apply_mmbench_patch.sh"
subprocess.run(["bash", str(patch)], check=True)
for relative, marker in (
    ("lmms_eval/tasks/mmbench/mmbench_evals.py", "def extract_final_answer"),
    ("lmms_eval/api/reasoning.py", "terminal answer"),
):
    path = root / relative
    if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"missing pinned MMBench patch marker: {path}")
print(f"verified lmms-eval: {root} @ {actual}")
PY
fi

mapfile -t ROWS < <("$PY" - "$PLAN" "$MODE" "$B6_EXPERTS" <<'PY'
import json
import os
from pathlib import Path
import sys
from dual_track_opd.eval.b6_pass1_report import load_plan
plan = load_plan(Path(sys.argv[1]), expected_experiments=4)
mode = sys.argv[2]
selected = {name.strip().upper() for name in sys.argv[3].split(",") if name.strip()}
experiments = plan["experiments"][:1] if mode == "smoke" else plan["experiments"]
if mode != "smoke" and selected:
    missing = selected - {row["expert"] for row in experiments}
    if missing:
        raise SystemExit(f"B6_EXPERTS names unknown experts: {sorted(missing)}")
    experiments = [row for row in experiments if row["expert"] in selected]
    if not experiments:
        raise SystemExit("B6_EXPERTS filter selected no experiments")
print("\t".join((plan["judge_model"], plan["judge_checkpoint"])))
for experiment in experiments:
    if not Path(experiment["checkpoint"]).is_dir():
        raise SystemExit(f"missing HF checkpoint: {experiment['checkpoint']}")
    print("\t".join(experiment[key] for key in ("expert", "run_name", "checkpoint")))
PY
)
if [[ "$MODE" == smoke ]]; then
  EXPECTED_ROWS=2
elif [[ -n "$B6_EXPERTS" ]]; then
  EXPECTED_ROWS=$(($(printf '%s' "$B6_EXPERTS" | awk -F, '{print NF}') + 1))
else
  EXPECTED_ROWS=5
fi
[[ ${#ROWS[@]} -eq "$EXPECTED_ROWS" ]] || {
  echo "FATAL: invalid queue entries" >&2; exit 1;
}
IFS=$'\t' read -r JUDGE_MODEL JUDGE_CHECKPOINT <<<"${ROWS[0]}"
unset 'ROWS[0]'
if [[ "$DRY_RUN" == 1 ]]; then
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r expert run_name checkpoint <<<"$row"
    printf '%s\t%s\t%s\n' "$expert" "$run_name" "$checkpoint"
  done
  echo "Dry run only: no servers, exports, scoring, or result files were started." >&2
  exit 0
fi

QUEUE_ROOT="${DTOPD_EVAL_ROOT:?set DTOPD_EVAL_ROOT}"
LOG_ROOT="${B6_PASS1_LOG_ROOT:-${QUEUE_ROOT}/_scoped_b6_pass1_logs}"
if [[ "$MODE" == full && "${B6_SKIP_SMOKE_GATE:-0}" != "1" ]]; then
  # Helper instances on a second node set B6_SKIP_SMOKE_GATE=1: the primary
  # instance has already validated the C1 smoke receipt on this shared root.
  "$PY" - "$PLAN" "$QUEUE_ROOT" <<'PY'
from pathlib import Path
import sys
from dual_track_opd.eval.b6_pass1_report import build_report, load_plan, receipt_is_valid, receipt_path
plan = load_plan(Path(sys.argv[1]))
root = Path(sys.argv[2])
smoke = build_report(plan, root, smoke=True, expected_experiments=4)
if smoke["status"] != "complete":
    raise SystemExit("a complete C1 smoke audit is required before the full queue: " + str(smoke["errors"]))
experiment = plan["experiments"][0]
if not receipt_is_valid(receipt_path(root, experiment, smoke=True), experiment, root,
                        smoke=True, judge=plan):
    raise SystemExit("the C1 smoke receipt must match a fresh raw-output audit")
PY
fi
# Ports follow the immutable plan. Generation and limit overrides are removed
# by the scoped runner rather than inherited from an interactive shell.
export SFT_RL_JUDGE_HF="$JUDGE_CHECKPOINT" JUDGE_CHECKPOINT="$JUDGE_CHECKPOINT"
mkdir -p "$QUEUE_ROOT" "$LOG_ROOT"
LOCK="$LOG_ROOT/.b6_pass1_${MODE}.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "FATAL: another $MODE queue holds $LOCK" >&2; exit 1; }
printf 'pid=%s\nplan=%s\n' "$$" "$PLAN" >&9
CHILD_PID=""
forward_signal() {
  local signal="$1"
  [[ -n "$CHILD_PID" ]] && kill -s "$signal" -- "-$CHILD_PID" 2>/dev/null || true
}
trap 'forward_signal TERM; exit 143' TERM
trap 'forward_signal INT; exit 130' INT

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r expert run_name checkpoint <<<"$row"
  # Per-expert lock on the shared output root: two instances (this 8-GPU node
  # plus a helper node) may share DTOPD_EVAL_ROOT over GPFS. The lock prevents
  # both from running the SAME expert concurrently; different experts never
  # conflict because every lane writes under its own run_name directory.
  EXPERT_LOCK="$LOG_ROOT/.b6_pass1_${MODE}_${expert}.lock"
  exec 8>"$EXPERT_LOCK"
  flock -n 8 || { echo "FATAL: another instance is already running ${expert} (${MODE});" \
    "use B6_EXPERTS to partition experts across instances" >&2; exit 1; }
  printf 'pid=%s\nplan=%s\n' "$$" "$PLAN" >&8
  if "$PY" - "$expert" "$run_name" "$MODE" "$QUEUE_ROOT" "$PLAN" <<'PY'
from pathlib import Path
import sys
from dual_track_opd.eval.b6_pass1_report import load_plan, receipt_is_valid, receipt_path
expert, run_name, mode, root, plan_path = sys.argv[1:]
plan = load_plan(Path(plan_path))
experiment = next(row for row in plan["experiments"]
                  if row["expert"] == expert and row["run_name"] == run_name)
root = Path(root)
smoke = mode == "smoke"
valid = receipt_is_valid(receipt_path(root, experiment, smoke=smoke), experiment, root,
                         smoke=smoke, judge=plan)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "== ${expert}: validated existing receipt; skip =="
    continue
  fi
  echo "== ${expert}: B6 single-sample pass@1 (${MODE}) =="
  log="$LOG_ROOT/${run_name}_${MODE}.log"
  stage_log="$LOG_ROOT/${run_name}_${MODE}_stages"
  mkdir -p "$stage_log"
  setsid env -u EVAL_LIMIT -u SFT_RL_LIMIT -u SFT_RL_SMOKE \
    -u SFT_RL_EVAL_CONFIG -u SFT_RL_BENCHMARKS -u SFT_RL_JUDGE_BENCHMARKS \
    -u EVAL_MAX_NEW_TOKENS_OVERRIDE -u SFT_RL_MAX_NEW_TOKENS_OVERRIDE \
    -u EVAL_V2_SECOND_GPU -u EVAL_V2_SECOND_PORT -u EVAL_V2_PARALLEL \
    -u EVAL_SMOKE -u SFT_RL_THINK_MODE -u EVAL_KEEP_GOING -u SFT_RL_KEEP_GOING \
    EVAL_CKPT="$checkpoint" EVAL_RUN_NAME="$run_name" \
    EVAL_SERVED_MODEL="SCOPED-${expert}" EVAL_THINK_MODE=auto \
    B6_STAGE_LOG_ROOT="$stage_log" \
    $(if [[ "$MODE" == smoke ]]; then printf 'B6_SMOKE=1'; fi) \
    "$PY" "$REPO_ROOT/scripts/eval/run_scoped_b6_expert.py" \
    >"$log" 2>&1 &
  CHILD_PID=$!
  status=0
  wait "$CHILD_PID" || status=$?
  CHILD_PID=""
  if [[ "$status" != 0 ]]; then
    echo "FATAL: ${expert} failed with status ${status}; see ${log}" >&2
    exit "$status"
  fi
  # Audit only the newly completed expert; the full 4x6 table is built below.
  args=(--plan "$PLAN" --output-root "$QUEUE_ROOT" --experiment "$expert")
  if [[ "$MODE" == smoke ]]; then
    args+=(--smoke --report-name scoped_b6_pass1)
  fi
  "$PY" -m dual_track_opd.eval.b6_pass1_report "${args[@]}" | tee "$LOG_ROOT/report_${expert}_${MODE}.log"
done

args=(--plan "$PLAN" --output-root "$QUEUE_ROOT")
[[ "$MODE" == smoke ]] && args+=(--smoke)
"$PY" -m dual_track_opd.eval.b6_pass1_report "${args[@]}" | tee "$LOG_ROOT/report_${MODE}.log"
echo "B6 single-sample pass@1 ${MODE} queue complete; plan: $PLAN" >&2
