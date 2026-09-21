---
name: hw-eval-operations
description: Run and audit HW SFT/RL evaluations using the canonical protocols
---

# HW SFT/RL evaluation operations

Use this skill whenever running, modifying, or reporting an evaluation.

## Before running

1. Read `environment/KEY_VERSIONS.md`.
2. Confirm the model path or Hugging Face repository.
3. Confirm the benchmark assets are local.
4. Set `HF_HUB_OFFLINE=1` on networkless GPU instances.
5. Never run full evaluations before a smoke test when a runner or protocol has
   changed.

## Required paths

Prefer environment variables:

```bash
export HW_EVAL_ROOT=/path/to/HW-SFT-RL-Eval
export DTOPD_EVAL_ROOT=/path/to/outputs
export DTOPD_DATASET_ROOT=/path/to/local/assets
export HF_HOME=/path/to/hf_cache
```

## Protocols

### B-segment v2

Four deterministic tasks:

- GQA
- DynaMath
- ViewSpatial
- MMMU-Pro

Use 16384 output tokens for DynaMath/MMMU-Pro and 8192 for GQA/ViewSpatial,
with `<answer>`-tag-first deterministic scoring. ReMI uses 16384 and MMBench
8192 in the canonical v1 suite. All prompt modes share these per-task budgets;
use the repository adapter to preserve them past the pinned backend clamp.

```bash
scripts/eval/run_target_benchmarks_v2.sh
```

### Project15 v1 / offline

Use the v1 or offline suite only when the task is explicitly in that protocol.
Do not mix native v1 scores with v2 scores without labeling both.

### ReMI

Generate replay output if needed, then always rescore with:

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <run>/replay/remi.jsonl \
  --label-jsonl <remi_replay>/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

Rules:

- denominator is all 2,600 rows;
- use task-aware extraction;
- do not use `summary.json` `normalized_exact_diagnostic`;
- report truncation count alongside accuracy.

### MV-MATH

Use `dual_track_opd.eval.score_mv_math`.

- `--mode official` is protocol diagnostics only.
- `--mode strict` requires completed generation before a judge verdict can
  count.
- Report completion-gated rows with the strict score.

## Result validity

Every report must include:

- model checkpoint identity
- protocol version
- backend and judge model
- dataset or replay source
- sample count
- truncation count
- scoring command

If truncation is high, mark the result as format-limited rather than
capability-comparable.

## After running

1. Inspect result files, not only process exit codes.
2. Treat return code 0 with no result JSON as failure.
3. Run smoke tests before claiming a runner is fixed.
4. Record output paths and completion rates in the task plan.
