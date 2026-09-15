# Project-15 v1 decoding-budget truncation audit (2026-09-09)

## Summary

The historical v1 contract protocol used low `max_new_tokens` values for several
short-answer tasks. This was acceptable for the original short-answer models but
systematically penalizes long-CoT checkpoints such as TailSFT. The artifact is
not new: it was already present in the 2026-08/09 B-segment results and was
quantified in `docs/mmf_sft_reproduction_audit_20260903.md`.

The v2 protocol already fixes the four highest-impact tasks by using a uniform
4096-token budget and answer-tag-aware deterministic scoring. The current
lmms-eval OpenAI backend clamps generation to 4096 tokens, so 4096 is also the
practical maximum for this stack.

## Measured artifact in the current run

Samples were read directly from the current project15 `samples_*.jsonl` files.

| Arm | Bench | Average output tokens | At v1 cap | Effect |
|---|---|---:|---:|---|
| base | ViewSpatial | 4.32 | 0/5,712 (0%) | Short direct answers; no truncation |
| PTD-PO r4 step390 | ViewSpatial | 4.16 | 0/5,712 (0%) | Short direct answers; no truncation |
| TailSFT | ViewSpatial | 255.99 | 5,711/5,712 (100%) | Long CoT cut mid-reasoning |
| base | GQA | 2.26 | 0/12,578 (0%) | Short direct answers; no truncation |
| PTD-PO r4 step390 | GQA | 2.26 | 0/12,578 (0%) | Short direct answers; no truncation |
| TailSFT | GQA | 93.20 | 8,986/12,578 (71.4%) | Long CoT cut before final answer |

Thus the current v1 run remains valid as the contract protocol, but TailSFT's
v1 short-answer scores must be interpreted with this artifact. Base and PTD-PO
are essentially unaffected on ViewSpatial/GQA.

MMF-14 uses `max_new_tokens=32768` and is therefore not affected by this v1
short-budget issue. Its long run time is expected from long CoT outputs. For
MMU, the observed 32k-character response share is approximately 12.6% for base,
11.9% for PTD-PO, and 23.7% for TailSFT.

## v1 budget map

| v1 benchmark | v1 cap | Risk for long-CoT checkpoints |
|---|---:|---|
| ViewSpatial | 256 | Very high |
| MindCube | 256 | High |
| GQA | 128 | Very high |
| VQAv2 | 128 | High |
| ScienceQA | 256 | High |
| ReMI replay | 2048 | Medium |
| DynaMath | 4096 | Low/medium |
| MMSI-Bench | 2048 | Medium |
| BLINK | 1024 | High |
| MMMU-Pro | 2048 | High |
| MathVerse | 4096 | Low/medium |
| MathVista | 4096 | Low/medium |
| MMBench | 1024 | High |
| MMVet | 2048 | Medium |

## Recommended alignment

- v1 remains the immutable contract baseline.
- v2 four-task diagnostic remains mandatory for TailSFT and recommended for all
  arms: GQA, DynaMath, ViewSpatial, MMMU-Pro.
- For a full 15-task aligned run, use
  `configs/eval/project_vision_opd_v1_aligned.yaml`. It keeps the v1 task and
  scoring semantics but raises every task to a uniform 4096-token budget.
- Use 4096 tokens as the aligned upper bound for this lmms-eval OpenAI backend.
- Keep separate output roots (`vision_opd_project_baseline` and
  `vision_opd_project_v2`) so v1 and v2 results remain independently auditable.

## Why no automatic `resume-truncated`

The lmms-eval response cache keys requests by a fingerprint that includes the
request shape/generation settings. It has no first-class mode for replaying only
samples whose previous response reached `max_new_tokens`. The safe equivalent is:

1. retain v1 output;
2. start a new v2 run with 4096 tokens;
3. use a new run/cache directory so old truncated responses cannot be reused;
4. optionally use `scripts/eval/audit_truncation.py` to quantify exact at-cap
   sample counts before and after the v2 run.

The new audit tool reads the sample JSONL files and reports cap counts by
benchmark, without modifying raw outputs.

## Launcher hardening

`scripts/eval/launch_project15_eval_pack.sh` now records one stdout/stderr log,
PID, and status file per arm under `fc-opd-storage/logs/project15_state/`.
It also passes `--keep-going` to `benchmark_suite`, so one failed benchmark no
longer stops the remaining tasks in that arm.

For a full aligned v1 rerun:

```bash
PROJECT15_CONFIG=/path/to/configs/eval/project_vision_opd_v1_aligned.yaml \
PROJECT15_RUN_SUFFIX=_v1_aligned \
  bash scripts/eval/launch_project15_eval_pack.sh 0 base,tailsft,ptdpo 3
```

Use new run names for the aligned protocol; do not resume an old low-cap v1 run
and mix completed low-cap results with new 4096-token results.
