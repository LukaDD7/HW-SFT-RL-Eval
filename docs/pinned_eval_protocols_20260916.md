# Pinned evaluation protocols — 2026-09-16

This document pins the two canonical result sets and the single user-facing
evaluation entrypoint.

## Canonical entrypoint

Use `scripts/eval/run_pinned_eval.sh` for both pinned protocols:

```bash
EVAL_CKPT=/path/to/model \
EVAL_RUN_NAME=my_run \
HW_EVAL_ENV=/path/to/env \
bash scripts/eval/run_pinned_eval.sh --protocol b6-mixed
```

or:

```bash
EVAL_CKPT=/path/to/model \
EVAL_RUN_NAME=my_run \
HW_EVAL_ENV=/path/to/env \
bash scripts/eval/run_pinned_eval.sh --protocol project15
```

The protocol argument is mandatory. There is no default protocol.

## Output roots and resume

`DTOPD_EVAL_ROOT` must be a clean path with no literal `}` characters. If a
historical output root contains accidental closing braces, create a clean
symlink to it and export the symlink path:

```bash
ln -sfn /path/to/historical-output-root /path/to/clean-output-root
export DTOPD_EVAL_ROOT=/path/to/clean-output-root
```

`run_pinned_eval.sh` rejects `DTOPD_EVAL_ROOT` values containing `}` so a
copy-pasted brace cannot silently create another output directory.

## B6-mixed

B6-mixed is the official six-benchmark comparison protocol:

- v2: GQA, DynaMath, ViewSpatial, MMMMU-Pro
- v1: ReMI, MMBench
- Four independent sampled generations per benchmark at temperature 1.0
- Primary values are avg@4, not pass@1 or pass@4

Pinned result summary:

```text
results/20260916/b6_mixed/summary.json
```

Important boundaries:

- Current generation budgets (updated 2026-09-21): MMMU-Pro/DynaMath/ReMI
  16384 tokens; GQA/ViewSpatial/MMBench 8192. All prompt modes share these
  budgets. The v2 tasks retain deterministic `<answer>`-tag-first scoring.
- These budgets supersede the historical settings used for the pinned results
  above. Use a new run name; historical scores are not recomputed by this change.
- ReMI uses strict exact full-denominator scoring over 2,600 rows.
- ReMI strict exact is averaged over four replay files and written to
  `remi_avg4.json`.
- MV-MATH replay, official-compatible judging, and strict completed-answer
  gating are likewise repeated four times; `mv_math_avg4.json` is the mean of
  four `strict_weighted_accuracy` values.
- MMBench uses the v1 judged protocol.
- Do not compare B6-mixed scores directly with Project15 scores except at the
  individual task level when the task version and scoring chain are identical.

## Project15

Project15 is the 15-task v1/offline protocol:

- ViewSpatial
- MindCube
- GQA
- VQAv2
- ScienceQA
- DynaMath
- MathVerse
- MathVista
- MMSI-Bench
- BLINK
- MMBench
- MMMU-Pro
- MMVet
- ReMI
- MV-MATH

Pinned result summary:

```text
results/20260916/project15/summary.json
```

Important boundaries:

- ReMI and MV-MATH are diagnostics, not native Project15 rows.
- The six B6 tasks in the canonical v1 config share the current budgets above.
  Other v1 budgets remain: MindCube/ScienceQA 1024, BLINK 2048, VQAv2 128.
  Historical aligned/offline configs retain their archived settings.
- Project15 is not equivalent to B6-mixed; it is a different task set.
- Do not average Project15 and B6-mixed scores in the same table.

## MMBench clarification

Base and PTD-PO r4 step390 both report:

```text
85.13745704467354
```

This is not a copied result file:

- the two result JSON files have different SHA-256 hashes;
- 460 of 4,329 filtered predictions differ;
- the model aliases and server identities are different.

The identical aggregate score is a coincidence after aggregation, not a
checkpoint-identity error.
