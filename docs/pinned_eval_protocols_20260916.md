# Pinned evaluation protocols — 2026-09-16

This document records the canonical protocols and single user-facing evaluation
entrypoint, amended for the seven-task 2026-11 midterm B6. Historical result
files retain their original six-task semantics.

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

For the 2026-11 midterm, B6-mixed contains exactly seven benchmarks:

- v2 no-judge: GQA, MMMU-Pro
- v1 no-judge: ReMI, ScienceQA, MindCube
- v1 judged: MMBench, MathVerse
- Four independent sampled generations per benchmark at temperature 1.0, seeds
  42–45; primary values are avg@4, not pass@1 or pass@4.
- The v2 stage uses two eval GPUs by default: GQA on one server and MMMU-Pro on
  another. Set `EVAL_V2_PARALLEL=0` for the single-server path.
- The new default run name is `b6_midterm7_avg4`. Use a unique run name per
  checkpoint and prompt mode; never resume an old six-task B6 directory.

| Benchmark | Task / scorer | Max new tokens |
| --- | --- | ---: |
| GQA | `gqa_v2`; normalized exact tag-or-direct | 8192 |
| MMMU-Pro | `mmmu_pro_standard_v2`; deterministic MCQ tag-first | 16384 |
| ReMI | strict `remi_reeval.py --mode exact`; full 2,600-row denominator | 16384 |
| ScienceQA | `scienceqa_img`; deterministic MCQ | 1024 |
| MindCube | `mindcube_full`; deterministic MCQ with subcategories | 1024 |
| MMBench | `mmbench_en_dev`; judged choice extractor | 8192 |
| MathVerse | `mathverse_testmini`; LLM judge | 4096 |

ReMI strict exact is averaged over four replay files and written to
`remi_avg4.json`. Record all four sample counts and truncation counts/rates.
MMBench and MathVerse require the judge model/version and judged raw outputs to
be recorded. All prompt modes share the budgets above.

Historical pinned result summary (legacy six-task protocol only):

```text
results/20260916/b6_mixed/summary.json
```

That file is not a midterm seven-task result and must not be silently relabeled
or combined into a seven-task aggregate. Historical generation-budget documents
and scores remain unchanged. Manifests now record `requested_benchmarks`;
resume rejects changed selections and legacy outputs containing tasks outside
the requested set.

The independent v2 runner still supports its historical four tasks, including
DynaMath and ViewSpatial, when called explicitly for that protocol. Their
removal from midterm B6 does not remove those benchmark configurations. Do not
compare B6-mixed scores directly with Project15 scores except at the individual
task level when the task version and scoring chain are identical.

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
- B6 v1 tasks retain the budgets above; GQA/MMMU-Pro use the v2 protocol in B6.
  Other v1 budgets remain: BLINK 2048, VQAv2 128.
  Historical aligned/offline configs retain their archived settings.
- Project15 is not equivalent to B6-mixed; it is a different task set.
- Do not average Project15 and B6-mixed scores in the same table.
- Project15 supports explicit `think` and `no-think` modes with the same
  prompt-protocol record as B6. Pair a mode-specific B6 run with the same-mode
  Project15 complement run; do not mix `auto` with explicit comparison arms.

### Project15 complement

When a matching B6-mixed run already exists, `project15-complement` avoids
regenerating the seven overlapping tasks and runs only:

```text
VQAv2, MV-MATH, MMSI-Bench, BLINK, MathVista, MMVet
```

Use the same checkpoint and generation protocol as the existing B6 run. Keep
B6 and complement rows labeled by their source run and task version when
combining them. This entrypoint is explicit compute saving; it does not
automatically import or mutate an existing B6 result.

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
