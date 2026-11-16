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
- The v2 stage uses two eval GPUs by default: GQA+DynaMath on one server and
  ViewSpatial+MMMU-Pro on another. Shard outputs are merged into the canonical
  run; set `EVAL_V2_PARALLEL=0` for the historical single-server path.

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

### SCOPED four-expert single-sample pass@1

This is a separate protocol, `b6-mixed-single-sample-pass1-seed42-v1`, not the
canonical avg@4 result set. On the eight-GPU evaluation node, run from the
Dual-Track-OPD repository root:

```bash
bash scripts/hpc/run_scoped_b6_pass1.sh
```

No arguments or interactive variables are required. The launcher verifies the
selected C1=300/C2=222/C3=200/C4=444 identities, exports missing C1/C2 weights
with the pinned official merger, verifies existing C3/C4 exports read-only,
then runs C1 smoke (eight samples per task) followed by the full C1→C4 queue.
A smoke or expert failure stops later work. `--prepare-only`, `--dry-run`, and
`--run-full` are optional diagnostic/explicit alternatives.

The protocol fixes native `auto` prompts, seed 42, one generation, and
temperature 1.0. GQA/ViewSpatial/MMBench use 8192 tokens;
DynaMath/MMMU-Pro/ReMI use 16384. ReMI uses exact full-denominator scoring;
MMBench uses the pinned local Qwen3-VL-32B-Instruct-FP8 judge.

| Stage | GPUs | TP | Port |
|---|---|---:|---:|
| v2a: GQA + DynaMath | 0,1 | 2 | 8000 |
| v2b: ViewSpatial + MMMU-Pro | 2,3 | 2 | 8001 |
| ReMI, after v2 | 0,1 | 2 | 8000 |
| MMBench, after v2 | 2,3 | 2 | 8002 |
| MMBench judge | 4,5 | 2 | 8004 |

Experts are sequential; lanes within an expert are parallel. GPUs 6,7 are
reserve. The queue applies/checks the MMBench patch once before starting the
two v1 lanes, and audits the endpoint recorded for each shard.

Every receipt requires a fresh raw-output audit of the exact sample count,
metric, checkpoint, served alias, endpoint, generation settings, ReMI replay,
judge provenance, and truncation counts. A process exit code or a stale
receipt alone is never sufficient. Repeating the same command preserves and
resumes partial outputs; only a receipt matching a fresh audit is skipped.

Final reports (created only after successful evaluation) are:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/scoped_four_expert_b6_pass1/scoped_b6_pass1_report.json
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/scoped_four_expert_b6_pass1/scoped_b6_pass1_report.md
```

Logs are under the SCOPED storage `logs/scoped_b6_pass1/` directory. Local
synthetic tests do not verify actual exports, vLLM TP=2 startup, or benchmark
scores; those require the GPU-node run.

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
- Project15 supports explicit `think` and `no-think` modes with the same
  prompt-protocol record as B6. Pair a mode-specific B6 run with the same-mode
  Project15 complement run; do not mix `auto` with explicit comparison arms.

### Project15 complement

When a matching B6-mixed run already exists, `project15-complement` avoids
regenerating the six overlapping tasks and runs only:

```text
MindCube, VQAv2, ScienceQA, MV-MATH, MathVerse, MathVista,
MMSI-Bench, BLINK, MMVet
```

Use the same checkpoint and generation protocol as the existing B6 run. Keep
B6 and complement rows labeled by their source run and task version when
combining them. This entrypoint is explicit compute saving; it does not
automatically import or mutate an existing B6 result.

### Project15 complement 4-GPU

`project15-complement-4gpu` runs the same nine complement tasks on four
networkless-GPU shards: one VQAv2 shard, one rule-based shard, one MV-MATH
shard, and one judge shard. It uses the offline local-parquet complement
configuration, and the four output directories are merged into one canonical
run directory and summary.

### Project15 complement offline

`project15-complement-offline` is the networkless-GPU variant of the same nine
complement tasks. Seven tasks load local parquet files, MathVerse uses the
shared datasets cache, and MV-MATH uses the local replay source. The runner
still expects the shared HuggingFace cache root to contain the `datasets/`
directory, but it does not require outbound network access.

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
