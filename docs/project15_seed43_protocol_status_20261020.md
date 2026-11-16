# Project15 seed43 protocol status (2026-10-20)

This snapshot records which results are usable after the GPU instance was
reclaimed and which historical protocols must not be reused.

## Current comparison contract

- Generation: seed 43, repeat count 1, explicit think mode
  `open-mopd-b6-prompt-v4`.
- Project15 mean: normalize 0–100 metrics to 0–1, then take a simple mean over
  the 15 benchmark values. Missing values are never counted as zero.
- A mean over incomplete coverage is diagnostic only and is not a final B15
  comparison.

## Benchmark coverage

| Arm | Latest-protocol coverage | Remaining work | Historical raw coverage |
|---|---:|---|---:|
| ISO-MOPD step300 | 11/15 | VQAv2, ViewSpatial v2, DynaMath v2, MMMU-Pro v2 | 14/15 (missing VQAv2) |
| Pure RL step100 | 14/15 | VQAv2 | 14/15 |
| VA-OPD-RL GRPO200 step200 | Not counted as current | Historical auto/native row; not a unified explicit-think comparison | 15/15 historical |
| Base 8B Instruct | Reference only | Use only as a historical v1 baseline, not a current seed43 comparison | 15/15 historical |

For ISO-MOPD, the raw 14/15 count includes legacy v1 scores for ViewSpatial,
DynaMath, and MMMU-Pro. Those three values are not usable under the current
v2 comparison contract. GQA is usable because the seed43 output was rescored
with the v2 tag-first rule.

For Pure RL, ScienceQA is usable after offline answer-tag rescoring:
`0.8170550322260783` over 2,017 rows. The original `0.0` result was an
upstream v1 parser artifact and is deprecated.

For VA-OPD-RL, ReMI used the current strict full-denominator exact scorer, but
the generation was the historical auto/native protocol rather than explicit
think v4. Therefore its 0.2953846154 is a valid historical score, not a
same-protocol comparison against ISO-MOPD or Pure RL.

## Current scoring rules

| Benchmark family | Required current rule |
|---|---|
| GQA, ViewSpatial, DynaMath, MMMU-Pro | v2 tasks with `<answer>`-tag-first extraction and no random-choice fallback |
| MindCube, VQAv2, ScienceQA, MMSI, BLINK | offline complement tasks with answer-tag extraction before the upstream evaluator |
| MathVerse, MathVista, MMVet | current judge scoring |
| ReMI | `scripts/sft_rl/remi_reeval.py --mode exact`, full 2,600-row denominator |
| MMBench | PR12 rescore where applicable; never mix raw legacy MMBench scores with PR12 |
| MV-MATH | official strict scoring sidecar |

## Deprecated protocols

These are marked unusable for current Project15/B6 comparisons:

- v1 direct-parser scores for MindCube, VQAv2, ScienceQA, MMSI, and BLINK in
  explicit-think runs.
- v1 GQA, ViewSpatial, DynaMath, and especially MMMU-Pro scores when comparing
  against a v2 arm.
- ReMI `normalized_exact_diagnostic` or any partial-denominator score.
- Mixing avg@4 means with seed43/repeat1 values.
- Raw legacy MMBench scores where a PR12 rescore exists.
- VA-OPD-RL auto/native ReMI generation as if it were explicit think v4.

## Current pending work

1. Restart Pure RL VQAv2 from its durable response cache; its previous vLLM
   engine died with `EngineDeadError`, so the displayed progress was failed
   retries rather than healthy generation.
2. Resume ISO VQAv2 from its durable response cache.
3. Run ISO ViewSpatial v2, DynaMath v2, and MMMU-Pro v2 under seed43/repeat1
   think v4.
4. Do not rerun completed complement tasks or completed B6 tasks that already
   satisfy the current scoring contract.

