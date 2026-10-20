# Project15 B6 seed43 unified comparison protocol

This protocol defines the seed43 single-generation comparison used for
ISO-MOPD, VA-OPD-RL, and Pure RL. It intentionally avoids mixing avg@4 scores
with seed43 scores.

## Scope

- Seed: 43
- Repeat count: 1
- Think mode: explicit think
- Benchmark set: ViewSpatial-Bench, GQA, DynaMath, MMMU-Pro, ReMI, MMBench
- Average: arithmetic mean over six normalized scores; MMBench is divided by
  100 only for the average.

## Answer extraction and scoring

All rule-based tasks use terminal/tag-aware answer extraction:

1. If the response contains `<answer>...</answer>`, use the last tag content.
2. Otherwise apply the benchmark's deterministic fallback.
3. Unextractable or truncated responses count as incorrect.

Specific rules:

- GQA: GQA v2 `short_answer` and `normalize_word`; full response is the
  fallback when no answer tag exists.
- DynaMath: last mathematical answer tag; full denominator.
- ViewSpatial / MMMU-Pro: deterministic MCQ with tag-first letter extraction.
- ReMI: `scripts/sft_rl/remi_reeval.py --mode exact`
  (`exact_accuracy_full_denominator`) over all 2,600 rows.
- MMBench: official LLM choice extractor; report the native 0–100 score and
  divide by 100 only for the six-benchmark average.

For an existing avg@4 Pure RL run, seed43 corresponds to `repeat_1` because
the configured base seed is 42 and repeats are 42, 43, 44, and 45.

## Current seed43 B6 scores

| Arm | ViewSpatial | GQA | DynaMath | MMMU-Pro | ReMI | MMBench | Six-benchmark average |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISO-MOPD step300 | 0.4513 | 0.5926 | 0.6938 | 0.3208 | 0.5365 | 87.2852 | 0.5780 |
| VA-OPD-RL GRPO200 step200 | 0.4097 | 0.6111 | 0.6639 | 0.3370 | 0.2954 | 84.6220 | 0.5272 |
| Pure RL step100 | 0.4459 | 0.5454 | 0.6691 | 0.5462 | 0.4892 | 75.6014 | 0.5745 |

## Provenance notes

- ISO-MOPD GQA was rescored from logged sample responses using the GQA v2
  extraction rule because the legacy GQA task scored long reasoning as the
  prediction and returned 0.
- ISO-MOPD ReMI was rescored from the completed replay JSONL using the same
  full-denominator exact evaluator as Pure RL and VA-OPD-RL.
- VA-OPD-RL ReMI already used the same strict full-denominator evaluator; no
  scoring change was required.
- Pure RL seed43 values are extracted from `repeat_1` of the completed avg@4
  artifact, not from a new generation run.
