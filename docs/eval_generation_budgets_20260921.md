# Eval generation budget update (2026-09-21)

## Current shared Bench6 budgets

The canonical v1 and v2 suite configs use these output limits in `auto`,
`think`, and `no-think`. Prompt selection does not change the token budget.

| Benchmark | Output token limit |
|---|---:|
| MMMU-Pro | 16384 |
| DynaMath | 16384 |
| ReMI | 16384 |
| MMBench | 8192 |
| ViewSpatial | 8192 |
| GQA | 8192 |

The four v2 task YAMLs use the same values. The common lmms OpenAI adapter
preserves each request's configured limit at the SDK boundary, bypassing the
pinned backend's 4096-token clamp for every prompt mode. ReMI sends its
configured limit directly. The total context limit remains 65536.

The prompt text, scorer selection, and avg@4 settings (temperature 1.0,
seeds 42–45) are preserved. Use a new run name: manifests record resolved
budgets and replay rows record the budget, temperature, and seed. Resume
rejects missing or mismatched settings rather than combining old and new
generations. Diagnostic `SFT_RL_MAX_NEW_TOKENS_OVERRIDE` applies equally to
lmms and replay runners in all prompt modes and is recorded in the manifest.

These budgets aim to reduce truncation bias; they do not establish that any
model's truncation rate will be below 1%. Report truncation alongside scores.
Historical aligned/offline configs and archived results retain their original
settings. Unrelated Project15 task budgets retain upstream values.

## Earlier upstream change (superseded for the six tasks above)

Upstream commits 730e66c and b06701c previously set the following per-request
output budgets. At that point the v2 B6 tasks remained at 4096 tokens.

| Benchmark | Before | After |
|---|---:|---:|
| ViewSpatial (Project15 v1) | 256 | 1024 |
| MindCube | 256 | 1024 |
| ScienceQA | 256 | 1024 |
| ReMI | 2048 | 8192 |
| BLINK | 1024 | 2048 |
| MMBench | 1024 | 2048 |
| MMMU-Pro (Project15 v1) | 2048 | 4096 |
| GQA/DynaMath/ViewSpatial/MMMU-Pro (B6 v2) | 4096 | 4096 |

GQA and VQAv2 remained at 128 tokens because that change request explicitly
covered the 256-token tasks, not the 128-token tasks. Historical archived
aligned/offline configs are not rewritten.

The value is 1024, correcting the earlier transient 1048 value before any
new evaluation run used it.

## Historical truncation evidence

The lmms-eval rows use the existing conservative proxy
`token_counts[0].output_tokens >= max_new_tokens`. ReMI uses its explicit
`finish_reason == "length"` field.

| Benchmark | Base | PTD-PO r4 step390 | TailSFT |
|---|---:|---:|---:|
| ViewSpatial | 0 / 5,712 (0.00%) | 0 / 5,712 (0.00%) | 5,711 / 5,712 (99.98%) |
| MindCube | 0 / 21,154 (0.00%) | 6,657 / 21,154 (31.47%) | 21,154 / 21,154 (100.00%) |
| ScienceQA | 2 / 2,017 (0.10%) | 54 / 2,017 (2.68%) | 1,950 / 2,017 (96.68%) |
| BLINK | 0 / 1,901 (0.00%) | 146 / 1,901 (7.68%) | 1,671 / 1,901 (87.90%) |
| MMBench | 3 / 4,329 (0.07%) | 3 / 4,329 (0.07%) | 1,684 / 4,329 (38.90%) |
| MMMU-Pro | 0 / 1,730 (0.00%) | 0 / 1,730 (0.00%) | 1,031 / 1,730 (59.60%) |
| ReMI | 7 / 2,600 (0.27%) | 18 / 2,600 (0.69%) | 1,365 / 2,600 (52.50%) |

The audit confirms that the low caps mainly affected verbose checkpoints,
especially TailSFT. Base was largely unaffected; PTD-PO was already materially
affected on MindCube and moderately affected on BLINK.

Raw sample files remain outside Git under
`eval_runs/vision_opd_project_baseline/`.
