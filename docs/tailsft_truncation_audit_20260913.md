# TailSFT Project15 truncation audit

Date: 2026-09-13

Checkpoint: `qwen3vl_sft_tailsft_mmf122k_1ep/global_step_1774/huggingface`

## Purpose

TailSFT learned a verbose reasoning format, but several Project15 evaluation
tasks use short output caps. This audit marks every completed benchmark where
the model hit the cap, so the current scores are not misread as pure capability
drops.

For the native `lmms-eval` sample files, generation logs do not expose
`finish_reason`. The audit therefore uses the conservative proxy:

```text
token_counts[0].output_tokens >= task max_new_tokens
```

For ReMI and MV-MATH replay JSONLs, the explicit `finish_reason == "length"`
field is used directly.

## Completed benchmark audit

| Benchmark | Reported score | Truncated / rows | Rate | Reporting status |
|---|---:|---:|---:|---|
| ViewSpatial | 6.79 | 5,711 / 5,712 | 99.98% | Not capability-comparable |
| MindCube | 5.34 | 21,154 / 21,154 | 100.00% | Not capability-comparable |
| GQA | 13.85 | 8,986 / 12,578 | 71.45% | Not capability-comparable |
| ScienceQA | 0.00 | 1,950 / 2,017 | 96.68% | Not capability-comparable |
| DynaMath | 57.88 | 1,026 / 5,010 | 20.48% | Materially affected diagnostic |
| MathVerse | 61.02 | 744 / 3,940 | 18.88% | Materially affected diagnostic |
| MathVista | 66.80 | 644 / 3,000 | 21.47% | Materially affected diagnostic |
| MMSI-Bench | 18.70 | 872 / 1,000 | 87.20% | Not capability-comparable |
| BLINK | 0.00 | 1,671 / 1,901 | 87.90% | Not capability-comparable |
| MMBench | 47.94 | 1,684 / 4,329 | 38.90% | Materially affected |
| MMMU-Pro | 23.53 | 1,031 / 1,730 | 59.60% | Not capability-comparable |
| MMVet | 56.73 | 85 / 218 | 38.99% | Materially affected |
| ReMI (exact full denominator) | 29.04 | 1,365 / 2,600 | 52.50% | Not capability-comparable |
| MV-MATH (strict completed-answer) | 22.65 | 1,429 / 2,009 | 71.13% | Not capability-comparable |

MathVista contains three 1,000-row prompt variants (`cot`, `solution`, and
`format`). Their truncation counts are 201, 198, and 245 respectively; the
reported score is the current summary value for the grouped task.

## VQAv2 status

VQAv2 is not complete. The final resumed run was reclaimed at approximately:

```text
91,264 / 214,354 requests (42.6%)
```

No final VQAv2 score is available. Because VQAv2 uses `max_new_tokens=128`
and TailSFT is verbose, the completed portion is expected to be highly
truncation-sensitive as well, but no final sample-level audit exists yet.

## Interpretation

1. Any TailSFT benchmark above roughly 50% truncation should be marked as
   **format-limited**, not as a direct capability result.
2. Even the 18–39% truncation benchmarks can be shifted by several points;
   they remain useful diagnostics but are not clean comparisons to Base or
   PTD-PO.
3. The severe TailSFT-specific pattern is confirmed by the two replay-based
   diagnostics: ReMI has 1,365 truncated rows for TailSFT versus 7 for Base and
   18 for PTD-PO; MV-MATH has 1,429 versus 952 for Base and 1,104 for PTD-PO.
4. Before re-running TailSFT, the evaluation prompt/output contract needs an
   explicit answer-only or final-answer-marker instruction, and either a larger
   cap or a task-appropriate cap. Re-running the same protocol would mostly
   reproduce the truncation artifact.

## Format-pilot protocol

`benchmark_suite.py` supports an environment-only override for a controlled
pilot:

```text
SFT_RL_SYSTEM_INSTRUCTION
```

When set, the suite switches lmms-eval from the sync `openai` backend to
`async_openai`, passes the prompt as `system_prompt`, and enables Qwen3-VL
message formatting. The earlier `--apply_chat_template` path is not used: it is
incompatible with multimodal chat tasks in this lmms-eval runtime and can fail
before any request reaches vLLM.

The suite also treats an lmms-eval process that exits 0 without writing result
JSON as failed, because this runtime can catch task-construction exceptions and
return success.

The pilot is not an official protocol; it is a format-alignment diagnostic. If
it removes truncation, Base and PTD-PO must be rerun under the same prompt
contract before any cross-arm conclusion is drawn.

## Source paths

All raw outputs and sidecars remain outside Git. The primary TailSFT Project15
run roots are:

```text
${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline/project15_tailsft_mmf122k_1ep_nojudge
${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline/project15_tailsft_mmf122k_1ep_offline_core_nojudge
${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline/project15_tailsft_mmf122k_1ep_judged
${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline/project15_tailsft_mmf122k_1ep_offline_judged
```
