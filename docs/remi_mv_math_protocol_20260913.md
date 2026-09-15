# ReMI and MV-MATH Scoring Protocol Audit

Date: 2026-09-13

## Conclusion

Both benchmarks should remain **internal diagnostics** in the Project15 v1 table.
They are useful signal, but they are not pinned `lmms-eval v0.7.1` community
tasks and must not use the old lenient `normalized_exact_diagnostic` number.

## MV-MATH

The official MV-MATH repository uses an LLM equivalence judge, not a pure exact
match:

* choice (1,109) and single-step (800): DeepSeek-Chat returns `true`/`false`;
* multi-step (100): DeepSeek-Chat returns `correct_steps/total_steps`;
* the headline score counts choice+single-step true rows plus multi-step rows
  where all steps are correct, divided by 2,009;
* multi-step also reports SAR (Step Accuracy Rate) and QCR (Question
  Completeness Rate).

We added `dual_track_opd.eval.score_mv_math`, which implements that protocol
against any OpenAI-compatible judge (default: the existing local
`Qwen3-VL-32B-Instruct` endpoint). It supports a resumable per-row sidecar, so
an interruption does not restart the judge drain.

Official-compatible command:

```bash
python -m dual_track_opd.eval.score_mv_math \
  --mode official \
  --replay-jsonl <project15_run>/replay/mv_math.jsonl \
  --metadata-json ${DTOPD_ROOT}/dataset/MV-MATH/MV-MATH.json \
  --judge-url http://127.0.0.1:8801/v1 \
  --judge-model Qwen3-VL-32B-Instruct \
  --judge-sidecar <project15_run>/replay/mv_math_official_judge.jsonl \
  --resume \
  --output-json <project15_run>/replay/mv_math_official_summary.json
```

When no judge is online, the new `--mode choice` gives a deterministic
full-denominator diagnostic for the 1,109 choice rows only. It is useful for
sanity checks but is not the official 2,009-row MV-MATH metric.

Current choice-only diagnostics from the completed raw replays:

| Arm | Correct / choice rows | Accuracy | Choice rows truncated at 4,096 |
|---|---:|---:|---:|
| Base | 569 / 1,109 | 51.31% | 451 |
| PTD-PO r4 step390 | 551 / 1,109 | 49.68% | 523 |
| TailSFT | 442 / 1,109 | 39.86% | 784 |

The high truncation rate means these numbers are completion-biased diagnostics,
not capability-equivalent scores. TailSFT is especially affected.

The official-compatible judge drain completed on 2026-09-13 using the local
Qwen3-VL-32B-Instruct endpoint. Official MV-MATH uses DeepSeek-Chat, so this is
protocol-compatible but not judge-model-identical.

| Arm | Official-compatible correct / rows | Weighted accuracy |
|---|---:|---:|
| Base | 1,176 / 2,009 | 58.54% |
| PTD-PO r4 step390 | 1,203 / 2,009 | 59.88% |
| TailSFT | 1,525 / 2,009 | 75.91% |

Finish-reason diagnostics from the judge sidecars:

| Arm | `length` rows | Correct among `length` | `stop` rows | Correct among `stop` |
|---|---:|---:|---:|---:|
| Base | 952 | 457 | 1,057 | 719 |
| PTD-PO r4 step390 | 1,104 | 559 | 905 | 644 |
| TailSFT | 1,429 | 1,070 | 580 | 455 |

The official-style judge prompt asks only whether the model's final answer
matches the standard answer. In a sample audit, a response with
`finish_reason=length` and no explicit final answer was still judged `true`.
Therefore these official-compatible numbers must remain diagnostics, especially
for models with high truncation rates; they are not a strict completed-answer
score and must not be mixed directly with the strict B-segment v2 protocol.

### Strict completed-answer correction

The Project15 reporting rule now treats a judge `true` on a truncated response
as invalid for the headline metric. `score_mv_math --mode strict` reuses the
completed judge sidecar and requires `finish_reason == "stop"` before a verdict
can count. Multi-step SAR is reported only over completed multi-step rows.

Strict results from the same 2,009-row replays and judge sidecars:

| Arm | Strict correct / rows | Strict weighted accuracy | Completion-gated rows |
|---|---:|---:|---:|
| Base | 719 / 2,009 | 35.79% | 952 |
| PTD-PO r4 step390 | 644 / 2,009 | 32.06% | 1,104 |
| TailSFT | 455 / 2,009 | 22.65% | 1,429 |

By answer type:

| Arm | Choice | Single-step | Multi-step |
|---|---:|---:|---:|
| Base | 447 / 1,109 | 250 / 800 | 22 / 100 |
| PTD-PO r4 step390 | 408 / 1,109 | 221 / 800 | 15 / 100 |
| TailSFT | 234 / 1,109 | 216 / 800 | 5 / 100 |

The strict results are the Project15 diagnostic rows to track. The
official-compatible table above is retained only to document the protocol
difference and judge behavior.

See `docs/tailsft_truncation_audit_20260913.md` for the full TailSFT
truncation-by-benchmark audit.

## ReMI

The ReMI paper reports:

* exact match for textual outputs after lowercase/spacing and a small set of
  documented postprocessing rules;
* relaxed numeric accuracy with 1% tolerance generally;
* 3% tolerance for GeomShapes and GeomCost;
* 10-minute tolerance for Clocks.

The paper prompted for JSON containing `explanation` and `answer`, with 512
output tokens. Our historical replay prompts are answer-only, so we do not claim
paper-prompt equivalence.

Use the existing honest sidecar rather than `summary.json`:

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <project15_run>/replay/remi.jsonl \
  --label-jsonl ${DTOPD_ROOT}/eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

This uses a full 2,600-row denominator. Its task-aware extraction and numeric
tolerances are closer to the paper than the old `normalized_exact_diagnostic`,
which counted only an extractable subset and inflated the score. LLM-judge mode
remains a diagnostic drain and is not the primary metric.

Current exact/relaxed full-denominator diagnostics from the completed raw
replays:

| Arm | Correct / rows | Full-denominator accuracy | Rows truncated |
|---|---:|---:|---:|
| Base | 726 / 2,600 | 27.92% | 7 |
| PTD-PO r4 step390 | 842 / 2,600 | 32.38% | 18 |
| TailSFT | 755 / 2,600 | 29.04% | 1,365 |

TailSFT's ReMI raw output is completion-biased because 1,365 rows hit the
2,048-token ceiling; its number is not capability-comparable.

These exact/full-denominator values were rerun on 2026-09-13 and are unchanged:
Base 726/2,600 (27.92%), PTD-PO 842/2,600 (32.38%), and TailSFT 755/2,600
(29.04%). ReMI remains task-aware exact/relaxed scoring with the full 2,600-row
denominator.

## Formal reporting rule

1. Project15 v1's 13 native `lmms-eval` benchmarks remain the formal v1 rows.
2. Report ReMI and MV-MATH only as clearly labeled diagnostics.
3. MV-MATH's Project15 diagnostic uses `score_mv_math --mode strict` over the
   completed LLM judge sidecar; do not use the choice-only diagnostic or the
   ungated official-compatible value for the primary row.
4. ReMI must use `remi_reeval.py --mode exact`, not the old
   `normalized_exact_diagnostic`.
