# Eval generation budget update (2026-09-21)

## Canonical protocol change

The canonical v1 suite now uses the following per-request output budgets. The
v2 B6 tasks already used 4096 tokens and are unchanged.

| Benchmark | Before | After |
|---|---:|---:|
| ViewSpatial (Project15 v1) | 256 | 1048 |
| MindCube | 256 | 1048 |
| ScienceQA | 256 | 1048 |
| ReMI | 2048 | 8192 |
| BLINK | 1024 | 2048 |
| MMBench | 1024 | 2048 |
| MMMU-Pro (Project15 v1) | 2048 | 4096 |
| GQA/DynaMath/ViewSpatial/MMMU-Pro (B6 v2) | 4096 | 4096 |

GQA and VQAv2 remain at 128 tokens because the user change request explicitly
covered the 256-token tasks, not the 128-token tasks. Historical archived
aligned/offline configs are not rewritten.

The value 1048 is intentional in this protocol update. If the intended value
was the more conventional 1024, it must be changed before merging.

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
