# 20260915 complete metric results

This directory stores the complete small metric artifacts for the evaluations
documented in `docs/eval_results_20260915.md`.

## Contents

- `ptdpo_r4_step390_project15_offline/`
  - Corrected Project15 offline run for PTD-PO r4 step390.
  - `summary.json`
  - Per-benchmark metric JSON: MindCube, ScienceQA, MMSI-Bench, BLINK.
- `qwen3vl_2b_base_target6/`
  - Qwen3-VL-2B Base six-benchmark results.
  - `summary.json`
  - Per-benchmark metric JSON: GQA, DynaMath, ViewSpatial, MMMU-Pro, MMBench.
- `qwen3vl_2b_va_opd_step175_target6/`
  - VA-OPD step175 six-benchmark results.
  - `summary.json`
  - Per-benchmark metric JSON: GQA, DynaMath, ViewSpatial, MMMU-Pro, MMBench.
- `qwen3vl_2b_va_opd_step300_target6/`
  - VA-OPD step300 six-benchmark results.
  - `summary.json`
  - Per-benchmark metric JSON: GQA, DynaMath, ViewSpatial, MMMU-Pro, MMBench.
- `remi_exact.json`
  - Strict full-denominator ReMI exact results and per-task breakdowns for
    Base, step175, and step300.

## Exclusions

Raw sample JSONL, response caches, model weights, and dataset files are not
included. They remain in the external evaluation run roots documented in the
report.
