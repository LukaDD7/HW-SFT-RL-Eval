# Evaluation results and model-identity correction — 2026-09-15

This report records two result sets:

1. The corrected PTD-PO r4 step390 Project15 offline run.
2. The Qwen3-VL-2B Base versus VA-OPD step175/step300 six-benchmark comparison.

All raw outputs, response caches, model weights, and benchmark datasets remain
outside Git.

Complete per-benchmark metric JSON files and the strict ReMI exact per-task
breakdown are stored under `results/20260915/`.

## Common protocol

- lmms-eval: `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`
- vLLM: 0.18.0
- Decoding: temperature 0, seed 42
- Judge: Qwen3-VL-32B-Instruct-FP8 for MMBench

Project15 offline tasks use local parquet snapshots under
`${DTOPD_ROOT}/dataset`. The six-benchmark protocol uses the existing
Project15/B-segment task definitions and ReMI replay source.

## PTD-PO Project15 offline correction

### Invalid prior run

The previous PTD-PO-labelled Project15 offline run was invalid:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_ptdpo_r4_step390_offline_core_nojudge
```

Its manifest and vLLM server log both show that
`Vision-OPD-Qwen3.5-4B/global_step_65` was loaded, although the output and
OpenAI alias were labelled PTD-PO r4 step390. Those scores must not be cited as
PTD-PO.

### Corrected run

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_ptdpo_r4_step390_offline_core_corrected_r2_gpu2_nojudge
```

Checkpoint:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_ptdpo_r4_step390
```

The vLLM log confirms this checkpoint was loaded. The run completed all four
benchmarks with return code 0.

| Benchmark | Base | Invalid prior “PTD-PO” | Corrected PTD-PO | Corrected Δ vs Base |
|---|---:|---:|---:|---:|
| MindCube | 0.29749 | 0.29957 | **0.30259** | +0.00511 |
| ScienceQA | 0.94646 | 0.69460 | **0.93852** | -0.00793 |
| MMSI-Bench | 0.30700 | 0.29800 | **0.31300** | +0.00600 |
| BLINK | 0.65124 | 0.13940 | **0.65018** | -0.00105 |

Four-task macro average:

| Arm | Macro |
|---|---:|
| Base | 0.55054 |
| Corrected PTD-PO | **0.55107** |
| Δ | +0.00053 |

The apparent 25.19-point ScienceQA drop and 51.18-point BLINK drop in the
invalid run were therefore model-identity artifacts, not PTD-PO capability
changes.

Corrected-run sample counts and generation-cap rates:

| Benchmark | Samples | At cap | Rate |
|---|---:|---:|---:|
| MindCube | 21,154 | 0 | 0.00% |
| ScienceQA | 2,017 | 6 | 0.30% |
| MMSI-Bench | 1,000 | 18 | 1.80% |
| BLINK | 1,901 | 0 | 0.00% |

## Qwen3-VL-2B Base versus VA-OPD checkpoints

Checkpoints:

| Arm | Path |
|---|---|
| Base | `${DTOPD_ROOT}/models/Qwen3-VL-2B-Instruct` |
| VA-OPD step175 | `${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_va_opd_step175` |
| VA-OPD step300 | `${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_va_opd_step300` |

Run roots:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/qwen3vl_2b_base_target6_nojudge
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/qwen3vl_2b_base_target6_judged
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/qwen3vl_2b_va_opd_step175_target6_nojudge
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/qwen3vl_2b_va_opd_step175_target6_judged
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/qwen3vl_2b_va_opd_step300_target6_nojudge
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/qwen3vl_2b_va_opd_step300_target6_judged
```

ReMI uses the strict full-denominator exact protocol:

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <run>/replay/remi.jsonl \
  --label-jsonl /inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

### Scores

| Benchmark | Base | VA-OPD step175 | VA-OPD step300 |
|---|---:|---:|---:|
| GQA | 0.59262 | 0.59350 | 0.59350 |
| DynaMath | 0.49202 | 0.49621 | 0.50339 |
| ViewSpatial | 0.36800 | 0.36940 | 0.36590 |
| MMMU-Pro | 0.27399 | 0.26647 | 0.25260 |
| ReMI exact | 0.25731 | 0.26654 | 0.28731 |
| MMBench | 0.76117 | 0.76460 | 0.77062 |
| Six-benchmark macro | 0.33192 | 0.33329 | **0.33507** |

Relative to Base:

- step175 macro: **+0.00137**
- step300 macro: **+0.00314**

Step300 relative to Base:

- ReMI: **+3.00 points**
- DynaMath: **+1.14 points**
- MMBench: **+0.95 points**
- GQA: **+0.09 points**
- ViewSpatial: **-0.21 points**
- MMMU-Pro: **-2.14 points**

The main trend is improved ReMI and DynaMath, offset by a larger MMMU-Pro
decline. Step300 is an interim checkpoint, not the final training result.

### Sample counts and truncation

| Benchmark | Samples | Base at cap | Step175 at cap | Step300 at cap |
|---|---:|---:|---:|---:|
| GQA | 12,578 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| DynaMath | 5,010 | 846 (16.89%) | 974 (19.44%) | 1,032 (20.60%) |
| ViewSpatial | 5,712 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| MMMU-Pro | 1,730 | 596 (34.45%) | 666 (38.50%) | 671 (38.79%) |
| ReMI | 2,600 | 103 (3.96%) | 140 (5.38%) | 163 (6.27%) |
| MMBench | 4,329 | 32 (0.74%) | 80 (1.85%) | 96 (2.22%) |

MMMU-Pro has the highest cap-hit rate and should be interpreted with the v1
short-output contract. DynaMath and ReMI are also materially affected by
length truncation.

## Training status at report time

The VA-OPD paper-primary Geometry3K run uses 2,101 training prompts for five
epochs, or 655 optimizer steps. At the time of this report, the resumed run had
passed step 375 and saved `global_step_375`; final checkpoint selection remains
pending until the 655-step run completes.
