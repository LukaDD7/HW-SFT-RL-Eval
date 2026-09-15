# Project15 offline model-identity audit — 2026-09-15

## Invalid run

The following Project15 offline run is not a valid PTD-PO arm:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_ptdpo_r4_step390_offline_core_nojudge
```

Its manifest records:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/third_party/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65
```

The vLLM server log for the same run confirms that this checkpoint was actually
loaded. The OpenAI model alias and output directory nevertheless used the
PTD-PO step390 name. This is an actual model mismatch, not merely a manifest
typo.

## Affected reported scores

These numbers came from Vision-OPD gs65 and must not be reported as PTD-PO:

| Benchmark | Base | Mislabelled “PTD-PO” | Gap |
|---|---:|---:|---:|
| ScienceQA | 0.9465 | 0.6946 | -25.19 points |
| BLINK | 0.6512 | 0.1394 | -51.18 points |

The same invalid run also contains MindCube and MMSI-Bench results.

ScienceQA is independently contradicted by the valid MMF-14 protocol, where
Base scored 93.37 and PTD-PO step390 scored 93.42. BLINK has no valid PTD-PO
Project15 offline result yet and must be rerun.

## Cause

The migrated Project15 launcher still called the legacy `Dual-Track-OPD`
runner and config. A direct or partially specified run could therefore let the
suite resolve its default Vision-OPD gs65 checkpoint while the run/served name
remained PTD-PO. Output names and OpenAI aliases do not prove which local
checkpoint was served.

## Fixes

1. The Project15 launcher now resolves the runner and config from
   `HW_EVAL_ROOT`/this repository instead of the legacy training repository.
2. `benchmark_suite` canonicalizes the checkpoint path before execution.
3. Known run-name identities are checked against their expected checkpoint
   tokens. A run named `ptdpo_r4_step390` cannot use Vision-OPD gs65.
4. Resume requires the exact canonical checkpoint recorded in the prior
   manifest.

Use a new run and cache directory for the corrected PTD-PO Project15 offline
results. Do not resume or merge the invalid run.
