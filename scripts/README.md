# Scripts

## Canonical pinned evaluation entrypoint

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

`b6-mixed` runs v2 GQA/DynaMath/ViewSpatial/MMMU-Pro and v1 ReMI/MMBench.
`project15` runs the full 15-task Project15 v1/offline suite. Both protocols
run four independent sampled generations at temperature 1.0 and report avg@4.

The v1 wrapper applies the repository-owned MMBench tag-first patch to the
pinned external `lmms-eval` checkout before starting the judge stage.

The underlying `run_target_benchmarks*.sh` scripts are internal implementation
details. Do not call them directly for new evaluations.

## ReMI strict rescoring

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <run>/replay/remi.jsonl \
  --label-jsonl <assets>/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

The canonical entrypoint performs this automatically for the four
`remi.repeat_*.jsonl` files and writes `<run>_nojudge/remi_avg4.json`.
It also scores the four MV-MATH repeats with the official-compatible judge and
strict completed-answer gate, then writes `<run>_nojudge/mv_math_avg4.json`.

## Dataset download

Run only on a networked machine:

```bash
HF_HOME=/path/to/hf_cache \
PYTHON=/path/to/python \
bash scripts/sft_rl/download_bench_datasets.sh
```

GPU instances should then set `HF_HUB_OFFLINE=1`.
