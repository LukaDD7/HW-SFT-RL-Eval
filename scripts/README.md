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
`project15` runs the full 15-task Project15 v1/offline suite.

The underlying `run_target_benchmarks*.sh` scripts are internal implementation
details. Do not call them directly for new evaluations.

## ReMI strict rescoring

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <run>/replay/remi.jsonl \
  --label-jsonl <assets>/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

## Dataset download

Run only on a networked machine:

```bash
HF_HOME=/path/to/hf_cache \
PYTHON=/path/to/python \
bash scripts/sft_rl/download_bench_datasets.sh
```

GPU instances should then set `HF_HUB_OFFLINE=1`.
