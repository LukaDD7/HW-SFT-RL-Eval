# Scripts

## B-segment v2

```bash
EVAL_CKPT=/path/to/model \
EVAL_RUN_NAME=my_run \
HW_EVAL_ENV=/path/to/env \
bash scripts/eval/run_target_benchmarks_v2.sh
```

Runs GQA, DynaMath, ViewSpatial, and MMMU-Pro with 4096-token decoding and
deterministic answer extraction.

## Project15 / v1

```bash
EVAL_CKPT=/path/to/model \
EVAL_RUN_NAME=my_run \
HW_EVAL_ENV=/path/to/env \
SFT_RL_BENCHMARKS=viewspatial,gqa,dynamath,mmmu_pro,remi \
SFT_RL_JUDGE_BENCHMARKS= \
bash scripts/eval/run_target_benchmarks.sh
```

For judged tasks, also set `SFT_RL_JUDGE_HF` and GPU/port variables.

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
