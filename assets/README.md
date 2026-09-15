# External asset policy

Do not store assets in Git.

## Hugging Face layout

### Models

Use one model repository per checkpoint, for example:

```text
<org>/qwen3-vl-8b-base
<org>/qwen3-vl-8b-tailsft
<org>/qwen3-vl-8b-ptdpo
<org>/qwen3-vl-32b-instruct-fp8-judge
```

Each model card must record:

- source training repository and commit
- base model
 intended evaluation role
- known output-format characteristics
- reproducibility manifest hash, if available

### Evaluation assets

Use a dataset repository such as:

```text
<org>/hw-sft-rl-eval-assets
```

Suggested layout:

```text
benchmarks/
  gqa/
  dynamath/
  viewspatial/
  mmmu_pro/
  project15_offline/
remi_replay/
  raw_responses/
  parquet/
mv_math/
```

### Results

Use a dataset repository such as:

```text
<org>/hw-sft-rl-eval-results
```

Store compact result JSON, summaries, and audit tables. Do not upload large
raw JSONL unless a release explicitly requires it.
