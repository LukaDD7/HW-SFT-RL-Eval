# Evaluation environment contract

The canonical evaluated environment used by the migration source was:

| Component | Version / pin |
|---|---|
| Python | 3.12.13 |
| `lmms-eval` | `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56` |
| vLLM | 0.18.0 |
| Torch | 2.10.0 (CUDA 12.8 wheels) |
| Transformers | 5.5.0 |
| datasets | 5.0.0 |
| pyarrow | 22.0.0 |
| openai | 2.24.0 |
| flash-attn | 2.8.3 (optional for OpenAI-backend evaluation) |

The source runtime was an editable `lmms-eval` checkout with local fixes for
MMBench judge URL construction and empty response handling. Those fixes are
required for judged MMBench runs; they do not affect the deterministic v2 four
task protocol.

## Recommended install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[runtime,dev]"
python -m pip install -e path/to/local/lmms-eval
```

Use the `lmms-eval` commit above. If using a clean upstream checkout, apply the
same local MMBench fixes before running judge-dependent tasks.

## Environment variables

Use environment variables instead of absolute paths:

| Variable | Purpose |
|---|---|
| `HW_EVAL_ROOT` | This repository checkout |
| `HF_HOME` | Hugging Face cache |
| `HF_HUB_OFFLINE` | Set to `1` on networkless GPU instances |
| `DTOPD_ROOT` | Optional shared NFS root for legacy runners |
| `DTOPD_EVAL_ROOT` | Output root |
| `DTOPD_DATASET_ROOT` | Local benchmark/replay data root |
