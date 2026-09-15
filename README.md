# HW-SFT-RL Eval

Lightweight, reproducible evaluation stack for the HW SFT/RL project.

This repository contains only protocol definitions, scoring code, runner scripts,
tests, audits, and agent collaboration rules. Model checkpoints, benchmark
datasets, ReMI replay sources, and raw outputs are intentionally external and
are managed as Hugging Face assets.

## Scope

**Included**

- `dual_track_opd.eval` scheduler and scorers
- Project15 / B-segment v1, v1-aligned, offline, and v2 task configurations
- `lmms-eval` task overrides under `eval_tasks/`
- ReMI strict full-denominator rescoring
- MV-MATH official-compatible and strict completed-answer scoring
- Truncation and protocol audit documents
- Reproducibility tests and CI

**Not included**

- Model weights or checkpoints
- Dataset snapshots, parquet files, or ReMI replay JSONL
- Raw evaluation outputs, logs, caches, or tarballs
- Training code

## Install

Use Python 3.12.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The evaluation runtime also requires a pinned `lmms-eval` checkout. See
[environment/KEY_VERSIONS.md](environment/KEY_VERSIONS.md).

## Quick protocol map

| Protocol | Use case | Entry point |
|---|---|---|
| B-segment v2 | Four deterministic tasks: GQA, DynaMath, ViewSpatial, MMMU-Pro | `scripts/eval/run_target_benchmarks_v2.sh` |
| Project15 v1 / offline | Native `lmms-eval` Project15 tasks | `scripts/eval/run_target_benchmarks.sh` |
| ReMI strict | Full 2,600-row denominator, task-aware extraction | `scripts/sft_rl/remi_reeval.py --mode exact` |
| MV-MATH strict | Completed-answer gate over judge sidecar | `python -m dual_track_opd.eval.score_mv_math --mode strict` |

Detailed protocol rules are in [docs/remi_mv_math_protocol_20260913.md](docs/remi_mv_math_protocol_20260913.md)
and [docs/eval_protocol_v2_20260904.md](docs/eval_protocol_v2_20260904.md).

## External assets

Use the manifests under `manifests/` to bind this repository to model and data
assets. Keep large artifacts on Hugging Face, not in Git.

Recommended layout:

- Models: one Hugging Face model repository per checkpoint
- Data/replay: one Hugging Face dataset repository for eval assets
- Results: one Hugging Face dataset repository per result release

See [assets/README.md](assets/README.md).

## Agent collaboration

Agents working in this repository must follow [AGENTS.md](AGENTS.md).
Long-running tasks must use the persistent planning skill under
`.agents/skills/planning-with-files/` and the evaluation-specific skill under
`.agents/skills/hw-eval-operations/`.

## Safety

Never commit secrets, tokens, `.env` files, model weights, datasets, raw
benchmark outputs, caches, or generated tarballs.
