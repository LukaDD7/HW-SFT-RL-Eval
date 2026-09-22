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
| B6-mixed avg@4 | v2 GQA/DynaMath/ViewSpatial/MMMU-Pro + v1 ReMI/MMBench, four sampled generations | `scripts/eval/run_pinned_eval.sh --protocol b6-mixed` |
| Project15 avg@4 | Native `lmms-eval` Project15 tasks, four sampled generations | `scripts/eval/run_pinned_eval.sh --protocol project15` |
| Project15 complement avg@4 | Run only the nine Project15 tasks not covered by B6-mixed; use after an existing B6-mixed run to avoid duplicate generation | `scripts/eval/run_pinned_eval.sh --protocol project15-complement` |
| ReMI strict | Full 2,600-row denominator, task-aware extraction | `scripts/sft_rl/remi_reeval.py --mode exact` |
| MV-MATH strict | Completed-answer gate over judge sidecar | `python -m dual_track_opd.eval.score_mv_math --mode strict` |

Use `scripts/eval/run_pinned_eval.sh` as the only user-facing evaluation
entrypoint. The underlying v1/v2 runners remain internal implementation
details.

Both canonical protocols use `repeat_count=4` and `sampling_temperature=1.0`.
The reported primary value is the mean of four independent repeat metrics. This
is an engineering avg@4 protocol, not pass@4.

`project15-complement` is an explicit compute-saving protocol. It runs the nine
Project15 tasks that are not in B6-mixed. Use the same checkpoint and protocol
generation as the existing B6-mixed run, then combine the two result sets in a
report. It does not silently import or rewrite an old B6 run; the user must
select the matching B6 run and keep task-version/scoring labels explicit.

### Optional B6 thinking prompt

The B6 adapter supports three modes without changing the upstream scorers or
avg@4 aggregation:

| Mode | Prompt |
|---|---|
| `auto` (default) | Native benchmark prompts; reasoning may occur naturally. |
| `think` | Open-MOPD training instruction and assistant `<think>\n` prefill. |
| `no-think` | Explicit direct-answer instruction, no thinking prefill. |

Output budgets are identical in all three modes and come from the benchmark
configuration, with no Think-specific override:

| Benchmarks | `max_new_tokens` |
|---|---:|
| MMMU-Pro, DynaMath, ReMI | **16384** |
| MMBench, ViewSpatial, GQA | **8192** |

The OpenAI adapter preserves these values in the actual API payload after the
pinned lmms backend's historical 4096-token clamp, including in `auto` mode.
The total server context remains 65536 tokens, including image/prompt tokens.

With the environment and asset paths configured as described in
[environment/KEY_VERSIONS.md](environment/KEY_VERSIONS.md), select a mode through
the canonical entrypoint:

```bash
EVAL_CKPT=/path/to/model EVAL_RUN_NAME=my_model_think_avg4 \
  bash scripts/eval/run_pinned_eval.sh --protocol b6-mixed --think-mode think
```

`EVAL_THINK_MODE` (or `SFT_RL_THINK_MODE`) provides the same selection when no
CLI mode is specified. All modes retain four repeats at temperature 1.0 with
configured seeds 42, 43, 44, and 45, followed by the upstream mean aggregation.
Use new run names for the shared budgets; do not resume old single-generation
or differently budgeted outputs. Manifests record resolved budgets and replay
rows record generation settings; resume rejects mismatches or missing settings.

Think requests use this exact system instruction on all six benchmarks:

```text
Explain your reasoning inside <think>...</think>, then give the final answer inside <answer>...</answer>. Any request in the question for a short or direct response applies to the final answer only. Keep the reasoning proportional to the question; simple questions need only brief reasoning.
```

DynaMath additionally requests `Put the final mathematical result in \boxed{}
inside the answer section.` Known direct-answer task footers are adjusted to
apply to the final answer only. Each explicit mode generates a separate server
chat template, verifies its rendered prompt before evaluation, and records the
prompt protocol for safe resume. Project15 retains native prompts (`auto`).

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
