# B6 think avg@4 results — 2026-09-29

This report records the completed B6-mixed `think` comparison and the
per-repeat resume fix used for recovery after GPU-instance reclamation.

Raw model outputs, response caches, model weights, and datasets remain outside
Git. The compact metric record is in `results/20260929/`.

## Protocol

- Protocol: B6-mixed, explicit `think` mode
- Aggregation: avg@4
- Decoding: temperature 1.0, seeds 42–45
- Eval repo commit for resumed runs and future MMBench execution:
  `83bdcbb97a9a1f83a5b0219846a742cc143fbbbc`
- Raw output root:
  `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/hweval_pr5_budget`

## MMBench implementation marker

The current repo state is MMBench-fixed. `83bdcbb` is the merge of
PR #12 (`codex/mmbench-patch-reliability`) and carries
`patches/lmms_eval_mmbench_tag_first.patch`. The launcher applies this patch to
the pinned external lmms-eval runtime, so MMBench runs launched from this
commit use terminal tag-first extraction instead of scanning the full
reasoning trace for an option letter.

SFT6939 and GRPO50 have not reached MMBench yet. Their MMBench results, when
run, will use this fixed implementation.

The already-complete 8B direct-RL arm records an earlier repo commit
(`5242d9ddeec2fb6e9c0ee6241301ca0cc949bbaf`) in its MMBench run manifest and
predates the PR #12 merge in that manifest. Its MMBench score is therefore
retained as a legacy result, not relabeled as PR #12-fixed without a runtime
patch audit.

## Current scores

MMBench is reported on its native 0–100 scale in the table; the macro uses
MMBench divided by 100. Pending values are not imputed.

For the two completed arms, the reported MMBench value is the formal PR #12
offline rescore: existing model generations are retained, terminal answer tags
are parsed directly, and unresolved rows are resolved with the Qwen3-VL-32B
LLM judge. Do not mix this value with the legacy score from the pre-PR #12
evaluator.

| Arm | Status | GQA | DynaMath | ViewSpatial | MMMU-Pro | ReMI exact | MMBench | Six-benchmark macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base 8B Instruct | complete, PR #12 official MMBench rescore | 0.549432 | 0.663723 | 0.447041 | 0.551012 | 0.476538 | 84.664948 | 0.589058 |
| 8B direct RL step100 | complete, PR #12 official MMBench rescore | 0.544482 | 0.669112 | 0.446166 | 0.551301 | 0.486058 | 83.977663 | 0.589483 |
| SFT6939 | complete | 0.466549 | 0.671856 | 0.386423 | 0.475434 | 0.395769 | 68.298969 | 0.513170 |
| SFT6939→GRPO50 | complete | 0.467662 | 0.666168 | 0.386992 | 0.471387 | 0.393750 | 68.277491 | 0.511456 |

The earlier PR #12 audit-only bounds are superseded by the official LLM-judge
rescores above:

| Arm | Lower bound | Upper bound | Unresolved rows |
|---|---:|---:|---:|
| Arm | Official PR #12 rescore |
|---|---:|
| Base 8B Instruct | 84.664948 |
| 8B direct RL step100 | 83.977663 |



## Recovery state

The per-repeat resume change skips a repeat when a complete result file
already exists and reruns only missing repeats. This avoids recomputing
completed avg@4 repeats after an instance is reclaimed.

All B6 arms in the table are complete. The per-repeat resume boundary remains
relevant for future interrupted runs.
