# Shared acceptance contract for T1–T5

This contract applies to every training-paradigm Issue. The owner executes in
the training repository; HWEval stores only compact provenance, results, and
links. Do not attach raw logs, JSONL, datasets, checkpoints, or secrets to Issues
or Git.

## 1. Claim and pre-register before compute

1. Assign the matching T1–T5 Issue to yourself according to the meeting split.
   Do not create a duplicate Issue for the same paradigm.
2. Comment with your branch, planned start time, and independent reviewer. Use
   `docs/midterm-t<N>-<owner>` based on `docs/midterm-report-202611`.
3. Copy `../report_template.md` to `t<N>_<topic>_<yyyymmdd>.md` and link the
   draft from the Issue. Fill the plan tables before training.
4. For every slash-separated name, explicitly state whether it means separate
   arms, alternative algorithms, or sequential stages. Do not interpret
   `RLSD/RLRT` or `ISO/MOPD` as one run or two runs without recording the
   decision. Give every concrete arm a unique `arm_id`.
5. Record the parent checkpoint and input/output checkpoint for each stage,
   which checkpoints are reused, and which stages are retrained. Reuse requires
   a verifiable immutable checkpoint and compatible data/config provenance; it
   is not just a matching directory name.
6. Post the resolved data/config references, training-volume calculation,
   hardware allocation, and smoke plan in the Issue before the full run. Ask the
   reviewer to check this plan before spending the full training budget.

## 2. Verify actual data and training volume

Follow `../training_alignment.md` for the matched numerical budget, pairwise
data/initialization requirements, and comparability limits. For T1/T5, use the
linked aligned recipe rather than the superseded short-run drafts. Compare every
resolved config against that contract and record any intentional algorithm or
input difference. Keep global rollout steps and actual optimizer updates
separate; do not treat equal prompt/trajectory totals as equal epochs, equal
end-to-end SFT/MOPD compute, or proof of a controlled result.

For each concrete arm and stage, record:

- Hugging Face dataset/model ID and full immutable revision, or the equivalent
  immutable local manifest; do not record only `main` or a mutable alias.
- Resolved data config, manifest SHA-256, original/filtered row counts, sampling
  ratio or mixture weight, and the actual filter/sampling behavior.
- Whether each new 15k dataset was actually used. Neither is a default input for
  all five tasks. MOPD-route-v2 and RL-RSTG-ZVP have independent revisions and
  must not be conflated.
- Planned/actual epochs, unique prompts or examples consumed, rollouts per
  prompt where applicable, world size, per-device batch, gradient accumulation,
  effective global batch, and expected/actual optimizer steps.
- The framework's exact batch and step formula, including `drop_last`, sampler,
  filtering, resume, early stop, and rollout-group behavior. Reconcile the
  calculation with logs; do not infer training volume from a requested step
  count alone.
- Code commit and dirty patch, backend commit/version, complete resolved config,
  seed, hardware, final checkpoint, and external training-log locations.

The two candidate new-data revisions and their verified row/shard counts are in
`../README.md`. New-data use must still be established from each arm's resolved
config and logs.

## 3. Train and check the final artifact

- Run a training smoke before the full run. Check the configured algorithm,
  data order/mixture, masks/reductions where relevant, output dimensions,
  checkpoint identity, and gradient/optimizer behavior as applicable. A finite
  loss or successful launch is not sufficient verification.
- Record smoke, full-run, and checkpoint-export status separately. Preserve
  failures and invalid runs with their evidence; never replace them silently.
- Verify the final checkpoint against the intended arm, stage, actual optimizer
  step, code/config, and data revisions. Give every evaluated checkpoint an
  immutable external location or content-addressed identity.
- If an existing result or checkpoint is reused, record the original run and
  independently verify its provenance instead of rerunning solely for formality.

## 4. Run smoke and then the seven-task B6

Use the canonical HWEval entrypoint at the reviewed commit:

```bash
EVAL_CKPT=/path/to/immutable/checkpoint \
EVAL_RUN_NAME=<arm_id>_<mode>_midterm7_avg4 \
HW_EVAL_ENV=/path/to/env \
DTOPD_EVAL_ROOT=/path/to/external/results \
bash scripts/eval/run_pinned_eval.sh --protocol b6-mixed --think-mode auto
```

Use `EVAL_SMOKE=1` with a distinct smoke output root or run name for the
seven-task smoke first. Inspect real outputs and judge/scorer logs; only then
start full evaluation. Do not use smoke metrics in the report's result table.
Keep prompt mode, checkpoint, data assets, judge, and scoring configuration
consistent across a comparison cohort. Any deviation requires a new run name
and an explicit comparability note.

Required full B6 set and output locations:

| Group | Benchmarks | Max new tokens | Scoring |
| --- | --- | --- | --- |
| v2 merged `${EVAL_RUN_NAME}` | GQA, MMMU-Pro | 8192, 16384 | deterministic tag/direct or tag-first |
| v1 no-judge `${EVAL_RUN_NAME}_nojudge` | ReMI, ScienceQA, MindCube | 16384, 1024, 1024 | ReMI strict exact; deterministic MCQ |
| v1 judged `${EVAL_RUN_NAME}_judged` | MMBench, MathVerse | 8192, 4096 | judged choice extraction; LLM judge |

Every benchmark uses four independent generations at temperature 1.0 with seeds
42–45 and reports the mean (avg@4), not pass@4. ReMI requires `--mode exact`,
four complete replays, and 2,600 rows in every repeat. DynaMath and ViewSpatial
are not midterm B6 rows; historical six-task aggregates are not substitutes.
The independent v2 four-task protocol remains available but is not this B6.

For every benchmark, provide four per-repeat metrics or a linked complete table,
the mean, each repeat's completed/expected samples, truncation count/rate, judge
or scorer identity, raw-output links, and a validity assessment. Record server
prompt checks where applicable. A zero exit code without complete result files
is a failure. High truncation, missing samples, judge failures, or changed task
versions must be marked invalid or not capability-comparable, with rationale.

## 5. Report, review, and close

- Complete `../report_template.md` for each concrete arm and link all three B6
  output roots, ReMI aggregate, final checkpoint, resolved configs, and logs.
- Include negative, blocked, incomplete, and invalid outcomes. Do not silently
  drop them or average incompatible historical and current protocols.
- Open a PR targeting `docs/midterm-report-202611`; link it in the Issue.
- An independent reviewer checks data/training-volume reconciliation, config and
  checkpoint identity, all seven results, truncations/completeness, and external
  evidence. The owner cannot be the sole approver.
- Close the Issue only after the report PR is merged and the reviewer records
  acceptance. If the team accepts a blocked/incomplete endpoint, document that
  decision, remaining work, and owner explicitly rather than calling it done.
