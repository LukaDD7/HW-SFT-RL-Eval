# Midterm training and B6 report — `<arm_id>`

## Ownership and status

| Field | Value |
| --- | --- |
| Task / concrete arm | T`<N>` / `<arm_id>` |
| Paradigm and stage sequence | `<resolved notation; do not leave slash semantics ambiguous>` |
| Matched comparison pair / comparability limits | `<training_alignment.md pair; intentional differences and invalid comparisons>` |
| Issue / report PR | `<URL>` / `<URL>` |
| Owner / independent reviewer | `<name>` / `<name>` |
| Status | `planned / smoke / running / evaluating / review / completed / blocked / invalid` |
| Report date / last updated | `<date>` / `<date>` |
| Summary conclusion | `<result plus limitations; no unsupported claim>` |

## Arm decomposition and reuse

| Arm ID | Stage | Objective/algorithm | Parent checkpoint | New training or reuse | Output checkpoint | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `<arm_id>` | `<stage>` | `<objective>` | `<immutable ID/revision>` | `<new/reuse>` | `<immutable ID/revision>` | `<reason>` |

For `RLSD/RLRT` and `ISO/MOPD`, document whether each slash means separate arms,
alternatives, or sequential stages. Provide one report per concrete arm or an
explicit per-arm result table; do not combine incompatible runs.

## Training provenance

| Field | Value |
| --- | --- |
| Training repository commit and dirty patch | `<commit; external patch/diff link>` |
| HWEval / backend commit or package version | `<commit/version>` |
| Complete resolved config | `<external immutable link>` |
| Matched-recipe config diff | `<comparison with training_alignment.md recipe; only declared algorithm/input differences>` |
| Base model and every reused checkpoint | `<IDs and full revisions>` |
| Seed(s) | `<values>` |
| Hardware / world size / wall time | `<GPU type, count, hours>` |
| Training and export commands | `<exact commands or immutable script links>` |
| Training / smoke / export logs | `<external links>` |
| Final checkpoint and export verification | `<location, step, hash/revision, checked by>` |

## Data and training-volume audit

Complete one table per training stage. For RL, distinguish unique prompts from
prompt × rollout-group samples. Record the framework formula rather than assuming
that every framework defines batch and optimizer steps identically.

| Data/config reference | Full revision + manifest SHA-256 | Raw rows | Used/filtered rows | Mixture/sampling | Planned/actual epochs | Consumed prompts/examples |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `<dataset and resolved config>` | `<immutable identity>` | | | | | |

| Stage | Rollouts/prompt (RL only) | Planned/actual global rollout steps | World size | Per-device batch | Grad accumulation | Effective global batch + formula | Expected/actual optimizer updates | Evidence |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| `<stage>` | | | | | | | | |

- Batch/step accounting, sampler, `drop_last`, filtering, resume/early-stop and zero-advantage skip effects:
  `<calculation reconciled with logs>`
- MOPD-route-v2 used? `<no / yes: stage, revision, rows, mixture>`
- RL-RSTG-ZVP-train-15k used? `<no / yes: stage, revision, rows, mixture>`
- Reused-data/checkpoint verification and original run links: `<evidence>`
- Smoke and full-run observations: `<behavior checked, not only finite loss>`
- Deviations, failed runs, and unresolved questions: `<explicit record>`

## Evaluation protocol

| Field | Value |
| --- | --- |
| Evaluated checkpoint identity | `<matches final checkpoint above>` |
| HWEval commit / config / `lmms-eval` commit | `<values>` |
| Seven-task smoke command and evidence | `<command, external outputs, validity>` |
| Full B6 command / run name / prompt mode | `<command; unique midterm7 name; mode>` |
| Seeds / temperature / aggregation | `42,43,44,45 / 1.0 / mean avg@4` |
| Judge and scorer identities | `<judge model/version, MMBench patch, ReMI exact>` |
| v2 merged raw output | `<external ${EVAL_RUN_NAME} link>` |
| v1 no-judge raw output + `remi_avg4.json` | `<external ${EVAL_RUN_NAME}_nojudge links>` |
| v1 judged raw output | `<external ${EVAL_RUN_NAME}_judged link>` |
| Server prompt checks / resolved eval configs | `<external links>` |

## Seven-task B6 results

All rows are required. Link a per-repeat table if it is too wide for this report.
`Samples` means completed/expected for every repeat, not just one repeat.
Truncation count and rate are required for every repeat. ReMI must show 2,600
rows per repeat and use strict exact full-denominator scoring.

| Benchmark | Four repeat scores | Avg@4 | Samples per repeat | Truncation n/rate per repeat | Judge/scorer | Raw output | Validity |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| MMMU-Pro | | | | | | | |
| MMBench | | | | | | | |
| ReMI | | | | | | | |
| MathVerse | | | | | | | |
| GQA | | | | | | | |
| ScienceQA | | | | | | | |
| MindCube | | | | | | | |

- Smoke results are excluded from this table: `<confirmed / exception>`
- Task versions and generation budgets checked against B6: `<evidence>`
- Invalid, incomplete, high-truncation, or judge-failed rows: `<status and rationale>`
- Historical or reused results and exact compatibility checks: `<links; no silent relabeling>`
- Comparison with the other midterm arms / baseline: `<per-task comparison and limitations>`
- Composite averages, if any: `<predeclared formula and exclusions; do not mix incompatible protocols>`

## Independent review and handoff

- [ ] Owner's data, global-rollout-step, and optimizer-update calculations reconcile with resolved configs and logs.
- [ ] Reviewer confirmed the matched pair, resolved-config equality/intentional differences and limitations, every parent/final checkpoint, code/backend revision, seed, and reuse decision.
- [ ] All seven full B6 results have four repeats, complete denominators, truncation evidence, and valid scoring.
- [ ] ReMI is strict exact with 2,600 rows per repeat; judged rows have judge provenance.
- [ ] Negative/blocked/invalid outcomes and comparability limits are retained.
- [ ] External artifact links are accessible; no raw artifacts or secrets were committed.
- [ ] Consolidated midterm conclusions accurately reflect the evidence.

Reviewer sign-off: `<name, date, PR review URL>`

Remaining work / accepted blocker decision: `<owner, due date, explicit team decision if applicable>`
