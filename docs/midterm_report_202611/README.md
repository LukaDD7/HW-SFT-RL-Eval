# 2026-11 midterm report workspace

Integration branch: `docs/midterm-report-202611`.

This workspace coordinates five training-paradigm reviews/retrains, their
seven-task B6 evaluations, and independent result review. It is not an artifact
store. Keep datasets, weights, raw outputs, logs, archives, and training code
outside this repository; link immutable external artifacts instead.

## Five claimable tasks

There is exactly one GitHub Issue per research task. Follow the meeting split:
assign yourself to the matching Issue, add a short claim comment with your
branch/start time/reviewer, and do not open a duplicate paradigm Issue. Use Issue
comments for pre-registration and status/evidence links, not large logs.

| Task | Paradigm | Contract | Issue | Owner / reviewer | Status |
| --- | --- | --- | --- | --- | --- |
| T1 | Base + GRPO | [Task](retraining/t1_base_grpo.md) | Pending browser submission | Unassigned / TBD | Not started |
| T2 | Base + RLSD/RLRT | [Task](retraining/t2_base_rlsd_rlrt.md) | Pending browser submission | Unassigned / TBD | Not started |
| T3 | Base + ISO/MOPD + RLSD/RLRT | [Task](retraining/t3_base_iso_mopd_rlsd_rlrt.md) | Pending browser submission | Unassigned / TBD | Not started |
| T4 | Base + ISO/MOPD | [Task](retraining/t4_base_iso_mopd.md) | Pending browser submission | Unassigned / TBD | Not started |
| T5 | Base + SFT + GRPO | [Task](retraining/t5_base_sft_grpo.md) | Pending browser submission | Unassigned / TBD | Not started |

The five Issues have not yet been submitted. The coordinator can create them
without an API token or device code using the [browser creation guide](issue_creation.md).
Until the five Issues are linked above, these task contracts are the canonical
drafts; do not create duplicate Issues for the same paradigm.

`IOS` in the original request is normalized to `ISO`. Slash notation is not an
arm definition: each owner must state whether `RLSD/RLRT` and `ISO/MOPD` mean
separate arms, alternatives, or sequential stages before allocating compute.
Each concrete arm gets its own data/config/training-volume/checkpoint/B6 record.

All owners follow [the shared acceptance contract](retraining/task_acceptance.md)
and copy [the report template](report_template.md) into `retraining/` using the
filename specified in their task. Required completion evidence includes:

1. Arm/stage decomposition and justified checkpoint reuse.
2. Actual data revisions, resolved filtering/mixture, and log-reconciled training volume.
3. Code/backend revisions, complete resolved configs, seeds, hardware, and final checkpoints.
4. Training and seven-task B6 smoke checks, final-run/export status, and external evidence.
5. Seven complete avg@4 results with sample completeness and truncation counts/rates.
6. A compact report PR, independent review, and explicit blocked/invalid/incomplete outcomes.

A launched job, finite loss, generated checkpoint, or zero runner exit code alone
is not completion. An owner cannot be the sole reviewer of their own result.

## Training-volume and configuration alignment

Use the [comparison-cohort contract](training_alignment.md) before allocating
compute or comparing arms. It separates global rollout steps from optimizer
updates, records the current T1/T5 and T2/T3 numerical budgets, and identifies
intentional data/initialization differences and non-controlled cross-cohort
comparisons. Do not copy one cohort's settings into another or relabel existing
runs to manufacture comparability.

The PR #18 [RLSD/RLRT reproduction handoff](rlsd_rlrt_reproduction_20261124/)
provides the T2/T3 hybrid recipe and data-preparation tools. It is not a complete
training repository or evidence of completed GPU training.

## Branch and PR workflow

1. Create a task branch from this integration branch:

   ```bash
   git switch -c docs/midterm-t<N>-<owner> docs/midterm-report-202611
   ```

2. Copy the report template, fill the pre-registration tables before compute,
   and keep the Issue linked to current external logs/configs/checkpoints.
3. Open the report PR targeting `docs/midterm-report-202611`, not directly to
   `main`. Request independent review and link the PR in the Issue.
4. Update the task status here through the PR. Preserve negative, blocked,
   invalid, and incomplete results; do not silently remove unfavorable rows.
5. Consolidate only reviewed evidence into the midterm report. Close a task
   Issue after its report is merged and accepted, or record an explicit team
   decision and remaining owner for an accepted blocker.

Small blocking defects may get separate linked Issues/PRs, but they do not
replace one of the five research-task Issues or its evidence deliverables.

## Review and testing responsibilities

| Directory | Responsibility | Required record |
| --- | --- | --- |
| `review/` | Independent checkpoint, data/config, training-volume, scoring, metric, and validity review | Reviewer, immutable model/data/config revisions, scorer/judge versions, raw-output links, completeness/truncation assessment, decision |
| `retraining/` | Five task contracts and compact owner reports | Arm/stage graph, data/config provenance, log-reconciled training volume, final checkpoint, B6 and artifact links |
| `testing/` | Local/server smoke and regression checks | Commit/environment, exact command, expected/observed behavior, status, external evidence; distinguish local from GPU/server verification |

Training execution stays in the training repository. Reviewers must inspect
actual manifests/results and not infer correctness from successful startup.

## Shared candidate training inputs

The following public datasets were downloaded and validated: 20/20 shards each,
15,000 rows each, matching SHA-256/byte/row counts, unique `sample_id`, and
successful first/last image decoding. Use immutable revisions where an arm's
resolved config actually selects them; do not assume every arm uses either one.

| Dataset | Full revision | Rows / shards | `dataset` column counts |
| --- | --- | ---: | --- |
| `lzy-vlm-lab-opd/MOPD-route-v2-train-15k` | `6374961a4945f5ee7462520ecd54bcc93ee9d0bc` | 15,000 / 20 | `the_cauldron=9514`, `mmfinereason=5486` |
| `lzy-vlm-lab-opd/RL-RSTG-ZVP-train-15k` | `df005b52aead28acf57b05a1f30487a220d398bb` | 15,000 / 20 | `the_cauldron=9000`, `mmfinereason=6000` |

`source` is a finer-grained field and is not interchangeable with these
`dataset` counts. Every owner must still record their actual manifest hash,
filtering/sampling, epochs, rollout multiplicity, and optimizer steps.

## Current seven-task B6

Use `scripts/eval/run_pinned_eval.sh --protocol b6-mixed` at the reviewed
commit. The midterm B6 contains exactly:

| Stage | Tasks |
| --- | --- |
| v2 no-judge | MMMU-Pro, GQA |
| v1 no-judge | ReMI, ScienceQA, MindCube |
| v1 judged | MMBench, MathVerse |

All seven use four independent generations at temperature 1.0, seeds 42–45, and
report mean avg@4, not pass@4. ReMI formal numbers use `remi_reeval.py --mode
exact` with a complete 2,600-row denominator in every repeat. MMBench and
MathVerse need judge/scorer provenance. Every result needs four repeat metrics,
sample counts, truncation counts/rates, and external raw-output links.

Use a unique run name; the new default is `b6_midterm7_avg4`. Run a seven-task
smoke before full evaluation. DynaMath and ViewSpatial remain available in the
independent historical v2 protocol but are not midterm B6. Historical six-task
results are not equivalent to the new seven-task aggregate. Manifests record
the requested set and reject incompatible resume directories.

Full budgets, task versions, scorer rules, output locations, and Project15
boundaries are in [the canonical protocol](../pinned_eval_protocols_20260916.md).
Actual GPU evaluations and all five training outcomes remain pending owner
execution; this workspace currently defines process and verification, not
experimental results.
