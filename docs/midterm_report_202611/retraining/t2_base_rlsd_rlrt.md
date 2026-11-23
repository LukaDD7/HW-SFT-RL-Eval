# T2 — Base + RLSD/RLRT

- Issue: not created yet; GitHub API authentication is required, then assign yourself.
- Owner: unassigned; independent reviewer: TBD.
- Status: not started.
- Branch: `docs/midterm-t2-<owner>` from `docs/midterm-report-202611`.

## Scope

Verify the Base checkpoint and actual RLSD/RLRT data/configuration, retrain or
justify reuse, and produce seven-task B6 results and a report. Before compute,
state whether `RLSD/RLRT` denotes two arms, alternative algorithms, or sequential
stages. Give each concrete arm its own data, configuration, training-volume,
checkpoint, and B6 record; do not merge them into an ambiguous single run.

## Required handoff

- [ ] Follow `task_acceptance.md`; publish the RLSD/RLRT arm decomposition and reuse graph before training.
- [ ] Reconcile per-arm data revisions, prompts/rollouts, batches, epochs, and optimizer steps with logs.
- [ ] Complete training smoke, final-run/export checks, and immutable per-arm artifact links.
- [ ] Run seven-task smoke and full B6 for every concrete arm, including strict ReMI and validity audits.
- [ ] Submit `t2_base_rlsd_rlrt_<yyyymmdd>.md` from `../report_template.md`, obtain independent review, and merge the report PR.

Completion is defined by `task_acceptance.md`, not by a launched job or finite loss.
