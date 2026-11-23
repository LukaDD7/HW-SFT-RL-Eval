# T4 — Base + ISO/MOPD

- Issue: not created yet; GitHub API authentication is required, then assign yourself.
- Owner: unassigned; independent reviewer: TBD.
- Status: not started.
- Branch: `docs/midterm-t4-<owner>` from `docs/midterm-report-202611`.

The request's `IOS` is normalized to the repository's existing `ISO` name.

## Scope

Verify Base and the actual ISO/MOPD data/configuration, retrain or justify reuse,
and produce seven-task B6 results and a report for each concrete arm. Before
compute, state whether `ISO/MOPD` denotes separate arms, alternative algorithms,
or sequential stages. Record each stage's parent and final checkpoint and its
own data/config/training volume; do not silently combine ISO and MOPD.

## Required handoff

- [ ] Follow `task_acceptance.md`; publish the ISO/MOPD arm decomposition and checkpoint reuse graph before training.
- [ ] Reconcile per-arm data revisions, samples/prompts, batches, epochs, and optimizer steps with logs.
- [ ] Complete training smoke, final-run/export checks, and immutable per-arm artifact links.
- [ ] Run seven-task smoke and full B6 for every concrete arm, including strict ReMI and validity audits.
- [ ] Submit `t4_base_iso_mopd_<yyyymmdd>.md` from `../report_template.md`, obtain independent review, and merge the report PR.

Completion is defined by `task_acceptance.md`, not by a launched job or finite loss.
