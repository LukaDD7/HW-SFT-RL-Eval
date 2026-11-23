# T3 — Base + ISO/MOPD + RLSD/RLRT

- Issue: not created yet; GitHub API authentication is required, then assign yourself.
- Owner: unassigned; independent reviewer: TBD.
- Status: not started.
- Branch: `docs/midterm-t3-<owner>` from `docs/midterm-report-202611`.

## Scope

Verify the complete stage chain from Base through ISO/MOPD to RLSD/RLRT, retrain
or justify reuse of each stage, and evaluate every concrete final arm on the
seven-task B6. Before compute, resolve both slash notations explicitly: separate
arms versus sequential stages, stage ordering, all parent/output checkpoints,
and whether combinations are separate runs. Do not infer a single pipeline from
the shorthand.

## Required handoff

- [ ] Follow `task_acceptance.md`; publish the stage/arm graph, checkpoint provenance, and reuse decisions before training.
- [ ] Reconcile each stage's data revisions, prompts/rollouts, batches, epochs, and optimizer steps with logs.
- [ ] Complete per-stage smoke/final checks and link immutable intermediate/final checkpoints, configs, and logs.
- [ ] Run seven-task smoke and full B6 for every concrete final arm, including strict ReMI and validity audits.
- [ ] Submit `t3_base_iso_mopd_rlsd_rlrt_<yyyymmdd>.md` from `../report_template.md`, obtain independent review, and merge the report PR.

Completion is defined by `task_acceptance.md`, not by a launched job or finite loss.
