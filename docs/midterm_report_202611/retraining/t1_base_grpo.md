# T1 — Base + GRPO

- Issue: not created yet; GitHub API authentication is required, then assign yourself.
- Owner: unassigned; independent reviewer: TBD.
- Status: not started.
- Branch: `docs/midterm-t1-<owner>` from `docs/midterm-report-202611`.

## Scope

Verify the Base checkpoint and the actual GRPO data/configuration, retrain or
justify reuse, and produce the seven-task B6 evaluation and report. Record the
prompt dataset, rollout count/group semantics, batch/step calculation, KL and
other resolved algorithm parameters, and final optimizer step. Do not assume
that either new 15k dataset was used without evidence from the resolved config.

## Required handoff

- [ ] Follow the pre-registration in `task_acceptance.md`; record concrete arm ID and checkpoint reuse.
- [ ] Reconcile data revisions, prompt/rollout counts, batches, epochs, and optimizer steps with logs.
- [ ] Complete training smoke, final-run/export checks, and immutable checkpoint/config/log links.
- [ ] Run seven-task smoke and full B6, including strict ReMI and truncation/completeness audits.
- [ ] Submit `t1_base_grpo_<yyyymmdd>.md` from `../report_template.md`, obtain independent review, and merge the report PR.

Completion is defined by `task_acceptance.md`, not by a launched job or finite loss.
