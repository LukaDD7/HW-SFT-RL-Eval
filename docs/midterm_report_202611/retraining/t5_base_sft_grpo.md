# T5 — Base + SFT + GRPO

- Issue: not created yet; GitHub API authentication is required, then assign yourself.
- Owner: unassigned; independent reviewer: TBD.
- Status: not started.
- Branch: `docs/midterm-t5-<owner>` from `docs/midterm-report-202611`.

## Scope

Verify Base, the SFT stage, and the subsequent GRPO stage as a sequential chain.
Retrain or justify reuse separately for SFT and GRPO, then evaluate the exact
final GRPO checkpoint on seven-task B6. Record SFT and GRPO data revisions and
training volumes separately, including the exported SFT parent used by GRPO and
GRPO prompt/rollout accounting. Do not assume the new 15k data applies to both
stages without resolved-config evidence.

## Required handoff

- [ ] Follow `task_acceptance.md`; publish the Base→SFT→GRPO checkpoint graph and stage-reuse decisions before training.
- [ ] Reconcile each stage's data revisions, samples/prompts/rollouts, batches, epochs, and optimizer steps with logs.
- [ ] Complete per-stage smoke/final/export checks and link immutable intermediate/final artifacts.
- [ ] Run seven-task smoke and full B6 on the verified final checkpoint, including strict ReMI and validity audits.
- [ ] Submit `t5_base_sft_grpo_<yyyymmdd>.md` from `../report_template.md`, obtain independent review, and merge the report PR.

Completion is defined by `task_acceptance.md`, not by a launched job or finite loss.
