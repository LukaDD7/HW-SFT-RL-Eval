---
name: planning-with-files
description: Persistent file-based planning for long-running agent tasks and evaluation audits
---

# Planning with Files

Use this skill for any task that spans multiple turns, needs evidence tracking,
or can be interrupted.

## Files

Create a local, Git-ignored plan directory:

```text
.agents/plans/<task-slug>/
  task_plan.md
  findings.md
  progress.md
```

### `task_plan.md`

Use this structure:

```markdown
# Task plan

## Objective
<one sentence>

## Context
- source files
- protocol documents
- external assets

## Steps
- [ ] step

## Completion gate
- [ ] tests pass
- [ ] protocol reviewed
- [ ] no forbidden artifacts
- [ ] summary updated
```

### `findings.md`

Record facts with evidence:

```markdown
## Finding
- evidence:
- implication:
```

Never record opinions without evidence.

### `progress.md`

Append-only during a session:

```markdown
## <timestamp> — action
- command/result
- next action
```

## Session recovery

At the beginning of every turn:

1. If an active plan exists, read `task_plan.md`, `findings.md`, and
   `progress.md` before doing anything else.
2. Continue the existing objective. Do not start a fresh plan unless the user
   explicitly changed the goal.
3. Treat the files as the source of truth when the conversation context is
   compacted or cleared.

## Per-turn update

After each meaningful action, update `progress.md`. When new evidence changes
the approach, update `task_plan.md` and `findings.md`.

## Completion gate

Do not declare success until:

1. every completion-gate item is checked or explicitly marked blocked;
2. tests and required checks are recorded in `progress.md`;
3. `git status --short` and `git diff --stat` are inspected;
4. no secret, data, raw output, model, or large artifact is staged.

This skill is adapted from the MIT-licensed
`OthmanAdi/planning-with-files` project.
