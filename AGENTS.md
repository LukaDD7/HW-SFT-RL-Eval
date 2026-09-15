# Agent operating contract

All coding agents and human collaborators working in this repository must use
the same conventions.

## Repository boundaries

This repository is protocol and code only.

Do not add:

- secrets or tokens
- `.env` files
- model weights
- datasets, parquet files, or replay JSONL
- raw benchmark outputs
- logs, caches, or generated tarballs

Large assets belong in the Hugging Face repositories described by
`manifests/` and `assets/README.md`.

## Mandatory planning workflow

For any task expected to take more than one turn, use the
`planning-with-files` skill in `.agents/skills/planning-with-files/`.

The working plan is:

```text
.agents/plans/<task-slug>/
  task_plan.md
  findings.md
  progress.md
```

These files are local and ignored by Git.

At the start of a session:

1. locate the active plan;
2. read all three files before making changes;
3. continue the existing plan instead of inventing a new one.

During work:

1. update `findings.md` with evidence and decisions;
2. update `progress.md` after every meaningful command or test;
3. keep `task_plan.md` aligned with the actual remaining work.

Before declaring completion:

1. verify every unchecked item is either complete or explicitly blocked;
2. record test results;
3. inspect `git status --short` and `git diff --stat`;
4. confirm no data, weights, secrets, or raw outputs are staged.

## Evaluation-specific rules

Use `.agents/skills/hw-eval-operations/SKILL.md` for benchmark work. In
particular:

- B-segment v2 is the deterministic four-task protocol.
- ReMI formal numbers use `remi_reeval.py --mode exact`, full 2,600-row
  denominator.
- Do not report the old `normalized_exact_diagnostic` ReMI value.
- MV-MATH Project15 diagnostics use the strict completed-answer gate.
- Truncation rates are part of result validity, not optional metadata.

## Code quality

- Keep research and protocol logic in Python modules under `src/`.
- Keep task YAML under `eval_tasks/`.
- Keep configurations under `configs/`.
- Add or update tests for scoring and protocol changes.
- Run `pytest -q` before committing.
- Prefer environment variables over absolute paths.

## Git conventions

- Branch from `main`.
- Use concise, imperative commits.
- Do not rewrite published history.
- Do not commit directly to `main` for non-trivial changes; open a branch and
  a pull request.
