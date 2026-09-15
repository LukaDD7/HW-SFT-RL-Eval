# Agent collaboration

This repository uses a single operating contract for all agents.

## Required files

- Root contract: `AGENTS.md`
- Persistent planning skill: `.agents/skills/planning-with-files/SKILL.md`
- Evaluation protocol skill: `.agents/skills/hw-eval-operations/SKILL.md`

Agents should discover skills through the standard `.agents/skills` layout.
If an agent does not support automatic skill discovery, read the `SKILL.md`
files directly at the start of the task.

## Long-running task workflow

1. Create `.agents/plans/<task-slug>/`.
2. Maintain `task_plan.md`, `findings.md`, and `progress.md`.
3. Re-read all three files after context loss or a new session.
4. Update progress after every significant command.
5. Run the completion gate before reporting success.

Plans are local and ignored by Git.

## Shared protocol rules

All agents must use:

- B-segment v2 for the deterministic four-task protocol;
- ReMI `--mode exact` full-denominator scoring;
- strict MV-MATH completed-answer reporting;
- truncation-aware validity labels;
- external Hugging Face assets for models and data.

No agent may silently reinterpret a benchmark protocol. Protocol changes must
be documented and tested.
