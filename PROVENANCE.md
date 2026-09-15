# Provenance

| Field | Value |
|---|---|
| Migration date | 2026-09-15 |
| Source repository | `LukaDD7/Dual-Track-OPD` |
| Source commit | `360c850` |
| History policy | Clean history; source commit recorded here |

## Migrated paths

- `src/dual_track_opd/eval`
- `configs/eval`
- `eval_tasks/opd_v1_offline`
- `eval_tasks/opd_v2`
- selected scripts under `scripts/eval`
- `scripts/sft_rl/remi_reeval.py`
- evaluation runner and dataset download scripts
- selected evaluation tests
- selected protocol and audit documents

## Excluded on purpose

- model checkpoints
- datasets and benchmark snapshots
- ReMI replay raw JSONL
- raw evaluation outputs
- logs and caches
- training and research code
- internal incident history not needed for protocol reproduction
