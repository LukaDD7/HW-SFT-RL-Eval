# Project15 B15 current usable results

This table separates latest-protocol usable values from historical reference
rows. It was updated after the 2026-10-20 GPU reclamation.

The mean normalizes 0–100 metrics to 0–1 and averages available values only.
Because coverage differs, these means are diagnostic, not final B15 rankings.

| Arm | VS | MC | GQA | VQA | SQA | DM | MVerse | MVista | MMSI | BLINK | MMB | MMMU | MMVet | ReMI | MVM | Usable | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ISO-MOPD step300 | legacy v1 | 0.1684 | 0.5926 | pending | 0.8096 | legacy v1 | 0.6360 | 77.5 | 0.2830 | 0.6218 | 87.2852 | legacy v1 | 75.9679 | 0.5365 | 0.3917 | 11/15 | 0.5861 |
| Pure RL step100 | 0.4459 | 0.2268 | 0.5454 | pending | 0.8171 | 0.6691 | 0.6058 | 75.2 | 0.2900 | 0.6260 | 84.6220 | 0.5462 | 76.9266 | 0.4892 | 0.3659 | 14/15 | 0.5711 |
| VA-OPD-RL step200 | historical | historical | historical | historical | historical | historical | historical | historical | historical | historical | historical | historical | historical | historical | historical | not current | — |
| Base 8B Instruct | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | reference | — |

## Pending under the latest protocol

- ISO-MOPD: VQAv2, ViewSpatial v2, DynaMath v2, MMMU-Pro v2.
- Pure RL: VQAv2 only.
- VA-OPD-RL: the existing row is historical auto/native generation and is not
  a unified explicit-think seed43 comparison.

## Deprecated values

- ISO ViewSpatial, DynaMath, and MMMU-Pro entries marked `legacy v1` must not
  be used in the current comparison.
- Pure RL ScienceQA `0.0` was the old v1 parser artifact; the current
  answer-tag rescore is `0.8170550322260783`.
- ReMI must use `exact_accuracy_full_denominator`; the old
  `normalized_exact_diagnostic` is unusable.
- MMBench must use the PR12 rescore where applicable; raw legacy MMBench
  scores must not be mixed with it.
- avg@4 and seed43/repeat1 values must not be mixed.

The machine-readable protocol status is in
`docs/project15_seed43_protocol_status_20261020.json`.

