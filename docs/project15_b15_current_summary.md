# Project15 B15 current usable results

This table records the currently usable Project15 values for the three arms
that have completed or partially completed evaluation. Missing values are
shown as `–`.

The `Avg` column is the mean of the available values after normalizing the
0–100 metrics (`MathVista`, `MMBench`, `MMVet`) to the 0–1 scale. It is not
directly comparable across rows with different coverage.

| Arm | VS | MC | GQA | VQA | SQA | DM | MVerse | MVista | MMSI | BLINK | MMB | MMMU | MMVet | ReMI | MVM | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base 8B Instruct (15/15) | 0.4231 | 0.2975 | 0.6164 | 0.8182 | 0.9465 | 0.6279 | 0.5236 | 72.7 | 0.3070 | 0.6512 | 85.1375 | 0.3983 | 74.2661 | 0.2792 | 0.3579 | 0.5712 |
| ISO-MOPD step300 (7/15) | – | 0.1684 | – | – | – | – | 0.6360 | 77.5 | 0.2830 | 0.6218 | – | – | 75.9679 | – | 0.3917 | 0.5194 |
| Pure RL step100 (10/15) | 0.4462 | – | 0.5445 | – | – | 0.6691 | 0.6058 | 75.2 | – | – | 83.9777 | 0.5513 | 76.9266 | 0.4861 | 0.3659 | 0.6030 |

## Notes

- `Base` values come from the completed Project15-v1 baseline.
- `ISO-MOPD step300` values come from the tag-fixed Project15 complement run;
  VQAv2 and ScienceQA are still pending.
- `Pure RL step100` values combine the completed B6 arm with the currently
  usable Project15 complement values; the remaining complement benchmarks are
  pending.
