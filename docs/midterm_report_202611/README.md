# 2026-11 midterm report workspace

Integration branch: `docs/midterm-report-202611`.

This workspace coordinates compact review, retraining, and testing summaries.
It is not an artifact store. Keep datasets, model weights, raw outputs, logs,
archives, and training code outside this repository. Link to immutable Hugging
Face revisions or other external artifacts instead of copying them into Git.

## Workstreams

| Directory | Scope | Required record |
| --- | --- | --- |
| `review/` | Checkpoint identity, benchmark protocol, scoring, metric review, and result validity | Owner, model revision, protocol/config revisions, scorer/environment versions, external raw-output links, validity and truncation assessment |
| `retraining/` | Retraining provenance and compact outcome summaries | Owner, objective, training repository and backend commits, resolved config, data/model revisions, seeds, hardware, status, and external log/checkpoint links |
| `testing/` | Reproducibility checks and regression tests | Owner, repository commit, environment, exact command, expected and observed behavior, status, and external evidence links |

Training execution remains in the training repository. Only provenance and
compact summaries belong under `retraining/`.

## Collaboration workflow

1. Create a short workstream branch from the integration branch:

   ```bash
   git switch -c docs/midterm-<workstream>-<owner> docs/midterm-report-202611
   ```

2. Add one compact Markdown summary per work item under the appropriate
   directory. Use `<topic>_<yyyymmdd>.md` as the filename pattern.
3. Open a pull request targeting `docs/midterm-report-202611`, request an
   independent review, and update the status table below.
4. Consolidate only verified results into the midterm report. Mark missing,
   blocked, invalid, and still-running work explicitly; do not silently drop
   unfavorable or incomplete results.

## Status

| Workstream | Owner | Entry | Status |
| --- | --- | --- | --- |
| Review | TBD | `review/` | Not started |
| Retraining | TBD | `retraining/` | Not started |
| Testing | TBD | `testing/` | Not started |

## Shared training inputs

Use these immutable revisions for the new-data retraining and review arms:

| Stage | Hugging Face dataset | Revision | Rows |
| --- | --- | --- | ---: |
| MOPD route-v2 | `lzy-vlm-lab-opd/MOPD-route-v2-train-15k` | `6374961a4945f5ee7462520ecd54bcc93ee9d0bc` | 15,000 |
| RL RSTG + RL-ZVP | `lzy-vlm-lab-opd/RL-RSTG-ZVP-train-15k` | `df005b52aead28acf57b05a1f30487a220d398bb` | 15,000 |

Record the full revision in resolved configs and run manifests. Do not
substitute a mutable branch such as `main` when reproducing these arms.

## Result validity

- Use `scripts/eval/run_pinned_eval.sh` as the canonical evaluation entrypoint.
- ReMI formal numbers require `remi_reeval.py --mode exact` and the full
  2,600-row denominator.
- MV-MATH Project15 diagnostics require the strict completed-answer gate.
- Report truncation rates and invalid or incomplete runs alongside metrics.
- Record the checkpoint, resolved config, code/backend revisions, dataset
  manifest hash, and external raw-output location for every reported result.
