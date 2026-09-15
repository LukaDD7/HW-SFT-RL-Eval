# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Branches

Use short, descriptive branches:

```text
eval/fix-remi-denominator
docs/clarify-v2-protocol
ci/add-protocol-regression
```

## Commits

Use imperative commit subjects:

```text
fix(eval): enforce full ReMI denominator
docs(protocol): document strict MV-MATH gate
```

## Pull requests

Before requesting review:

1. run `pytest -q`;
2. run `git diff --check`;
3. inspect `git status --short`;
4. confirm no forbidden artifacts are included.

Forbidden artifacts include secrets, weights, datasets, raw outputs, logs,
caches, and tarballs.

## Protocol changes

Protocol changes require:

1. a short rationale;
2. updated tests;
3. updated protocol documentation;
4. explicit note on comparability with prior runs.
